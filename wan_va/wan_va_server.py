# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
import argparse
import os
import sys
import time
from functools import partial
from PIL import Image
from diffusers.video_processor import VideoProcessor
from diffusers.utils import export_to_video

import numpy as np
import torch
import torch.nn.functional as F
from diffusers.pipelines.wan.pipeline_wan import prompt_clean
from einops import rearrange
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from configs import VA_CONFIGS
from distributed.fsdp import shard_model
from distributed.util import _configure_model, init_distributed
from modules.utils import (
    WanVAEStreamingWrapper,
    WanVAEStreamingDecoder,
    load_text_encoder,
    load_tokenizer,
    load_transformer,
    load_vae,
)
from utils import (
    FlowMatchScheduler,
    data_seq_to_patch,
    get_mesh_id,
    init_logger,
    logger,
    run_async_server_mode,
    save_async,
)


class VA_Server:

    def __init__(self, job_config):
        self.cache_name = 'pos'
        self.job_config = job_config
        self.save_root = job_config.save_root
        self.dtype = job_config.param_dtype
        self.device = torch.device(f"cuda:{job_config.local_rank}")
        self.enable_offload = getattr(job_config, 'enable_offload', True)  # offload vae & text_encoder to save vram

        self.scheduler = FlowMatchScheduler(shift=self.job_config.snr_shift,
                                            sigma_min=0.0,
                                            extra_one_step=True)
        self.action_scheduler = FlowMatchScheduler(
            shift=self.job_config.action_snr_shift,
            sigma_min=0.0,
            extra_one_step=True)
        self.scheduler.set_timesteps(1000, training=True)
        self.action_scheduler.set_timesteps(1000, training=True)

        self.vae = load_vae(
            os.path.join(job_config.wan22_pretrained_model_name_or_path,
                         'vae'),
            torch_dtype=self.dtype,
            torch_device='cpu' if self.enable_offload else self.device,
        )
        self.streaming_vae = WanVAEStreamingWrapper(self.vae)
        # Decoder-side counterpart, so the imagined video is one continuous
        # stream rather than a cold 5-frame restart per chunk. Costs a constant
        # ~418 MiB of causal-conv cache; see WanVAEStreamingDecoder.
        self.streaming_decoder = WanVAEStreamingDecoder(self.vae)
        # Also set in _reset; initialised here so any path that reaches
        # _compute_kv_cache without a preceding reset does not AttributeError.
        self._video_stream_active = False
        self.last_video_gen_ms = 0.0
        self.last_action_gen_ms = 0.0
        self.last_real_decode_ms = 0.0
        self.video_processor = VideoProcessor(vae_scale_factor=1)

        self.tokenizer = load_tokenizer(
            os.path.join(job_config.wan22_pretrained_model_name_or_path,
                         'tokenizer'), )

        self.text_encoder = load_text_encoder(
            os.path.join(job_config.wan22_pretrained_model_name_or_path,
                         'text_encoder'),
            torch_dtype=self.dtype,
            torch_device='cpu' if self.enable_offload else self.device,
        )

        self.transformer = load_transformer(
            os.path.join(job_config.wan22_pretrained_model_name_or_path,
                         'transformer'),
            torch_dtype=self.dtype,
            torch_device=self.device,
            attn_mode="torch"
        )
        shard_fn = shard_model
        self.transformer = _configure_model(model=self.transformer,
                                            shard_fn=shard_fn,
                                            param_dtype=self.dtype,
                                            device=self.device,
                                            eval_mode=True,
                                            )

        self.env_type = job_config.env_type
        self.streaming_vae_half = None
        if self.env_type == 'robotwin_tshape':
            vae_half = load_vae(
                os.path.join(job_config.wan22_pretrained_model_name_or_path,
                             'vae'),
                torch_dtype=self.dtype,
                torch_device='cpu' if self.enable_offload else self.device,
            )
            self.streaming_vae_half = WanVAEStreamingWrapper(vae_half)

    def _get_t5_prompt_embeds(
        self,
        prompt=None,
        num_videos_per_prompt=1,
        max_sequence_length=512,
        device=None,
        dtype=None,
    ):
        device = device or self.device
        dtype = dtype or self.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt = [prompt_clean(u) for u in prompt]
        batch_size = len(prompt)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
        seq_lens = mask.gt(0).sum(dim=1).long()

        text_encoder_device = next(self.text_encoder.parameters()).device
        prompt_embeds = self.text_encoder(text_input_ids.to(text_encoder_device),
                                          mask.to(text_encoder_device)).last_hidden_state
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
        prompt_embeds = torch.stack([
            torch.cat(
                [u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))])
            for u in prompt_embeds
        ],
                                    dim=0)

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt,
                                           seq_len, -1)

        return prompt_embeds.to(device)

    def encode_prompt(
        self,
        prompt,
        negative_prompt=None,
        do_classifier_free_guidance=True,
        num_videos_per_prompt=1,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        max_sequence_length=226,
        device=None,
        dtype=None,
    ):
        r"""
        TODO
        """
        device = device or self.device
        dtype = dtype or self.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_embeds = self._get_t5_prompt_embeds(
                prompt=prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt = batch_size * [negative_prompt] if isinstance(
                negative_prompt, str) else negative_prompt

            if prompt is not None and type(prompt) is not type(
                    negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}.")
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`.")

            negative_prompt_embeds = self._get_t5_prompt_embeds(
                prompt=negative_prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )
        return prompt_embeds, negative_prompt_embeds

    def normalize_latents(
        self,
        latents: torch.Tensor,
        latents_mean: torch.Tensor,
        latents_std: torch.Tensor,
    ) -> torch.Tensor:
        latents_mean = latents_mean.view(1, -1, 1, 1,
                                         1).to(device=latents.device)
        latents_std = latents_std.view(1, -1, 1, 1,
                                       1).to(device=latents.device)
        latents = ((latents.float() - latents_mean) * latents_std).to(latents)
        return latents

    def preprocess_action(self, action):
        action_model_input = torch.from_numpy(action)
        CA, FA, HA = action_model_input.shape  # C, F, H
        action_model_input_paded = F.pad(action_model_input,
                                         [0, 0, 0, 0, 0, 1],
                                         mode='constant',
                                         value=0)

        action_model_input = action_model_input_paded[
            self.job_config.inverse_used_action_channel_ids]

        if self.action_norm_method == 'quantiles':
            action_model_input = (action_model_input - self.actions_q01) / (
                self.actions_q99 - self.actions_q01 + 1e-6) * 2. - 1.
        else:
            raise NotImplementedError
        return action_model_input.unsqueeze(0).unsqueeze(-1)  # B, C, F, H, W

    def postprocess_action(self, action):
        action = action.cpu()  # B, C, F, H, W

        action = action[0, ..., 0]  #C, F, H
        if self.action_norm_method == 'quantiles':
            action = (action + 1) / 2 * (self.actions_q99 - self.actions_q01 +
                                         1e-6) + self.actions_q01
        else:
            raise NotImplementedError
        action = action.squeeze(0).detach().cpu().numpy()
        return action[self.job_config.used_action_channel_ids]
    
    def _repeat_input_for_cfg(self, input_dict):
        if self.use_cfg:
            input_dict['noisy_latents'] = input_dict['noisy_latents'].repeat(2, 1, 1, 1, 1)
            input_dict['text_emb'] = torch.cat([self.prompt_embeds.to(self.dtype).clone(), self.negative_prompt_embeds.to(self.dtype).clone()], dim=0)
            input_dict['grid_id'] = input_dict['grid_id'][None].repeat(2, 1, 1)
            input_dict['timesteps'] = input_dict['timesteps'][None].repeat(2, 1)
        else:
            input_dict['grid_id'] = input_dict['grid_id'][None]
            input_dict['timesteps'] = input_dict['timesteps'][None]
        return input_dict

    def _prepare_latent_input(self,
                              latent_model_input,
                              action_model_input,
                              latent_t=0,
                              action_t=0,
                              latent_cond=None,
                              action_cond=None,
                              frame_st_id=0,
                              patch_size=(1, 2, 2)):
        logger.info(f"FRAME START ID: {frame_st_id}")
        input_dict = dict()
        if latent_model_input is not None:
            input_dict['latent_res_lst'] = {
                'noisy_latents':
                latent_model_input,
                'timesteps':
                torch.ones([latent_model_input.shape[2]],
                           dtype=torch.float32,
                           device=self.device) * latent_t,
                'grid_id':
                get_mesh_id(latent_model_input.shape[-3] // patch_size[0],
                            latent_model_input.shape[-2] // patch_size[1],
                            latent_model_input.shape[-1] // patch_size[2], 0,
                            1, frame_st_id).to(self.device),
                'text_emb':
                self.prompt_embeds.to(self.dtype).clone(),
            }
            if latent_cond is not None:
                input_dict['latent_res_lst'][
                    'noisy_latents'][:, :, 0:1] = latent_cond[:, :, 0:1]
                input_dict['latent_res_lst']['timesteps'][0:1] *= 0

        if action_model_input is not None:
            input_dict['action_res_lst'] = {
                'noisy_latents':
                action_model_input,
                'timesteps':
                torch.ones([action_model_input.shape[2]],
                           dtype=torch.float32,
                           device=self.device) * action_t,
                'grid_id':
                get_mesh_id(action_model_input.shape[-3],
                            action_model_input.shape[-2],
                            action_model_input.shape[-1],
                            1,
                            1,
                            frame_st_id,
                            action=True).to(self.device),
                'text_emb':
                self.prompt_embeds.to(self.dtype).clone(),
            }

            if action_cond is not None:
                input_dict['action_res_lst'][
                    'noisy_latents'][:, :, 0:1] = action_cond[:, :, 0:1]
                input_dict['action_res_lst']['timesteps'][0:1] *= 0
            input_dict['action_res_lst']['noisy_latents'][:, ~self.
                                                          action_mask] *= 0
        return input_dict

    def _encode_obs(self, obs):
        images = obs['obs']
        if not isinstance(images, list):
            images = [images]
        if len(images) < 1:
            return None
        videos = []
        for k_i, k in enumerate(self.job_config.obs_cam_keys):
            if self.env_type == 'robotwin_tshape':
                if k_i == 0:  # camera high
                    height_i, width_i = self.height, self.width
                else:
                    height_i, width_i = self.height // 2, self.width // 2
            else:
                height_i, width_i = self.height, self.width

            history_video_k = torch.from_numpy(
                np.stack([each[k]
                          for each in images])).float().permute(3, 0, 1, 2)
            history_video_k = F.interpolate(history_video_k,
                                            size=(height_i, width_i),
                                            mode='bilinear',
                                            align_corners=False).unsqueeze(0)
            videos.append(history_video_k)

        if self.env_type == 'robotwin_tshape':
            videos_high = videos[0] / 255.0 * 2.0 - 1.0
            videos_left_and_right = torch.cat(videos[1:],
                                              dim=0) / 255.0 * 2.0 - 1.0
            vae_device = next(self.streaming_vae.vae.parameters()).device
            enc_out_high = self.streaming_vae.encode_chunk(
                videos_high.to(vae_device).to(self.dtype))
            enc_out_left_and_right = self.streaming_vae_half.encode_chunk(
                videos_left_and_right.to(vae_device).to(self.dtype))
            enc_out = torch.cat([
                torch.cat(enc_out_left_and_right.split(1, dim=0), dim=-1),
                enc_out_high
            ],
                                dim=-2)
        else:
            videos = torch.cat(videos, dim=0) / 255.0 * 2.0 - 1.0
            vae_device = next(self.streaming_vae.vae.parameters()).device
            videos_chunk = videos.to(vae_device).to(self.dtype)
            enc_out = self.streaming_vae.encode_chunk(videos_chunk)

        mu, logvar = torch.chunk(enc_out, 2, dim=1)
        latents_mean = torch.tensor(self.vae.config.latents_mean).to(mu.device)
        latents_std = torch.tensor(self.vae.config.latents_std).to(mu.device)
        mu_norm = self.normalize_latents(mu, latents_mean, 1.0 / latents_std)
        video_latent = torch.cat(mu_norm.split(1, dim=0), dim=-1)
        return video_latent.to(self.device)

    def _reset(self, prompt=None):
        logger.info('Reset.')
        self.use_cfg = (self.job_config.guidance_scale > 1) or (self.job_config.action_guidance_scale > 1)
        #### Reset all parameters
        self.frame_st_id = 0
        self.init_latent = None
        #### clean vae and transformer cache
        self.transformer.clear_cache(self.cache_name)
        self.streaming_vae.clear_cache()
        self.streaming_decoder.clear_cache()
        # Advancing the decoder's real stream costs a second VAE decode per chunk, so
        # it is only done once a client has actually asked for video.
        self._video_stream_active = False
        self.last_video_gen_ms = 0.0
        self.last_action_gen_ms = 0.0
        self.last_real_decode_ms = 0.0

        self.action_per_frame = self.job_config.action_per_frame
        self.height, self.width = self.job_config.height, self.job_config.width

        if self.env_type == 'robotwin_tshape':
            self.latent_height, self.latent_width = (
                (self.height // 16) * 3) // 2, self.width // 16
            self.streaming_vae_half.clear_cache()
        else:
            self.latent_height, self.latent_width = self.height // 16, self.width // 16 * len(
                self.job_config.obs_cam_keys)

        patch_size = self.job_config.patch_size
        latent_token_per_chunk = (self.job_config.frame_chunk_size *
                                  self.latent_height * self.latent_width) // (
                                      patch_size[0] * patch_size[1] *
                                      patch_size[2])
        action_token_per_chunk = self.job_config.frame_chunk_size * self.action_per_frame
        self.transformer.create_empty_cache(self.cache_name,
                                            self.job_config.attn_window,
                                            latent_token_per_chunk,
                                            action_token_per_chunk,
                                            dtype=self.dtype,
                                            device=self.device,
                                            batch_size = 2 if self.use_cfg else 1
                                            )

        self.action_mask = torch.zeros([self.job_config.action_dim]).bool()
        self.action_mask[self.job_config.used_action_channel_ids] = True

        self.actions_q01 = torch.tensor(self.job_config.norm_stat['q01'],
                                        dtype=torch.float32).reshape(-1, 1, 1)
        self.actions_q99 = torch.tensor(self.job_config.norm_stat['q99'],
                                        dtype=torch.float32).reshape(-1, 1, 1)
        self.action_norm_method = self.job_config.action_norm_method

        # Latent frame 0's actions, as training builds them: _action_post_process pads
        # frame 0 with RAW zeros and only then normalizes and clips. Normalized zeros are
        # something else -- the midpoint of every q01..q99 range, which on YAM's one-sided
        # relative-pose ranges reads as both arms ~18 cm into a reach, and the first chunk
        # starts by moving both arms toward it. Always restored over frame 0 of the
        # history fed back by the client; the first chunk's action condition only under
        # frame0_action='train_pad' (see shared_config). The clip is needed only here,
        # and only for EEF actions: raw zero quaternion w normalizes to -8.5, which
        # training clips to -1.5 (joint actions' raw zeros all fall inside it). The
        # fed-back predictions themselves are left unclipped.
        if self.job_config.frame0_action not in ('denoised', 'train_pad'):
            raise ValueError(f"frame0_action must be 'denoised' or 'train_pad', got "
                             f"{self.job_config.frame0_action!r}")
        self.frame0_pad = self.preprocess_action(
            np.zeros((len(self.job_config.used_action_channel_ids), 1,
                      self.action_per_frame), dtype=np.float32)).clamp(-1.5, 1.5)
        self.frame0_pad[:, ~self.action_mask] = 0

        ##### get prompt
        if prompt is None:
            self.prompt_embeds = self.negative_prompt_embeds = None
        else:
            self.prompt_embeds, self.negative_prompt_embeds = self.encode_prompt(
                prompt=prompt,
                negative_prompt=None,
                do_classifier_free_guidance=self.job_config.guidance_scale > 1,
                num_videos_per_prompt=1,
                prompt_embeds=None,
                negative_prompt_embeds=None,
                max_sequence_length=512,
                device=self.device,
                dtype=self.dtype,
            )

        self.exp_name = f"{prompt}_{time.strftime('%Y%m%d_%H%M%S')}" if prompt else "default"
        self.exp_save_root = os.path.join(self.save_root, 'real', self.exp_name)
        os.makedirs(self.exp_save_root, exist_ok=True)
        torch.cuda.empty_cache()

    def _infer(self, obs, frame_st_id=0):
        frame_chunk_size = self.job_config.frame_chunk_size
        if frame_st_id == 0:
            init_latent = self._encode_obs(obs)
            self.init_latent = init_latent

        latents = torch.randn(1,
                              48,
                              frame_chunk_size,
                              self.latent_height,
                              self.latent_width,
                              device=self.device,
                              dtype=self.dtype)
        actions = torch.randn(1,
                              self.job_config.action_dim,
                              frame_chunk_size,
                              self.action_per_frame,
                              1,
                              device=self.device,
                              dtype=self.dtype)

        video_inference_step = self.job_config.num_inference_steps
        action_inference_step = self.job_config.action_num_inference_steps
        video_step = self.job_config.video_exec_step

        self.scheduler.set_timesteps(video_inference_step)
        self.action_scheduler.set_timesteps(action_inference_step)
        timesteps = self.scheduler.timesteps
        action_timesteps = self.action_scheduler.timesteps

        timesteps = F.pad(timesteps, (0, 1), mode='constant', value=0)

        if video_step != -1:
            timesteps = timesteps[:video_step]

        action_timesteps = F.pad(
            action_timesteps,
            (0,
             1),  # pad 1 element at the end (right side) of the last dimension
            mode='constant',
            value=0)

        with (
                torch.no_grad(),
        ):
            # 1. Video Generation Loop
            # Timed separately from the action loop below. They are different models'
            # worth of work under two independent knobs -- num_inference_steps and
            # action_num_inference_steps -- and lumping them into one "infer" number
            # hides which of the two a latency change came from. Note this loop runs
            # even when no video was requested: the action loop attends to the latents
            # it produces, so it is not optional work.
            torch.cuda.synchronize()
            t_vgen = time.perf_counter()
            for i, t in enumerate(tqdm(timesteps)):
                last_step = i == len(timesteps) - 1
                latent_cond = init_latent[:, :, 0:1].to(
                    self.dtype) if frame_st_id == 0 else None
                input_dict = self._prepare_latent_input(
                    latents,
                    None,
                    t,
                    t,
                    latent_cond,
                    None,
                    frame_st_id=frame_st_id)

                video_velocity_pred = self.transformer(
                    self._repeat_input_for_cfg(input_dict['latent_res_lst']),
                    update_cache=1 if last_step else 0,
                    cache_name=self.cache_name,
                    action_mode=False)

                if not last_step or video_step != -1:
                    video_velocity_pred = data_seq_to_patch(
                        self.job_config.patch_size, video_velocity_pred,
                        frame_chunk_size, self.latent_height,
                        self.latent_width, batch_size=2 if self.use_cfg else 1)
                    if self.job_config.guidance_scale > 1:
                        video_velocity_pred = video_velocity_pred[1:] + self.job_config.guidance_scale * (video_velocity_pred[:1] - video_velocity_pred[1:])
                    else:
                        video_velocity_pred = video_velocity_pred[:1]
                    latents = self.scheduler.step(video_velocity_pred,
                                                  t,
                                                  latents,
                                                  return_dict=False)

                latents[:, :, 0:1] = latent_cond if frame_st_id == 0 else latents[:, :, 0:1]

            # 2. Action Generation Loop
            torch.cuda.synchronize()
            self.last_video_gen_ms = (time.perf_counter() - t_vgen) * 1e3
            t_agen = time.perf_counter()
            for i, t in enumerate(tqdm(action_timesteps)):
                last_step = i == len(action_timesteps) - 1
                action_cond = self.frame0_pad.to(
                    device=self.device,
                    dtype=self.dtype) if (
                        frame_st_id == 0
                        and self.job_config.frame0_action == 'train_pad') else None

                input_dict = self._prepare_latent_input(
                    None,
                    actions,
                    t,
                    t,
                    None,
                    action_cond,
                    frame_st_id=frame_st_id)
                action_velocity_pred = self.transformer(
                    self._repeat_input_for_cfg(input_dict['action_res_lst']),
                    update_cache=1 if last_step else 0,
                    cache_name=self.cache_name,
                    action_mode=True)

                if not last_step:
                    action_velocity_pred = rearrange(action_velocity_pred,
                                                  'b (f n) c -> b c f n 1',
                                                  f=frame_chunk_size)
                    if self.job_config.action_guidance_scale > 1:
                        action_velocity_pred = action_velocity_pred[1:] + self.job_config.action_guidance_scale * (action_velocity_pred[:1] - action_velocity_pred[1:])
                    else:
                        action_velocity_pred = action_velocity_pred[:1]
                    actions = self.action_scheduler.step(action_velocity_pred,
                                                         t,
                                                         actions,
                                                         return_dict=False)

                if action_cond is not None:
                    actions[:, :, 0:1] = action_cond

            torch.cuda.synchronize()
            self.last_action_gen_ms = (time.perf_counter() - t_agen) * 1e3

        actions[:, ~self.action_mask] *= 0

        save_async(latents, os.path.join(self.exp_save_root, f'latents_{frame_st_id}.pt'))
        save_async(actions, os.path.join(self.exp_save_root, f'actions_{frame_st_id}.pt'))

        actions = self.postprocess_action(actions)
        torch.cuda.empty_cache()
        return actions, latents

    def _compute_kv_cache(self, obs):
        ### optional async save obs for debug
        self.transformer.clear_pred_cache(self.cache_name)
        save_async(obs['obs'], os.path.join(self.exp_save_root, f'obs_data_{self.frame_st_id}.pt'))
        latent_model_input = self._encode_obs(obs)
        if self.frame_st_id == 0:
            latent_model_input = torch.cat(
                [self.init_latent, latent_model_input],
                dim=2) if latent_model_input is not None else self.init_latent

        self.last_real_decode_ms = 0.0
        if self._video_stream_active:
            # The second decode per chunk, and the reason the video is honest: this
            # advances the decoder's causal cache with what the robot ACTUALLY saw, so
            # the next chunk's imagined frames are rendered as a continuation of reality
            # -- the same history the transformer is conditioned on, since the
            # clear_pred_cache above just evicted the previous chunk's prediction. The
            # pixels are thrown away; the cache side effect is the whole point.
            #
            # Timed and reported so it lands in the client's `vid` rather than in `adv`.
            # It exists only because --video is on, and leaving it inside the advance
            # would make "refresh the KV cache with the real observation" mean two
            # different things depending on a flag.
            torch.cuda.synchronize()
            t_rdec = time.perf_counter()
            self.streaming_decoder.decode_chunk(latent_model_input)
            torch.cuda.synchronize()
            self.last_real_decode_ms = (time.perf_counter() - t_rdec) * 1e3

        action_model_input = self.preprocess_action(obs['state'])
        if self.frame_st_id == 0:
            # The client hands frame 0 back after a bf16 round trip through
            # denormalization; put back exactly what training holds there.
            action_model_input[:, :, 0:1] = self.frame0_pad
        action_model_input = action_model_input.to(latent_model_input)
        logger.info(
            f"get KV cache obs: {latent_model_input.shape} {action_model_input.shape}"
        )
        input_dict = self._prepare_latent_input(latent_model_input,
                                                action_model_input,
                                                frame_st_id=self.frame_st_id)

        with (
                torch.no_grad(),
        ):
            self.transformer(self._repeat_input_for_cfg(input_dict['latent_res_lst']),
                             update_cache=2,
                             cache_name=self.cache_name,
                             action_mode=False)

            self.transformer(self._repeat_input_for_cfg(input_dict['action_res_lst']),
                             update_cache=2,
                             cache_name=self.cache_name,
                             action_mode=True)
        torch.cuda.empty_cache()
        self.frame_st_id += latent_model_input.shape[2]

    @torch.no_grad()
    def infer(self, obs):
        reset = obs.get('reset', False)
        prompt = obs.get('prompt', None)
        compute_kv_cache = obs.get('compute_kv_cache', False)

        if reset:
            logger.info(f"******************* Reset server ******************")
            self._reset(prompt=prompt)
            return dict()
        elif compute_kv_cache:
            logger.info(
                f"################# Compute KV Cache #################")
            self._compute_kv_cache(obs)
            return dict(real_decode_ms=self.last_real_decode_ms)
        else:
            logger.info(f"################# Infer One Chunk #################")
            save_visualization = obs.get('save_visualization', False)
            action, latents = self._infer(obs, frame_st_id=self.frame_st_id)
            # Split so the client can attribute latency to a knob rather than to
            # "inference": video_gen is num_inference_steps, action_gen is
            # action_num_inference_steps, and video_decode (below) is the optional VAE.
            out = dict(action=action,
                       video_gen_ms=self.last_video_gen_ms,
                       action_gen_ms=self.last_action_gen_ms)
            if save_visualization:
                # Reported separately so the client is not left calling a VAE decode
                # "inference". This is the IMAGINED half; its real-stream counterpart is
                # timed in _compute_kv_cache and reported from there. Folding either into
                # the surrounding stage would silently inflate a latency that is supposed
                # to be independent of --video.
                t_dec = time.perf_counter()
                video = self.decode_streaming_video(latents, output_type='np')
                out['video_decode_ms'] = (time.perf_counter() - t_dec) * 1e3
                out['video'] = video[0] if hasattr(video, 'shape') and video.ndim == 5 else video
            return out

    def decode_streaming_video(self, latents, output_type):
        """Decode this chunk's imagined latents as a continuation of the REAL past.

        Two things are going on here.

        First, continuity. `decode_one_video` restarts the decoder cold every call, and
        a cold start costs temporal resolution -- the first latent frame of a call
        expands to 1 pixel frame instead of 4 -- so every chunk came out as a
        disconnected 5-frame clip. Carrying the decoder's causal-conv cache across
        chunks gives 5 frames for the first chunk and 8 for each one after.

        Second, and less obvious: the cache is fed by the REAL observation latents, not
        by the imagined ones, and each chunk decodes against a *fork* of it. That is
        because the model's own context is real-observation-only. `_infer` commits its
        predictions as `is_pred` (:511), and the first thing `_compute_kv_cache` does is
        `clear_pred_cache` (:581) -- evicting them before writing the real camera
        latents for the *same* frame positions. So the model is teacher-forced on
        reality and imagines exactly one chunk ahead. Warming the decoder on the
        previous chunk's imagination would render each chunk as a continuation of a
        pixel history the model never conditioned on. Measured, that shifts pixels by
        ~0.5-1.3% of range on average (locally up to half of full scale), bounded rather
        than compounding because the causal cache is only 2 frames deep -- small, but
        wrong, and this video exists precisely to be compared against reality.

        Falls back to a cold decode if the caller skipped a chunk: the cache is "the
        video so far", so decoding a non-adjacent chunk against it would splice two
        distant moments into one smooth shot -- motion that never happened, with nothing
        in the picture to reveal it. Only reachable by requesting video on some chunks
        and not others, which neither shipped client does, so it is reported rather than
        silently patched over.
        """
        self._video_stream_active = True
        expected = self.streaming_decoder.frames_done
        if expected and expected != self.frame_st_id:
            logger.warning(
                f"streaming decode gap: cache holds {expected} real latent frames but "
                f"this chunk starts at {self.frame_st_id}; decoding it cold. Request "
                "video on every chunk to keep the video continuous.")
            self.streaming_decoder.clear_cache()
        # Fork, not the stream itself: these imagined latents sit at the same frame
        # positions the real observations are about to occupy, and the real ones are
        # what the stream must carry forward.
        video = self.streaming_decoder.fork().decode_chunk(latents)
        return self.video_processor.postprocess_video(video, output_type=output_type)
    
    def decode_one_video(self, latents, output_type):
        # Cold, whole-sequence decode. Still correct for the offline `generate()` path
        # below: that loop never calls _compute_kv_cache, so nothing ever evicts the
        # is_pred entries and the model really does dream on its own imagination there.
        # An imagination-continuing decode is the right rendering of an imagined-context
        # rollout -- which is exactly why the server path needs the other one.
        latents = latents.to(self.vae.dtype)
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            latents.device, latents.dtype
        )
        latents = latents / latents_std + latents_mean
        video = self.vae.decode(latents, return_dict=False)[0]
        video = self.video_processor.postprocess_video(video, output_type=output_type)
        return video
    
    def load_init_obs(self):
        imf_dict = {v: np.array(Image.open(os.path.join(self.job_config.input_img_path, f"{v}.png")).convert("RGB")) for v in self.job_config.obs_cam_keys}
        init_obs = {}
        init_obs['obs'] = [imf_dict]
        return init_obs
    
    @torch.no_grad()
    def generate(self):
        self._reset(self.job_config.prompt)
        init_obs = self.load_init_obs()
        pred_latent_lst = []
        pred_action_lst = []
        for chunk_id in range(self.job_config.num_chunks_to_infer):
            actions, latents = self._infer(init_obs, frame_st_id=(chunk_id * self.job_config.frame_chunk_size))
            actions = torch.from_numpy(actions)
            pred_latent_lst.append(latents)
            pred_action_lst.append(actions)
        pred_latent = torch.cat(pred_latent_lst, dim=2)
        pred_action = torch.cat(pred_action_lst, dim=1).flatten(1)
        self.transformer.clear_cache(self.cache_name)
        self.streaming_vae.clear_cache()
        if self.streaming_vae_half:
            self.streaming_vae_half.clear_cache()
        del self.transformer
        del self.streaming_vae_half
        del self.text_encoder
        torch.cuda.empty_cache()
        
        # Move VAE to GPU for decoding
        if self.enable_offload:
            self.vae = self.vae.to(self.device).to(self.dtype)
        
        decoded_video = self.decode_one_video(pred_latent, 'np')[0]
        export_to_video(decoded_video, os.path.join(self.save_root, "demo.mp4"), fps=10)

def served_metadata(config_name, config):
    """Who this server is, sent to every client in the websocket handshake.

    A client evaluating the policy has to file its results under the checkpoint being
    served, and the action stream alone cannot identify it: yam_place_cube and
    yam_place_cube_base share action_per_frame and every inference setting, differing
    only in serving directory. So the server says which config it was launched with and
    which weights that resolved to.

    `transformer_weights` is resolved through the serving directory's symlinks here, at
    load time, so it keeps naming the checkpoint actually in memory even if the
    directory is re-assembled while the server runs. None when the directory has no
    single-file transformer to resolve.

    `norm_stat` is the quantile fit the actions are denormalized with, which also
    names the dataset the model was trained on: an open-loop eval compares it against
    its --dst dataset's own norm_stat.json, so scoring one task's model against another
    task's episodes stops rather than producing numbers.
    """
    serving_dir = os.path.abspath(config.wan22_pretrained_model_name_or_path)
    weights = os.path.join(serving_dir, 'transformer',
                           'diffusion_pytorch_model.safetensors')
    return dict(
        config_name=config_name,
        serving_dir=serving_dir,
        transformer_weights=(os.path.realpath(weights)
                             if os.path.exists(weights) else None),
        action_per_frame=config.action_per_frame,
        num_inference_steps=config.num_inference_steps,
        video_exec_step=config.video_exec_step,
        action_num_inference_steps=config.action_num_inference_steps,
        attn_window=config.attn_window,
        guidance_scale=config.guidance_scale,
        action_guidance_scale=config.action_guidance_scale,
        # How the first chunk treats latent frame 0's action slot: 'denoised' or
        # 'train_pad' (shared_config). Absent from servers that conditioned it on
        # normalized zeros, which is how results from all three stay distinguishable.
        frame0_action=config.frame0_action,
        # 'eef' or 'joint' (shared_config.action_space): the client picks its action
        # converter from this. Absent from older servers, all of which served 'eef'.
        action_space=config.get('action_space', 'eef'),
        norm_stat={k: [float(x) for x in config.norm_stat[k]] for k in ('q01', 'q99')},
    )


def run(args):

    config = VA_CONFIGS[args.config_name]
    port = config.port if args.port is None else args.port
    if args.save_root is not None:
        config.save_root = args.save_root
    if args.frame0_action is not None:
        config.frame0_action = args.frame0_action
    rank = int(os.getenv("RANK", 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    init_distributed(world_size, local_rank, rank)
    config.rank = rank
    config.local_rank = local_rank
    config.world_size = world_size
    # Taken just before the weights load, so it names what VA_Server reads.
    metadata = served_metadata(args.config_name, config)
    model = VA_Server(config)
    if config.infer_mode == 'i2va':
        logger.info(f"******************************USE I2AV mode******************************")
        model.generate()
    elif config.infer_mode == 'server':
        logger.info(f"******************************USE Server mode******************************")
        logger.info(f"Serving {metadata}")
        run_async_server_mode(model, local_rank, config.host, port, metadata=metadata)
    else:
        raise ValueError(f"Unknown infer mode: {config.infer_mode}")

def main():
    """
    TODO
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-name",
        type=str,
        required=False,
        default='robotwin',
        help="config name.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help='(start) port'
    )
    parser.add_argument(
        "--save_root",
        type=str,
        default=None,
        help='save root'
    )
    parser.add_argument(
        "--frame0-action",
        choices=['denoised', 'train_pad'],
        default=None,
        help="how the first chunk treats latent frame 0's action slot; overrides the "
             "config's frame0_action (default 'denoised', see shared_config)"
    )
    args = parser.parse_args()
    run(args)
    logger.info("Finish all process!!!!!!!!!!!!")


if __name__ == "__main__":
    init_logger()
    main()
