# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
import torch
from diffusers import AutoencoderKLWan
from diffusers.models.autoencoders.autoencoder_kl_wan import unpatchify
from transformers import (
    T5TokenizerFast,
    UMT5EncoderModel,
)

from .model import WanTransformer3DModel


def load_vae(
    vae_path,
    torch_dtype,
    torch_device,
):
    vae = AutoencoderKLWan.from_pretrained(
        vae_path,
        torch_dtype=torch_dtype,
    )
    return vae.to(torch_device)


def load_text_encoder(
    text_encoder_path,
    torch_dtype,
    torch_device,
):
    text_encoder = UMT5EncoderModel.from_pretrained(
        text_encoder_path,
        torch_dtype=torch_dtype,
    )
    return text_encoder.to(torch_device)


def load_tokenizer(tokenizer_path, ):
    tokenizer = T5TokenizerFast.from_pretrained(tokenizer_path, )
    return tokenizer


def load_transformer(
    transformer_path,
    torch_dtype,
    torch_device,
    **kwargs
):
    model = WanTransformer3DModel.from_pretrained(
        transformer_path,
        torch_dtype=torch_dtype,
        **kwargs
    )
    return model.to(torch_device)


def patchify(x, patch_size):
    if patch_size is None or patch_size == 1:
        return x
    batch_size, channels, frames, height, width = x.shape
    x = x.view(batch_size, channels, frames, height // patch_size, patch_size,
               width // patch_size, patch_size)
    x = x.permute(0, 1, 6, 4, 2, 3, 5).contiguous()
    x = x.view(batch_size, channels * patch_size * patch_size, frames,
               height // patch_size, width // patch_size)
    return x


class WanVAEStreamingWrapper:

    def __init__(self, vae_model):
        self.vae = vae_model
        self.encoder = vae_model.encoder
        self.quant_conv = vae_model.quant_conv

        if hasattr(self.vae, "_cached_conv_counts"):
            self.enc_conv_num = self.vae._cached_conv_counts["encoder"]
        else:
            count = 0
            for m in self.encoder.modules():
                if m.__class__.__name__ == "WanCausalConv3d":
                    count += 1
            self.enc_conv_num = count

        self.clear_cache()

    def clear_cache(self):
        self.feat_cache = [None] * self.enc_conv_num

    def encode_chunk(self, x_chunk):
        if hasattr(self.vae.config,
                   "patch_size") and self.vae.config.patch_size is not None:
            x_chunk = patchify(x_chunk, self.vae.config.patch_size)
        feat_idx = [0]
        out = self.encoder(x_chunk,
                           feat_cache=self.feat_cache,
                           feat_idx=feat_idx)
        enc = self.quant_conv(out)
        return enc


class WanVAEStreamingDecoder:
    """Decoder-side mirror of `WanVAEStreamingWrapper`, so video decodes continuously.

    `AutoencoderKLWan._decode` is already a one-latent-frame-at-a-time loop over the
    decoder's causal convs, but it calls `self.clear_cache()` on entry *and* exit
    (autoencoder_kl_wan.py:1190, 1207). Every `vae.decode()` therefore restarts cold,
    and a cold start costs temporal resolution: `WanResample.upsample3d` skips its
    `time_conv` while the cache slot is empty, so the first latent frame of a call
    expands to 1 pixel frame instead of 4. Decoding a 2-latent chunk per call yields
    1 + 4 = 5 frames every time, and the chunks do not join up.

    Holding the cache across calls -- as the encoder wrapper above already does for
    `encode_chunk` -- makes the first latent of the ROLLOUT cost the 1-frame start and
    every latent after it expand to the full 4. A 2-latent chunk then yields 8 frames
    from the second chunk on, and concatenating the chunks reproduces the whole-sequence
    decode that `visualization/decode_latents.py` does offline, frame for frame.

    The cache is this object's own list, never `vae._feat_map`, for the same reason the
    encoder wrapper keeps its own: `vae.clear_cache()` wipes the encoder and decoder
    maps together, so any incidental `vae.encode()`/`vae.decode()` elsewhere would
    silently reset a rollout mid-flight.

    Cost, measured on the robotwin_tshape canvas (24x20 latents, bf16): 418 MiB held
    between chunks, constant -- it is one cache slot per causal conv holding CACHE_T=2
    temporal frames, not a growing buffer of latents -- and +116 MiB on the transient
    peak, from producing 8 output frames per chunk rather than 5.
    """

    def __init__(self, vae_model):
        self.vae = vae_model
        self.decoder = vae_model.decoder
        self.post_quant_conv = vae_model.post_quant_conv

        if hasattr(self.vae, "_cached_conv_counts"):
            self.dec_conv_num = self.vae._cached_conv_counts["decoder"]
        else:
            count = 0
            for m in self.decoder.modules():
                if m.__class__.__name__ == "WanCausalConv3d":
                    count += 1
            self.dec_conv_num = count

        self.clear_cache()

    def clear_cache(self):
        self.feat_cache = [None] * self.dec_conv_num
        # Latent frames consumed so far this rollout. Drives `first_chunk` and, in the
        # server, the contiguity check -- a cache warmed on frames 0..3 is only valid
        # for frame 4, and decoding frame 6 against it would silently invent motion.
        self.frames_done = 0

    def fork(self):
        """A throwaway copy of this stream, for decoding a branch off the real past.

        The server keeps ONE canonical cache, fed by the real observation latents, and
        decodes each chunk's imagined latents against a fork of it. A fork is needed
        rather than a second cache because the imagined latent for position k and the
        real latent for position k are the SAME position -- the model predicts k, the
        robot executes, and the real observation for k arrives and replaces the
        prediction in the transformer's context (wan_va_server.py:581). Two values for
        one position cannot both advance one cache.

        The tensors are cloned, not aliased: `WanCausalConv3d` writes its cached frames
        into these slots in place, so a shallow copy would let the imagined decode
        corrupt the real stream it was supposed to branch from.
        """
        other = self.__class__.__new__(self.__class__)
        other.vae = self.vae
        other.decoder = self.decoder
        other.post_quant_conv = self.post_quant_conv
        other.dec_conv_num = self.dec_conv_num
        other.feat_cache = [None if c is None else c.clone() for c in self.feat_cache]
        other.frames_done = self.frames_done
        return other

    def decode_chunk(self, latents):
        """(B, z, F, h, w) normalized latents -> (B, 3, T, H, W) pixels in [-1, 1].

        Denormalization matches `VA_Server.decode_one_video` exactly; it is elementwise,
        so applying it per chunk is identical to applying it to the whole sequence.
        """
        latents = latents.to(self.vae.dtype)
        latents_mean = (torch.tensor(self.vae.config.latents_mean)
                        .view(1, self.vae.config.z_dim, 1, 1, 1)
                        .to(latents.device, latents.dtype))
        latents_std = (1.0 / torch.tensor(self.vae.config.latents_std)
                       .view(1, self.vae.config.z_dim, 1, 1, 1)
                       .to(latents.device, latents.dtype))
        latents = latents / latents_std + latents_mean

        x = self.post_quant_conv(latents)
        frames = []
        for i in range(x.shape[2]):
            # feat_idx is a fresh moving write pointer per frame; the decoder graph is
            # traversed in a fixed order, so conv k always reads and writes slot k.
            frames.append(self.decoder(x[:, :, i:i + 1], feat_cache=self.feat_cache,
                                       feat_idx=[0],
                                       first_chunk=(self.frames_done == 0 and i == 0)))
        self.frames_done += x.shape[2]

        out = torch.cat(frames, dim=2)
        if getattr(self.vae.config, "patch_size", None) is not None:
            out = unpatchify(out, patch_size=self.vae.config.patch_size)
        return torch.clamp(out, min=-1.0, max=1.0)
