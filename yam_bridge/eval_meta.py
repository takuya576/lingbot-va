# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""What an evaluation needs to know about the model it is evaluating.

Which checkpoint is being served, the settings it is being served at, and therefore
where the artifacts go and what they are called. The policy server states all of it in
the websocket handshake (wan_va_server.served_metadata), and `resolve_served` takes it
from there; the serving directory and the config are only the fallback for a server
that predates the handshake metadata.

Eval output mirrors the training layout under `eval_out/` rather than living beside the
weights, so pruning a checkpoint does not take its evaluation with it:

    train_out/yam_place_cube/20260922_150000/checkpoints/checkpoint_step_1000/
    eval_out/ yam_place_cube/20260922_150000/checkpoints/checkpoint_step_1000/open_loop/

The checkpoint is not passed in by hand in the common case. The policy server is launched
against a config whose `wan22_pretrained_model_name_or_path` is an assembled serving
directory (script/assemble_yam_serving_model.sh), and that directory's transformer files
are symlinks into the training checkpoint. Following one of those links names the
checkpoint the server is ACTUALLY serving, which is the thing an eval should be filed
under -- a hand-typed path can silently disagree with what is loaded. So can a
hand-typed config name: yam_place_cube and yam_place_cube_base differ only in serving
directory, so nothing in the action stream gives a mismatch away. That is why the link
is followed by the server, at load time, and the client's flags are only checked
against the result.
"""
import importlib.util
import os
import re
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_OUT = os.path.join(REPO_ROOT, 'train_out')
EVAL_OUT = os.path.join(REPO_ROOT, 'eval_out')

# The link that is followed to find the checkpoint. config.json is also a link into the
# same directory, but the weights are what the server actually loads.
_SERVED_WEIGHTS = os.path.join('transformer', 'diffusion_pytorch_model.safetensors')


_VA_CONFIGS = None


def _va_configs():
    """VA_CONFIGS, loaded without importing the `wan_va` package.

    The eval scripts run under the raiden venv, which has torch but not diffusers, so
    `from wan_va.configs import VA_CONFIGS` fails: wan_va/__init__.py imports .modules
    (-> diffusers) and .distributed before the subpackage is reached. Reading three
    scalars out of a config should not require the inference stack.

    Loading configs/__init__.py under its own name keeps its `from .va_x import y`
    relative imports working via submodule_search_locations, and -- unlike putting
    wan_va/ on sys.path -- does not shadow `utils`, `dataset` or `modules` for whatever
    imports next.
    """
    global _VA_CONFIGS
    if _VA_CONFIGS is not None:
        return _VA_CONFIGS

    pkg = os.path.join(REPO_ROOT, 'wan_va', 'configs')
    spec = importlib.util.spec_from_file_location(
        '_va_configs', os.path.join(pkg, '__init__.py'),
        submodule_search_locations=[pkg])
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except ImportError as exc:
        del sys.modules[spec.name]
        missing = getattr(exc, 'name', None) or '<dependency>'
        raise SystemExit(
            f"cannot read the VA configs: {exc}\n"
            f"This interpreter ({sys.executable}) is missing {missing!r}, which "
            f"wan_va/configs/ needs. Either install it there:\n"
            f"    {sys.executable} -m pip install {missing}\n"
            f"or run this script with an interpreter that already has it -- the eval "
            f"scripts are meant for the raiden venv.") from exc
    _VA_CONFIGS = module.VA_CONFIGS
    return _VA_CONFIGS


def _config(config_name):
    VA_CONFIGS = _va_configs()

    if config_name not in VA_CONFIGS:
        raise SystemExit(
            f"unknown --config-name {config_name!r}; choose from "
            f"{', '.join(sorted(VA_CONFIGS))}")
    return VA_CONFIGS[config_name]


def serving_dir_for_config(config_name):
    """The assembled serving directory a config's policy server loads from."""
    path = _config(config_name).wan22_pretrained_model_name_or_path
    return path if os.path.isabs(path) else os.path.join(REPO_ROOT, path)


def action_per_frame(config_name):
    """Control steps the server packs into one latent frame.

    Sets both the chunk length and the keyframe rate of an open-loop rollout: the causal
    VAE takes 4 frames per latent frame, so a keyframe falls every action_per_frame/4
    control steps. 24 for cover_blocks (stride 6 at 5fps), 16 for place_cube (stride 4).
    """
    return _config(config_name).action_per_frame


def inference_param_tag(config_name):
    """`<num_inference_steps>_<video_exec_step>_<action_num_inference_steps>`.

    The three denoising knobs that change what a rollout looks like without changing
    the checkpoint (wan_va_server.py:465-467), so two evaluations of one checkpoint at
    different settings land in the same directory under different names instead of
    overwriting each other. Not every such setting is here: attn_window, guidance and
    the frame-0 condition are recorded per result instead (`_SERVED_SETTINGS`), so an
    open-loop re-run that differs only in those overwrites the previous one unless it
    is given its own --out.

    Read from the config, so only as accurate as `config_name` matching the
    CONFIG_NAME the server was launched with. `resolve_served` uses the server's own
    report instead and falls back to this only for a server that predates it.
    """
    cfg = _config(config_name)
    return (f"{cfg.num_inference_steps}_{cfg.video_exec_step}"
            f"_{cfg.action_num_inference_steps}")


# Inference settings that change a rollout without changing the checkpoint. Only the
# first three are in the filename tag; all of them go into every results.json record,
# since attn_window, guidance and the frame-0 condition have changed between rollouts
# of one checkpoint before, and nothing else would tell those rollouts apart.
_SERVED_SETTINGS = ('num_inference_steps', 'video_exec_step',
                    'action_num_inference_steps', 'attn_window', 'guidance_scale',
                    'action_guidance_scale', 'frame0_action')


def checkpoint_for_serving_dir(serving_dir):
    """Resolve a serving directory to the training checkpoint it symlinks into.

    Returns the checkpoint directory (the parent of `transformer/`), or None when the
    serving directory holds real files rather than links -- a hand-assembled or copied
    model has no run to file the evaluation under.
    """
    weights = os.path.join(serving_dir, _SERVED_WEIGHTS)
    if not os.path.exists(weights):
        # exists() follows links, so this catches a dangling one too. Same breakage the
        # server would hit at load time, reported here instead of after a long startup.
        raise SystemExit(
            f"{weights} does not resolve.\n"
            f"Re-assemble it: script/assemble_yam_serving_model.sh <checkpoint_dir> "
            f"{serving_dir}")
    return _checkpoint_for_weights(weights, os.path.realpath(weights))


def _checkpoint_for_weights(link, real):
    """The checkpoint directory `real` lies in, or None if `link` was not a link."""
    if real is None or real == os.path.abspath(link):
        return None
    return os.path.dirname(os.path.dirname(real))


def resolve_served(server_metadata, config_name=None, checkpoint=None):
    """Identify the model a connected policy server is serving.

    Returns dict(config_name, checkpoint, param_tag, action_per_frame, action_space,
    norm_stat, settings), taken from what the server reported in the handshake
    (VAClient.server_metadata). `checkpoint` is None when the serving directory holds
    real files rather than links. `settings` maps each of _SERVED_SETTINGS to the
    server's value, None where an older server did not report it -- unknown, not a
    default, since the config on disk may have changed since that server launched.

    `config_name` and `checkpoint` are what the caller was told on the command line.
    Against a server that reports itself they are assertions, and a disagreement stops
    the run before anything is recorded: the results would otherwise be filed, silently,
    under a different model's history -- one whose numbering and results.json they
    would extend. `checkpoint` is still needed to name a serving directory of real
    files, which has no link to report.

    A server from before the handshake metadata sends nothing. Then `config_name` is
    the only source and cannot be checked, so it is required rather than defaulted.
    """
    served_name = (server_metadata or {}).get('config_name')
    if served_name is None:
        if config_name is None:
            raise SystemExit(
                "the policy server does not report which config it is serving (it "
                "predates the handshake metadata). Relaunch it, or pass --config-name "
                "with the CONFIG_NAME it was launched with.")
        print(f"  NOTE: the policy server predates handshake metadata, so "
              f"--config-name {config_name} cannot be verified against it.")
        if checkpoint is None:
            checkpoint = checkpoint_for_serving_dir(serving_dir_for_config(config_name))
        cfg = _config(config_name)
        return dict(config_name=config_name, checkpoint=checkpoint,
                    param_tag=inference_param_tag(config_name),
                    action_per_frame=action_per_frame(config_name),
                    action_space=cfg.get('action_space', 'eef'),
                    norm_stat=cfg.norm_stat,
                    # Unverifiable here for the same reason config_name is. frame0_action
                    # is not even that: a server this old ignores the setting and
                    # conditions frame 0 on normalized zeros.
                    settings={k: (None if k == 'frame0_action' else cfg.get(k))
                              for k in _SERVED_SETTINGS})

    if config_name is not None and config_name != served_name:
        raise SystemExit(
            f"--config-name {config_name}, but the policy server was launched with "
            f"CONFIG_NAME={served_name}. Drop --config-name to evaluate what is being "
            f"served, or relaunch the server with the config you meant.")

    m = server_metadata
    served_ckpt = _checkpoint_for_weights(
        os.path.join(m['serving_dir'], _SERVED_WEIGHTS), m['transformer_weights'])
    if checkpoint is None:
        checkpoint = served_ckpt
    elif (served_ckpt is not None
          and os.path.realpath(checkpoint) != os.path.realpath(served_ckpt)):
        raise SystemExit(
            f"--checkpoint {checkpoint}, but the policy server loaded {served_ckpt}. "
            f"Drop --checkpoint to file the results under what is being served.")
    return dict(config_name=served_name, checkpoint=checkpoint,
                param_tag=(f"{m['num_inference_steps']}_{m['video_exec_step']}"
                           f"_{m['action_num_inference_steps']}"),
                action_per_frame=int(m['action_per_frame']),
                # Servers from before this field all served EEF checkpoints.
                action_space=m.get('action_space', 'eef'),
                # From the config if the server's report predates this field.
                norm_stat=m.get('norm_stat') or _config(served_name).norm_stat,
                settings={k: m.get(k) for k in _SERVED_SETTINGS})


def eval_out_dir(loop, checkpoint):
    """Directory for one evaluation's artifacts.

    `loop` is 'open_loop' or 'closed_loop' and becomes the leaf directory, which is what
    separates the two kinds of evaluation of one checkpoint -- the filenames underneath
    are identical between them, so the parent directory is the distinction.

    `checkpoint` is the checkpoint directory, normally `resolve_served`'s. The returned
    path mirrors its position under train_out/, so a checkpoint outside that tree falls
    back to its own basename, and None (a serving directory of real files) to
    eval_out/unlinked/.
    """
    if loop not in ('open_loop', 'closed_loop'):
        raise ValueError(f"loop must be 'open_loop' or 'closed_loop', got {loop!r}")
    if checkpoint is None:
        return os.path.join(EVAL_OUT, 'unlinked', loop)

    rel = os.path.relpath(os.path.abspath(checkpoint), TRAIN_OUT)
    if rel.startswith(os.pardir):
        # Not under train_out/ -- nothing to mirror, so file it by name alone.
        rel = os.path.basename(os.path.normpath(checkpoint))
    return os.path.join(EVAL_OUT, rel, loop)


def artifact_stem(ep, param_tag):
    """Filename stem for one episode's artifacts, e.g. ep50_5_-1_10.

    No loop prefix: open_loop/ and closed_loop/ are separate directories, so the same
    filename in each is unambiguous and the two are directly comparable by name.

    Unpadded, so these sort lexically as ep0, ep1, ep10, ep2 -- fine for picking a file
    out by name, misleading if you read an `ls` as chronological.
    """
    return f"ep{ep}_{param_tag}"


def next_rollout_index(out_dir):
    """Lowest unused rollout number in `out_dir`, so sessions accumulate.

    A closed-loop session numbers its rollouts from 0, so without this a second session
    against the same checkpoint would overwrite the first one's plots. Real-robot
    rollouts cannot be reproduced -- the scene is reset by hand and the arms never
    repeat a trajectory exactly -- so losing one to a filename collision loses it for
    good. Numbering resumes past the highest `ep<N>_` already on disk.

    Counts from the filenames rather than from results.json: the plots are what must not
    collide, and a run killed before it wrote its json still leaves them behind.
    """
    if not os.path.isdir(out_dir):
        return 0
    highest = -1
    for name in os.listdir(out_dir):
        match = re.match(r'ep(\d+)_', name)
        if match:
            highest = max(highest, int(match.group(1)))
    return highest + 1
