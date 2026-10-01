# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""What an evaluation needs to know about the model it is evaluating.

Which checkpoint is being served, the settings it is being served at, and therefore
where the artifacts go and what they are called. The eval scripts talk to the policy
server over a socket that carries no metadata, so all of it is recovered here from the
serving directory and the config rather than asked for.

Eval output mirrors the training layout under `eval_out/` rather than living beside the
weights, so pruning a checkpoint does not take its evaluation with it:

    train_out/yam_place_cube/20260922_150000/checkpoints/checkpoint_step_1000/
    eval_out/ yam_place_cube/20260922_150000/checkpoints/checkpoint_step_1000/open_loop/

The checkpoint is not passed in by hand in the common case. The policy server is launched
against a config whose `wan22_pretrained_model_name_or_path` is an assembled serving
directory (script/assemble_yam_serving_model.sh), and that directory's transformer files
are symlinks into the training checkpoint. Following one of those links names the
checkpoint the server is ACTUALLY serving, which is the thing an eval should be filed
under -- a hand-typed path can silently disagree with what is loaded.
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

    The three knobs that change what a rollout looks like without changing the
    checkpoint (wan_va_server.py:465-467), so two evaluations of one checkpoint at
    different settings land in the same directory under different names instead of
    overwriting each other.

    Read from the config, because the VA protocol has no metadata call to ask the
    server with (yam_bridge/va_client.py). That makes it only as accurate as
    --config-name matching the CONFIG_NAME the server was launched with, and blind to
    per-launch overrides appended to run_launch_va_server_sync.sh.
    """
    cfg = _config(config_name)
    return (f"{cfg.num_inference_steps}_{cfg.video_exec_step}"
            f"_{cfg.action_num_inference_steps}")


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
    real = os.path.realpath(weights)
    if real == os.path.abspath(weights):
        return None
    return os.path.dirname(os.path.dirname(real))


def eval_out_dir(loop, config_name=None, checkpoint=None):
    """Directory for one evaluation's artifacts.

    `loop` is 'open_loop' or 'closed_loop' and becomes the leaf directory, which is what
    separates the two kinds of evaluation of one checkpoint -- the filenames underneath
    are identical between them, so the parent directory is the distinction.

    `checkpoint` names the checkpoint directory directly; otherwise it is resolved from
    `config_name`'s serving directory. The returned path mirrors the checkpoint's
    position under train_out/, so a checkpoint outside that tree (or a serving directory
    of real files) falls back to its own basename.
    """
    if loop not in ('open_loop', 'closed_loop'):
        raise ValueError(f"loop must be 'open_loop' or 'closed_loop', got {loop!r}")
    if checkpoint is None:
        if config_name is None:
            raise ValueError('pass checkpoint or config_name')
        checkpoint = checkpoint_for_serving_dir(serving_dir_for_config(config_name))
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
