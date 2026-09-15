# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Thin wrapper over the LingBot-VA websocket protocol, importable without `wan_va`.

The deployment client runs in the robot environment (raiden's venv), which has
websockets, msgpack, mujoco and mink but not torch, diffusers or easydict. Importing
`wan_va.utils.Simple_Remote_Infer.deploy.websocket_client_policy` would execute
`wan_va/__init__.py` and pull in the whole training stack, so this module loads that
package directly instead: `Simple_Remote_Infer/` has no `__init__.py` while `deploy/`
does, so putting the former on `sys.path` makes `deploy` a top-level package and its
relative `from .msgpack_numpy import ...` resolves normally.

It also wraps the two protocol sharp edges that are easy to get wrong:

  - `WebsocketClientPolicy.reset()` sends no prompt. The server then leaves
    `prompt_embeds = None` and the *next* inference dies dereferencing it, far from
    the cause. `VAClient.reset()` always sends the prompt.
  - Any server-side exception comes back as a websocket *text* frame, after which the
    server closes the socket and re-raises -- the process is gone. There is nothing to
    retry against, so it is surfaced as a distinct, fatal error type.
"""
import importlib
import os
import sys
from typing import Dict, List, Optional

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEPLOY_PARENT = os.path.join(
    _REPO_ROOT, "wan_va", "utils", "Simple_Remote_Infer")

DEFAULT_PORT = 29536  # wan_va/configs/shared_config.py:8. Note run_launch_va_server_sync.sh
                      # defines PORT but never forwards it, so the server uses this.


class VAServerDied(RuntimeError):
    """The policy server raised and terminated. Relaunch it; do not retry in place."""


def _load_websocket_client_cls():
    """Import `WebsocketClientPolicy` without importing the `wan_va` package."""
    if not os.path.isdir(os.path.join(_DEPLOY_PARENT, "deploy")):
        raise FileNotFoundError(f"no deploy/ package under {_DEPLOY_PARENT}")
    if _DEPLOY_PARENT not in sys.path:
        sys.path.insert(0, _DEPLOY_PARENT)
    module = importlib.import_module("deploy.websocket_client_policy")
    return module.WebsocketClientPolicy


class VAClient:
    """One rollout's worth of conversation with the LingBot-VA policy server.

    The server is stateful and the call order is load-bearing:
    `reset(prompt)` then `(infer, advance)` repeatedly. `infer` does not advance the
    world model -- only `advance` (`compute_kv_cache`) does, so skipping it makes the
    server regenerate from the same step forever.
    """

    def __init__(self, host: str = "0.0.0.0", port: int = DEFAULT_PORT):
        cls = _load_websocket_client_cls()
        # Blocks until the server accepts, retrying every 5s.
        self._client = cls(host=host, port=port)
        self.prompt: Optional[str] = None

    def _call(self, payload: Dict) -> Dict:
        try:
            return self._client.infer(payload)
        except RuntimeError as exc:
            # The client raises RuntimeError with the server's traceback for a text
            # frame; by then the server has already closed and exited.
            raise VAServerDied(str(exc)) from exc

    def reset(self, prompt: str) -> None:
        """Start a new rollout. Clears the KV cache, streaming VAE and frame counter.

        The prompt is T5-encoded server-side and cached for the whole rollout; it
        cannot be changed without another reset, which would discard the world-model
        state along with it.
        """
        if not prompt:
            raise ValueError("a non-empty prompt is required: the server builds "
                             "classifier-free-guidance embeddings from it at reset, "
                             "and inference dereferences them unconditionally")
        self.prompt = prompt
        self._call(dict(reset=True, prompt=prompt))

    def infer(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        """Predict the next action chunk. Returns `(16, frame_chunk_size, action_per_frame)`.

        `obs` is only consumed while `frame_st_id == 0`, i.e. on the first call after a
        reset; later calls ignore it. Passing the same initial observation every time is
        what the reference clients do.
        """
        out = self._call(dict(obs=obs, prompt=self.prompt))
        action = out["action"]
        # msgpack hands back arrays backed by the receive buffer, so they are read-only
        # and non-owning; the converter and the caller both want a normal array.
        return np.array(action, dtype=np.float64, copy=True)

    def advance(self, keyframes: List[Dict[str, np.ndarray]],
                action: np.ndarray) -> None:
        """Fold the executed chunk back into the world model.

        `keyframes` must be exactly the observations captured during execution, and its
        length must be a multiple of 4 -- the causal VAE consumes 4 frames per latent
        frame and silently desyncs on a ragged list. `action` must be the **full**
        array `infer` returned, not the subset actually executed: on the first chunk the
        server prepends its stored init latent so the latent stream again matches this
        array's frame axis.
        """
        if not keyframes or len(keyframes) % 4 != 0:
            raise ValueError(
                f"keyframe count must be a non-zero multiple of 4, got {len(keyframes)}; "
                "a partial list desyncs the server's streaming VAE cache")
        self._call(dict(obs=keyframes, compute_kv_cache=True, state=action))
