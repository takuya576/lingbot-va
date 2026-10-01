# Deploying LingBot-VA on real YAM arms

Operating manual for the live policy and the safeguards around it.

Every number here was measured on `cover_blocks` (74 episodes, 106165 steps) and on the
`mlcav` YAM cell. Re-measure on different hardware before trusting them.

---

## 1. What runs where

Three processes, two machines:

| process | host | what it does |
|---|---|---|
| **wan_va policy server** | `lambda-scalar` (GPUs) | predicts 16-dim relative EEF-pose chunks, port 29536 |
| **`rd serve`** | `mlcav` / Dell-server | cameras + measured joints + the arms, port 8765 |
| **`run_yam_policy.py`** | `lambda-scalar`, raiden venv | converts poses to joints via IK, paces dispatch, watches everything |

The representation gap the client bridges:

```
model emits   16-dim bimanual EEF pose, RELATIVE to the pose at rollout start
YAM accepts   14-dim absolute joints  = [left q6, left grip, right q6, right grip]

chunk -> un-relativize against the latched anchor -> renormalize quats
      -> per-arm 4x4 -> IK on grasp_site, seeded from the previous command -> (14,)
```

Two properties shape everything downstream, and both are worth internalising before
you touch a flag:

- **Inference cannot overlap execution.** The next `infer` needs a world model advanced
  by observations captured *during* the previous chunk, so the stages are strictly
  sequential. The arms hold position while the model thinks.
- **The policy never sees the robot.** `format_obs`'s `observation.state` key is never
  read (`wan_va/wan_va_server.py:583`) — the model is fed back the chunk it just
  emitted. IK residual, silent joint clipping, tracking error and slew limiting are all
  invisible to it. **Only the cameras close the loop.** Every safeguard in §4 exists
  because nothing else is watching.

---

## 2. Before the first run

**Certify the hardware path first.** `script/replay_yam_episode.py` replays a recorded
episode through the same converter and transport with no policy involved. On a new
machine, run it before the policy — if it cannot hold 30 Hz, neither can this.

```bash
# free, no hardware
/data2/takuya/raiden/.venv/bin/python script/replay_yam_episode.py --no-robot --episode 0

# then, against the arms (see §6 for what good looks like)
rd serve --action-type joint --control-hz 30 --max-joint-delta 0.35 \
         --resize-images 64x64 --no-depth
/data2/takuya/raiden/.venv/bin/python script/replay_yam_episode.py \
    --robot-uri ws://mlcav.seas.ucla.edu:8765 --source raw --episode 0 --max-steps 300
```

`--source raw` first (transport only), then `--source converted` (adds the converter).
`--resize-images 64x64` because the replay reads only proprios — see §5.

**Check disk.** `/data2` runs near full and the policy server writes ~25 MB per chunk
(`obs_data_*.pt` is 99.6% of it), roughly **25 GB per hour of rollout**. Each reset
creates a new timestamped directory; nothing is ever pruned.

```bash
df -h /data2
rm -rf /data2/takuya/lingbot-va/example/yam_cover_blocks/real/*
```

`save_root` is hardcoded in `script/run_launch_va_server_sync.sh` — passing
`--save_root` loses to argparse last-wins because the script appends its own after your
overrides.

**Check the serving model is assembled:**

```bash
ls -la Pretrained_models/lingbot-va-posttrain-yam-cover-blocks/transformer/  # -> checkpoint_step_N
script/assemble_yam_serving_model.sh --check Pretrained_models/lingbot-va-posttrain-yam-cover-blocks

# to re-point it (out_dir defaults to the place_cube serving dir, so pass it):
script/assemble_yam_serving_model.sh \
    train_out/yam_cover_blocks/<run>/checkpoints/checkpoint_step_N \
    Pretrained_models/lingbot-va-posttrain-yam-cover-blocks
```

---

## 3. Launch sequence

### 3.1 Policy server (`lambda-scalar`)

```bash
cd /data2/takuya/lingbot-va
NGPU=1 CONFIG_NAME=yam_cover_blocks bash script/run_launch_va_server_sync.sh
```

Wait for it to listen on **29536** (`run_launch_va_server_sync.sh` defines `PORT` but
never forwards it, so the config value wins). Loading 10.2 GB takes a while.

`CONFIG_NAME=yam_cover_blocks` is not optional. Serving under the `robotwin` config would
denormalize with the wrong q01/q99 and emit plausible-looking garbage.
`va_yam_cover_blocks_cfg.norm_stat` has been verified to match `norm_stat.json` exactly on all 30
channels.

### 3.2 Robot server (`mlcav`)

```bash
rd serve --action-type joint --control-hz 30 --max-joint-delta 0.35 \
         --resize-images 480x640 --no-depth
```

| flag | why this value |
|---|---|
| `--action-type joint` | the client sends 14-D joints; `apply_action` then does no server-side IK |
| `--control-hz 30` | the model's actions are at the dataset's native 30 fps. 10 Hz would discard 2 of every 3 |
| `--max-joint-delta 0.35` | **measured.** See §7 — raiden's CLI default is 0.8, and the docs' 0.2 would e-stop on known-good motion |
| `--resize-images 480x640` | the training resolution. Parsed as `(h, w)`, and it is the RealSense native size, so the resize is a no-op |
| `--no-depth` | nothing consumes depth, and it is more than half the payload |

Confirm all three cameras open at 640×480 before continuing.

### 3.3 Client — dry run first (`lambda-scalar`)

```bash
/data2/takuya/raiden/.venv/bin/python script/run_yam_policy.py \
  --robot-uri ws://mlcav.seas.ucla.edu:8765 --dry-run --max-chunks 3 \
  --prompt "On the table, red, green, and blue blocks are arranged randomly along with three cups. From the current view point, cover the blocks from left to right using the cups, and then uncover them again in sequence of red, green, and blue."
```

Use the prompt **verbatim** from `Datasets/yam-real-world-lerobot/cover_blocks/meta/tasks.jsonl`
— the model was conditioned on that exact string, and it is T5-encoded once at reset and
cached for the whole rollout.

`--dry-run` drives the full protocol and reads real observations but sends nothing to
the arms. It exists to measure **inference latency**, which nothing else establishes.
Then drop `--dry-run`, arms clear of the table, `--max-chunks 3`.

`--robot-uri` has no default by design. Note `Dell-server` is only an SSH alias; the
resolvable name is `mlcav.seas.ucla.edu`.

---

## 4. The safeguard system

Seven layers. The first six are client-side and stop the run gracefully; the last two
are the hard backstops that take the server down with them.

### Layer 1 — configuration, once at connect

`check_server_metadata` compares the client's assumptions against what `rd serve`
actually reports.

- **`control_hz` mismatch → fatal.** Nothing detects this at run time. `apply_action` is
  fire-and-forget with no response, so a client pacing at 30 Hz against a server
  interpolating for 10 Hz gets no error — just motion the arms cannot follow.
- **`action_type` / `action_shape` mismatch → fatal.**
- **`max_joint_delta`** is reported so the e-stop threshold is visible before anything
  moves.

Requires a `rd serve` built after the metadata commit (`5516c5e`). An older server omits
the keys, which is reported as *unverifiable* rather than treated as agreement.

### Layer 2 — conversion failure, per chunk

A degenerate quaternion or malformed shape halts with the arms holding, rather than
propagating out and killing the process. (An uncaught exception here disconnects from
raiden, which e-stops and homes — safe, but it discards the diagnostic.)

### Layer 3 — IK branch switch, per chunk

`conv.stats["unsafe_jumps"]` increments when a stalled IK seed is rescued from a
fallback seed. That switches solution branch: measured at **0.734 rad** in one step.

**This is refused, never smoothed.** A branch switch reaches the same end-effector pose
from a different arm configuration; interpolating between the two in joint space
traverses poses nobody asked for, so spreading it over more steps makes it slower, not
safer. The gate runs *before* the slew limiter for exactly this reason.

### Layer 4 — slew limit, per chunk

`--max-step-delta 0.35` clips per-step arm-joint change, matching the server's
threshold. Grippers are excluded (normalized [0,1]; a radian bound is meaningless).

Clipping **rewrites the trajectory**, and the policy cannot tell. So it is paired with
`--max-slew-deviation 0.3`, which halts when the executed path has drifted that far from
the requested one. When clipping fires you get an immediate line naming the joints:

```
SLEW CLIP: 5/24 steps limited to 0.35 rad (largest request 1.840) on L1x5, L2x2, L3x3;
           trajectory is 1.490 rad behind what was asked for
```

One line is informational. **Repeated lines across consecutive chunks mean the rollout
has quietly diverged from what the model intends**, while every other number looks fine.

### Layer 5 — action backlog, per keyframe

`--max-backlog 10`. Queued actions are latency the arms have not begun working through,
and a growing queue invalidates every timing figure in the report.

This exists because chiral's dispatcher is **not self-correcting**: `_aaction_dispatch`
sleeps `interval - send_time` with no absolute deadline, so `asyncio.sleep` overshoot
accumulates at ~0.72 ms/tick — 29.37 Hz when asked for 30. The mitigation is
`--dispatch-mult 4`, which ticks the dispatcher at 120 Hz so *our* deadline-corrected
loop sets the cadence and the dispatcher only drains. Backlog then sits at 0–1.

Lowering the control rate does **not** help: the error is per tick and an episode is the
same number of ticks at any rate (~1.3 s of accumulated lag at either 30 or 10 Hz).

### Layer 6 — tracking error, per keyframe

`--max-tracking-error`, compared **lag-compensated** against a rolling command history.

Raw comparison does not work. At `--obs-hz 10` the observation is up to 100 ms stale,
plus transport, plus the fact that `_make_obs` stamps observations with the reference
camera's arrival time and interpolates proprios onto it. At 30 Hz that is 0.3+ rad of
command travel, which reads as tracking failure on a perfectly healthy arm.
`tracking_error()` searches recent commands and takes the minimum; the argmin is the
pipeline latency, the minimum is the real error. A large error at *every* lag is a
genuine failure and still halts.

### Layer 7 — server and hardware backstops

These are outside your control and they are not graceful.

- **`_check_joint_delta`** (`raiden/server.py`) compares the commanded action against
  *measured* joints and calls `emergency_stop()` on exceedance → 5 s hold, home,
  `os._exit(0)`. The `rd serve` process dies.
- **Silent joint clipping.** `command_joint_pos` clips the 6 arm joints to the YAM
  limits with **no log, no warning, no return flag**
  (`i2rt/robots/motor_chain_robot.py:485-502`). Achieved ≠ commanded with zero
  indication. The converter's FK residual and the tracking monitor are the only way
  you would ever see it.
- **Torque clipping** and a **gripper force limiter**, both also silent.
- **There is no velocity or acceleration limit anywhere** in the stack.

---

## 5. Bandwidth

Observations and actions share one socket, and raiden's handler is sequential per
connection — an observation on the wire delays the actions queued behind it.

| config | per obs | at 30 Hz | at 10 Hz |
|---|---|---|---|
| 480×640×3, 3 cameras | 2.76 MB | 82.9 MB/s (**75% of 1 GbE**) | 27.6 MB/s |
| + depth (float32) | 6.5 MB | — | — |
| 64×64×3, 3 cameras | 0.04 MB | 1.1 MB/s | — |

The client prints this at reset and warns past a third of the link. The **policy** needs
480×640 and must buy headroom from `--obs-hz` (10 is the default). The **replay** reads
only proprios, so shrink the images instead and poll at 30.

---

## 6. Reading the output

```
[chunk   7] vgen   1400ms  act    700ms  ik    64.2ms  adv   310ms  48 steps  8 keyframes  duty 39.4%
            track 0.038 rad  lag 6  slew 0/48@0.000
```

| field | healthy | what it means when it is not |
|---|---|---|
| `vgen` | — | denoising the **video** latents, `num_inference_steps` |
| `act` | — | denoising the **action** chunk, `action_num_inference_steps` |
| `ik` | 40–90 ms | >1000 ms means targets are drifting out of reach (§8) |
| `adv` | 200–400 ms | refresh the KV cache with the real observation: VAE **encode** of the keyframes + 2 transformer forwards. Independent of `--video` |
| `vid` | only with `--video` | **both** VAE **decodes** — the imagined chunk, and the real latents that keep the video continuing from reality |
| `oth` | shown above 50 ms | round-trip remainder: transport, msgpack, chunk-0 obs encode |
| `steps` | `action_per_frame` then 2x that | first chunk skips latent frame 0. cover_blocks (stride 6, apf 24): **24 then 48**. place_cube (stride 4, apf 16): **16 then 32** |
| `keyframes` | **4 then 8** | anything else halts — a ragged list desyncs the streaming VAE |
| `duty` | higher is better | fraction of wall-clock the arms actually move |

**`vgen` runs even without `--video`.** The action loop attends to the video latents the
first loop produces (`wan_va_server.py`, action loop follows the video loop with no clear
in between), so cutting `num_inference_steps` is a real latency lever on the *action*
path, not just a visualisation setting. `vid` is the only genuinely optional stage.

**The budget `duty` is measured against.** One chunk's execution is
`steps / control_hz`, and that is the whole time the pipeline has to produce the next
chunk before the arms run dry:

| config | steps (chunk 0 / steady) | execution at 30 Hz |
|---|---|---|
| cover_blocks, `action_per_frame` 24 | 24 / 48 | 0.80 s / **1.60 s** |
| place_cube, `action_per_frame` 16 | 16 / 32 | 0.53 s / **1.07 s** |

Halving `action_per_frame` halves the budget: place_cube re-plans twice as often, so the
same per-chunk latency buys a markedly worse duty cycle than it did on cover_blocks.

> **Duty-cycle figures from before 2026-09-24 are not comparable.** `adv` was omitted
> from the hold entirely and `infer` silently absorbed the video decode, so the old
> `duty` overstated the fraction of time the arms move — by ~26 points in a
> representative chunk without video, more with it. Every stage between two dispatch
> bursts is now counted: `infer + ik + adv + vid`.
| `track` | 0.02–0.13 rad | rising = collision or torque limit, invisible to the policy |
| `lag` | **5–6 steps** | see below. Far higher = queueing |
| `slew` | `0/48@0.000` | anything else means the trajectory is being overridden |

**Why `lag` is 5–6 and not 1–2.** Roughly half of it is the arm, not the software:

| source | steps |
|---|---|
| actuator / position-loop response | **3** (measured from the dataset itself; kp=80 → 3, kp=40 → 2, kp=10 → 5) |
| `_smooth_command` interpolation | 0.9 — `_SMOOTH_DUTY` *is* the lag in control periods |
| client obs staleness | 0–1 |
| camera frame age + transport | ~1 |

Only about two steps are pipeline, and the rest was equally present in training. Lowering
`_SMOOTH_DUTY` shortens the second term (0.9 → 0.6 buys 10 ms), but the push rate to the
arms rises correspondingly and the hardware only consumes ~150–200 Hz.

---

## 7. Measured reference values

**Converter, episode-anchored** (the healthy case):

| metric | value |
|---|---|
| IK failures | 0 / 106165 |
| FK position residual | 0.10 mm (at solver threshold) |
| IK cost | 1.22–1.36 ms/step → ~60 ms per 48-action chunk vs 1600 ms budget |
| command step delta | p99.99 = 0.083 rad, max 0.086 |

**Tracking, from the demonstrations themselves** (`action` vs `observation.state`, which
is the closest proxy for what `_check_joint_delta` sees):

| percentile | rad |
|---|---|
| p50 | 0.0485 |
| p99 | 0.1547 |
| p99.99 | 0.2905 |
| max | **0.3914** |

This is why `--max-joint-delta 0.35` and not the plan's original 0.3: the threshold
tests commanded-vs-*measured*, not step-to-step, and 0.3 would e-stop on motion the arms
demonstrably executed. Raiden's own defaults disagree with each other —
`_DEFAULT_MAX_JOINT_DELTA = 0.2` (module), `--max-joint-delta 0.8` (CLI, and what
`rd serve` actually uses), `0.2` (docs). **Always pass it explicitly.**

**Full 60 s replay, converted, 30 Hz** — the certification run:

```
1820/1820 steps in 60.7s     effective send rate 30.00 Hz    backlog max 1, final 0
tracking err  p50=0.0214  p99=0.1032  max=0.1241   (lag-compensated)
dispatch lateness p50=+0.08ms  p99=+0.25ms  max=+5.50ms
```

The arms follow the IK-converted trajectory **better** than they followed the original
teleop commands (0.124 vs 0.234 rad).

**Joint limits.** Commands are clipped to `_ARM_JOINT_LIMITS` widened by ±0.15 rad. The
demonstrations press **j3 to within 5 milliradians** of that boundary, and 0.145 rad past
the *stock* XML limit. The ±0.15 buffer is load-bearing and nearly exhausted — this is
why `widen_arm_joint_limits` exists in `yam_bridge/kinematics.py`.

---

## 8. Known issues

### Anchor drift and the IK latency cliff — **deferred, not solved**

The anchor is latched from the measured pose at reset, which never exactly matches a
training episode's frame 0. `add_eef_pose` adds translation in the arm base frame, so an
anchor offset translates the **entire trajectory** rigidly.

Measured: arms homing within 0.013 rad of episode 0's start is a **14 mm shift at the
right gripper** — enough to push the reach-limit portion of the demonstration outside the
workspace.

| | IK failures | residual | ms/step |
|---|---|---|---|
| episode anchor | 0 / 3552 | 0.10 mm | 1.3 |
| measured-home anchor | 432 / 3552 | 2.62 mm | **48.6** |

The accuracy cost is trivial (2.62 mm). **The latency cost is not:** a failed solve costs
~196 ms because `ik_robust` burns every fallback seed on a target no seed can reach. A
chunk tolerates only **~8 unreachable targets out of 96** before IK alone exceeds the
1.6 s budget.

A closed-loop policy may steer away from this — it sees the objects and is not blindly
replaying a fixed trajectory. But if it does happen, it breaks the real-time loop
outright rather than degrading gracefully. **Watch the `ik` field**; the client warns
when IK exceeds the motion it is planning.

The one-line fix, when you want it: loosen the converter's IK tolerance. A sweep showed
`pos_threshold`/`ori_threshold` of `3e-3` gives **0 failures at 0.87 ms/step with the
residual unchanged at 2.62 mm** — a 55× speedup for 1 µm of accuracy, because IK
converges to the same solution either way and the tolerance only decides whether it is
*labelled* a failure. Pair it with a real gate on `stats["max_pos_residual_m"]`, which is
currently tracked but never acted on.

### `--max-tracking-error` vs the server threshold

The client's gate is lag-compensated; the server's is not (≈2.8× smaller on the measured
run). With the client at 0.45 and the server at 0.35, **the server e-stops first** and
you lose the graceful halt. Consider `--max-tracking-error 0.25`.

### One-shot failure modes

- **Any client disconnect** e-stops `rd serve` and exits it (`_handle`'s `finally`, on
  every exit path). One `rd serve` per run.
- **Any wan_va server exception** kills that server too.
- Both need relaunching after every run. The policy server is slow to reload.

### No timeout anywhere

A client that dies mid-rollout leaves the arms holding their last command indefinitely
under PD control — stiff, not limp, and a closed gripper stays closed. Only the
disconnect-triggered `emergency_stop()` recovers them.

---

## 9. Caveat on what a good run proves

All 74 episodes were trained on; there is no held-out set. A successful rollout
demonstrates that the **pipeline** works end to end — adapter loaded, norm stats right,
observation pipeline correct, conversion sound, transport stable. It is **not** a
success-rate estimate and says nothing about generalization.

---

## 10. File map

| path | role |
|---|---|
| `script/run_yam_policy.py` | the live client |
| `script/replay_yam_episode.py` | hardware-path certification, no policy |
| `script/validate_yam_action_convert.py` | offline converter validation, no hardware |
| `yam_bridge/action_convert.py` | relative EEF → absolute joints |
| `yam_bridge/kinematics.py` | shared FK/IK, `widen_arm_joint_limits` |
| `yam_bridge/robot_io.py` | observation decoding, tracking/lag/slew/bandwidth monitors |
| `yam_bridge/va_client.py` | wan_va websocket protocol wrapper |
| `wan_va/configs/va_yam_cover_blocks_cfg.py` | serving config — norm stats live here |

`yam_bridge/` is deliberately torch-free so it imports in raiden's venv.
