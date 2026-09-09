import numpy as np
from wan_va.utils.Simple_Remote_Infer.deploy.websocket_client_policy import WebsocketClientPolicy
import argparse
from libero.libero import benchmark
import time
from libero.libero.envs import OffScreenRenderEnv
from pathlib import Path
from tqdm import tqdm
from lerobot.datasets.utils import write_json
import os
import imageio
import cv2


def save_video(real_obs_list, save_path, fps=15, video_names=["observation.images.agentview_rgb", "observation.images.eye_in_hand_rgb"]):
    if not real_obs_list:
        print("❌ No real observation frames")
        return

    first_obs = real_obs_list[0]
    base_h, width_base = first_obs[video_names[0]].shape[:2]
    target_size = (width_base, base_h)
    
    print(f"Saving video: {len(real_obs_list)} frames...")

    final_frames = [
        np.hstack([cv2.resize(obs[name], target_size) for name in video_names]).astype(np.uint8)
        for obs in real_obs_list
    ]

    imageio.mimsave(save_path, final_frames, fps=fps)
    print(f"✅ Video saved to: {save_path}")


def add_title_bar(img, text, font_scale=0.4, thickness=1, bar_height=20):
    """Add a black title bar with centered white text above an image."""
    h, w, _ = img.shape
    title_bar = np.zeros((bar_height, w, 3), dtype=np.uint8)
    (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    text_x = max(0, (w - text_w) // 2)
    text_y = (bar_height + text_h) // 2 - 2
    cv2.putText(title_bar, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return np.vstack([title_bar, img])


def resize_h(img, h):
    if img.shape[0] != h:
        w = int(img.shape[1] * h / img.shape[0])
        img = cv2.resize(img, (w, h))
    img = np.ascontiguousarray(img)
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)
    return img


def split_two_panel_imagined(img_frame):
    """Server (env_type='none') concatenates obs_cam_keys along width.
    For LIBERO that's [agentview ‖ eye_in_hand]. Returns (agentview, eye_in_hand) as uint8."""
    if img_frame.dtype != np.uint8:
        if img_frame.max() <= 1.0001:
            img_frame = (img_frame * 255).astype(np.uint8)
        else:
            img_frame = img_frame.astype(np.uint8)
    W = img_frame.shape[1]
    half = W // 2
    return img_frame[:, :half], img_frame[:, half:]


def save_comparison_video(real_obs_list, imagined_video, save_path, fps=15,
                          video_names=("observation.images.agentview_rgb",
                                       "observation.images.eye_in_hand_rgb")):
    if not real_obs_list:
        print("❌ No real observation frames")
        return

    n_real = len(real_obs_list)
    if imagined_video is not None:
        if isinstance(imagined_video, list):
            imagined_video = np.concatenate(imagined_video, 0)
        n_imagined = len(imagined_video)
    else:
        n_imagined = 0

    print(f"Saving comparison video: Real {n_real} frames, Imagined {n_imagined} frames...")

    first_obs = real_obs_list[0]
    base_h = first_obs[video_names[0]].shape[0]
    row_real_template = np.hstack([resize_h(first_obs[name], base_h) for name in video_names])
    target_width = row_real_template.shape[1]

    final_frames = []
    for i in range(n_real):
        obs = real_obs_list[i]
        row_real = np.hstack([resize_h(obs[name], base_h) for name in video_names])
        if row_real.shape[1] != target_width:
            row_real = cv2.resize(row_real, (target_width, base_h))
        row_real = np.ascontiguousarray(row_real)
        row_real = add_title_bar(row_real, "Real Obs (Agentview / Eye-in-hand)")

        if imagined_video is not None and i < n_imagined:
            ag_im, eih_im = split_two_panel_imagined(imagined_video[i])
            row_imagined = np.hstack([resize_h(ag_im, base_h), resize_h(eih_im, base_h)])
            if row_imagined.shape[1] != target_width:
                row_imagined = cv2.resize(row_imagined, (target_width, base_h))
        else:
            row_imagined = np.zeros((base_h, target_width, 3), dtype=np.uint8)
            cv2.putText(row_imagined, "Coming soon",
                        (target_width // 2 - 50, base_h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

        row_imagined = np.ascontiguousarray(row_imagined)
        row_imagined = add_title_bar(row_imagined, "Imagined Video Stream")
        full_frame = np.ascontiguousarray(np.vstack([row_real, row_imagined]))
        final_frames.append(full_frame)

    imageio.mimsave(save_path, final_frames, fps=fps)
    print(f"✅ Comparison video saved to: {save_path}")


def construct_single_env(env_args):
    count = 0
    env = None
    env_creation = False
    while not env_creation and count < 5:
        try:
            env = OffScreenRenderEnv(**env_args)
            env_creation = True
        except Exception as e:
            print(f"Error!!!  construct env failed: {e}")
            time.sleep(5)
            count += 1
    if count >= 5:
        return None
    return env


def _extract_obs(obs):
    """
    Extract agentview and eye_in_hand images from raw env obs dict.

    Avoids torch round-trip: the env already returns uint8 numpy arrays [H, W, C].
    We just flip the vertical axis ([::-1]) and make a contiguous copy once.
    """
    agentview = np.ascontiguousarray(obs["agentview_image"][::-1])
    eye_in_hand = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1])
    return {"observation.images.agentview_rgb": agentview, "observation.images.eye_in_hand_rgb": eye_in_hand}


def init_single_env(env_in, init_state):
    env_in.reset()
    env_in.set_init_state(init_state)
    for _ in range(5):
        obs, _, _, _ = env_in.step([0.] * 7)
    return _extract_obs(obs)


def env_one_step(env_in, action):
    obs, _, done, _ = env_in.step(action)
    return _extract_obs(obs), done


def run_one(model, libero_benchmark, task_idx, out_dir, episode_idx, save_visualization=False):
    benchmark_dict = benchmark.get_benchmark_dict()
    benchmark_instance = benchmark_dict[libero_benchmark]()
    num_tasks = benchmark_instance.get_num_tasks()
    assert task_idx < num_tasks, f"Error: error id must smaller than {num_tasks}"
    prompt = benchmark_instance.get_task(task_idx).language
    env_args = {
                "bddl_file_name": benchmark_instance.get_task_bddl_file_path(task_idx),
                "camera_heights": 128,
                "camera_widths": 128,
            }
    init_states = benchmark_instance.get_task_init_states(task_idx)

    cur_env = construct_single_env(env_args)
    first_obs = init_single_env(cur_env, init_states[episode_idx % init_states.shape[0]])

    ret = model.infer(dict(reset=True, prompt=prompt))

    full_obs_list = []
    gen_video_list = []
    done = False
    first = True
    while cur_env.env.timestep < 800:
        ret = model.infer(dict(obs=first_obs, prompt=prompt,
                               save_visualization=save_visualization))
        action = ret['action']
        if save_visualization and 'video' in ret:
            gen_video_list.append(ret['video'])

        key_frame_list = []
        assert action.shape[2] % 4 == 0
        action_per_frame = action.shape[2] // 4
        start_idx = 1 if first else 0
        for i in range(start_idx, action.shape[1]):
            for j in range(action.shape[2]):
                ee_action = action[:, i, j]
                observes, done = env_one_step(cur_env, ee_action)
                if done:
                    break
                if (j+1) % action_per_frame == 0:
                    full_obs_list.append(observes)
                    key_frame_list.append(observes)

            if done:
                break

        first = False

        if done:
            break
        else:
            model.infer(dict(obs=key_frame_list, compute_kv_cache=True, imagine=False, state=action))

    out_file = Path(out_dir) / libero_benchmark / f"{task_idx}_{prompt.replace(' ', '_')}" / f"{episode_idx}_{done}.mp4"
    out_file.parent.mkdir(exist_ok=True, parents=True)

    save_video(
        real_obs_list=full_obs_list,
        save_path=out_file,
        fps=60,
        video_names=["observation.images.agentview_rgb", "observation.images.eye_in_hand_rgb"]
    )

    if save_visualization:
        cmp_path = out_file.parent / "comparison" / out_file.name
        cmp_path.parent.mkdir(exist_ok=True, parents=True)
        save_comparison_video(
            real_obs_list=[first_obs] + full_obs_list,
            imagined_video=gen_video_list if gen_video_list else None,
            save_path=str(cmp_path),
            fps=15,
        )

    cur_env.close()
    return done


def run(libero_benchmark, port, out_dir, test_num, task_range=None, save_visualization=False):
    '''
        task_range: [start, end) for splitting tasks
    '''
    if task_range is None:
        benchmark_dict = benchmark.get_benchmark_dict()
        benchmark_instance = benchmark_dict[libero_benchmark]()
        num_tasks = benchmark_instance.get_num_tasks()
        progress_bar = tqdm(range(num_tasks), total=num_tasks)
    else:
        assert len(task_range) == 2, f'task_range: [start, end) for splitting tasks, however, task_range: {task_range}'
        num_tasks = task_range[1] - task_range[0]
        progress_bar = tqdm(range(task_range[0], task_range[1]), total=num_tasks)

    print(f"#################### Use benchmark: {libero_benchmark}, num_tasks: {num_tasks} #############")
    model = WebsocketClientPolicy(port=port)

    video_save_root_dict = None

    episode_list = range(test_num)
    for task_idx in progress_bar:
        if video_save_root_dict is not None and task_idx in video_save_root_dict:
            video_save_list = os.listdir(os.path.join(out_dir, libero_benchmark, video_save_root_dict[task_idx]))
            video_states = [1 for file in video_save_list if file.split('_')[1].split('.')[0] == 'True']
            succ_num = float(len(video_states))
            episode_list = range(len(video_save_list), test_num)
        else:
            succ_num = 0.

        for episode_idx in tqdm(episode_list, total=len(episode_list)):
            res_i = run_one(model, libero_benchmark, task_idx, out_dir, episode_idx,
                            save_visualization=save_visualization)
            succ_num += res_i
            succ_rate = succ_num / (episode_idx + 1)
            print(f"Success rate: {succ_rate}, success num: {succ_num}, total num: {episode_idx + 1}")
            out_file = Path(out_dir) / f"{libero_benchmark}_{task_idx}.json"
            out_file.parent.mkdir(exist_ok=True, parents=True)
            write_json({
                "succ_num": succ_num,
                "total_num": episode_idx + 1.,
                "succ_rate": succ_rate,
                }, out_file
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--libero-benchmark",
        type=str,
        default="libero_10",
        choices=["libero_10", "libero_goal", "libero_spatial", "libero_object"],
        help="Benchmark name",
    )
    parser.add_argument(
        "--task-range",
        type=int,
        nargs="+",
        default=[0, 10],
        help="Task range [start, end) for splitting tasks",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=23908,
        help="WebSocket port",
    )
    parser.add_argument(
        "--test-num",
        type=int,
        default=50,
        help="Number of test episodes",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="outputs/libero",
        help="Output directory for results",
    )
    parser.add_argument(
        "--save-visualization",
        type=bool,
        default=False,
        help="Also save real-vs-imagined comparison MP4 to <out_dir>/<benchmark>/<task>/comparison/",
    )
    args = parser.parse_args()
    run(**vars(args))
    print("Finish all process!!!!!!!!!!!!")


if __name__ == "__main__":
    main()
