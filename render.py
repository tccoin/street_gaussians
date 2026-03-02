import torch
import os
import json
import copy
import numpy as np
from tqdm import tqdm

from lib.models.street_gaussian_model import StreetGaussianModel
from lib.models.street_gaussian_renderer import StreetGaussianRenderer
from lib.datasets.dataset import Dataset
from lib.models.scene import Scene
from lib.utils.general_utils import safe_state
from lib.config import cfg
from lib.visualizers.base_visualizer import BaseVisualizer as Visualizer
from lib.visualizers.street_gaussian_visualizer import StreetGaussianVisualizer
import time


def render_sets():
    cfg.render.save_image = True
    cfg.render.save_video = True  # save .mp4 videos for visualization

    with torch.no_grad():
        dataset = Dataset()
        gaussians = StreetGaussianModel(dataset.scene_info.metadata)
        scene = Scene(gaussians=gaussians, dataset=dataset)
        renderer = StreetGaussianRenderer()

        times = []
        if not cfg.eval.skip_train:
            save_dir = os.path.join(
                cfg.model_path, "train", f"ours_{scene.loaded_iter}"
            )
            visualizer = Visualizer(save_dir)
            cameras = scene.getTrainCameras()
            for idx, camera in enumerate(
                tqdm(cameras, desc="Rendering Training View")
            ):
                torch.cuda.synchronize()
                start_time = time.time()
                result = renderer.render(camera, gaussians)

                torch.cuda.synchronize()
                end_time = time.time()
                times.append((end_time - start_time) * 1000)

                visualizer.visualize(result, camera)
            if hasattr(visualizer, "summarize"):
                visualizer.summarize()

        if not cfg.eval.skip_test:
            save_dir = os.path.join(
                cfg.model_path, "test", f"ours_{scene.loaded_iter}"
            )
            visualizer = Visualizer(save_dir)
            cameras = scene.getTestCameras()
            for idx, camera in enumerate(
                tqdm(cameras, desc="Rendering Testing View")
            ):
                torch.cuda.synchronize()
                start_time = time.time()

                result = renderer.render(camera, gaussians)

                torch.cuda.synchronize()
                end_time = time.time()
                times.append((end_time - start_time) * 1000)

                visualizer.visualize(result, camera)
            if hasattr(visualizer, "summarize"):
                visualizer.summarize()

        print(times)
        if len(times) > 1:
            print("average rendering time: ", sum(times[1:]) / len(times[1:]))
        elif len(times) > 0:
            print("average rendering time: ", times[0])


def offset_camera(camera, height_offset=2.0, pitch_degrees=-15.0):
    """
    Create a copy of camera with an offset in the **world / dataset** frame.

    Behavior depends on dataset coordinate convention:
    - If cfg.data.type == "Waymo": world frame is X forward, Y left, Z up
      → "height" is along +Z.
    - Otherwise: infer "up" from the camera rotation (Y axis of c2w).

    - height_offset: meters to move camera *up* in world space.
    - pitch_degrees: degrees to rotate camera down around its local X axis
      (negative values look more towards the ground).
    """
    # Camera-to-world matrix (world frame = dataset frame)
    c2w = camera.get_extrinsic()

    # Current camera position and rotation
    cam_pos = c2w[:3, 3]
    cam_rot = c2w[:3, :3]

    # Height offset in world space.
    if getattr(cfg.data, "type", None) == "Waymo":
        # Waymo vehicle/world frame: X forward, Y left, Z up
        world_offset = np.array([0.0, 0.0, height_offset], dtype=np.float32)
    else:
        # Generic case: use camera's "up" direction mapped to world.
        # Second column of rotation is camera's Y axis in world frame.
        cam_up_world = cam_rot[:, 1]  # shape (3,)
        world_y_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        # If camera-up roughly aligns with +Y, treat that as up; otherwise flip.
        if np.dot(cam_up_world, world_y_up) >= 0:
            world_up = cam_up_world
        else:
            world_up = -cam_up_world
        world_offset = world_up * float(height_offset)

    new_cam_pos = cam_pos + world_offset

    # Pitch rotation around camera's local X-axis.
    pitch_rad = np.radians(pitch_degrees)
    pitch_rot_x = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(pitch_rad), -np.sin(pitch_rad)],
            [0.0, np.sin(pitch_rad), np.cos(pitch_rad)],
        ],
        dtype=np.float32,
    )

    # Apply pitch in camera frame
    new_cam_rot = cam_rot @ pitch_rot_x

    # New camera-to-world matrix
    new_c2w = np.eye(4, dtype=np.float32)
    new_c2w[:3, :3] = new_cam_rot
    new_c2w[:3, 3] = new_cam_pos

    # Create new camera with modified pose
    new_camera = copy.deepcopy(camera)
    new_camera.set_extrinsic(new_c2w)

    return new_camera


def render_trajectory():
    cfg.render.save_image = False
    cfg.render.save_video = True

    # Get offset parameters from config or use defaults
    height_offset = cfg.render.get("camera_height_offset", 2.0)
    pitch_degrees = cfg.render.get("camera_pitch_offset", -15.0)

    with torch.no_grad():
        dataset = Dataset()
        gaussians = StreetGaussianModel(dataset.scene_info.metadata)

        scene = Scene(gaussians=gaussians, dataset=dataset)
        renderer = StreetGaussianRenderer()

        train_cameras = scene.getTrainCameras()
        test_cameras = scene.getTestCameras()
        cameras = train_cameras + test_cameras
        cameras = list(sorted(cameras, key=lambda x: x.id))

        # Render original trajectory
        save_dir = os.path.join(
            cfg.model_path, "trajectory", f"ours_{scene.loaded_iter}"
        )
        visualizer = StreetGaussianVisualizer(save_dir)

        for idx, camera in enumerate(
            tqdm(cameras, desc="Rendering Trajectory (original)")
        ):
            result = renderer.render_all(camera, gaussians)
            visualizer.visualize(result, camera)
        visualizer.summarize()

        # Render offset trajectory
        if height_offset != 0.0 or pitch_degrees != 0.0:
            save_dir_offset = os.path.join(
                cfg.model_path, "trajectory_offset", f"ours_{scene.loaded_iter}"
            )
            visualizer_offset = StreetGaussianVisualizer(save_dir_offset)

            print(
                f"\nRendering offset trajectory: height={height_offset}m, pitch={pitch_degrees}deg"
            )
            for idx, camera in enumerate(
                tqdm(cameras, desc="Rendering Trajectory (offset)")
            ):
                offset_cam = offset_camera(
                    camera,
                    height_offset=height_offset,
                    pitch_degrees=pitch_degrees,
                )
                result = renderer.render_all(offset_cam, gaussians)
                visualizer_offset.visualize(result, offset_cam)
            visualizer_offset.summarize()
            print(f"Offset trajectory saved to: {save_dir_offset}")


if __name__ == "__main__":
    print("Rendering " + cfg.model_path)
    safe_state(cfg.eval.quiet)

    if cfg.mode == "evaluate":
        render_sets()
    elif cfg.mode == "trajectory":
        render_trajectory()
    else:
        raise NotImplementedError()

