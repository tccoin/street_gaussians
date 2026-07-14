import json
import os
import shutil
from glob import glob

import numpy as np

from lib.config import cfg
from lib.datasets.base_readers import get_Sphere_Norm


def image_filename_to_cam(filename):
    return int(filename.split(".")[0].split("_")[-1])


def image_filename_to_frame(filename):
    return int(filename.split(".")[0].split("_")[0])


def _load_json(path, default):
    if not os.path.exists(path):
        return default
    with open(path, "r") as handle:
        return json.load(handle)


def _available_camera_ids(datadir):
    intrinsics_dir = os.path.join(datadir, "intrinsics")
    ids = []
    for path in glob(os.path.join(intrinsics_dir, "*.txt")):
        stem = os.path.splitext(os.path.basename(path))[0]
        if stem.isdigit():
            ids.append(int(stem))
    return sorted(ids)


def _camera_name_by_id(datadir):
    raw = _load_json(os.path.join(datadir, "camera_names.json"), {})
    return {int(k): str(v) for k, v in raw.items()}


def _camera_models_by_id(datadir):
    raw = _load_json(os.path.join(datadir, "camera_models.json"), {})
    return {int(k): v for k, v in raw.items()}


def load_camera_info(datadir):
    ego_pose_dir = os.path.join(datadir, "ego_pose")
    intrinsics_dir = os.path.join(datadir, "intrinsics")
    extrinsics_dir = os.path.join(datadir, "extrinsics")
    camera_ids = _available_camera_ids(datadir)
    camera_models = _camera_models_by_id(datadir)

    intrinsics = {}
    extrinsics = {}
    image_sizes = {}
    for cam_id in camera_ids:
        intrinsic = np.loadtxt(os.path.join(intrinsics_dir, f"{cam_id}.txt"))
        fx, fy, cx, cy = intrinsic[0], intrinsic[1], intrinsic[2], intrinsic[3]
        intrinsics[cam_id] = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        extrinsics[cam_id] = np.loadtxt(os.path.join(extrinsics_dir, f"{cam_id}.txt"))
        model = camera_models.get(cam_id, {})
        if "height" in model and "width" in model:
            image_sizes[cam_id] = (int(model["height"]), int(model["width"]))

    ego_frame_poses = {}
    ego_cam_poses = {cam_id: {} for cam_id in camera_ids}
    for name in sorted(os.listdir(ego_pose_dir)):
        if not name.endswith(".txt"):
            continue
        stem = os.path.splitext(name)[0]
        pose = np.loadtxt(os.path.join(ego_pose_dir, name))
        if "_" not in stem:
            ego_frame_poses[int(stem)] = pose
        else:
            frame_text, cam_text = stem.split("_", 1)
            ego_cam_poses[int(cam_text)][int(frame_text)] = pose

    if ego_frame_poses:
        center_point = np.mean([pose[:3, 3] for pose in ego_frame_poses.values()], axis=0)
        for pose in ego_frame_poses.values():
            pose[:3, 3] -= center_point
        for per_cam in ego_cam_poses.values():
            for pose in per_cam.values():
                pose[:3, 3] -= center_point
    return intrinsics, extrinsics, ego_frame_poses, ego_cam_poses, image_sizes, _camera_name_by_id(datadir)


def _copy_init_ply(datadir):
    source = os.path.join(datadir, "init_ply", "points3D_bkgd.ply")
    target = os.path.join(cfg.model_path, "input_ply", "points3D_bkgd.ply")
    if os.path.exists(target) and not cfg.data.get("regenerate_pcd", False):
        return target
    if not os.path.exists(source):
        raise FileNotFoundError(f"OneDat init point cloud not found: {source}")
    os.makedirs(os.path.dirname(target), exist_ok=True)
    shutil.copy2(source, target)
    return target


def generate_dataparser_outputs(datadir, selected_frames=None, build_pointcloud=True, cameras=None):
    image_dir = os.path.join(datadir, "images")
    image_filenames_all = sorted(glob(os.path.join(image_dir, "*.jpg")) + glob(os.path.join(image_dir, "*.jpeg")) + glob(os.path.join(image_dir, "*.png")))
    if not image_filenames_all:
        raise FileNotFoundError(f"No images found in {image_dir}")

    intrinsics, extrinsics, ego_frame_poses, ego_cam_poses, image_sizes, camera_names = load_camera_info(datadir)
    if cameras is None:
        cameras = sorted(intrinsics.keys())
    cameras = [int(cam) for cam in cameras]

    max_frame_in_data = max(image_filename_to_frame(os.path.basename(path)) for path in image_filenames_all)
    if selected_frames is None:
        start_frame, end_frame = 0, max_frame_in_data
    else:
        start_frame, end_frame = int(selected_frames[0]), int(selected_frames[1])
    end_frame = min(end_frame, max_frame_in_data)
    start_frame = max(0, min(start_frame, end_frame))
    num_frames = end_frame - start_frame + 1

    with open(os.path.join(datadir, "timestamps.json"), "r") as handle:
        timestamps = json.load(handle)

    frames = []
    frames_idx = []
    cams = []
    image_filenames = []
    ixts = []
    exts = []
    poses = []
    c2ws = []
    frames_timestamps = []
    cams_timestamps = []

    for frame in range(start_frame, end_frame + 1):
        frames_timestamps.append(float(timestamps["FRAME"][f"{frame:06d}"]))

    for image_filename in image_filenames_all:
        image_basename = os.path.basename(image_filename)
        frame = image_filename_to_frame(image_basename)
        cam = image_filename_to_cam(image_basename)
        if frame < start_frame or frame > end_frame or cam not in cameras:
            continue
        camera_name = camera_names.get(cam, str(cam))
        ixt = intrinsics[cam]
        ext = extrinsics[cam]
        pose = ego_cam_poses[cam][frame]

        frames.append(frame)
        frames_idx.append(frame - start_frame)
        cams.append(cam)
        image_filenames.append(image_filename)
        ixts.append(ixt)
        exts.append(ext)
        poses.append(pose)
        c2ws.append(pose)
        cams_timestamps.append(float(timestamps[camera_name][f"{frame:06d}"]))

    if not image_filenames:
        raise RuntimeError(f"No OneDat images selected in {datadir} for cameras={cameras} frames={[start_frame, end_frame]}")

    timestamp_offset = min(cams_timestamps + frames_timestamps)
    cams_timestamps = np.asarray(cams_timestamps, dtype=np.float64) - timestamp_offset
    frames_timestamps = np.asarray(frames_timestamps, dtype=np.float64) - timestamp_offset

    if build_pointcloud:
        _copy_init_ply(datadir)

    return {
        "num_frames": num_frames,
        "exts": np.stack(exts, axis=0),
        "ixts": np.stack(ixts, axis=0),
        "poses": np.stack(poses, axis=0),
        "c2ws": np.stack(c2ws, axis=0),
        "obj_tracklets": np.ones((num_frames, 1, 8), dtype=np.float32) * -1.0,
        "obj_info": {},
        "frames": np.asarray(frames, dtype=np.int32),
        "cams": np.asarray(cams, dtype=np.int32),
        "frames_idx": np.asarray(frames_idx, dtype=np.int32),
        "image_filenames": np.asarray(image_filenames),
        "cams_timestamps": cams_timestamps,
        "tracklet_timestamps": frames_timestamps,
        "obj_bounds": [],
        "image_sizes": image_sizes,
        "camera_names": camera_names,
    }


def save_frame_camera_info(output, model_path, source_path="", selected_frames=None, cameras=None):
    os.makedirs(model_path, exist_ok=True)
    cache_path = os.path.join(model_path, "frame_camera_info.npz")
    payload = dict(output)
    payload["schema_version"] = np.asarray([1], dtype=np.int32)
    payload["source_path"] = np.asarray([source_path])
    payload["selected_frames"] = np.asarray(selected_frames if selected_frames is not None else [], dtype=np.int32)
    payload["cameras"] = np.asarray(cameras if cameras is not None else [], dtype=np.int32)
    payload["obj_info"] = np.asarray(payload.get("obj_info", {}), dtype=object)
    payload["image_sizes"] = np.asarray(payload.get("image_sizes", {}), dtype=object)
    payload["camera_names"] = np.asarray(payload.get("camera_names", {}), dtype=object)
    np.savez_compressed(cache_path, **payload)
    return cache_path


def ensure_background_ply(datadir):
    return _copy_init_ply(datadir)


def sphere_from_background_ply(path):
    from lib.datasets.base_readers import fetchPly

    pcd = fetchPly(path)
    return get_Sphere_Norm(pcd.points)
