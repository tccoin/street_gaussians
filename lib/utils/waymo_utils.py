import os
import numpy as np
import cv2
import torch
import json
import open3d as o3d
import math
from glob import glob
from tqdm import tqdm
from lib.config import cfg
from lib.utils.box_utils import bbox_to_corner3d, inbbox_points, get_bound_2d_mask
from lib.utils.colmap_utils import read_points3D_binary, read_extrinsics_binary, qvec2rotmat
from lib.utils.data_utils import get_val_frames
from lib.utils.graphics_utils import get_rays, sphere_intersection
from lib.utils.general_utils import matrix_to_quaternion, quaternion_to_matrix_numpy
from lib.datasets.base_readers import storePly, get_Sphere_Norm

waymo_track2label = {"vehicle": 0, "pedestrian": 1, "cyclist": 2, "sign": 3, "misc": -1}

_camera2label = {
    'FRONT': 0,
    'FRONT_LEFT': 1,
    'FRONT_RIGHT': 2,
    'SIDE_LEFT': 3,
    'SIDE_RIGHT': 4,
}

_label2camera = {
    0: 'FRONT',
    1: 'FRONT_LEFT',
    2: 'FRONT_RIGHT',
    3: 'SIDE_LEFT',
    4: 'SIDE_RIGHT',
}
image_heights = [1280, 1280, 1280, 886, 886]
image_widths = [1920, 1920, 1920, 1920, 1920]
image_filename_to_cam = lambda x: int(os.path.splitext(x)[0].split('_')[-1])
image_filename_to_frame = lambda x: int(os.path.splitext(x)[0].split('_')[0])
FRAME_CAMERA_INFO_FILENAME = 'frame_camera_info.npz'


def frame_camera_info_path(model_path):
    if str(model_path).endswith('.npz'):
        return str(model_path)
    return os.path.join(model_path, FRAME_CAMERA_INFO_FILENAME)


def should_keep_static_vehicle_actor(object_info, obj_pose_vehicle, min_frames, max_min_distance):
    if object_info.get('class') != 'vehicle':
        return False
    if obj_pose_vehicle.shape[0] < min_frames:
        return False
    min_distance = np.linalg.norm(obj_pose_vehicle[:, :2], axis=1).min()
    return min_distance <= max_min_distance


def _pi3x_track_point_counts(datadir):
    pi3x_path = os.path.join(datadir, 'pi3x_object_points.npz')
    if not os.path.exists(pi3x_path):
        return {}

    counts = {}
    with np.load(pi3x_path, allow_pickle=True) as payload:
        track_ids_by_frame = payload['track_ids'].item()

    for track_ids in track_ids_by_frame.values():
        track_ids = np.asarray(track_ids)
        if track_ids.size == 0:
            continue
        track_ids, frame_counts = np.unique(track_ids.astype(np.int32), return_counts=True)
        for track_id, count in zip(track_ids, frame_counts):
            counts[int(track_id)] = counts.get(int(track_id), 0) + int(count)
    return counts


def _project_vehicle_points(points_vehicle, K, ext, height, width):
    vehicle2cam = np.linalg.inv(ext)
    points_cam = points_vehicle @ vehicle2cam[:3, :3].T + vehicle2cam[:3, 3:].T
    valid = points_cam[:, 2] > 1e-3
    points_img = points_cam @ K.T
    uv = points_img[:, :2] / np.clip(points_img[:, 2:], 1e-3, None)
    uv = np.round(uv).astype(np.int32)
    valid &= (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
    return valid, uv


def _load_instance_mask_for_image(datadir, image_filename, height, width):
    stem = os.path.splitext(os.path.basename(image_filename))[0]
    path = os.path.join(datadir, 'instance_masks', f'{stem}.npz')
    if not os.path.exists(path):
        return None
    try:
        with np.load(path, allow_pickle=False) as payload:
            if 'instance_mask' not in payload.files:
                return None
            mask = np.asarray(payload['instance_mask'])
    except Exception:
        return None
    if mask.shape[:2] != (height, width):
        mask = cv2.resize(mask.astype(np.uint16), (width, height), interpolation=cv2.INTER_NEAREST)
    return mask


def _filter_object_points_by_instance_masks(
    datadir,
    track_id,
    points_xyz,
    points_rgb,
    object_tracklets_vehicle,
    image_filenames,
    frames_idx,
    ixts,
    exts,
    train_frames,
):
    if not bool(cfg.data.get('obj_point_mask_filter', False)):
        return points_xyz, points_rgb
    if points_xyz.shape[0] == 0:
        return points_xyz, points_rgb

    min_visible = int(cfg.data.get('obj_point_mask_filter_min_visible', 2))
    min_inside_ratio = float(cfg.data.get('obj_point_mask_filter_min_inside_ratio', 0.5))
    min_points = int(cfg.data.get('obj_point_mask_filter_min_points', 2000))
    max_images = int(cfg.data.get('obj_point_mask_filter_max_images', 64))
    max_outside = int(cfg.data.get('obj_point_mask_filter_max_outside', -1))
    track_id = int(track_id)
    train_frames = set(train_frames)

    image_indices = []
    for image_idx, frame_idx in enumerate(frames_idx):
        if int(frame_idx) not in train_frames:
            continue
        if np.any(object_tracklets_vehicle[int(frame_idx), :, 0] == track_id):
            image_indices.append(image_idx)
    if len(image_indices) > max_images:
        selected = np.linspace(0, len(image_indices) - 1, max_images).round().astype(np.int32)
        image_indices = [image_indices[i] for i in selected]
    if len(image_indices) == 0:
        return points_xyz, points_rgb

    points_h = np.concatenate(
        [points_xyz, np.ones((points_xyz.shape[0], 1), dtype=points_xyz.dtype)],
        axis=-1,
    )
    visible_count = np.zeros((points_xyz.shape[0],), dtype=np.int32)
    inside_count = np.zeros((points_xyz.shape[0],), dtype=np.int32)

    for image_idx in image_indices:
        frame_idx = int(frames_idx[image_idx])
        tracklet_idx = np.argwhere(object_tracklets_vehicle[frame_idx, :, 0] == track_id)
        if len(tracklet_idx) == 0:
            continue
        image = cv2.imread(image_filenames[image_idx], cv2.IMREAD_COLOR)
        if image is None:
            continue
        height, width = image.shape[:2]
        instance_mask = _load_instance_mask_for_image(datadir, image_filenames[image_idx], height, width)
        if instance_mask is None or not np.any(instance_mask == track_id):
            continue

        tracklet = object_tracklets_vehicle[frame_idx, tracklet_idx[0, 0]]
        obj_pose_vehicle = np.eye(4)
        obj_pose_vehicle[:3, :3] = quaternion_to_matrix_numpy(tracklet[4:8])
        obj_pose_vehicle[:3, 3] = tracklet[1:4]
        points_vehicle = (points_h @ obj_pose_vehicle.T)[:, :3]
        valid, uv = _project_vehicle_points(points_vehicle, ixts[image_idx], exts[image_idx], height, width)
        if not valid.any():
            continue
        visible_count[valid] += 1
        inside = np.zeros_like(valid)
        inside[valid] = instance_mask[uv[valid, 1], uv[valid, 0]] == track_id
        inside_count[inside] += 1

    observed = visible_count > 0
    if not observed.any():
        return points_xyz, points_rgb
    inside_ratio = np.zeros((points_xyz.shape[0],), dtype=np.float32)
    inside_ratio[observed] = inside_count[observed] / np.maximum(visible_count[observed], 1)
    outside_count = visible_count - inside_count
    keep = (visible_count >= min_visible) & (inside_ratio >= min_inside_ratio)
    if max_outside >= 0:
        keep &= outside_count <= max_outside

    if int(keep.sum()) < min_points:
        candidates = np.where(observed)[0]
        if candidates.size == 0:
            return points_xyz, points_rgb
        order = np.lexsort((outside_count[candidates], -visible_count[candidates], -inside_ratio[candidates]))
        fallback = candidates[order[: min(min_points, candidates.size)]]
        keep = np.zeros_like(keep)
        keep[fallback] = True

    print(
        f'filtered obj_{track_id:03d} pointcloud by instance masks: '
        f'{points_xyz.shape[0]} -> {int(keep.sum())} '
        f'(views={len(image_indices)}, min_visible={min_visible}, '
        f'min_inside_ratio={min_inside_ratio}, max_outside={max_outside})'
    )
    return points_xyz[keep], points_rgb[keep]


def _filter_object_local_pointcloud(track_id, points_xyz, points_rgb, object_info):
    if not bool(cfg.data.get('obj_local_robust_filter', False)):
        return points_xyz, points_rgb
    if points_xyz.shape[0] == 0:
        return points_xyz, points_rgb

    keep = np.isfinite(points_xyz).all(axis=1) & np.isfinite(points_rgb).all(axis=1)
    info = object_info.get(int(track_id), {})
    bbox_scale = float(cfg.data.get('obj_local_filter_bbox_scale', 1.1))
    if bbox_scale > 0.0 and all(k in info for k in ('length', 'width', 'height')):
        half_extent = np.array(
            [info['length'], info['width'], info['height']],
            dtype=np.float32,
        ) * 0.5 * bbox_scale
        keep &= np.all(np.abs(points_xyz) <= half_extent[None], axis=1)

    percentile_clip = float(cfg.data.get('obj_local_filter_percentile_clip', 0.0))
    if 0.0 < percentile_clip < 50.0 and keep.sum() > 0:
        kept_points = points_xyz[keep]
        lo = np.percentile(kept_points, percentile_clip, axis=0)
        hi = np.percentile(kept_points, 100.0 - percentile_clip, axis=0)
        keep &= np.all(points_xyz >= lo[None], axis=1) & np.all(points_xyz <= hi[None], axis=1)

    min_points = int(cfg.data.get('obj_local_filter_min_points', 0))
    if min_points > 0 and int(keep.sum()) < min_points:
        print(
            f'skip object-local robust filter for obj_{int(track_id):03d}: '
            f'{points_xyz.shape[0]} -> {int(keep.sum())} would fall below min_points={min_points}'
        )
        return points_xyz, points_rgb

    print(
        f'filtered obj_{int(track_id):03d} pointcloud in object-local frame: '
        f'{points_xyz.shape[0]} -> {int(keep.sum())} '
        f'(bbox_scale={bbox_scale}, percentile_clip={percentile_clip})'
    )
    return points_xyz[keep], points_rgb[keep]


def save_frame_camera_info(output, model_path, source_path=None, selected_frames=None, cameras=None):
    cache_path = frame_camera_info_path(model_path)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    image_filenames = [
        os.path.join('images', os.path.basename(path))
        for path in output['image_filenames']
    ]
    payload = {
        'schema_version': np.array([1], dtype=np.int32),
        'source_path': np.array([source_path or '']),
        'selected_frames': np.array(selected_frames if selected_frames is not None else [], dtype=np.int32),
        'cameras': np.array(cameras if cameras is not None else [], dtype=np.int32),
        'num_frames': np.array([int(output['num_frames'])], dtype=np.int32),
        'exts': np.asarray(output['exts']),
        'ixts': np.asarray(output['ixts']),
        'poses': np.asarray(output['poses']),
        'c2ws': np.asarray(output['c2ws']),
        'frames': np.asarray(output['frames'], dtype=np.int32),
        'cams': np.asarray(output['cams'], dtype=np.int32),
        'frames_idx': np.asarray(output['frames_idx'], dtype=np.int32),
        'image_filenames': np.asarray(image_filenames),
        'cams_timestamps': np.asarray(output['cams_timestamps'], dtype=np.float64),
        'tracklet_timestamps': np.asarray(output['tracklet_timestamps'], dtype=np.float64),
        'obj_tracklets': np.asarray(output['obj_tracklets']),
        'obj_info': np.array([output['obj_info']], dtype=object),
    }
    np.savez_compressed(cache_path, **payload)
    print(f'Saved frame/camera info cache to {cache_path}')
    return cache_path


def load_frame_camera_info(model_path):
    cache_path = frame_camera_info_path(model_path)
    with np.load(cache_path, allow_pickle=True) as data:
        output = {
            'num_frames': int(data['num_frames'][0]),
            'exts': data['exts'],
            'ixts': data['ixts'],
            'poses': data['poses'],
            'c2ws': data['c2ws'],
            'frames': data['frames'].astype(np.int32).tolist(),
            'cams': data['cams'].astype(np.int32).tolist(),
            'frames_idx': data['frames_idx'].astype(np.int32).tolist(),
            'image_filenames': data['image_filenames'].astype(str).tolist(),
            'cams_timestamps': data['cams_timestamps'],
            'tracklet_timestamps': data['tracklet_timestamps'],
            'obj_tracklets': data['obj_tracklets'],
            'obj_info': data['obj_info'][0].item() if hasattr(data['obj_info'][0], 'item') else data['obj_info'][0],
        }
    return output


def _camera_ids_from_dir(path):
    if not os.path.exists(path):
        return []
    ids = []
    for filename in os.listdir(path):
        stem, ext = os.path.splitext(filename)
        if ext == '.txt' and stem.isdigit():
            ids.append(int(stem))
    return sorted(ids)


def load_camera_names(datadir):
    camera_names_path = os.path.join(datadir, 'camera_names.json')
    if os.path.exists(camera_names_path):
        with open(camera_names_path, 'r') as f:
            payload = json.load(f)
        return {int(k): v for k, v in payload.items()}
    return dict(_label2camera)


def get_camera_ids(datadir):
    camera_ids = _camera_ids_from_dir(os.path.join(datadir, 'intrinsics'))
    if camera_ids:
        return camera_ids
    camera_names = load_camera_names(datadir)
    return sorted(camera_names.keys())


def _image_size(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f'Could not read image: {image_path}')
    h, w = image.shape[:2]
    return h, w

# load ego pose and camera calibration(extrinsic and intrinsic)
def load_camera_info(datadir):
    ego_pose_dir = os.path.join(datadir, 'ego_pose')
    extrinsics_dir = os.path.join(datadir, 'extrinsics')
    intrinsics_dir = os.path.join(datadir, 'intrinsics')
    camera_ids = get_camera_ids(datadir)
    if not camera_ids:
        raise FileNotFoundError(f'No camera intrinsics found in {intrinsics_dir}')
    max_cam_id = max(camera_ids)

    intrinsics = [None for _ in range(max_cam_id + 1)]
    extrinsics = [None for _ in range(max_cam_id + 1)]
    for i in camera_ids:
        intrinsic = np.loadtxt(os.path.join(intrinsics_dir,  f"{i}.txt"))
        fx, fy, cx, cy = intrinsic[0], intrinsic[1], intrinsic[2], intrinsic[3]
        intrinsic = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        intrinsics[i] = intrinsic

    for i in camera_ids:
        cam_to_ego = np.loadtxt(os.path.join(extrinsics_dir,  f"{i}.txt"))
        extrinsics[i] = cam_to_ego

    ego_frame_poses = []
    ego_cam_poses = [[] for i in range(max_cam_id + 1)]
    ego_pose_paths = sorted(os.listdir(ego_pose_dir))
    for ego_pose_path in ego_pose_paths:

        # frame pose
        if '_' not in ego_pose_path:
            ego_frame_pose = np.loadtxt(os.path.join(ego_pose_dir, ego_pose_path))
            ego_frame_poses.append(ego_frame_pose)
        else:
            cam = image_filename_to_cam(ego_pose_path)
            ego_cam_pose = np.loadtxt(os.path.join(ego_pose_dir, ego_pose_path))
            ego_cam_poses[cam].append(ego_cam_pose)

    # center ego pose
    ego_frame_poses = np.array(ego_frame_poses)
    center_point = np.mean(ego_frame_poses[:, :3, 3], axis=0)
    ego_frame_poses[:, :3, 3] -= center_point # [num_frames, 4, 4]

    ego_cam_poses = [
        np.array(ego_cam_poses[i]) if len(ego_cam_poses[i]) > 0 else np.array(ego_frame_poses)
        for i in range(max_cam_id + 1)
    ]
    ego_cam_poses = np.array(ego_cam_poses)
    ego_cam_poses[:, :, :3, 3] -= center_point # [5, num_frames, 4, 4]
    return intrinsics, extrinsics, ego_frame_poses, ego_cam_poses

# calculate obj pose in world frame
# box_info: box_center_x box_center_y box_center_z box_heading
def make_obj_pose(ego_pose, box_info):
    tx, ty, tz, heading = box_info
    c = math.cos(heading)
    s = math.sin(heading)
    rotz_matrix = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    obj_pose_vehicle = np.eye(4)
    obj_pose_vehicle[:3, :3] = rotz_matrix
    obj_pose_vehicle[:3, 3] = np.array([tx, ty, tz])
    obj_pose_world = np.matmul(ego_pose, obj_pose_vehicle)

    obj_rotation_vehicle = torch.from_numpy(obj_pose_vehicle[:3, :3]).float().unsqueeze(0)
    obj_quaternion_vehicle = matrix_to_quaternion(obj_rotation_vehicle).squeeze(0).numpy()
    obj_quaternion_vehicle = obj_quaternion_vehicle / np.linalg.norm(obj_quaternion_vehicle)
    obj_position_vehicle = obj_pose_vehicle[:3, 3]
    obj_pose_vehicle = np.concatenate([obj_position_vehicle, obj_quaternion_vehicle])

    obj_rotation_world = torch.from_numpy(obj_pose_world[:3, :3]).float().unsqueeze(0)
    obj_quaternion_world = matrix_to_quaternion(obj_rotation_world).squeeze(0).numpy()
    obj_quaternion_world = obj_quaternion_world / np.linalg.norm(obj_quaternion_world)
    obj_position_world = obj_pose_world[:3, 3]
    obj_pose_world = np.concatenate([obj_position_world, obj_quaternion_world])

    return obj_pose_vehicle, obj_pose_world




def get_obj_pose_tracking(datadir, selected_frames, ego_poses, cameras=[0, 1, 2, 3, 4]):
    tracklets_ls = []
    objects_info = {}

    if cfg.data.get('use_tracker', False):
        tracklet_path = os.path.join(datadir, 'track/track_info_castrack.txt')
        tracklet_camera_vis_path = os.path.join(datadir, 'track/track_camera_vis_castrack.json')
    else:
        tracklet_path = os.path.join(datadir, 'track/track_info.txt')
        tracklet_camera_vis_path = os.path.join(datadir, 'track/track_camera_vis.json')

    print(f'Loading from : {tracklet_path}')
    f = open(tracklet_path, 'r')
    tracklets_str = f.read().splitlines()
    tracklets_str = tracklets_str[1:]

    f = open(tracklet_camera_vis_path, 'r')
    tracklet_camera_vis = json.load(f)

    start_frame, end_frame = selected_frames[0], selected_frames[1]

    image_dir = os.path.join(datadir, 'images')
    n_frames = len(ego_poses)
    n_obj_in_frame = np.zeros(n_frames)

    for tracklet in tracklets_str:
        tracklet = tracklet.split()
        frame_id = int(tracklet[0])
        track_id = int(tracklet[1])
        object_class = tracklet[2]

        if object_class in ['sign', 'misc']:
            continue

        cameras_vis_list = tracklet_camera_vis[str(track_id)][str(frame_id)]
        join_cameras_list = list(set(cameras) & set(cameras_vis_list))
        if len(join_cameras_list) == 0:
            continue

        if track_id not in objects_info.keys():
            objects_info[track_id] = dict()
            objects_info[track_id]['track_id'] = track_id
            objects_info[track_id]['class'] = object_class
            objects_info[track_id]['class_label'] = waymo_track2label[object_class]
            objects_info[track_id]['height'] = float(tracklet[4])
            objects_info[track_id]['width'] = float(tracklet[5])
            objects_info[track_id]['length'] = float(tracklet[6])
        else:
            objects_info[track_id]['height'] = max(objects_info[track_id]['height'], float(tracklet[4]))
            objects_info[track_id]['width'] = max(objects_info[track_id]['width'], float(tracklet[5]))
            objects_info[track_id]['length'] = max(objects_info[track_id]['length'], float(tracklet[6]))

        tr_array = np.concatenate(
            [np.array(tracklet[:2]).astype(np.float64), np.array([type]), np.array(tracklet[4:]).astype(np.float64)]
        )
        tracklets_ls.append(tr_array)
        n_obj_in_frame[frame_id] += 1

    tracklets_array = np.array(tracklets_ls)
    max_obj_per_frame = int(n_obj_in_frame[start_frame:end_frame + 1].max())
    num_frames = end_frame - start_frame + 1
    visible_objects_ids = np.ones([num_frames, max_obj_per_frame]) * -1.0
    visible_objects_pose_vehicle = np.ones([num_frames, max_obj_per_frame, 7]) * -1.0
    visible_objects_pose_world = np.ones([num_frames, max_obj_per_frame, 7]) * -1.0

    # Iterate through the tracklets and process object data
    for tracklet in tracklets_array:
        frame_id = int(tracklet[0])
        track_id = int(tracklet[1])
        if start_frame <= frame_id <= end_frame:
            ego_pose = ego_poses[frame_id]
            obj_pose_vehicle, obj_pose_world = make_obj_pose(ego_pose, tracklet[6:10])

            frame_idx = frame_id - start_frame
            obj_column = np.argwhere(visible_objects_ids[frame_idx, :] < 0).min()

            visible_objects_ids[frame_idx, obj_column] = track_id
            visible_objects_pose_vehicle[frame_idx, obj_column] = obj_pose_vehicle
            visible_objects_pose_world[frame_idx, obj_column] = obj_pose_world

    # Remove static objects
    print("Removing static objects")
    keep_static_vehicle_actors = cfg.data.get('keep_static_vehicle_actors', False)
    static_vehicle_min_frames = int(cfg.data.get('static_vehicle_actor_min_visible_frames', 20))
    static_vehicle_max_dist = float(cfg.data.get('static_vehicle_actor_max_min_distance_m', 30.0))
    pi3x_actor_min_points = int(cfg.data.get('pi3x_object_actor_min_points', 0))
    pi3x_track_counts = {}
    if bool(cfg.data.get('disable_lidar_pointcloud', False)) and pi3x_actor_min_points > 0:
        pi3x_track_counts = _pi3x_track_point_counts(datadir)
        if pi3x_track_counts:
            print(f"Filtering vehicle actors with fewer than {pi3x_actor_min_points} PI3X object points")

    kept_static_vehicle_count = 0
    removed_pi3x_vehicle_count = 0
    for key in objects_info.copy().keys():
        all_obj_idx = np.where(visible_objects_ids == key)
        if len(all_obj_idx[0]) > 0:
            if (
                pi3x_track_counts
                and objects_info[key].get('class') == 'vehicle'
                and pi3x_track_counts.get(int(key), 0) < pi3x_actor_min_points
            ):
                visible_objects_ids[all_obj_idx] = -1.
                visible_objects_pose_vehicle[all_obj_idx] = -1.
                visible_objects_pose_world[all_obj_idx] = -1.
                objects_info.pop(key)
                removed_pi3x_vehicle_count += 1
                continue

            obj_world_postions = visible_objects_pose_world[all_obj_idx][:, :3]
            distance = np.linalg.norm(obj_world_postions[0] - obj_world_postions[-1])
            dynamic = np.any(np.std(obj_world_postions, axis=0) > 0.5) or distance > 2
            if not dynamic:
                keep_static_vehicle = keep_static_vehicle_actors and should_keep_static_vehicle_actor(
                    objects_info[key],
                    visible_objects_pose_vehicle[all_obj_idx][:, :3],
                    static_vehicle_min_frames,
                    static_vehicle_max_dist,
                )
                if keep_static_vehicle:
                    objects_info[key]['kept_static_vehicle_actor'] = True
                    kept_static_vehicle_count += 1
                    continue
                visible_objects_ids[all_obj_idx] = -1.
                visible_objects_pose_vehicle[all_obj_idx] = -1.
                visible_objects_pose_world[all_obj_idx] = -1.
                objects_info.pop(key)
        else:
            objects_info.pop(key)
    if keep_static_vehicle_actors:
        print("Kept static vehicle actors:", kept_static_vehicle_count)
    if pi3x_track_counts:
        print("Removed vehicle actors without enough PI3X object points:", removed_pi3x_vehicle_count)

    # Clip max_num_obj
    mask = visible_objects_ids >= 0
    max_obj_per_frame_new = np.sum(mask, axis=1).max()
    print("Max obj per frame:", max_obj_per_frame_new)

    if max_obj_per_frame_new == 0:
        print("No moving obj in current sequence, make dummy visible objects")
        visible_objects_ids = np.ones([num_frames, 1]) * -1.0
        visible_objects_pose_world = np.ones([num_frames, 1, 7]) * -1.0
        visible_objects_pose_vehicle = np.ones([num_frames, 1, 7]) * -1.0
    elif max_obj_per_frame_new < max_obj_per_frame:
        visible_objects_ids_new = np.ones([num_frames, max_obj_per_frame_new]) * -1.0
        visible_objects_pose_vehicle_new = np.ones([num_frames, max_obj_per_frame_new, 7]) * -1.0
        visible_objects_pose_world_new = np.ones([num_frames, max_obj_per_frame_new, 7]) * -1.0
        for frame_idx in range(num_frames):
            for y in range(max_obj_per_frame):
                obj_id = visible_objects_ids[frame_idx, y]
                if obj_id >= 0:
                    obj_column = np.argwhere(visible_objects_ids_new[frame_idx, :] < 0).min()
                    visible_objects_ids_new[frame_idx, obj_column] = obj_id
                    visible_objects_pose_vehicle_new[frame_idx, obj_column] = visible_objects_pose_vehicle[frame_idx, y]
                    visible_objects_pose_world_new[frame_idx, obj_column] = visible_objects_pose_world[frame_idx, y]

        visible_objects_ids = visible_objects_ids_new
        visible_objects_pose_vehicle = visible_objects_pose_vehicle_new
        visible_objects_pose_world = visible_objects_pose_world_new

    box_scale = cfg.data.get('box_scale', 1.0)
    print('box scale: ', box_scale)

    frames = list(range(start_frame, end_frame + 1))
    frames = np.array(frames).astype(np.int32)

    # postprocess object_info
    for key in objects_info.keys():
        obj = objects_info[key]
        if obj['class'] == 'pedestrian':
            obj['deformable'] = True
        else:
            obj['deformable'] = False

        obj['width'] = obj['width'] * box_scale
        obj['length'] = obj['length'] * box_scale

        obj_frame_idx = np.argwhere(visible_objects_ids == key)[:, 0]
        obj_frame_idx = obj_frame_idx.astype(np.int32)
        obj_frames = frames[obj_frame_idx]
        obj['start_frame'] = np.min(obj_frames)
        obj['end_frame'] = np.max(obj_frames)

        objects_info[key] = obj

    # [num_frames, max_obj, track_id, x, y, z, qw, qx, qy, qz]
    objects_tracklets_world = np.concatenate(
        [visible_objects_ids[..., None], visible_objects_pose_world], axis=-1
    )

    objects_tracklets_vehicle = np.concatenate(
        [visible_objects_ids[..., None], visible_objects_pose_vehicle], axis=-1
    )


    return objects_tracklets_world, objects_tracklets_vehicle, objects_info


def _load_ignore_mask(datadir, image_filename):
    mask_path = os.path.join(datadir, 'ignore_mask', f'{os.path.splitext(os.path.basename(image_filename))[0]}.png')
    if not os.path.exists(mask_path):
        return None
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    return mask > 0


def build_image_ray_pointcloud(datadir, image_filenames, ixts, c2ws, train_frames, frames_idx):
    rng = np.random.default_rng(int(cfg.data.get('image_pcd_seed', 0)))
    samples_per_image = int(cfg.data.get('image_pcd_samples_per_image', 512))
    min_depth = float(cfg.data.get('image_pcd_min_depth', 4.0))
    max_depth = float(cfg.data.get('image_pcd_max_depth', cfg.data.get('extent', 20)))
    max_points = int(cfg.data.get('image_pcd_max_points', 200000))
    points_xyz = []
    points_rgb = []

    for image_filename, ixt, c2w, frame_idx in tqdm(
        zip(image_filenames, ixts, c2ws, frames_idx),
        total=len(image_filenames),
        desc='initialize from image rays',
    ):
        if frame_idx not in train_frames:
            continue
        image = cv2.imread(image_filename, cv2.IMREAD_COLOR)
        if image is None:
            continue
        image = image[..., [2, 1, 0]] / 255.
        h, w = image.shape[:2]
        valid = np.ones((h, w), dtype=np.bool_)
        ignore = _load_ignore_mask(datadir, image_filename)
        if ignore is not None:
            valid &= ~ignore
        valid[: max(1, h // 100), :] = False
        valid[-max(1, h // 100):, :] = False
        valid[:, : max(1, w // 100)] = False
        valid[:, -max(1, w // 100):] = False
        ys, xs = np.nonzero(valid)
        if len(xs) == 0:
            continue
        count = min(samples_per_image, len(xs))
        sample_idx = rng.choice(len(xs), size=count, replace=False)
        xs = xs[sample_idx]
        ys = ys[sample_idx]
        depths = rng.uniform(min_depth, max_depth, size=count)

        pixels = np.stack([xs.astype(np.float64), ys.astype(np.float64), np.ones(count)], axis=0)
        rays = np.linalg.inv(ixt) @ pixels
        points_cam = (rays * depths[None]).T
        points_cam_h = np.concatenate([points_cam, np.ones((count, 1))], axis=1)
        points_world = points_cam_h @ c2w.T
        points_xyz.append(points_world[:, :3].astype(np.float32))
        points_rgb.append(image[ys, xs].astype(np.float32))

    if len(points_xyz) == 0:
        raise RuntimeError('Failed to initialize point cloud from image rays; no valid pixels were sampled.')

    points_xyz = np.concatenate(points_xyz, axis=0)
    points_rgb = np.concatenate(points_rgb, axis=0)
    if len(points_xyz) > max_points:
        sample_idx = rng.choice(len(points_xyz), size=max_points, replace=False)
        points_xyz = points_xyz[sample_idx]
        points_rgb = points_rgb[sample_idx]

    return points_xyz.astype(np.float32), points_rgb.astype(np.float32)


def load_pi3_static_pointcloud(datadir):
    pi3_path = cfg.data.get('pi3_pointcloud_path', 'pi3_static_points.npz')
    if not os.path.isabs(pi3_path):
        pi3_path = os.path.join(datadir, pi3_path)
    if not os.path.exists(pi3_path):
        raise FileNotFoundError(f'use_pi3=true but PI3 static pointcloud is missing: {pi3_path}')

    with np.load(pi3_path, allow_pickle=False) as payload:
        points_xyz = np.asarray(payload['points'], dtype=np.float32)
        points_rgb = np.asarray(payload['colors'], dtype=np.float32)

    if points_xyz.ndim != 2 or points_xyz.shape[1] != 3:
        raise ValueError(f'Invalid PI3 points shape in {pi3_path}: {points_xyz.shape}')
    if points_rgb.shape != points_xyz.shape:
        raise ValueError(f'Invalid PI3 colors shape in {pi3_path}: {points_rgb.shape}; expected {points_xyz.shape}')

    finite = np.isfinite(points_xyz).all(axis=1) & np.isfinite(points_rgb).all(axis=1)
    points_xyz = points_xyz[finite]
    points_rgb = np.clip(points_rgb[finite], 0.0, 1.0)
    if points_xyz.shape[0] == 0:
        raise ValueError(f'PI3 static pointcloud is empty after filtering: {pi3_path}')
    return points_xyz.astype(np.float32), points_rgb.astype(np.float32), pi3_path


def append_pi3x_object_pointcloud(
    datadir,
    points_xyz_dict,
    points_rgb_dict,
    object_tracklets_vehicle,
    object_info,
    image_filenames,
    frames_idx,
    start_frame,
    train_frames,
):
    pi3x_path = os.path.join(datadir, 'pi3x_object_points.npz')
    if not os.path.exists(pi3x_path):
        return

    with np.load(pi3x_path, allow_pickle=True) as payload:
        pts3d_dict = payload['pointcloud'].item()
        pts2d_dict = payload['camera_projection'].item()
        track_ids_dict = payload['track_ids'].item()

    image_by_frame_cam = {}
    for image_filename in image_filenames:
        basename = os.path.basename(image_filename)
        image_by_frame_cam[(image_filename_to_frame(basename), image_filename_to_cam(basename))] = image_filename

    train_frames = set(int(frame) for frame in train_frames)
    appended_points = 0
    appended_tracks = set()
    for frame, points_vehicle in pts3d_dict.items():
        frame = int(frame)
        frame_idx = frame - int(start_frame)
        if frame_idx not in train_frames:
            continue
        if frame_idx < 0 or frame_idx >= object_tracklets_vehicle.shape[0]:
            continue

        points_vehicle = np.asarray(points_vehicle, dtype=np.float64)
        projections = np.asarray(pts2d_dict.get(frame, np.zeros((0, 6), dtype=np.int16)), dtype=np.int16)
        track_ids = np.asarray(track_ids_dict.get(frame, np.zeros((0,), dtype=np.int32)), dtype=np.int32)
        if points_vehicle.shape[0] == 0 or projections.shape[0] != points_vehicle.shape[0] or track_ids.shape[0] != points_vehicle.shape[0]:
            continue

        colors = np.ones((points_vehicle.shape[0], 3), dtype=np.float32)
        for cam in sorted(set(int(v) for v in projections[:, 0] if int(v) >= 0)):
            image_filename = image_by_frame_cam.get((frame, cam))
            if image_filename is None:
                continue
            image = cv2.imread(image_filename, cv2.IMREAD_COLOR)
            if image is None:
                continue
            image = image[..., [2, 1, 0]] / 255.
            cam_mask = projections[:, 0] == cam
            xs = projections[cam_mask, 1].astype(np.int32)
            ys = projections[cam_mask, 2].astype(np.int32)
            valid = (xs >= 0) & (xs < image.shape[1]) & (ys >= 0) & (ys < image.shape[0])
            cam_indices = np.nonzero(cam_mask)[0][valid]
            colors[cam_indices] = image[ys[valid], xs[valid]].astype(np.float32)

        for track_id in sorted(set(int(v) for v in track_ids if int(v) in object_info)):
            tracklet_idx = np.argwhere(object_tracklets_vehicle[frame_idx, :, 0] == track_id)
            if len(tracklet_idx) == 0:
                continue
            track_mask = track_ids == track_id
            track_points_vehicle = points_vehicle[track_mask]
            if track_points_vehicle.shape[0] == 0:
                continue

            tracklet = object_tracklets_vehicle[frame_idx, tracklet_idx[0, 0]]
            obj_pose_vehicle = np.eye(4)
            obj_pose_vehicle[:3, :3] = quaternion_to_matrix_numpy(tracklet[4:8])
            obj_pose_vehicle[:3, 3] = tracklet[1:4]
            vehicle2local = np.linalg.inv(obj_pose_vehicle)
            track_points_h = np.concatenate(
                [track_points_vehicle, np.ones((track_points_vehicle.shape[0], 1), dtype=np.float64)],
                axis=-1,
            )
            track_points_local = (track_points_h @ vehicle2local.T)[:, :3]
            key = f'obj_{track_id:03d}'
            if key not in points_xyz_dict:
                continue
            points_xyz_dict[key].append(track_points_local)
            points_rgb_dict[key].append(colors[track_mask])
            appended_points += int(track_points_local.shape[0])
            appended_tracks.add(track_id)

    print(f'initialize objects from pi3x mask pointcloud, tracks={len(appended_tracks)}, points={appended_points}')


def padding_tracklets(tracklets, frame_timestamps, min_timestamp, max_timestamp):
    # tracklets: [num_frames, max_obj, ....]
    # frame_timestamps: [num_frames]

    # Clone instead of extrapolation
    if min_timestamp < frame_timestamps[0]:
        tracklets_first = tracklets[0]
        frame_timestamps = np.concatenate([[min_timestamp], frame_timestamps])
        tracklets = np.concatenate([tracklets_first[None], tracklets], axis=0)

    if max_timestamp > frame_timestamps[1]:
        tracklets_last = tracklets[-1]
        frame_timestamps = np.concatenate([frame_timestamps, [max_timestamp]])
        tracklets = np.concatenate([tracklets, tracklets_last[None]], axis=0)

    return tracklets, frame_timestamps

def generate_dataparser_outputs(
        datadir,
        selected_frames=None,
        build_pointcloud=True,
        cameras=[0, 1, 2, 3, 4]
    ):

    image_dir = os.path.join(datadir, 'images')
    image_filenames_all = sorted(
        glob(os.path.join(image_dir, '*.png')) + glob(os.path.join(image_dir, '*.jpg'))
    )
    # Actual frame count from data (works for any camera subset; //5 only correct for 5 cams)
    max_frame_in_data = max(image_filename_to_frame(os.path.basename(f)) for f in image_filenames_all) if image_filenames_all else -1
    num_frames_all = max_frame_in_data + 1 if max_frame_in_data >= 0 else 0
    camera_names = load_camera_names(datadir)
    num_cameras = len(cameras)

    if selected_frames is None:
        start_frame = 0
        end_frame = num_frames_all - 1
        selected_frames = [start_frame, end_frame]
    else:
        start_frame, end_frame = selected_frames[0], selected_frames[1]
    # Clamp to actual frame range so config [0,198] on a 0-197 scene never requests 198
    end_frame = min(end_frame, num_frames_all - 1) if num_frames_all > 0 else 0
    start_frame = max(0, min(start_frame, end_frame))
    num_frames = end_frame - start_frame + 1

    # load calibration and ego pose
    intrinsics, extrinsics, ego_frame_poses, ego_cam_poses = load_camera_info(datadir)

    # load camera, frame, path
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

    split_test = cfg.data.get('split_test', -1)
    split_train = cfg.data.get('split_train', -1)
    train_frames, test_frames = get_val_frames(
        num_frames,
        test_every=split_test if split_test > 0 else None,
        train_every=split_train if split_train > 0 else None,
    )

    timestamp_path = os.path.join(datadir, 'timestamps.json')
    with open(timestamp_path, 'r') as f:
        timestamps = json.load(f)

    for frame in range(start_frame, end_frame+1):
        frames_timestamps.append(timestamps['FRAME'][f'{frame:06d}'])

    for image_filename in image_filenames_all:
        image_basename = os.path.basename(image_filename)
        frame = image_filename_to_frame(image_basename)
        cam = image_filename_to_cam(image_basename)
        if frame >= start_frame and frame <= end_frame and cam in cameras:
            ixt = intrinsics[cam]
            ext = extrinsics[cam]
            pose = ego_cam_poses[cam, frame]
            c2w = pose @ ext

            frames.append(frame)
            frames_idx.append(frame - start_frame)
            cams.append(cam)
            image_filenames.append(image_filename)

            ixts.append(ixt)
            exts.append(ext)
            poses.append(pose)
            c2ws.append(c2w)

            camera_name = camera_names.get(cam, _label2camera.get(cam))
            if camera_name is None:
                raise KeyError(f'No camera name found for camera id {cam}')
            timestamp = timestamps[camera_name][f'{frame:06d}']
            cams_timestamps.append(timestamp)

    exts = np.stack(exts, axis=0)
    ixts = np.stack(ixts, axis=0)
    poses = np.stack(poses, axis=0)
    c2ws = np.stack(c2ws, axis=0)

    timestamp_offset = min(cams_timestamps + frames_timestamps)
    cams_timestamps = np.array(cams_timestamps) - timestamp_offset
    frames_timestamps = np.array(frames_timestamps) - timestamp_offset
    min_timestamp, max_timestamp = min(cams_timestamps.min(), frames_timestamps.min()), max(cams_timestamps.max(), frames_timestamps.max())

    _, object_tracklets_vehicle, object_info = get_obj_pose_tracking(
        datadir,
        selected_frames,
        ego_frame_poses,
        cameras,
    )

    for track_id in object_info.keys():
        object_start_frame = object_info[track_id]['start_frame']
        object_end_frame = object_info[track_id]['end_frame']
        object_start_timestamp = timestamps['FRAME'][f'{object_start_frame:06d}'] - timestamp_offset - 0.1
        object_end_timestamp = timestamps['FRAME'][f'{object_end_frame:06d}'] - timestamp_offset + 0.1
        object_info[track_id]['start_timestamp'] = max(object_start_timestamp, min_timestamp)
        object_info[track_id]['end_timestamp'] = min(object_end_timestamp, max_timestamp)

    result = dict()
    result['num_frames'] = num_frames
    result['exts'] = exts
    result['ixts'] = ixts
    result['poses'] = poses
    result['c2ws'] = c2ws
    result['obj_tracklets'] = object_tracklets_vehicle
    result['obj_info'] = object_info
    result['frames'] = frames
    result['cams'] = cams
    result['frames_idx'] = frames_idx
    result['image_filenames'] = image_filenames
    result['cams_timestamps'] = cams_timestamps
    result['tracklet_timestamps'] = frames_timestamps

    # get object bounding mask
    obj_bounds = []
    for i, image_filename in tqdm(enumerate(image_filenames)):
        cam = cams[i]
        h, w = _image_size(image_filename)
        obj_bound = np.zeros((h, w)).astype(np.uint8)
        instance_mask = _load_instance_mask_for_image(datadir, image_filename, h, w)
        obj_tracklets = object_tracklets_vehicle[frames_idx[i]]
        ixt, ext = ixts[i], exts[i]
        for obj_tracklet in obj_tracklets:
            track_id = int(obj_tracklet[0])
            if track_id >= 0:
                if instance_mask is not None and np.any(instance_mask == track_id):
                    obj_bound = np.logical_or(obj_bound, instance_mask == track_id)
                    continue
                obj_pose_vehicle = np.eye(4)
                obj_pose_vehicle[:3, :3] = quaternion_to_matrix_numpy(obj_tracklet[4:8])
                obj_pose_vehicle[:3, 3] = obj_tracklet[1:4]
                obj_length = object_info[track_id]['length']
                obj_width = object_info[track_id]['width']
                obj_height = object_info[track_id]['height']
                bbox = np.array([[-obj_length, -obj_width, -obj_height],
                                 [obj_length, obj_width, obj_height]]) * 0.5
                corners_local = bbox_to_corner3d(bbox)
                corners_local = np.concatenate([corners_local, np.ones_like(corners_local[..., :1])], axis=-1)
                corners_vehicle = corners_local @ obj_pose_vehicle.T # 3D bounding box in vehicle frame
                mask = get_bound_2d_mask(
                    corners_3d=corners_vehicle[..., :3],
                    K=ixt,
                    pose=np.linalg.inv(ext),
                    H=h, W=w
                )
                obj_bound = np.logical_or(obj_bound, mask)
        obj_bounds.append(obj_bound)
    result['obj_bounds'] = obj_bounds

    # os.makedirs('obj_bounds', exist_ok=True)
    # for i, x in enumerate(obj_bounds):
    #     x = x.astype(np.uint8) * 255
    #     cv2.imwrite(f'obj_bounds/{i}.png', x)

    # run colmap (skipped when use_colmap=False; downstream code in this
    # file already handles a missing colmap pointcloud via try/except and
    # falls back to a lidar-only initialisation).
    colmap_basedir = os.path.join(f'{cfg.model_path}/colmap')
    if cfg.data.get('use_colmap', True) and not os.path.exists(os.path.join(colmap_basedir, 'triangulated/sparse/model')):
        from script.waymo.colmap_waymo_full import run_colmap_waymo
        run_colmap_waymo(result)

    if build_pointcloud:
        print('build point cloud')
        pointcloud_dir = os.path.join(cfg.model_path, 'input_ply')
        os.makedirs(pointcloud_dir, exist_ok=True)

        points_xyz_dict = dict()
        points_rgb_dict = dict()
        points_xyz_dict['bkgd'] = []
        points_rgb_dict['bkgd'] = []
        for track_id in object_info.keys():
            points_xyz_dict[f'obj_{track_id:03d}'] = []
            points_rgb_dict[f'obj_{track_id:03d}'] = []

        print('initialize from sfm pointcloud')
        points_colmap_path = os.path.join(colmap_basedir, 'triangulated/sparse/model/points3D.bin')
        if cfg.data.get('use_colmap', True) and os.path.exists(points_colmap_path):
            points_colmap_xyz, points_colmap_rgb, points_colmap_error = read_points3D_binary(points_colmap_path)
            points_colmap_rgb = points_colmap_rgb / 255.
        else:
            # No colmap pointcloud — downstream filter_colmap try/except will
            # fall back to lidar-only background.
            points_colmap_xyz = np.zeros((0, 3), dtype=np.float32)
            points_colmap_rgb = np.zeros((0, 3), dtype=np.float32)
            points_colmap_error = np.zeros((0,), dtype=np.float32)

        pointcloud_path = os.path.join(datadir, 'pointcloud.npz')
        use_lidar_pointcloud = os.path.exists(pointcloud_path) and not bool(cfg.data.get('disable_lidar_pointcloud', False))
        use_pi3_pointcloud = bool(cfg.data.get('use_pi3', False))
        if use_lidar_pointcloud:
            print('initialize from lidar pointcloud')
            pts3d_dict = np.load(pointcloud_path, allow_pickle=True)['pointcloud'].item()
            pts2d_dict = np.load(pointcloud_path, allow_pickle=True)['camera_projection'].item()

            for i, frame in tqdm(enumerate(range(start_frame, end_frame+1))):
                idxs = [idx for idx, frame_idx in enumerate(frames_idx) if frame_idx == i]
                cams_frame = [cams[idx] for idx in idxs]
                image_filenames_frame = [image_filenames[idx] for idx in idxs]

                raw_3d = pts3d_dict[frame]
                raw_2d = pts2d_dict[frame]

                # use the first projection camera
                points_camera_all = raw_2d[..., 0]
                points_projw_all = raw_2d[..., 1]
                points_projh_all = raw_2d[..., 2]

                # each point should be observed by at least one camera in camera lists
                mask = np.array([c in cameras for c in points_camera_all]).astype(np.bool_)

                # get filtered LiDAR pointcloud position and color
                points_xyz_vehicle = raw_3d[mask]

                # transfrom LiDAR pointcloud from vehicle frame to world frame
                ego_pose = ego_frame_poses[frame]
                points_xyz_vehicle = np.concatenate(
                    [points_xyz_vehicle,
                    np.ones_like(points_xyz_vehicle[..., :1])], axis=-1
                )
                points_xyz_world = points_xyz_vehicle @ ego_pose.T

                points_rgb = np.ones_like(points_xyz_vehicle[:, :3])
                points_camera = points_camera_all[mask]
                points_projw = points_projw_all[mask]
                points_projh = points_projh_all[mask]

                for cam, image_filename in zip(cams_frame, image_filenames_frame):
                    mask_cam = (points_camera == cam)
                    image = cv2.imread(image_filename)[..., [2, 1, 0]] / 255.

                    mask_projw = points_projw[mask_cam]
                    mask_projh = points_projh[mask_cam]
                    mask_rgb = image[mask_projh, mask_projw]
                    points_rgb[mask_cam] = mask_rgb

                # filer points in tracking bbox
                points_xyz_obj_mask = np.zeros(points_xyz_vehicle.shape[0], dtype=np.bool_)

                for tracklet in object_tracklets_vehicle[i]:
                    track_id = int(tracklet[0])
                    if track_id >= 0:
                        obj_pose_vehicle = np.eye(4)
                        obj_pose_vehicle[:3, :3] = quaternion_to_matrix_numpy(tracklet[4:8])
                        obj_pose_vehicle[:3, 3] = tracklet[1:4]
                        vehicle2local = np.linalg.inv(obj_pose_vehicle)

                        points_xyz_obj = points_xyz_vehicle @ vehicle2local.T
                        points_xyz_obj = points_xyz_obj[..., :3]

                        length = object_info[track_id]['length']
                        width = object_info[track_id]['width']
                        height = object_info[track_id]['height']
                        bbox = [[-length/2, -width/2, -height/2], [length/2, width/2, height/2]]
                        obj_corners_3d_local = bbox_to_corner3d(bbox)

                        points_xyz_inbbox = inbbox_points(points_xyz_obj, obj_corners_3d_local)
                        points_xyz_obj_mask = np.logical_or(points_xyz_obj_mask, points_xyz_inbbox)
                        points_xyz_dict[f'obj_{track_id:03d}'].append(points_xyz_obj[points_xyz_inbbox])
                        points_rgb_dict[f'obj_{track_id:03d}'].append(points_rgb[points_xyz_inbbox])

                points_lidar_xyz = points_xyz_world[~points_xyz_obj_mask][..., :3]
                points_lidar_rgb = points_rgb[~points_xyz_obj_mask]

                points_xyz_dict['bkgd'].append(points_lidar_xyz)
                points_rgb_dict['bkgd'].append(points_lidar_rgb)
        else:
            if use_pi3_pointcloud:
                points_lidar_xyz, points_lidar_rgb, pi3_path = load_pi3_static_pointcloud(datadir)
                print(f'initialize from PI3 static pointcloud: {pi3_path}')
            else:
                reason = f'{pointcloud_path} is disabled' if os.path.exists(pointcloud_path) else f'{pointcloud_path} is missing'
                print(f'initialize from image-ray pointcloud because {reason}')
                points_lidar_xyz, points_lidar_rgb = build_image_ray_pointcloud(
                    datadir,
                    image_filenames,
                    ixts,
                    c2ws,
                    train_frames,
                    frames_idx,
                )
            points_xyz_dict['bkgd'].append(points_lidar_xyz)
            points_rgb_dict['bkgd'].append(points_lidar_rgb)
            append_pi3x_object_pointcloud(
                datadir,
                points_xyz_dict,
                points_rgb_dict,
                object_tracklets_vehicle,
                object_info,
                image_filenames,
                frames_idx,
                start_frame,
                train_frames,
            )

        initial_num_obj = 20000

        for k, v in points_xyz_dict.items():
            if len(v) == 0:
                continue
            else:
                points_xyz = np.concatenate(v, axis=0)
                points_rgb = np.concatenate(points_rgb_dict[k], axis=0)
                if k == 'bkgd':
                    if use_lidar_pointcloud:
                        # downsample lidar pointcloud with voxels
                        points_lidar = o3d.geometry.PointCloud()
                        points_lidar.points = o3d.utility.Vector3dVector(points_xyz)
                        points_lidar.colors = o3d.utility.Vector3dVector(points_rgb)
                        downsample_points_lidar = points_lidar.voxel_down_sample(voxel_size=0.15)
                        downsample_points_lidar, _ = downsample_points_lidar.remove_radius_outlier(nb_points=10, radius=0.5)
                        points_lidar_xyz = np.asarray(downsample_points_lidar.points).astype(np.float32)
                        points_lidar_rgb = np.asarray(downsample_points_lidar.colors).astype(np.float32)
                    else:
                        points_lidar_xyz = points_xyz.astype(np.float32)
                        points_lidar_rgb = points_rgb.astype(np.float32)
                elif k.startswith('obj'):
                    # points_obj = o3d.geometry.PointCloud()
                    # points_obj.points = o3d.utility.Vector3dVector(points_xyz)
                    # points_obj.colors = o3d.utility.Vector3dVector(points_rgb)
                    # downsample_points_lidar = points_obj.voxel_down_sample(voxel_size=0.05)
                    # points_xyz = np.asarray(downsample_points_lidar.points).astype(np.float32)
                    # points_rgb = np.asarray(downsample_points_lidar.colors).astype(np.float32)
                    track_id = int(k.split('_')[-1])
                    points_xyz, points_rgb = _filter_object_points_by_instance_masks(
                        datadir,
                        track_id,
                        points_xyz,
                        points_rgb,
                        object_tracklets_vehicle,
                        image_filenames,
                        frames_idx,
                        ixts,
                        exts,
                        train_frames,
                    )
                    points_xyz, points_rgb = _filter_object_local_pointcloud(
                        track_id,
                        points_xyz,
                        points_rgb,
                        object_info,
                    )

                    if len(points_xyz) > initial_num_obj:
                        random_indices = np.random.choice(len(points_xyz), initial_num_obj, replace=False)
                        points_xyz = points_xyz[random_indices]
                        points_rgb = points_rgb[random_indices]

                    points_xyz_dict[k] = points_xyz
                    points_rgb_dict[k] = points_rgb

                else:
                    raise NotImplementedError()

        # Get sphere center and radius
        lidar_sphere_normalization = get_Sphere_Norm(points_lidar_xyz)
        sphere_center = lidar_sphere_normalization['center']
        sphere_radius = lidar_sphere_normalization['radius']

        # combine SfM pointcloud with LiDAR pointcloud
        try:
            if cfg.data.filter_colmap:
                points_colmap_mask = np.ones(points_colmap_xyz.shape[0], dtype=np.bool_)
                for i, ext in enumerate(exts):
                    # if frames_idx[i] not in train_frames:
                    #     continue
                    camera_position = c2ws[i][:3, 3]
                    radius = np.linalg.norm(points_colmap_xyz - camera_position, axis=-1)
                    mask = np.logical_or(radius < cfg.data.get('extent', 10), points_colmap_xyz[:, 2] < camera_position[2])
                    points_colmap_mask = np.logical_and(points_colmap_mask, ~mask)
                points_colmap_xyz = points_colmap_xyz[points_colmap_mask]
                points_colmap_rgb = points_colmap_rgb[points_colmap_mask]

            points_colmap_dist = np.linalg.norm(points_colmap_xyz - sphere_center, axis=-1)
            mask = points_colmap_dist < 2 * sphere_radius
            points_colmap_xyz = points_colmap_xyz[mask]
            points_colmap_rgb = points_colmap_rgb[mask]

            points_bkgd_xyz = np.concatenate([points_lidar_xyz, points_colmap_xyz], axis=0)
            points_bkgd_rgb = np.concatenate([points_lidar_rgb, points_colmap_rgb], axis=0)
        except:
            print('No colmap pointcloud')
            points_bkgd_xyz = points_lidar_xyz
            points_bkgd_rgb = points_lidar_rgb

        range_init_key = 'lidar' if use_lidar_pointcloud else ('pi3' if use_pi3_pointcloud else 'image_rays')
        points_xyz_dict[range_init_key] = points_lidar_xyz
        points_rgb_dict[range_init_key] = points_lidar_rgb
        points_xyz_dict['colmap'] = points_colmap_xyz
        points_rgb_dict['colmap'] = points_colmap_rgb
        points_xyz_dict['bkgd'] = points_bkgd_xyz
        points_rgb_dict['bkgd'] = points_bkgd_rgb

        result['points_xyz_dict'] = points_xyz_dict
        result['points_rgb_dict'] = points_rgb_dict

        # Sample sky point cloud
        # if num_cameras < 3:
        #     background_sphere_points = 50000
        # else:
        #     background_sphere_points = 100000
        # background_sphere_distance = 2.5

        # if cfg.model.nsg.get('include_sky', False):
        #     sky_mask_dir = os.path.join(datadir, 'sky_mask')
        #     assert os.path.exists(sky_mask_dir)
        #     points_xyz_sky_mask = []
        #     points_rgb_sky_mask = []
        #     num_samples = background_sphere_points // len(train_frames)
        #     print('sample points from sky mask for background sphere')

        #     for i, frame in tqdm(enumerate(range(start_frame, end_frame+1))):
        #         idxs = list(range(i * num_cameras, (i+1) * num_cameras))
        #         sky_mask_path = [os.path.join(sky_mask_dir,  os.path.basename(image_filenames[idx])) for idx in idxs]
        #         sky_mask_list = [(cv2.imread(sky_mask_path_)[..., 0] > 0).reshape(-1) for sky_mask_path_ in sky_mask_path]
        #         sky_pixel_all = np.sum(np.stack(sky_mask_list, axis=0))
        #         for i, sky_mask in enumerate(sky_mask_list):
        #             image_path = image_filenames[idxs[i]]
        #             image = cv2.imread(image_path)[..., [2, 1, 0]] / 255.
        #             H, W, _ = image.shape

        #             sky_pixel = np.sum(sky_mask)
        #             sky_indices = np.argwhere(sky_mask == True)[..., 0]
        #             num_sample = int(num_samples * sky_pixel / sky_pixel_all)
        #             if len(sky_indices) == 0:
        #                 continue
        #             elif len(sky_indices) > num_sample:
        #                 random_indices = np.random.choice(len(sky_indices), num_sample, replace=False)
        #                 sky_indices = sky_indices[random_indices]

        #             idx = idxs[i]

        #             K = ixts[idx]
        #             w2c = np.linalg.inv(c2ws[idx])
        #             R, T = w2c[:3, :3], w2c[:3, 3]
        #             rays_o, rays_d = get_rays(H, W, K, R, T)
        #             rays_o = rays_o.reshape(-1, 3)[sky_indices]
        #             rays_d = rays_d.reshape(-1, 3)[sky_indices]

        #             p_sphere = sphere_intersection(rays_o, rays_d, sphere_center, sphere_radius * background_sphere_distance)
        #             points_xyz_sky_mask.append(p_sphere)

        #             pixel_value = image.reshape(-1, 3)[sky_indices]
        #             points_rgb_sky_mask.append(pixel_value)

        #     points_xyz_sky_mask = np.concatenate(points_xyz_sky_mask, axis=0)
        #     points_rgb_sky_mask = np.concatenate(points_rgb_sky_mask, axis=0)
        #     points_xyz_dict['sky'] = points_xyz_sky_mask
        #     points_rgb_dict['sky'] = points_rgb_sky_mask

        # elif cfg.data.get('add_background_sphere', False):
        #     print('sample random points for background sphere')

        #     # Random sample points on the sphere
        #     samples = np.arange(background_sphere_points)
        #     y = 1.0 - samples / float(background_sphere_points) * 2 # y in [-1, 1]
        #     radius = np.sqrt(1 - y * y) # radius at y
        #     phi = math.pi * (math.sqrt(5.) - 1.) # golden angle in radians
        #     theta = phi * samples # golden angle increment
        #     x = np.cos(theta) * radius
        #     z = np.sin(theta) * radius
        #     unit_sphere_points = np.concatenate([x[:, None], y[:, None], z[:, None]], axis=1)

        #     points_xyz_sky_random = (unit_sphere_points * sphere_center * background_sphere_distance) + sphere_radius
        #     points_rgb_sky_random = np.asarray(np.random.random(points_xyz_sky_random.shape) * 255, dtype=np.uint8)

        #     points_xyz_dict['sky'] = points_xyz_sky_random
        #     points_rgb_dict['sky'] = points_rgb_sky_random
        # else:
        #     pass


        # result['points_xyz_dict'] = points_xyz_dict
        # result['points_rgb_dict'] = points_rgb_dict

        for k in points_xyz_dict.keys():
            points_xyz = points_xyz_dict[k]
            points_rgb = points_rgb_dict[k]
            ply_path = os.path.join(pointcloud_dir, f'points3D_{k}.ply')
            try:
                storePly(ply_path, points_xyz, points_rgb)
                print(f'saving pointcloud for {k}, number of initial points is {points_xyz.shape}')
            except:
                print(f'failed to save pointcloud for {k}')
                continue
    return result
