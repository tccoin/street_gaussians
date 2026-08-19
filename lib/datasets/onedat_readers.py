import os

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from lib.config import cfg
from lib.datasets.base_readers import CameraInfo, SceneInfo, fetchPly, getNerfppNorm, get_Sphere_Norm
from lib.utils.data_utils import get_val_frames
from lib.utils.graphics_utils import focal2fov
from lib.utils.onedat_utils import ensure_background_ply, generate_dataparser_outputs


def readOneDatInfo(path, images="images", split_train=-1, split_test=-1, **kwargs):
    selected_frames = cfg.data.get("selected_frames", None)
    if cfg.debug:
        selected_frames = [0, 0]

    bkgd_ply_path = os.path.join(cfg.model_path, "input_ply", "points3D_bkgd.ply")
    build_pointcloud = cfg.mode == "train" and (not os.path.exists(bkgd_ply_path) or cfg.data.get("regenerate_pcd", False))
    output = generate_dataparser_outputs(
        datadir=path,
        selected_frames=selected_frames,
        build_pointcloud=build_pointcloud,
        cameras=cfg.data.get("cameras", None),
    )
    if cfg.mode != "train" and not os.path.exists(bkgd_ply_path):
        ensure_background_ply(path)

    exts = output["exts"]
    ixts = output["ixts"]
    poses = output["poses"]
    c2ws = output["c2ws"]
    image_filenames = output["image_filenames"]
    obj_tracklets = output["obj_tracklets"]
    obj_info = output["obj_info"]
    frames = output["frames"]
    cams = output["cams"]
    frames_idx = output["frames_idx"]
    num_frames = output["num_frames"]
    cams_timestamps = output["cams_timestamps"]
    tracklet_timestamps = output["tracklet_timestamps"]
    image_sizes = output["image_sizes"]

    train_frames, test_frames = get_val_frames(
        num_frames,
        test_every=split_test if split_test > 0 else None,
        train_every=split_train if split_train > 0 else None,
    )

    configured_cameras = cfg.data.get("cameras", None)
    if configured_cameras is None:
        configured_cameras = sorted(image_sizes.keys())

    scene_metadata = {
        "obj_tracklets": obj_tracklets,
        "tracklet_timestamps": tracklet_timestamps,
        "obj_meta": obj_info,
        "num_images": len(exts),
        "num_cams": len(configured_cameras),
        "num_frames": num_frames,
    }
    camera_timestamps = {}
    for cam in configured_cameras:
        camera_timestamps[int(cam)] = {"train_timestamps": [], "test_timestamps": []}

    sky_mask_dir = os.path.join(path, str(cfg.data.get("sky_masks", "sky_mask")))
    load_sky_mask = cfg.mode == "train" and os.path.exists(sky_mask_dir)

    cam_infos = []
    for i in tqdm(range(len(exts))):
        cam_id = int(cams[i])
        image_path = str(image_filenames[i])
        image_name = os.path.basename(image_path).split(".")[0]
        # PIL keeps the underlying file descriptor open while an Image remains
        # lazy.  A multi-camera OneDat scene can contain more images than the
        # process FD limit, so materialize the pixels and close the file here.
        with Image.open(image_path) as image_file:
            image = image_file.copy()
        width, height = image.size
        if cam_id in image_sizes:
            height, width = image_sizes[cam_id]
        ixt = ixts[i]
        c2w = c2ws[i]
        fx, fy = ixt[0, 0], ixt[1, 1]
        FovY = focal2fov(fy, height)
        FovX = focal2fov(fx, width)
        RT = np.linalg.inv(c2w)

        metadata = {
            "frame": int(frames[i]),
            "cam": cam_id,
            "frame_idx": int(frames_idx[i]),
            "ego_pose": poses[i],
            "extrinsic": exts[i],
            "timestamp": float(cams_timestamps[i]),
        }
        if frames_idx[i] in train_frames:
            metadata["is_val"] = False
            camera_timestamps[cam_id]["train_timestamps"].append(cams_timestamps[i])
        else:
            metadata["is_val"] = True
            camera_timestamps[cam_id]["test_timestamps"].append(cams_timestamps[i])

        guidance = {}
        if load_sky_mask:
            sky_mask_path = os.path.join(sky_mask_dir, f"{image_name}.png")
            if os.path.exists(sky_mask_path):
                sky_mask = cv2.imread(sky_mask_path)[..., 0] > 0.0
                guidance["sky_mask"] = Image.fromarray(sky_mask)

        cam_infos.append(
            CameraInfo(
                uid=i,
                R=RT[:3, :3].T,
                T=RT[:3, 3],
                FovY=FovY,
                FovX=FovX,
                K=ixt.copy(),
                image=image,
                image_path=image_path,
                image_name=image_name,
                width=width,
                height=height,
                metadata=metadata,
                guidance=guidance,
            )
        )

    train_cam_infos = [cam_info for cam_info in cam_infos if not cam_info.metadata["is_val"]]
    test_cam_infos = [cam_info for cam_info in cam_infos if cam_info.metadata["is_val"]]
    for cam_id in camera_timestamps:
        camera_timestamps[cam_id]["train_timestamps"] = sorted(camera_timestamps[cam_id]["train_timestamps"])
        camera_timestamps[cam_id]["test_timestamps"] = sorted(camera_timestamps[cam_id]["test_timestamps"])
    scene_metadata["camera_timestamps"] = camera_timestamps

    nerf_normalization = getNerfppNorm(train_cam_infos)
    nerf_normalization["radius"] = max(nerf_normalization["radius"], 10)
    if cfg.data.get("extent", False):
        nerf_normalization["radius"] = cfg.data.extent
    cfg.data.extent = float(nerf_normalization["radius"])
    scene_metadata["scene_center"] = nerf_normalization["center"]
    scene_metadata["scene_radius"] = nerf_normalization["radius"]
    print(f'Scene extent: {nerf_normalization["radius"]}')

    pcd = fetchPly(bkgd_ply_path)
    sphere_normalization = get_Sphere_Norm(pcd.points)
    scene_metadata["sphere_center"] = sphere_normalization["center"]
    scene_metadata["sphere_radius"] = sphere_normalization["radius"]
    print(f'Sphere extent: {sphere_normalization["radius"]}')

    scene_info = SceneInfo(
        point_cloud=pcd if cfg.mode == "train" else None,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=bkgd_ply_path if cfg.mode == "train" else None,
        metadata=scene_metadata,
        novel_view_cameras=[],
    )
    return scene_info
