#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
import torchvision.transforms.functional as tf
import json
from PIL import Image
from pathlib import Path

from tqdm import tqdm
from lib.config import cfg
from lib.utils.loss_utils import ssim, psnr
from lib.utils.lpipsPyTorch import lpips
from lib.datasets.dataset import Dataset
from lib.models.street_gaussian_model import StreetGaussianModel
from lib.models.scene import Scene
from lib.models.street_gaussian_renderer import StreetGaussianRenderer


def evaluate(split='test'):
    scene_dir = cfg.model_path
    dataset = Dataset()
    if split == 'test':
        test_dir = Path(scene_dir) / "test"
        cam_infos = dataset.test_cameras[1]
    else:
        test_dir = Path(scene_dir) / "train"
        cam_infos = dataset.train_cameras[1]
        
    cam_infos = list(sorted(cam_infos, key=lambda x: x.id))
    
    # Load model to get Gaussian count and measure memory usage
    print("Loading model to get Gaussian statistics...")
    gaussians = StreetGaussianModel(dataset.scene_info.metadata)
    scene = Scene(gaussians=gaussians, dataset=dataset)
    renderer = StreetGaussianRenderer()
    
    # Get total number of Gaussians by summing all components
    total_gaussians = 0
    if hasattr(gaussians, 'background') and gaussians.background is not None:
        total_gaussians += gaussians.background.get_xyz.shape[0]
    if hasattr(gaussians, 'obj_list'):
        for obj_name in gaussians.obj_list:
            if hasattr(gaussians, obj_name):
                obj_model = getattr(gaussians, obj_name)
                if hasattr(obj_model, 'get_xyz'):
                    total_gaussians += obj_model.get_xyz.shape[0]
    
    print(f"Total Gaussians: {total_gaussians:,}")
    
    # Get memory usage during rendering (render one test image)
    memory_used = 0.0
    peak_memory = 0.0
    if len(cam_infos) > 0 and torch.cuda.is_available():
        # Clear cache and reset stats before measurement
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        initial_memory = torch.cuda.memory_allocated() / 1024**3  # GB
        
        test_camera = cam_infos[0]
        # Set viewpoint camera for StreetGaussianModel
        gaussians.viewpoint_camera = test_camera
        gaussians._build_graph()
        
        # Measure memory before rendering (after building graph)
        memory_after_graph = torch.cuda.memory_allocated() / 1024**3  # GB
        
        # Measure memory during rendering
        with torch.no_grad():
            _ = renderer.render(test_camera, gaussians)
        torch.cuda.synchronize()
        
        peak_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
        current_memory = torch.cuda.memory_allocated() / 1024**3  # GB
        # Memory used is the peak memory during rendering
        memory_used = peak_memory - initial_memory
        
        print(f"GPU Memory - Initial: {initial_memory:.2f} GB, After graph: {memory_after_graph:.2f} GB, Peak: {peak_memory:.2f} GB, Used: {memory_used:.2f} GB")
    
    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    
    print(f"Scene: {scene_dir }")
    full_dict[scene_dir] = {
        "num_gaussians": total_gaussians,
        "gpu_memory_gb": memory_used,
        "gpu_memory_peak_gb": peak_memory
    }
    per_view_dict[scene_dir] = {}
    full_dict_polytopeonly[scene_dir] = {}
    per_view_dict_polytopeonly[scene_dir] = {}
    
    for method in os.listdir(test_dir):
        print("Method:", method)
        full_dict[scene_dir][method] = {}
        per_view_dict[scene_dir][method] = {}
        full_dict_polytopeonly[scene_dir][method] = {}
        per_view_dict_polytopeonly[scene_dir][method] = {}    


        renders = []
        gts = []
        image_names = []

        for cam_info in tqdm(cam_infos, desc="Reading image progress"):
            image_name = cam_info.image_name
            render_path = test_dir / method / f'{image_name}_rgb.png'
            gt_path = test_dir / method / f'{image_name}_gt.png'
            
            render = Image.open(render_path)
            gt = Image.open(gt_path)
            renders.append(tf.to_tensor(render)[:3, :, :])
            gts.append(tf.to_tensor(gt)[:3, :, :])
            image_names.append(image_name)

        psnrs = []
        ssims = []
        lpipss = []

        for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
            render = renders[idx].cuda()
            gt = gts[idx].cuda()
            ssims.append(ssim(render, gt))
            psnrs.append(psnr(render, gt))
            lpipss.append(lpips(render, gt, net_type='alex'))
        
        print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
        print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
        print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
        print("")
        
        full_dict[scene_dir][method].update({"SSIM": torch.tensor(ssims).mean().item(),
                                        "PSNR": torch.tensor(psnrs).mean().item(),
                                        "LPIPS": torch.tensor(lpipss).mean().item()})
        per_view_dict[scene_dir][method].update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                                                    "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                                                    "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)}})

    with open(scene_dir + f"/results_{split}.json", 'w') as fp:
        json.dump(full_dict[scene_dir], fp, indent=True)
    with open(scene_dir + f"/per_view_{split}.json", 'w') as fp:
        json.dump(per_view_dict[scene_dir], fp, indent=True)

if __name__ == "__main__":
    if cfg.eval.eval_train:
        evaluate(split='train')
    if cfg.eval.eval_test:
        evaluate(split='test')
