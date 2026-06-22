import torch
from lib.utils.sh_utils import eval_sh
from lib.models.street_gaussian_model import StreetGaussianModel
from lib.utils.camera_utils import Camera, make_rasterizer
from lib.config import cfg

class StreetGaussianRenderer():
    def __init__(
        self,         
    ):
        self.cfg = cfg.render
              
    def render_all(
        self, 
        viewpoint_camera: Camera,
        pc: StreetGaussianModel,
        convert_SHs_python = None, 
        compute_cov3D_python = None, 
        scaling_modifier = None, 
        override_color = None
    ):
        
        # render all
        render_composition = self.render(viewpoint_camera, pc, convert_SHs_python, compute_cov3D_python, scaling_modifier, override_color)

        # render background
        render_background = self.render_background(viewpoint_camera, pc, convert_SHs_python, compute_cov3D_python, scaling_modifier, override_color)
        
        # render object
        render_object = self.render_object(viewpoint_camera, pc, convert_SHs_python, compute_cov3D_python, scaling_modifier, override_color)
        
        result = render_composition
        result['rgb_background'] = render_background['rgb']
        result['acc_background'] = render_background['acc']
        result['rgb_object'] = render_object['rgb']
        result['acc_object'] = render_object['acc']
        
        # result['bboxes'], result['bboxes_input'] = pc.get_bbox_corner(viewpoint_camera)
    
        return result
    
    def render_object(
        self, 
        viewpoint_camera: Camera,
        pc: StreetGaussianModel,
        convert_SHs_python = None, 
        compute_cov3D_python = None, 
        scaling_modifier = None, 
        override_color = None,
        parse_camera_again: bool = True,
    ):        
        pc.set_visibility(include_list=pc.obj_list)
        if parse_camera_again: pc.parse_camera(viewpoint_camera)        
        result = self.render_kernel(viewpoint_camera, pc, convert_SHs_python, compute_cov3D_python, scaling_modifier, override_color, white_background=False)

        return result
    
    def render_background(
        self, 
        viewpoint_camera: Camera,
        pc: StreetGaussianModel,
        convert_SHs_python = None, 
        compute_cov3D_python = None, 
        scaling_modifier = None, 
        override_color = None,
        parse_camera_again: bool = True,
    ):
        pc.set_visibility(include_list=['background'])
        if parse_camera_again: pc.parse_camera(viewpoint_camera)
        result = self.render_kernel(viewpoint_camera, pc, convert_SHs_python, compute_cov3D_python, scaling_modifier, override_color, white_background=False)

        return result
    
    def render_sky(
        self, 
        viewpoint_camera: Camera,
        pc: StreetGaussianModel,
        convert_SHs_python = None, 
        compute_cov3D_python = None, 
        scaling_modifier = None, 
        override_color = None,
        parse_camera_again: bool = True,
    ):  
        if not pc.include_sky or pc.sky_cubemap is None:
            raise ValueError("Sky rendering not available for this scene")

        sky_color = pc.sky_cubemap(viewpoint_camera, acc=None)
        if pc.use_color_correction:
            sky_color = pc.color_correction(viewpoint_camera, sky_color, use_sky=True)
        if cfg.mode != 'train':
            sky_color = torch.clamp(sky_color, 0., 1.)

        result = {
            "rgb": sky_color,
            "acc": torch.ones(
                1,
                int(viewpoint_camera.image_height),
                int(viewpoint_camera.image_width),
                device=sky_color.device,
                dtype=sky_color.dtype,
            ),
        }
        return result
    
    def render(
        self, 
        viewpoint_camera: Camera,
        pc: StreetGaussianModel,
        convert_SHs_python = None, 
        compute_cov3D_python = None, 
        scaling_modifier = None, 
        override_color = None,
        exclude_list = [],
    ):   
        include_list = list(set(pc.model_name_id.keys()) - set(exclude_list))
                    
        # Step1: render foreground
        pc.set_visibility(include_list)
        pc.parse_camera(viewpoint_camera)
        
        result = self.render_kernel(viewpoint_camera, pc, convert_SHs_python, compute_cov3D_python, scaling_modifier, override_color)

        # Step2: render sky
        if pc.include_sky and bool(cfg.model.nsg.get('composite_sky', True)):
            sky_color = pc.sky_cubemap(viewpoint_camera, result['acc'].detach())

            result['rgb'] = result['rgb'] + sky_color * (1 - result['acc'])
            sky_mask = viewpoint_camera.guidance.get('sky_mask') if hasattr(viewpoint_camera, 'guidance') else None
            if cfg.mode != 'train' and bool(cfg.model.nsg.get('force_sky_mask_composite', True)):
                force_sky_mask = None
                if sky_mask is not None:
                    force_sky_mask = sky_mask.to(device=result['rgb'].device, dtype=torch.bool)
                if bool(cfg.model.nsg.get('force_sky_dark_top_mask', True)) and hasattr(viewpoint_camera, 'original_image'):
                    original_image = viewpoint_camera.original_image[:3].to(device=result['rgb'].device)
                    threshold = float(cfg.model.nsg.get('force_sky_dark_threshold', 35.0 / 255.0))
                    dark = original_image.max(dim=0, keepdim=True).values <= threshold
                    top_connected_dark = torch.cumprod(dark[0].to(torch.int32), dim=0).bool().unsqueeze(0)
                    force_sky_mask = top_connected_dark if force_sky_mask is None else torch.logical_or(force_sky_mask, top_connected_dark)
                if force_sky_mask is not None:
                    result['rgb'] = torch.where(force_sky_mask, pc.sky_cubemap(viewpoint_camera, acc=None), result['rgb'])

        if pc.use_color_correction:
            result['rgb'] = pc.color_correction(viewpoint_camera, result['rgb'])

        if cfg.mode != 'train':
            result['rgb'] = torch.clamp(result['rgb'], 0., 1.)

        return result
    
            
    def render_kernel(
        self, 
        viewpoint_camera: Camera,
        pc: StreetGaussianModel,
        convert_SHs_python = None, 
        compute_cov3D_python = None, 
        scaling_modifier = None, 
        override_color = None,
        white_background = cfg.data.white_background,
    ):
        try:
            means3D = pc.get_xyz
            num_gaussians = len(means3D)
        except:
            num_gaussians = 0
        
        if num_gaussians == 0:
            if white_background:
                rendered_color = torch.ones(3, int(viewpoint_camera.image_height), int(viewpoint_camera.image_width), device="cuda")
            else:
                rendered_color = torch.zeros(3, int(viewpoint_camera.image_height), int(viewpoint_camera.image_width), device="cuda")
            
            rendered_acc = torch.zeros(1, int(viewpoint_camera.image_height), int(viewpoint_camera.image_width), device="cuda")
            rendered_semantic = torch.zeros(0, int(viewpoint_camera.image_height), int(viewpoint_camera.image_width), device="cuda")
            
            return {
                "rgb": rendered_color,
                "acc": rendered_acc,
                "semantic": rendered_semantic,
            }

        # Set up rasterization configuration and make rasterizer
        bg_color = [1, 1, 1] if white_background else [0, 0, 0]
        bg_color = torch.tensor(bg_color).float().cuda()
        scaling_modifier = scaling_modifier or self.cfg.scaling_modifier
        rasterizer = make_rasterizer(viewpoint_camera, pc.max_sh_degree, bg_color, scaling_modifier)
        
        convert_SHs_python = convert_SHs_python or self.cfg.convert_SHs_python
        compute_cov3D_python = compute_cov3D_python or self.cfg.compute_cov3D_python

        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        # screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
        if cfg.mode == 'train':
            screenspace_points = torch.zeros((num_gaussians, 3), requires_grad=True).float().cuda() + 0
            try:
                screenspace_points.retain_grad()
            except:
                pass
        else:
            screenspace_points = None 

        means2D = screenspace_points
        opacity = pc.get_opacity

        # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # scaling / rotation by the rasterizer.
        scales = None
        rotations = None
        cov3D_precomp = None
        if compute_cov3D_python:
            cov3D_precomp = pc.get_covariance(scaling_modifier)
        else:
            scales = pc.get_scaling
            rotations = pc.get_rotation

        # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
        # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
        shs = None
        colors_precomp = None
        if override_color is None:
            if convert_SHs_python:
                shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
                dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
                dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
                sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
                colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
            else:
                try:
                    shs = pc.get_features
                except:
                    colors_precomp = pc.get_colors(viewpoint_camera.camera_center)
        else:
            colors_precomp = override_color

        # TODO: add more feature here
        feature_names = []
        feature_dims = []
        features = []
        
        if cfg.render.render_normal:
            normals = pc.get_normals(viewpoint_camera)
            feature_names.append('normals')
            feature_dims.append(normals.shape[-1])
            features.append(normals)

        if cfg.data.get('use_semantic', False):
            semantics = pc.get_semantic
            feature_names.append('semantic')
            feature_dims.append(semantics.shape[-1])
            features.append(semantics)
        
        if len(features) > 0:
            features = torch.cat(features, dim=-1)
        else:
            features = None
        
        
        # Rasterize visible Gaussians to image, obtain their radii (on screen). 
        rendered_color, radii, rendered_depth, rendered_acc, rendered_feature = rasterizer(
            means3D = means3D,
            means2D = means2D,
            opacities = opacity,
            shs = shs,
            colors_precomp = colors_precomp,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp,
            semantics = features,
        )  
        
        if cfg.mode != 'train':
            rendered_color = torch.clamp(rendered_color, 0., 1.)
        
        rendered_feature_dict = dict()
        if rendered_feature.shape[0] > 0:
            rendered_feature_list = torch.split(rendered_feature, feature_dims, dim=0)
            for i, feature_name in enumerate(feature_names):
                rendered_feature_dict[feature_name] = rendered_feature_list[i]
        
        if 'normals' in rendered_feature_dict:
            rendered_feature_dict['normals'] = torch.nn.functional.normalize(rendered_feature_dict['normals'], dim=0)
                
        if 'semantic' in rendered_feature_dict:
            rendered_semantic = rendered_feature_dict['semantic']
            semantic_mode = cfg.model.gaussian.get('semantic_mode', 'logits')
            assert semantic_mode in ['logits', 'probabilities']
            if semantic_mode == 'logits': 
                pass # return raw semantic logits
            else:
                rendered_semantic = rendered_semantic / (torch.sum(rendered_semantic, dim=0, keepdim=True) + 1e-8) # normalize to probabilities
                rendered_semantic = torch.log(rendered_semantic + 1e-8) # change for cross entropy loss

            rendered_feature_dict['semantic'] = rendered_semantic
        
        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        
        result = {
            "rgb": rendered_color,
            "acc": rendered_acc,
            "depth": rendered_depth,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii
        }
        
        result.update(rendered_feature_dict)
        
        return result
