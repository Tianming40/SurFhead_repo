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

import torch
import math
from typing import Union
from diff_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene import GaussianModel, FlameGaussianModel
from utils.sh_utils import eval_sh
from utils.mesh_utils import world_to_camera, compute_face_normals
from utils.loss_utils import hann_window
from utils.general_utils import build_rotation
from utils.point_utils import depth_to_normal


def render(viewpoint_camera, pc : Union[GaussianModel, FlameGaussianModel], pipe, 
    bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, specular_color = None
    ):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    render_bucket = {}
    # asset_bucket = {}
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass
    
    eyeball_mask = torch.isin(pc.binding, pc.flame_model.mask.f.eyeballs)
    eyeball_indices = torch.nonzero(eyeball_mask).squeeze(1)
    

    
    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform.cuda(),
        projmatrix=viewpoint_camera.full_proj_transform.cuda(),
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center.cuda(),
        prefiltered=False,
        debug=pipe.debug,
        tight_pruning_threshold = pipe.tight_pruning_threshold#*
    )
    # breakpoint()
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    if pipe.train_kinematic:
        means3D = pc.get_blended_xyz
        # if pipe.detach_eyeball_geometry:
        #     means3D[eyeball_indices] = means3D[eyeball_indices].detach()
    else:
        means3D = pc.get_xyz
        # if pipe.detach_eyeball_geometry:
        #     means3D[eyeball_indices] = means3D[eyeball_indices].detach()
    means2D = screenspace_points
    
    opacity = pc.get_opacity
    

    
    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
      
        rotations = pc.get_rotation
        # if pipe.detach_eyeball_geometry:
        #     rotations[eyeball_indices] = rotations[eyeball_indices].detach()
        

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    # print(rotations[eyeball_indices][0],'rotations_eb')
    pipe.convert_SHs_python = False
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            
            #! For exact directional queries for the SHs.
            if pipe.train_kinematic: 
                dir_pp_normalized = (pc.blended_R.permute(0,2,1) @ dir_pp_normalized[...,None]).squeeze(-1)
            else:
                dir_pp_normalized = (pc.face_R_mat.permute(0,2,1) @ dir_pp_normalized[...,None]).squeeze(-1)
                    
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color
        
    if pipe.SGs:
        shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
        dir_pp = (means3D - viewpoint_camera.camera_center.repeat(means3D.shape[0], 1).cuda())
        dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
        sh2rgb = eval_sh(pc.active_sh_degree,
                        shs_view,
                        dir_pp_normalized)
        if specular_color is None:
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
           
            specular_color_full = torch.zeros_like(sh2rgb).to(sh2rgb)
            specular_color_full[eyeball_indices] = specular_color
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0) + specular_color_full
        shs = None
   
    
    try:
        means3D.retain_grad()
    except:
        pass
    
    if pipe.DTF: #! to consensus with the convention with glm, matrices must be transposed as input. 
   
        if pipe.train_kinematic:

            Jacobians = pc.blended_Jacobian.permute(0,2,1)#[pc.binding]
            Jacobians_inv = torch.linalg.inv(Jacobians.permute(0,2,1))#.permute(0,2,1)

        else:
        
            Jacobians = pc.face_trans_mat[pc.binding].permute(0,2,1)
            Jacobians_inv = torch.linalg.inv(pc.face_trans_mat)[pc.binding]#.permute(0,2,1).permute(0,2,1)
    else:
        #! For Similarity transform, Jacobian is identity.
        Jacobians = torch.eye(3)[None].repeat(means3D.shape[0],1,1).to(means3D)
        Jacobians_inv = Jacobians

    
    rendered_image, radii, allmap, n_contrib_pixel, top_weights, top_depths, visible_points_tight = rasterizer(                                                                                   

        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp,
        Jacobians = Jacobians,
        Jacobians_inv = Jacobians_inv)#*
    
    
    render_bucket.update({"render": rendered_image, #* 0-1
            "viewspace_points": means2D,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "visibility_filter_tight": visible_points_tight,
            "top_weights": top_weights,
            "top_depths": top_depths,
            "n_contrib_pixel": n_contrib_pixel,#*
            })
    
  
    # additional regularizations
    render_alpha = allmap[1:2]

    # get normal map
    render_normal = allmap[2:5]

    #! world normal
    render_normal = (render_normal.permute(1,2,0) @ (viewpoint_camera.world_view_transform[:3,:3].T.cuda())).permute(2,0,1)

    # get median depth map
    render_depth_median = allmap[5:6]
    render_depth_median = torch.nan_to_num(render_depth_median, 0, 0)

    # get expected depth map
    render_depth_intersect = allmap[0:1]
   
    render_depth_expected = (render_depth_intersect / render_alpha)
    render_depth_expected = torch.nan_to_num(render_depth_expected, 0, 0)

    render_dist = allmap[6:7]
    

    #? psedo surface attributes
    #? surf depth is either median or expected by setting depth_ratio to 1 or 0
    #? for bounded scene, use median depth, i.e., depth_ratio = 1; 
    #? for unbounded scene, use expected depth, i.e., depth_ration = 0, to reduce disk anliasing.
    if pipe.depth_ratio == -1:
        assert False, "insersect depth is not valid, set 1(median) or 0(mean)"
        surf_depth = render_depth_intersect
    else:
        surf_depth = render_depth_expected * (1-pipe.depth_ratio) + (pipe.depth_ratio) * render_depth_median
    
    # assume the depth points form the 'surface' and generate psudo surface normal for regularizations.
    # breakpoint()
    surf_normal = depth_to_normal(viewpoint_camera, surf_depth)
    
    surf_normal = surf_normal.permute(2,0,1)
    # remember to multiply with accum_alpha since render_normal is unnormalized.
    surf_normal = surf_normal * (render_alpha).detach()
    
    render_bucket.update({
            'surfel_rend_alpha': render_alpha,
            'surfel_rend_normal': render_normal,
            'surfel_rend_dist': render_dist,
            'surfel_surf_depth': surf_depth,
            'surfel_surf_normal': surf_normal,
    })
    if pipe.SGs:
        shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
        dir_pp = (means3D - viewpoint_camera.camera_center.repeat(means3D.shape[0], 1).cuda())
        dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
        sh2rgb = eval_sh(pc.active_sh_degree,
                        shs_view,
                        dir_pp_normalized)
        # colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0) + mlp_color
        #! 여기서 분리 
        colors_precomp_diffuse = torch.clamp_min(sh2rgb + 0.5 , 0.0)
        
        rendered_diffuse = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = shs,
            colors_precomp = colors_precomp_diffuse,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp,
            Jacobians = Jacobians,
            Jacobians_inv = Jacobians_inv)[0]
        
        if specular_color is None:
            colors_precomp_specular = torch.zeros_like(colors_precomp_diffuse).to(colors_precomp_diffuse)
        else:
            # if spec_only_eyeball:
            #     # specular_color_full = torch.zeros_like(sh2rgb).to(sh2rgb)
            #     # specular_color_full[eyeball_indices] = specular_color
            #     colors_precomp_specular = specular_color_full
            # else:
            colors_precomp_specular = specular_color

        
       
        rendered_specular = rasterizer(
            means3D = means3D[eyeball_indices],
            means2D = means2D[eyeball_indices],
            shs = shs,
            colors_precomp = colors_precomp_specular,
            opacities = opacity[eyeball_indices],
            scales = scales[eyeball_indices],
            rotations = rotations[eyeball_indices],
            cov3D_precomp = cov3D_precomp,
            Jacobians = Jacobians[eyeball_indices],
            Jacobians_inv = Jacobians_inv[eyeball_indices])[0]
    
            
        render_bucket.update({
                'rend_diffuse': rendered_diffuse,
                'rend_specular': rendered_specular,
        })

    return render_bucket

 