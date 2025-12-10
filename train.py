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
import numpy as np
import os
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from random import randint
from utils.loss_utils import l1_loss, ssim, laplacian_loss, laplacian_loss_U, get_effective_rank

from gaussian_renderer import render, network_gui
from mesh_renderer import NVDiffRenderer
import sys
from scene import Scene, GaussianModel, FlameGaussianModel, SpecularModel
from utils.general_utils import safe_state, colormap
import uuid
from tqdm import tqdm
from npz2mesh3 import nvdiffrecrender


from utils.image_utils import psnr, error_map, visualize_gaussians_with_tensor
from lpipsPyTorch import lpips
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from kornia.losses import inverse_depth_smoothness_loss

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    
    if pipe.SGs:
        # tb_writer = prepare_output_and_logger(dataset)
        specular_mlp = SpecularModel(sg_type=pipe.sg_type)
        specular_mlp.train_setting(opt)
    else:
        specular_mlp = None
    tb_writer = prepare_output_and_logger(dataset)
    if dataset.bind_to_mesh:
        
        train_normal = False
        if os.path.basename(dataset.source_path).startswith("FaceTalk"):
            # breakpoint()
            n_shape = 100    
            n_expr = 50
        else:
            n_shape = 300
            n_expr = 100    
        
       
    
        gaussians = FlameGaussianModel(dataset.sh_degree, dataset.sg_degree, dataset.disable_flame_static_offset, dataset.not_finetune_flame_params, n_shape=n_shape, n_expr=n_expr, 
            train_kinematic=pipe.train_kinematic, DTF = pipe.DTF,
            densification_type=opt.densification_type, detach_eyeball_geometry = pipe.detach_eyeball_geometry)
        try:
            mesh_renderer = NVDiffRenderer()
        except:
            mesh_renderer = None
            print("Mesh renderer not available")
    else:
        gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians,shuffle=False)

    train_cameras = scene.getTrainCameras()
    print(f"total {len(train_cameras)} camera_flames")

    for i, camera in enumerate(train_cameras):
        timestep = camera.timestep
        # camera_id = camera.uid

        # print(f"{i}: timestep{timestep}, camera{camera_id}")


        nvdiffrecrender(scene.gaussians, camera, timestep, total_frame_num=120)


















def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, losses, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, specular_mlp):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', losses['l1'].item(), iteration)
        tb_writer.add_scalar('train_loss_patches/ssim_loss', losses['ssim'].item(), iteration)
        if 'xyz' in losses:
            tb_writer.add_scalar('train_loss_patches/xyz_loss', losses['xyz'].item(), iteration)
        if 'scale' in losses:
            tb_writer.add_scalar('train_loss_patches/scale_loss', losses['scale'].item(), iteration)
        if 'dynamic_offset' in losses:
            tb_writer.add_scalar('train_loss_patches/dynamic_offset', losses['dynamic_offset'].item(), iteration)
        if 'laplacian' in losses:
            tb_writer.add_scalar('train_loss_patches/laplacian', losses['laplacian'].item(), iteration)
        if 'dynamic_offset_std' in losses:
            tb_writer.add_scalar('train_loss_patches/dynamic_offset_std', losses['dynamic_offset_std'].item(), iteration)

        # underlyind loss specially for surfhead


        if 'alpha' in losses:
            tb_writer.add_scalar('train_loss_patches/alpha_loss', losses['alpha'].item(), iteration)
        if 'normal' in losses:
            tb_writer.add_scalar('train_loss_patches/normal_loss', losses['normal'].item(), iteration)
        if 'surfel_normal_loss' in losses:
            tb_writer.add_scalar('train_loss_patches/surfel_normal_loss', losses['surfel_normal_loss'].item(), iteration)
        if 'surfel_dist_loss' in losses:
            tb_writer.add_scalar('train_loss_patches/surfel_dist_loss', losses['surfel_dist_loss'].item(), iteration)

        if 'blend_weight' in losses:
            tb_writer.add_scalar('train_loss_patches/blend_weight', losses['blend_weight'].item(), iteration)
        if 'normal_norm' in losses:
            tb_writer.add_scalar('train_loss_patches/normal_norm', losses['normal_norm'].item(), iteration)
     
        if 'convex_eyeballs' in losses:
            tb_writer.add_scalar('train_loss_patches/convex_eyeballs', losses['convex_eyeballs'].item(), iteration)
        if 'eye_alpha' in losses:
            tb_writer.add_scalar('train_loss_patches/eye_alpha', losses['eye_alpha'].item(), iteration)

        tb_writer.add_scalar('train_loss_patches/total_loss', losses['total'].item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    testing_iterations_rough = [_iter * 2 for _iter in testing_iterations]
    if iteration in testing_iterations: #or iteration in [5000, 10000]:
        print("[ITER {}] Evaluating".format(iteration))
        torch.cuda.empty_cache()
        validation_configs = (
            {'name': 'val', 'cameras' : scene.getValCameras()},
            {'name': 'test', 'cameras' : scene.getTestCameras()},
        )

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                ssim_test = 0.0
                lpips_test = 0.0
                num_vis_img = 10
                image_cache = []
                gt_image_cache = []
                vis_ct = 0
                for idx, viewpoint in tqdm(enumerate(DataLoader(config['cameras'], shuffle=False, batch_size=None, num_workers=8)), total=len(config['cameras'])):
                    if scene.gaussians.num_timesteps > 1:
                        scene.gaussians.select_mesh_by_timestep(viewpoint.timestep)
                    
                    
                    
              
                    specular_color=None
                    try:
                        if renderArgs[0].train_kinematic or renderArgs[0].train_kinematic_dist:
                            dir_pp = (scene.gaussians.get_blended_xyz - viewpoint.camera_center.repeat(
                                scene.gaussians.get_features.shape[0], 1).cuda())
                            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
                            normal_normalised=  F.normalize(scene.gaussians.get_normal,dim=-1).detach()
                            normal = scene.gaussians.get_normal
                            normal = normal * ((((dir_pp_normalized * normal_normalised).sum(dim=-1) < 0) * 1 - 0.5) * 2)[...,None]
                            
                            dir_pp_normalized = (scene.gaussians.blended_R.permute(0,2,1) @ dir_pp_normalized[...,None]).squeeze(-1)

                        else:
                            dir_pp = (scene.gaussians.get_xyz - viewpoint.camera_center.repeat(
                                scene.gaussians.get_features.shape[0], 1).cuda())
                            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
                            normal_normalised=  F.normalize(scene.gaussians.get_normal,dim=-1).detach()
                            normal = scene.gaussians.get_normal
                            normal = normal * ((((dir_pp_normalized * normal_normalised).sum(dim=-1) < 0) * 1 - 0.5) * 2)[...,None]
                            
                            dir_pp_normalized = (scene.gaussians.face_R_mat[scene.gaussians.binding].permute(0,2,1) @ dir_pp_normalized[...,None]).squeeze(-1)
                                
                        if renderArgs[0].spec_only_eyeball:
                            mask = torch.isin(scene.gaussians.binding, scene.gaussians.flame_model.mask.f.eyeballs)

                            points_indices = torch.nonzero(mask).squeeze(1)
                            
                            specular_color_eyeballs = specular_mlp.step(scene.gaussians.get_sg_features[points_indices], dir_pp_normalized[points_indices], normal[points_indices], sg_type = renderArgs[0].sg_type)
                            
                          
                            specular_color = specular_color_eyeballs
                        
                        else:
                            specular_color = specular_mlp.step(scene.gaussians.get_sg_features, dir_pp_normalized, normal.detach(), sg_type = renderArgs[0].sg_type)
                        # print('Specular Forwarded')
                    except:
                        pass
                    # breakpoint()
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs, specular_color, renderArgs[0].spec_only_eyeball)
                    
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    gt_mask = torch.clamp(viewpoint.original_mask.to("cuda"), 0.0, 1.0)
                    
                    if tb_writer and (idx % (len(config['cameras']) // num_vis_img) == 0):
                        tb_writer.add_images(config['name'] + "_{}/render".format(vis_ct), image[None], global_step=iteration)
                        error_image = error_map(image, gt_image)
                        tb_writer.add_images(config['name'] + "_{}/error".format(vis_ct), error_image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_{}/ground_truth".format(vis_ct), gt_image[None], global_step=iteration)
                        #! from NRFF
                    
                        try:
                            diffuse = render_pkg["rend_diffuse"]
                            specular = render_pkg['rend_specular']
                            tb_writer.add_images(config['name'] + "_{}/render_diffuse".format(vis_ct), diffuse[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_{}/render_specular".format(vis_ct), specular[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_{}/render_specular_2x".format(vis_ct), specular[None] * 2, global_step=iteration)
                            tb_writer.add_images(config['name'] + "_{}/render_specular_4x".format(vis_ct), specular[None] * 4, global_step=iteration)
                        except:
                            pass
                
                        #! from 2dgs
                        
                        depth = render_pkg["surfel_surf_depth"]
                        alpha = render_pkg['surfel_rend_alpha']
                        # breakpoint()
                        valid_depth_area = depth[gt_mask > 0.1]
                        max_depth_value = valid_depth_area.max()
                        min_depth_value = valid_depth_area.min()
                        
                        norm = max_depth_value - min_depth_value
                        depth[alpha < 0.1] = max_depth_value #! fill bg with max depth
                        depth = (depth - min_depth_value) / norm
                        # from torchvision.utils import save_image as si
                        # breakpoint()
                        # depth = depth / norm
                        depth = colormap(depth.cpu().numpy()[0], cmap='turbo')
                        depth = depth * gt_mask.repeat(3,1,1).cpu() + (1 - gt_mask.repeat(3,1,1).cpu())
                        tb_writer.add_images(config['name'] + "_{}/surfel_depth".format(vis_ct), depth[None], global_step=iteration)
                
                            
                        try:
                            rend_alpha = render_pkg['surfel_rend_alpha']
                            rend_normal = render_pkg["surfel_rend_normal"] * 0.5 + 0.5
                            surf_normal = (render_pkg["surfel_surf_normal"] * 0.5 + 0.5) * gt_mask.repeat(3,1,1).cuda + (1 - gt_mask.repeat(3,1,1).cuda())
                            tb_writer.add_images(config['name'] + "_{}/surfel_surfel_render_normal".format(vis_ct), rend_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_{}/surfel_surfel_normal".format(vis_ct), surf_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_{}/surfel_surfel_alpha".format(vis_ct), rend_alpha[None], global_step=iteration)
                          
                            rend_dist = render_pkg["surfel_rend_dist"]
                            rend_dist = colormap(rend_dist.cpu().numpy()[0])
                            tb_writer.add_images(config['name'] + "_view_{}/rend_dist".format(viewpoint.image_name), rend_dist[None], global_step=iteration)
                        except:
                            pass
                        
                        vis_ct += 1
                        
                        
                    image = viewpoint.original_mask_face.cuda() * image
                    gt_image = viewpoint.original_mask_face.cuda() * gt_image
                    # breakpoint()
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    ssim_test += ssim(image, gt_image).mean().double()

                    image_cache.append(image)
                    gt_image_cache.append(gt_image)

                    if idx == len(config['cameras']) - 1 or len(image_cache) == 16:
                        batch_img = torch.stack(image_cache, dim=0)
                        batch_gt_img = torch.stack(gt_image_cache, dim=0)
                        lpips_test += lpips(batch_img, batch_gt_img).sum().double()
                        image_cache = []
                        gt_image_cache = []

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                lpips_test /= len(config['cameras'])          
                ssim_test /= len(config['cameras'])          
                print("[ITER {}] Evaluating {}: L1 {:.4f} PSNR {:.4f} SSIM {:.4f} LPIPS {:.4f}".format(iteration, config['name'], l1_test, psnr_test, ssim_test, lpips_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - ssim', ssim_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - lpips', lpips_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_histogram("scene/scale_histogram", torch.mean(scene.gaussians.get_scaling, dim=-1), iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
            try:
                tb_writer.add_histogram('scene/blend_weight_primary',\
                    F.normalize(scene.gaussians.get_blend_weight, dim=-1)[...,:1], iteration)
            except:
                pass
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--interval", type=int, default=30_000, help="A shared iteration interval for test and saving results and checkpoints.")
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    if args.interval > op.iterations:
        args.interval = op.iterations // 5
    if len(args.test_iterations) == 0:
        args.test_iterations.extend(list(range(args.interval, args.iterations+1, args.interval)))
    if len(args.save_iterations) == 0:
        args.save_iterations.extend(list(range(args.interval, args.iterations+1, args.interval)))
    if len(args.checkpoint_iterations) == 0:
        args.checkpoint_iterations.extend(list(range(args.interval, args.iterations+1, args.interval)))
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)


    # print parameters
    print("\n" + "=" * 80)
    print("TRAINING CONFIGURATION")
    print("=" * 80)


    model_args = lp.extract(args)
    opt_args = op.extract(args)
    pipe_args = pp.extract(args)

    print("\n=== Model Parameters ===")
    for attr in sorted(dir(model_args)):
        if not attr.startswith('_'):
            value = getattr(model_args, attr)
            print(f"  {attr:30} : {value}")

    print("\n=== Optimization Parameters ===")
    for attr in sorted(dir(opt_args)):
        if not attr.startswith('_'):
            value = getattr(opt_args, attr)
            print(f"  {attr:30} : {value}")

    print("\n=== Pipeline Parameters ===")
    for attr in sorted(dir(pipe_args)):
        if not attr.startswith('_'):
            value = getattr(pipe_args, attr)
            print(f"  {attr:30} : {value}")

    print("\n=== Other Parameters ===")
    other_params = ['ip', 'port', 'debug_from', 'detect_anomaly', 'interval',
                    'test_iterations', 'save_iterations', 'quiet', 'checkpoint_iterations', 'start_checkpoint']
    for param in other_params:
        if hasattr(args, param):
            value = getattr(args, param)
            print(f"  {param:30} : {value}")

    print("=" * 80 + "\n")

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
