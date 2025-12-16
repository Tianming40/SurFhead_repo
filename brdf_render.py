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
from scene import Scene, RelightingScene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render, render_lighting,brdf_render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel, FlameGaussianModel
from utils.image_utils import apply_depth_colormap
from utils.general_utils import get_minimum_axis
from submodules.nvdiffrec.render.util import save_image_raw
import numpy as np


def render_lightings(model_path, name, iteration, gaussians, sample_num):
    lighting_path = os.path.join(model_path, name, "ours_{}".format(iteration))
    makedirs(lighting_path, exist_ok=True)
    # sampled_indicies = torch.randperm(gaussians.get_xyz.shape[0])[:sample_num]
    sampled_indicies = torch.arange(gaussians.get_xyz.shape[0], dtype=torch.long)[:sample_num]
    for sampled_index in tqdm(sampled_indicies, desc="Rendering lighting progress"):
        lighting = render_lighting(gaussians, sampled_index=sampled_index)
        torchvision.utils.save_image(lighting, os.path.join(lighting_path, '{0:05d}'.format(sampled_index) + ".png"))
        save_image_raw(os.path.join(lighting_path, '{0:05d}'.format(sampled_index) + ".hdr"),
                       lighting.permute(1, 2, 0).detach().cpu().numpy())


def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        torch.cuda.synchronize()

        render_pkg = brdf_render(view, gaussians, pipeline, background, speed=False)

        torch.cuda.synchronize()

        gt = view.original_image[0:3, :, :]
        gt_mask = view.original_mask[0:1, :, :]
        torchvision.utils.save_image(render_pkg["render"], os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        for k in render_pkg.keys():
            if render_pkg[k].dim() < 3 or k == "render" or k == "delta_normal_norm":
                continue
            save_path = os.path.join(model_path, name, "ours_{}".format(iteration), k)
            makedirs(save_path, exist_ok=True)
            img = render_pkg[k].detach().cpu().float()

            if k == "alpha":

                img = apply_depth_colormap(img[0][..., None], min=0., max=1.).permute(2, 0, 1)
            elif k == "surfel_rend_alpha" or k == "surfel_surf_depth" or k == "render_rend_dist":
                    # img: [1, H, W] → [3, H, W]
                img = img.repeat(3, 1, 1)


            elif k == "depth":
                img = apply_depth_colormap(-img[0][..., None]).permute(2, 0, 1)

            elif "normal" in k:
                img = 0.5 + 0.5 * img
            elif k == "surfel_rend_normal":
                img_vis = 0.5 + 0.5 * img
                mask = gt_mask.repeat(3, 1, 1).cpu()
                img_vis = img_vis * mask + (1 - mask)

                torchvision.utils.save_image(
                    img_vis,
                    os.path.join(save_path, f'{idx:05d}.png')
                )
                np.save(
                    os.path.join(save_path, f'{idx:05d}.npy'),
                    img.numpy()
                )
                continue
            elif k == "surfel_surf_normal":

                normal = render_pkg[k].detach().cpu().float()


                normal_vis = 0.5 + 0.5 * normal


                torchvision.utils.save_image(
                    normal_vis,
                    os.path.join(save_path, f'{idx:05d}.png')
                )

                np.save(
                    os.path.join(save_path, f'{idx:05d}.npy'),
                    normal.numpy()
                )
                continue
            elif img.shape[0] not in [1, 3]:

                print(f"Skipping {k} with shape {img.shape}")
                continue
            torchvision.utils.save_image(img, os.path.join(save_path, '{0:05d}.png'.format(idx)))


def render_sets(dataset: ModelParams, iteration: int, pipeline: PipelineParams, skip_train: bool, skip_test: bool):
    with torch.no_grad():
        if dataset.bind_to_mesh:
            if os.path.basename(dataset.source_path).startswith("FaceTalk"):
                # breakpoint()
                n_shape = 100
                n_expr = 50
            else:
                n_shape = 300
                n_expr = 100
        gaussians = FlameGaussianModel(sh_degree=-1, sg_degree=dataset.sg_degree, brdf_dim=3,
                                       brdf_mode=dataset.brdf_mode,
                                       brdf_envmap_res=dataset.brdf_envmap_res,
                                       disable_flame_static_offset=dataset.disable_flame_static_offset,
                                       not_finetune_flame_params=dataset.not_finetune_flame_params, n_shape=n_shape,
                                       n_expr=n_expr,
                                       train_kinematic=pipeline.train_kinematic, DTF=pipeline.DTF)
        scene = RelightingScene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline,
                       background)
            print("render_train done")

        if not skip_test:
            render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline,
                       background)
            print("render_test done")
        if pipeline.brdf:
            render_lightings(dataset.model_path, "lighting", scene.loaded_iter, gaussians, sample_num=1)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=False)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--brdf_iterations", default=20000, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.brdf_iterations, pipeline.extract(args), args.skip_train, args.skip_test)