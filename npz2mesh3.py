import torch
import sys
import os
from typing import Tuple, Literal
import cv2
from submodules.nvdiffrec.render.render import render_mesh, render_uv, render_layer
from submodules.nvdiffrec.render import  light, texture,material,util
from submodules.nvdiffrec.render import mesh  as nv_mesh
import nvdiffrast.torch as dr

import imageio
import numpy as np
import OpenEXR
import Imath
import pyvista as pv
import numpy as np
from submodules.nvdiffrec.render import renderutils as ru







def flame_to_nvdiffrec_mesh(flame_gaussian_model, timestep=0):

    flame_gaussian_model.select_mesh_by_timestep(timestep)

    vertices = flame_gaussian_model.verts.squeeze(0)  # [V, 3]
    faces = flame_gaussian_model.faces.squeeze(0)  # [F, 3]


    if hasattr(flame_gaussian_model.flame_model, 'verts_uvs'):
        v_tex = flame_gaussian_model.flame_model.verts_uvs  # [V_uv, 2]
        t_tex_idx = flame_gaussian_model.flame_model.textures_idx.squeeze(0)  # [F, 3]
    else:
        v_tex = None
        t_tex_idx = None


    mesh = nv_mesh.Mesh(
        v_pos=vertices,
        t_pos_idx=faces,
        v_tex=v_tex,
        t_tex_idx=t_tex_idx
    )

    mesh = nv_mesh.auto_normals(mesh)


    mesh = nv_mesh.compute_tangents(mesh)

    return mesh
def test_camer(camera_info,cam_pos=np.array([ 0.7654/2, 0.0, 0.6832/2], dtype=np.float32), target=np.array([0.0, 0.0, 0.0], dtype=np.float32), fovy_deg=60.0):

    forward = target - cam_pos
    forward = forward / np.linalg.norm(forward)


    world_up = np.array([0, -1, 0], dtype=np.float32)

    right = np.cross(world_up, forward)
    right = right / np.linalg.norm(right)

    up = np.cross(forward, right)


    R_world2cam = np.vstack([right, up, forward]).T  # world-to-camera rotation
    T_world2cam = -R_world2cam @ cam_pos  # w2c[:3,3]


    camera_info.R[:] = R_world2cam.astype(np.float32)
    camera_info.T[:] = T_world2cam.astype(np.float32)


    camera_info.camera_center[:] = torch.tensor(cam_pos, dtype=torch.float32, device=camera_info.camera_center.device)
    camera_info.image_width = 256
    camera_info.image_height = 256

    camera_info.FoVy = np.radians(fovy_deg)
    return
def create_light():
    base_res = 256  # > EnvironmentLight.LIGHT_MIN_RES (16)
    white_base = torch.ones(6, base_res, base_res, 3, device='cuda', dtype=torch.float32)
    # white_base[0] = 0.2
    # white_base[1] = 0.1
    # white_base[2] = 0.9
    # white_base[3] = 0.05
    # white_base[4] = 0.3
    # white_base[5] = 0.1
    env_light = light.EnvironmentLight(white_base)
    env_light.build_mips()

    return env_light
def nvdiffrecrender(gaussians, camera_info, timestep, total_frame_num, use_test_camera=False, bg_rotation= False, white_bg = True):

    if use_test_camera:
        test_camer(camera_info)

    mesh_obj = flame_to_nvdiffrec_mesh(gaussians, timestep=timestep)

    kd_texture = texture.load_texture2D("/home/tzhang/texture.jpg")
    Highth, Weidth = kd_texture.data.shape[1:3]
    specular_map = torch.zeros(Highth, Weidth, 3, device='cuda')
    specular_map[..., 0] = 0.04  # specular intensity for non-metal
    specular_map[..., 1] = 0.2  # roughness
    specular_map[..., 2] = 0.2  # metallic
    simple_material = material.Material({
        'bsdf': 'pbr',
        'kd': kd_texture,
        'ks': texture.Texture2D(specular_map)
    })

    mesh_obj.material = simple_material


    env_path = "/home/tzhang/012_hdrmaps_com_free_2K.exr"


    if not white_bg:
        env_light = light.load_env(env_path)
    else:
        env_light = create_light()



    # mtx_in = camera_info.full_proj_transform.T.unsqueeze(0).cuda().float()
    view_pos = camera_info.camera_center.cuda().float()
    view_pos = view_pos.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # [1,1,1,3]


    gbpos = mesh_obj.v_pos
    normal = mesh_obj.v_nrm

    vertex_uv_indices = torch.full((mesh_obj.v_pos.shape[0],), -1, dtype=torch.long, device='cuda')
    for face_idx in range(10144):
        for i in range(3):
            vertex_idx = mesh_obj.t_pos_idx[face_idx, i]
            uv_idx = mesh_obj.t_tex_idx[face_idx, i]
            if vertex_uv_indices[vertex_idx] == -1:
                vertex_uv_indices[vertex_idx] = uv_idx


    correct_uvs = mesh_obj.v_tex[vertex_uv_indices]  # [5143, 2]
    uv_coords = correct_uvs.unsqueeze(0).unsqueeze(0)  # [1, 1, 5143, 2]
    derivs = torch.zeros(1, 1, 5143, 4, device='cuda')
    kd_colors = kd_texture.sample(uv_coords, derivs).squeeze(0).squeeze(0)
    ks_values = torch.tensor([0.04, 0.4, 0.2], device='cuda').repeat(mesh_obj.v_pos.shape[0], 1)

    os.makedirs('video_material', exist_ok=True)
    save_root = 'video_material'
    if use_test_camera:
        os.makedirs('test_camera', exist_ok=True)
        save_root = 'test_camera'
    picture_name = camera_info.image_name
    save_dir = os.path.join(save_root, picture_name)
    os.makedirs(save_dir, exist_ok=True)
    if not bg_rotation:
        os.makedirs('no_rotation', exist_ok=True)
        save_dir = 'no_rotation'


    print(f"Saved for data{picture_name}image to: {save_dir}")
    total_dir = os.path.join(save_dir, 'total')
    diffuse_dir = os.path.join(save_dir, 'diffuse')
    specular_dir = os.path.join(save_dir, 'specular')
    point_dir = os.path.join(save_dir, 'point')


    for d in [total_dir, diffuse_dir, specular_dir, point_dir]:
        os.makedirs(d, exist_ok=True)

    if bg_rotation:
        for frame_idx in range(total_frame_num):

            theta = 2 * np.pi * frame_idx / total_frame_num
            rotation_matrix = torch.tensor([
                [np.cos(theta), 0, np.sin(theta), 0],
                [0, 1, 0, 0],
                [-np.sin(theta), 0, np.cos(theta), 0],
                [0, 0, 0, 1]
            ], device='cuda', dtype=torch.float32).unsqueeze(0)
            env_light.xfm(rotation_matrix)
            if not white_bg:
                env_map = load_env_map_exr(env_path, device='cuda', scale=1.0)
                background = render_background_from_env(env_map, camera_info, rotation_matrix=rotation_matrix)
                background_np = background.squeeze(0).cpu().numpy()
            else:
                background_np = np.ones((camera_info.image_height, camera_info.image_width, 3), dtype=np.float32)
            color, brdf_pkg = env_light.shade3(gbpos[None, None, ...], normal[None, None, ...], kd_colors[None, None, ...], ks_values[None, None, ...], view_pos)

            colors_precomp = color.squeeze()  # (N, 3)
            diffuse_color = brdf_pkg['diffuse'].squeeze()  # (N, 3)
            specular_color = brdf_pkg['specular'].squeeze()  # (N, 3)


            colors_np = colors_precomp.detach().cpu().numpy()
            vertices_np = mesh_obj.v_pos.detach().cpu().numpy()
            faces_np = mesh_obj.t_pos_idx.detach().cpu().numpy()
            diffuse_color_np = diffuse_color.detach().cpu().numpy()
            specular_color_np = specular_color.detach().cpu().numpy()

            # env_map = load_env_map_exr(env_path, device='cuda', scale=1.0)
            # background = render_background_from_env(env_map, camera_info)
            # vertices_flipped = vertices_np.copy()
            # vertices_flipped[:, 0] *= -1
            pv_mesh = pv.PolyData(
                vertices_np,
                np.hstack([np.full((len(faces_np), 1), 3), faces_np])
            )


            pv_mesh['colors'] = colors_np
            pv_mesh['diffuse'] = diffuse_color_np
            pv_mesh['specular'] = specular_color_np


            camera_pos = view_pos.squeeze().detach().cpu().numpy()


            R = camera_info.R
            t = camera_info.T
            Rt = np.zeros((4, 4))
            Rt[:3, :3] = R.transpose()
            Rt[:3, 3] = t
            Rt[3, 3] = 1.0

            C2W = np.linalg.inv(Rt)
            C2W[:3, 1] *= -1

            forward_vec = C2W[:3, 2]
            focal_point = camera_pos + forward_vec
            up_vector = C2W[:3, 1]



            # C2W same as josn but -z




            ##########################################
            # render total
            ##########################################

            plotter = pv.Plotter(off_screen=True, window_size=(camera_info.image_width, camera_info.image_height))
            plotter.add_mesh(pv_mesh, scalars='colors', rgb=True)
            plotter.add_axes()

            plotter.camera_position = [camera_pos, focal_point, up_vector]
            plotter.camera.view_angle = np.degrees(camera_info.FoVy)
            # plotter.camera.view_angle = np.degrees(1.0)
            plotter.add_text("total", font_size=16, color="black")
            pic_path = os.path.join(total_dir, f"{picture_name}_{frame_idx}_total.png")
            plotter.screenshot(pic_path)
            html_path = os.path.join(total_dir, f"{picture_name}_{frame_idx}_total.html")
            plotter.export_html(html_path)
            plotter.close()

            back_pic_path = os.path.join(total_dir, f"{picture_name}_{frame_idx}_total_background.png")
            img_mesh = imageio.imread(pic_path)

            img_mesh_float = img_mesh / 255.0  # [0,1]


            mask = (np.any(img_mesh_float < 0.99, axis=-1, keepdims=True)).astype(np.float32)


            composite = img_mesh_float * mask + background_np * (1 - mask)

            composite_uint8 = (np.clip(composite, 0, 1) * 255).astype(np.uint8)
            imageio.imwrite(back_pic_path, composite_uint8)
            ##########################################
            # render diffuse
            ##########################################

            plotter = pv.Plotter(off_screen=True,window_size=(camera_info.image_width,camera_info.image_height))

            plotter.add_mesh(pv_mesh, scalars='diffuse', rgb=True)

            plotter.camera_position = [camera_pos, focal_point, up_vector]
            plotter.camera.view_angle = np.degrees(camera_info.FoVy)
            # plotter.camera.view_angle = np.degrees(1.0)
            plotter.add_text("diffuse_only", font_size=16, color="black")

            pic_path = os.path.join(diffuse_dir, f"{picture_name}_{frame_idx}_diffuse.png")
            plotter.screenshot(pic_path)
            html_path = os.path.join(diffuse_dir, f"{picture_name}_{frame_idx}_diffuse.html")
            plotter.export_html(html_path)
            plotter.close()



            back_pic_path = os.path.join(diffuse_dir, f"{picture_name}_{frame_idx}_diffuse_background.png")
            img_mesh = imageio.imread(pic_path)

            img_mesh_float = img_mesh / 255.0  # [0,1]

            mask = (np.any(img_mesh_float < 0.99, axis=-1, keepdims=True)).astype(np.float32)

            composite = img_mesh_float * mask + background_np * (1 - mask)

            composite_uint8 = (np.clip(composite, 0, 1) * 255).astype(np.uint8)
            imageio.imwrite(back_pic_path, composite_uint8)
            ##########################################
            # render specular
            ##########################################

            plotter = pv.Plotter(off_screen=True, window_size=(camera_info.image_width,camera_info.image_height))

            plotter.add_mesh(pv_mesh, scalars='specular', rgb=True)


            focal_point = camera_pos + forward_vec
            plotter.camera_position = [camera_pos, focal_point, up_vector]
            plotter.camera.view_angle = np.degrees(camera_info.FoVy)
            # plotter.camera.view_angle = np.degrees(1.0)
            plotter.add_text("specular_only", font_size=16, color="black")
            pic_path = os.path.join(specular_dir, f"{picture_name}_{frame_idx}_specular.png")
            plotter.screenshot(pic_path)
            html_path = os.path.join(specular_dir, f"{picture_name}_{frame_idx}_specular.html")
            plotter.export_html(html_path)
            plotter.close()

            back_pic_path = os.path.join(specular_dir, f"{picture_name}_{frame_idx}_specular_background.png")
            img_mesh = imageio.imread(pic_path)

            img_mesh_float = img_mesh / 255.0  # [0,1]

            mask = (np.any(img_mesh_float < 0.99, axis=-1, keepdims=True)).astype(np.float32)

            composite = img_mesh_float * mask + background_np * (1 - mask)

            composite_uint8 = (np.clip(composite, 0, 1) * 255).astype(np.uint8)
            imageio.imwrite(back_pic_path, composite_uint8)

        subfolders = ['total', 'diffuse', 'specular']

        for sub in subfolders:
            folder_path = os.path.join(save_root, picture_name, sub)
            if not os.path.exists(folder_path):
                continue

            image_files = sorted(
                [f for f in os.listdir(folder_path) if f.endswith('.png') and "_background" not in f],
                key=lambda x: int(x.split('_')[2]))  # pictureName_frameIdx_XXX.png

            images = [imageio.imread(os.path.join(folder_path, f)) for f in image_files]

            video_path = os.path.join(save_root, picture_name, f"{sub}.mp4")
            imageio.mimsave(video_path, images, fps=8, quality=8)
            print(f"Saved video: {video_path}")

            background_image_files = sorted(
                [f for f in os.listdir(folder_path) if f.endswith('.png') and "_background" in f],
                key=lambda x: int(x.split('_')[2])
            )

            background_images = [imageio.imread(os.path.join(folder_path, f)) for f in background_image_files]
            background_video_path = os.path.join(save_root, picture_name, f"{sub}_background.mp4")
            imageio.mimsave(background_video_path, background_images, fps=8, quality=8)
            print(f"Saved video: {background_video_path}")

    else:

        if not white_bg:
            env_map = load_env_map_exr(env_path, device='cuda', scale=1.0)
            background = render_background_from_env(env_map, camera_info, rotation_matrix=None)
            background_np = background.squeeze(0).cpu().numpy()
        else:
            background_np = np.ones((camera_info.image_height, camera_info.image_width, 3), dtype=np.float32)

        color, brdf_pkg = env_light.shade3(gbpos[None, None, ...], normal[None, None, ...], kd_colors[None, None, ...],
                                           ks_values[None, None, ...], view_pos)

        colors_precomp = color.squeeze()  # (N, 3)
        diffuse_color = brdf_pkg['diffuse'].squeeze()  # (N, 3)
        specular_color = brdf_pkg['specular'].squeeze()  # (N, 3)

        colors_np = colors_precomp.detach().cpu().numpy()
        vertices_np = mesh_obj.v_pos.detach().cpu().numpy()
        faces_np = mesh_obj.t_pos_idx.detach().cpu().numpy()
        diffuse_color_np = diffuse_color.detach().cpu().numpy()
        specular_color_np = specular_color.detach().cpu().numpy()

        # env_map = load_env_map_exr(env_path, device='cuda', scale=1.0)
        # background = render_background_from_env(env_map, camera_info)
        # vertices_flipped = vertices_np.copy()
        # vertices_flipped[:, 0] *= -1
        pv_mesh = pv.PolyData(
            vertices_np,
            np.hstack([np.full((len(faces_np), 1), 3), faces_np])
        )

        pv_mesh['colors'] = colors_np
        pv_mesh['diffuse'] = diffuse_color_np
        pv_mesh['specular'] = specular_color_np

        camera_pos = view_pos.squeeze().detach().cpu().numpy()

        R = camera_info.R
        t = camera_info.T
        Rt = np.zeros((4, 4))
        Rt[:3, :3] = R.transpose()
        Rt[:3, 3] = t
        Rt[3, 3] = 1.0

        C2W = np.linalg.inv(Rt)
        C2W[:3, 1] *= -1

        forward_vec = C2W[:3, 2]
        focal_point = camera_pos + forward_vec
        up_vector = C2W[:3, 1]

        # C2W same as josn but -z

        ##########################################
        # render total
        ##########################################

        plotter = pv.Plotter(off_screen=True, window_size=(camera_info.image_width, camera_info.image_height))
        plotter.add_mesh(pv_mesh, scalars='colors', rgb=True)
        plotter.add_axes()

        plotter.camera_position = [camera_pos, focal_point, up_vector]
        plotter.camera.view_angle = np.degrees(camera_info.FoVy)
        # plotter.camera.view_angle = np.degrees(1.0)
        plotter.add_text("total", font_size=8, color="black")
        pic_path = os.path.join(total_dir, f"{picture_name}_total.png")
        plotter.screenshot(pic_path)

        plotter.deep_clean()
        plotter.close()


        # html_path = os.path.join(total_dir, f"{picture_name}_total.html")
        # plotter.export_html(html_path)
        # plotter.close()
        #
        # back_pic_path = os.path.join(total_dir, f"{picture_name}_total_background.png")
        # img_mesh = imageio.imread(pic_path)
        #
        # img_mesh_float = img_mesh / 255.0  # [0,1]
        #
        # mask = (np.any(img_mesh_float < 0.99, axis=-1, keepdims=True)).astype(np.float32)
        #
        # composite = img_mesh_float * mask + background_np * (1 - mask)
        #
        # composite_uint8 = (np.clip(composite, 0, 1) * 255).astype(np.uint8)
        # imageio.imwrite(back_pic_path, composite_uint8)
        ##########################################
        # render diffuse
        ##########################################

        plotter = pv.Plotter(off_screen=True, window_size=(camera_info.image_width, camera_info.image_height))

        plotter.add_mesh(pv_mesh, scalars='diffuse', rgb=True)

        plotter.camera_position = [camera_pos, focal_point, up_vector]
        plotter.camera.view_angle = np.degrees(camera_info.FoVy)
        # plotter.camera.view_angle = np.degrees(1.0)
        plotter.add_text("diffuse_only", font_size=8, color="black")

        pic_path = os.path.join(diffuse_dir, f"{picture_name}_diffuse.png")
        plotter.screenshot(pic_path)

        plotter.deep_clean()
        plotter.close()


        # html_path = os.path.join(diffuse_dir, f"{picture_name}_diffuse.html")
        # plotter.export_html(html_path)
        # plotter.close()
        #
        # back_pic_path = os.path.join(diffuse_dir, f"{picture_name}_diffuse_background.png")
        # img_mesh = imageio.imread(pic_path)
        #
        # img_mesh_float = img_mesh / 255.0  # [0,1]
        #
        # mask = (np.any(img_mesh_float < 0.99, axis=-1, keepdims=True)).astype(np.float32)
        #
        # composite = img_mesh_float * mask + background_np * (1 - mask)
        #
        # composite_uint8 = (np.clip(composite, 0, 1) * 255).astype(np.uint8)
        # imageio.imwrite(back_pic_path, composite_uint8)
        ##########################################
        # render specular
        ##########################################

        plotter = pv.Plotter(off_screen=True, window_size=(camera_info.image_width, camera_info.image_height))

        plotter.add_mesh(pv_mesh, scalars='specular', rgb=True)

        focal_point = camera_pos + forward_vec
        plotter.camera_position = [camera_pos, focal_point, up_vector]
        plotter.camera.view_angle = np.degrees(camera_info.FoVy)
        # plotter.camera.view_angle = np.degrees(1.0)
        plotter.add_text("specular_only", font_size=8, color="black")
        pic_path = os.path.join(specular_dir, f"{picture_name}_specular.png")
        plotter.screenshot(pic_path)

        plotter.deep_clean()
        plotter.close()

        #
        # html_path = os.path.join(specular_dir, f"{picture_name}_specular.html")
        # plotter.export_html(html_path)
        # plotter.close()
        #
        # back_pic_path = os.path.join(specular_dir, f"{picture_name}_specular_background.png")
        # img_mesh = imageio.imread(pic_path)
        #
        # img_mesh_float = img_mesh / 255.0  # [0,1]
        #
        # mask = (np.any(img_mesh_float < 0.99, axis=-1, keepdims=True)).astype(np.float32)
        #
        # composite = img_mesh_float * mask + background_np * (1 - mask)
        #
        # composite_uint8 = (np.clip(composite, 0, 1) * 255).astype(np.uint8)
        # imageio.imwrite(back_pic_path, composite_uint8)


    # plotter = pv.Plotter(off_screen=True)
    # plotter.add_points(pv_mesh, scalars='colors', rgb=True, point_size=5)
    # plotter.screenshot('vertex_points_only.png')
    # plotter.export_html('vertex_points_only.html')
    # plotter.close()



        # img = render_points_as_image(mesh_obj.v_pos, colors_precomp, mtx_in, camera_info.image_height, camera_info.image_width)
        #
        # img_np = (img.clamp(0, 1).detach().cpu().numpy() * 255).astype('uint8')
        # pic_path = os.path.join(point_dir, f"{picture_name}_{frame_idx}_pointrender.png")
        #
        # imageio.imwrite(pic_path, img_np)



    return color




def render_points_as_image(vertices, colors, mtx_in, H, W):

    device = vertices.device

    # 1. Homogeneous coords
    verts_clip = ru.xfm_points(vertices[None, ...], mtx_in)  # [1, N, 4]

    # -----------------------------
    # 2. Perspective divide → NDC
    # -----------------------------
    ndc = verts_clip[0, :, :3] / verts_clip[0, :, 3:4]  # [N,3]       # [N,3], x,y,z ∈ [-1,1]

    # 4. NDC → screen pixels
    x_img = (((ndc[:,0] + 1) * 0.5)) * W
    y_img = (( (ndc[:,1] + 1) * 0.5)) * H

    px = x_img.long().clamp(0, W-1)
    py = y_img.long().clamp(0, H-1)


    img = torch.ones(H, W, 3, device=device)

    # 6. Draw points (no depth test)
    img[py, px] = colors

    return img





def load_env_map_exr(path, device='cuda', scale=1.0):


    img = imageio.imread(path).astype(np.float32)
    img = torch.tensor(img, device=device)
    return img


def render_background_from_env(env_latlong, camera_info, rotation_matrix=None):
    """

        env_latlong: [H_env, W_env, 3], float32, CUDA, latlong HDR
        camera_info:  full_proj_transform, FoVx, FoVy, image_width, image_height, R, T

        background: [1, H, W, 3], float32, CUDA
    """
    import torch
    import numpy as np
    import math
    import nvdiffrast.torch as dr

    def to_torch(x):
        if torch.is_tensor(x):
            return x.to("cuda", dtype=torch.float32)
        return torch.tensor(x, dtype=torch.float32, device='cuda')

    R = to_torch(camera_info.R)  # [3,3]

    H, W = camera_info.image_height, camera_info.image_width


    gy, gx = torch.meshgrid(
        torch.linspace(-1, 1, H, device="cuda"),
        torch.linspace(-1, 1, W, device="cuda"),
        indexing='ij'
    )
    aspect = W / H

    tan_fovy = math.tan(np.radians(60) * 0.5)
    tan_fovx = aspect * tan_fovy

    # build rays in camera space (now +Z is forward)
    x = gx * tan_fovx
    y = -gy * tan_fovy  # flip so +Y goes up in 3D
    z = torch.ones_like(x)

    rays_cam = torch.stack((x, y, z), dim=-1)
    rays_cam = rays_cam / torch.norm(rays_cam, dim=-1, keepdim=True)

    # transform to world
    rays_world = rays_cam @ R
    rays_world = rays_world / torch.norm(rays_world, dim=-1, keepdim=True)


    # rays = camera_info.get_rays()
    # OPTIONAL user rotation (no initial_rot needed!)
    if rotation_matrix is not None:
        rot = rotation_matrix
        if isinstance(rot, torch.Tensor):
            rot = rot.to("cuda", dtype=torch.float32)
        if rot.ndim == 3 and rot.shape[0] == 1:
            rot = rot[0]

        rot3 = rot[:3, :3]  # shape = [3,3]

        rays_world = torch.einsum('hwc,cd->hwd', rays_world, rot3.T)
        rays_world = rays_world / torch.norm(rays_world, dim=-1, keepdim=True)






    vx, vy, vz = rays_world[..., 0], rays_world[..., 1], rays_world[..., 2]
    tu = torch.atan2(vx, -vz) / (2*np.pi) + 0.5
    tv =1- torch.acos(torch.clamp(vy, -1, 1)) / np.pi
    texcoords = torch.stack((tu, tv), dim=-1)  # [H, W, 2]


    if env_latlong.dim() == 3:
        env_latlong = env_latlong.unsqueeze(0)  # [1,H_env,W_env,3]

    background = dr.texture(env_latlong, texcoords.unsqueeze(0), filter_mode='linear')[0]  # [H,W,3]

    return background.unsqueeze(0)  # [1,H,W,3]



def check_mesh_in_frustum(mesh_obj, mtx_in):

    v_pos_homo = torch.cat([mesh_obj.v_pos, torch.ones(mesh_obj.v_pos.shape[0], 1, device='cuda')], dim=-1)
    v_clip = (mtx_in.squeeze(0) @ v_pos_homo.T).T

    v_ndc = v_clip[:, :3] / v_clip[:, 3:]

    print("=== Frustum Check ===")
    print(f"NDC bounds: min {v_ndc.min(dim=0)[0]}, max {v_ndc.max(dim=0)[0]}")
    print(f"Vertices in view (|x|,|y|,|z| <= 1): {((v_ndc.abs() <= 1.0).all(dim=1)).sum()}/{v_ndc.shape[0]}")


    outside_x = (v_ndc[:, 0].abs() > 1.0).sum()
    outside_y = (v_ndc[:, 1].abs() > 1.0).sum()
    outside_z = (v_ndc[:, 2].abs() > 1.0).sum()
    print(f"Outside X: {outside_x}, Y: {outside_y}, Z: {outside_z}")




def debug_mesh_info(mesh_obj):
    print("=== Mesh Debug Info ===")
    print(f"Vertices: {mesh_obj.v_pos.shape}")
    print(f"Faces: {mesh_obj.t_pos_idx.shape}")
    print(f"Vertex bounds:")
    print(f"  Min: {mesh_obj.v_pos.min(dim=0)[0]}")
    print(f"  Max: {mesh_obj.v_pos.max(dim=0)[0]}")
    print(f"  Center: {mesh_obj.v_pos.mean(dim=0)}")



def debug_camera_info(camera_info):
    print("=== Camera Debug Info ===")
    print(f"Projection matrix:\n{camera_info.full_proj_transform}")
    print(f"Camera center: {camera_info.camera_center}")
    print(f"World view transform:\n{camera_info.world_view_transform}")






def view_rendered_result(rendered_image, name):



    image_data = (rendered_image.detach().cpu().numpy() * 255).astype(np.uint8)


    pv_texture = pv.numpy_to_texture(image_data)

    plane = pv.Plane(center=(0, 0, 0), direction=(0, 0, 1),
                     i_size=5, j_size=5)


    plotter = pv.Plotter()
    plotter.add_mesh(plane, texture=pv_texture)
    plotter.add_title("nvdiffrec render result")



    plotter.show()
    # plotter.screenshot('picture/name.png', transparent_background=False)
    # print("store in : picture/name.png")
    import imageio
    os.makedirs('picture', exist_ok=True)
    name = f"{name}.png"
    img = (rendered_image.detach().cpu().numpy() * 255).astype('uint8')  # [H,W,3]
    path = os.path.join('picture', name)
    imageio.imwrite(path, img)
    print(f"Saved image to: {path}")

def test_render_and_view():
    pass


if __name__ == "__main__":
    test_render_and_view()