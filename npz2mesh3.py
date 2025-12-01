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


def nvdiffrecrender(gaussians, camera_info, timestep=0):


    mesh_obj = flame_to_nvdiffrec_mesh(gaussians, timestep=timestep)

    kd_texture = texture.load_texture2D("/home/tzhang/texture.jpg")
    Highth, Weidth = kd_texture.data.shape[1:3]
    specular_map = torch.zeros(Highth, Weidth, 3, device='cuda')
    specular_map[..., 0] = 0.04  # specular intensity for non-metal
    specular_map[..., 1] = 0.4  # roughness
    specular_map[..., 2] = 0.2  # metallic
    simple_material = material.Material({
        'bsdf': 'pbr',
        'kd': kd_texture,
        'ks': texture.Texture2D(specular_map)
    })

    mesh_obj.material = simple_material


    env_path = "/home/tzhang/012_hdrmaps_com_free_2K.exr"


    env_light = light.load_env(env_path)

    mtx_in = camera_info.full_proj_transform.T.unsqueeze(0).cuda().float()
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
    roughness = torch.full((mesh_obj.v_pos.shape[0], 1), 0.4, device='cuda')

    # view_pos_flat = view_pos.squeeze()  # [1,1,1,3] -> [3]
    # view_pos_expanded = view_pos_flat.repeat(mesh_obj.v_pos.shape[0], 1)

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
    plotter = pv.Plotter(off_screen=True, window_size=(camera_info.image_width,camera_info.image_height))
    plotter.add_mesh(pv_mesh, scalars='colors', rgb=True)



    R = camera_info.R
    t = camera_info.T
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)

      # 相机的上方向 Y 轴

    C2W[:3, 1] *= -1
    # arrow = pv.Arrow(start=camera_pos, direction=forward_vec, scale=0.005)
    # plotter.add_mesh(arrow, color='red')
    forward_vec = C2W[:3, 2]  # Z 方向


    focal_point = camera_pos + forward_vec
    up_vector = C2W[:3, 1]

    plotter.add_axes()

    plotter.camera_position = [camera_pos, focal_point, up_vector]
    plotter.camera.view_angle = np.degrees(camera_info.FoVy)

    plotter.screenshot('rendered_debug.png')
    plotter.export_html('vertex_colors_only.html')
    plotter.close()


    # plotter = pv.Plotter(off_screen=True)
    # plotter.add_mesh(pv_mesh, scalars='diffuse', rgb=True)
    # plotter.screenshot('diffuse_only.png')
    # plotter.close()
    #
    # plotter = pv.Plotter(off_screen=True)
    # plotter.add_mesh(pv_mesh, scalars='specular', rgb=True)
    # plotter.screenshot('specular_only.png')
    # plotter.close()
    #
    # plotter = pv.Plotter(off_screen=True)
    # plotter.add_points(pv_mesh, scalars='colors', rgb=True, point_size=5)
    # plotter.screenshot('vertex_points_only.png')
    # plotter.export_html('vertex_points_only.html')
    # plotter.close()



    img = render_points_as_image(mesh_obj.v_pos, colors_precomp, mtx_in, camera_info.image_height, camera_info.image_width)

    img_np = (img.clamp(0, 1).detach().cpu().numpy() * 255).astype('uint8')
    imageio.imwrite("pointrender.png", img_np)
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


def render_background_from_env(env_latlong, camera_info):
    """

        env_latlong: [H_env, W_env, 3], float32, CUDA, latlong HDR
        camera_info: 含 full_proj_transform, FoVx, FoVy, image_width, image_height, R, T

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

    # tan_fovx = math.tan(camera_info.FoVx * 0.5)
    # tan_fovy = math.tan(camera_info.FoVy * 0.5)
    tan_fovx = math.tan(1.5 * 0.5)
    tan_fovy = math.tan(1.5 * H/W * 0.5)
    x = gx * tan_fovx
    y = gy * tan_fovy
    z = -torch.ones_like(x)

    rays_cam = torch.stack((x, y, z), dim=-1)
    rays_cam = rays_cam / torch.norm(rays_cam, dim=-1, keepdim=True)


    rays_world = rays_cam @ R.T
    rays_world = rays_world / torch.norm(rays_world, dim=-1, keepdim=True)


    vx, vy, vz = rays_world[..., 0], rays_world[..., 1], rays_world[..., 2]
    tu = torch.atan2(vx, -vz) / (2*np.pi) + 0.5
    tv = torch.acos(torch.clamp(vy, -1, 1)) / np.pi
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