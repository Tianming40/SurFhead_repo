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

def exr_to_hdr(exr_path, hdr_path):
    # 打开 EXR 文件
    exr_file = OpenEXR.InputFile(exr_path)
    dw = exr_file.header()['dataWindow']
    H = dw.max.y - dw.min.y + 1
    W = dw.max.x - dw.min.x + 1

    # 读取 RGB 通道
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    R = np.frombuffer(exr_file.channel('R', pt), dtype=np.float32).reshape(H, W)
    G = np.frombuffer(exr_file.channel('G', pt), dtype=np.float32).reshape(H, W)
    B = np.frombuffer(exr_file.channel('B', pt), dtype=np.float32).reshape(H, W)

    # 拼接成 HWC
    img = np.stack([R, G, B], axis=-1)

    # 保存为 HDR
    imageio.imwrite(hdr_path, img, format='HDR-FI')
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
def load_env_map(path):
    img = cv2.imread(path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)   # HDR
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torch.tensor(img, dtype=torch.float32, device='cuda')
    return img

def nvdiffrecrender(gaussians, camera_info, timestep=0):


    mesh_obj = flame_to_nvdiffrec_mesh(gaussians, timestep=timestep)
    # mesh_obj = nv_mesh.unit_size(mesh_obj)

    # debug_mesh_info(mesh_obj)
    # debug_camera_info(camera_info)

    # if mesh_obj.v_pos is not None and mesh_obj.t_pos_idx is not None:
    #     print(f"✓ Successfully converted to Mesh:")
    #     print(f"  Vertices: {mesh_obj.v_pos.shape[0]}")
    #     print(f"  Faces: {mesh_obj.t_pos_idx.shape[0]}")
    #     print(f"  Has normals: {mesh_obj.v_nrm is not None}")
    #     print(f"  Has texcoords: {mesh_obj.v_tex is not None}")
    #     print(f"  Has tangents: {mesh_obj.v_tng is not None}")
    # else:
    #     print("✗ Mesh conversion failed!")


    # specular_color = torch.tensor([0.3, 0.3, 0.3], device='cuda')
    kd_texture = texture.load_texture2D("/home/tzhang/texture.jpg")
    Highth, Weidth = kd_texture.data.shape[1:3]
    specular_map = torch.zeros(Highth, Weidth, 3, device='cuda')
    specular_map[..., 0] = 0.04  # specular intensity for non-metal
    specular_map[..., 1] = 0.6  # roughness
    specular_map[..., 2] = 0.0  # metallic
    simple_material = material.Material({
        'bsdf': 'pbr',
        'kd': kd_texture,
        'ks': texture.Texture2D(specular_map)
    })

    mesh_obj.material = simple_material
    ctx = dr.RasterizeCudaContext()

    exr_to_hdr('/home/tzhang/012_hdrmaps_com_free_2K.exr', '/home/tzhang/012_hdrmaps_com_free_2K.hdr')

    env_light = light.load_env("/home/tzhang/012_hdrmaps_com_free_2K.hdr")


    mtx_in = camera_info.full_proj_transform.T.unsqueeze(0).cuda().float()
    view_pos = camera_info.camera_center.cuda().float()
    view_pos = view_pos.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # [1,1,1,3]

    env_map = load_env_map("/home/tzhang/012_hdrmaps_com_free_2K.hdr")
    background = render_background_from_env(env_map, camera_info)

    ########################################################
    # mtx_in, view_pos = create_test_camera()

    # test_background = create_background_from_env_light(env_light, mtx_in,view_pos,(camera_info.image_width, camera_info.image_height))
    # check_mesh_in_frustum(mesh_obj, mtx_in)
    # light.save_env_map("env.hdr", env_light)
    buffers = render_mesh(
        ctx=ctx,
        mesh=mesh_obj,
        mtx_in=mtx_in,
        view_pos=view_pos,
        lgt=env_light,
        resolution=(camera_info.image_height, camera_info.image_width),
        num_layers=3,
        background= background
    )
    # print(buffers)
    rendered_image = buffers['shaded'][0, ..., :3].clamp(0, 1)
    picture_name = camera_info.image_name
    view_rendered_result(rendered_image,picture_name)
    return buffers



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

    # 1. 生成每个像素的视线方向（相机坐标系）
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

    # 2. 将射线从相机坐标变换到世界坐标
    rays_world = rays_cam @ R.T
    rays_world = rays_world / torch.norm(rays_world, dim=-1, keepdim=True)

    # 3. 将方向转成 latlong 采样坐标
    vx, vy, vz = rays_world[..., 0], rays_world[..., 1], rays_world[..., 2]
    tu = torch.atan2(vx, -vz) / (2*np.pi) + 0.5
    tv = torch.acos(torch.clamp(vy, -1, 1)) / np.pi
    texcoords = torch.stack((tu, tv), dim=-1)  # [H, W, 2]

    # 4. 用 dr.texture 采样 HDR 环境贴图
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

    import pyvista as pv
    import numpy as np

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