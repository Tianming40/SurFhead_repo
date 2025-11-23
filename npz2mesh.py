import torch
import sys
import os
from typing import Tuple, Literal

from submodules.nvdiffrec.render.render import render_mesh, render_uv, render_layer
from submodules.nvdiffrec.render import  light, texture,material,util
from submodules.nvdiffrec.render import mesh  as nv_mesh
import nvdiffrast.torch as dr
from submodules.nvdiffrec.render import renderutils as ru
import numpy as np
from flame_model import flame
import math
from mesh_renderer import NVDiffRenderer

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
    # mesh_obj = nv_mesh.unit_size(mesh_obj)

    debug_mesh_info(mesh_obj)
    debug_camera_info(camera_info)

    if mesh_obj.v_pos is not None and mesh_obj.t_pos_idx is not None:
        print(f"✓ Successfully converted to Mesh:")
        print(f"  Vertices: {mesh_obj.v_pos.shape[0]}")
        print(f"  Faces: {mesh_obj.t_pos_idx.shape[0]}")
        print(f"  Has normals: {mesh_obj.v_nrm is not None}")
        print(f"  Has texcoords: {mesh_obj.v_tex is not None}")
        print(f"  Has tangents: {mesh_obj.v_tng is not None}")
    else:
        print("✗ Mesh conversion failed!")


    specular_color = torch.tensor([0.3, 0.3, 0.3], device='cuda')
    kd_texture = texture.load_texture2D("/home/tzhang/texture.jpg")

    simple_material = material.Material({
        'bsdf': 'pbr',
        'kd': kd_texture,
        'ks': texture.Texture2D(specular_color[None, None, :])
    })

    mesh_obj.material = simple_material
    ctx = dr.RasterizeCudaContext()


    env_light = light.load_env("/home/tzhang/109_hdrmaps_com_free_2K.hdr")

    mtx_in = camera_info.full_proj_transform.T.unsqueeze(0).cuda().float()
    view_pos = camera_info.camera_center.cuda().float()
    view_pos = view_pos.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # [1,1,1,3]



########################################################
    # mtx_in, view_pos = create_test_camera()

    test_background = create_background_from_env_light(env_light, mtx_in,view_pos,(camera_info.image_width, camera_info.image_height))
    check_mesh_in_frustum(mesh_obj, mtx_in)
    light.save_env_map("env.hdr", env_light)
    buffers = render_mesh(
        ctx=ctx,
        mesh=mesh_obj,
        mtx_in=mtx_in,
        view_pos=view_pos,
        lgt=env_light,
        resolution=(camera_info.image_height, camera_info.image_width),
        num_layers=3,
        background= test_background
    )
    # print(buffers)
    rendered_image = buffers['shaded'][0, ..., :3].clamp(0, 1)
    view_rendered_result(rendered_image,"picture")
    return buffers

# def create_background_from_env_light(env_light,  resolution):
#     # 直接将立方体贴图转换为经纬图作为背景
#     background = util.cubemap_to_latlong(
#         env_light.base,
#         [resolution[1], resolution[0]]  # [width, height]
#     )
#     return background.unsqueeze(0)
def create_background_from_env_light(env_light, mtx_in, view_pos, resolution, use_base=True):
    """
    将环境光转换为背景图像

    Args:
        env_light: EnvironmentLight对象
        camera_info: 相机信息
        resolution: 分辨率 (width, height)

    Returns:
        background_tensor: 背景图像张量 [1, H, W, 3]
    """

    width, height = resolution

    # 生成屏幕空间坐标 [-1, 1]
    u = torch.linspace(-1.0, 1.0, width, device='cuda')
    v = torch.linspace(1.0, -1.0, height, device='cuda')  # OpenGL坐标系y向下
    v, u = torch.meshgrid(v, u, indexing='ij')

    # 创建3D坐标 [H, W, 3] - 只包含x,y,z，不要第4个分量
    screen_coords = torch.stack([u, v, torch.ones_like(u)], dim=-1)  # [H, W, 3]

    # 重塑为 [1, H*W, 3] 以满足 xfm_points 的输入要求
    screen_coords_flat = screen_coords.reshape(1, -1, 3)  # [1, H*W, 3]

    # 使用逆投影矩阵将屏幕坐标转换到世界空间
    inv_proj = torch.inverse(mtx_in)
    world_coords_flat = ru.xfm_points(screen_coords_flat, inv_proj)  # [1, H*W, 4]

    # 现在world_coords_flat是齐次坐标 [1, H*W, 4]，需要转换为3D坐标
    # 通过除以w分量进行透视除法
    world_coords_flat = world_coords_flat[..., :3] / world_coords_flat[..., 3:4]

    # 重塑回图像形状 [1, H, W, 3]
    world_coords = world_coords_flat.reshape(1, height, width, 3)

    # 计算从相机出发的射线方向
    # 确保view_pos的形状正确 [1, 1, 1, 3]
    if len(view_pos.shape) == 1:
        view_pos = view_pos.unsqueeze(0).unsqueeze(0).unsqueeze(0)
    elif len(view_pos.shape) == 2:
        view_pos = view_pos.unsqueeze(0).unsqueeze(0)

    ray_dirs = util.safe_normalize(world_coords[0, :, :, :3] - view_pos[0, 0, 0])




    cubemap = env_light.specular[0][None, ...]


    # 从立方体贴图采样
    background = dr.texture(
        cubemap,
        ray_dirs.unsqueeze(0).contiguous(),
        filter_mode='linear',
        boundary_mode='cube'
    )

    # 确保形状正确 [1, H, W, 3]
    # 注意：dr.texture返回的形状是 [1, H, W, 3]，通常不需要permute
    # 但如果需要调整，可以取消下面的注释
    # background = background.permute(0, 2, 1, 3)

    return background

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

    os.makedirs('picture', exist_ok=True)


    plotter.show()
    # plotter.screenshot('picture/name.png', transparent_background=False)
    print("store in : picture/name.png")
    import imageio
    img = (rendered_image.detach().cpu().numpy() * 255).astype('uint8')  # [H,W,3]
    imageio.imwrite('picture/shaded_direct.png', img)
    print("wrote picture/shaded_direct.png")

def test_render_and_view():
    pass


if __name__ == "__main__":
    test_render_and_view()