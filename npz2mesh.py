import torch
import sys
import os


from submodules.nvdiffrec.render.render import render_mesh, render_uv, render_layer
from submodules.nvdiffrec.render import  light, texture,material,util
from submodules.nvdiffrec.render import mesh  as nv_mesh
import nvdiffrast.torch as dr
import numpy as np
from flame_model import flame
import math


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
    mesh_obj = nv_mesh.unit_size(mesh_obj)

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




    color = torch.tensor([0.9, 0.2, 0.2], device='cuda')
    specular_color = torch.tensor([0.3, 0.3, 0.3], device='cuda')
    kd_texture = texture.load_texture2D("/home/tzhang/texture.jpg")

    simple_material = material.Material({
        'bsdf': 'pbr',
        # 'kd': texture.Texture2D(color[None, None, :]) ,
        'kd': kd_texture,
        'ks': texture.Texture2D(specular_color[None, None, :])
    })

    mesh_obj.material = simple_material
    ctx = dr.RasterizeCudaContext()

    def create_background_from_env_light(env_light, resolution):
        """将环境光照转换为2D背景图像"""
        # 将立方体贴图转换为等距柱状投影（全景图）
        background = util.cubemap_to_latlong(env_light.base, resolution)
        if len(background.shape) == 3:
            background = background[None, ...]  # 添加batch维度

            # 检查尺寸
        assert background.shape[1] == resolution[1], f"背景高度 {background.shape[1]} 不等于分辨率高度 {resolution[1]}"
        assert background.shape[2] == resolution[0], f"背景宽度 {background.shape[2]} 不等于分辨率宽度 {resolution[0]}"

        return background


    def create_black_light():
        base_res = 256  # > EnvironmentLight.LIGHT_MIN_RES (16)
        white_base = torch.ones(6, base_res, base_res, 3, device='cuda', dtype=torch.float32)
        white_base[0] = 0.2
        white_base[1] = 0.1
        white_base[2] = 0.9
        white_base[3] = 0.05
        white_base[4] = 0.3
        white_base[5] = 0.1
        env_light = light.EnvironmentLight(white_base)
        env_light.build_mips()

        return env_light


    black_light = create_black_light()
    env_light = light.load_env("/home/tzhang/109_hdrmaps_com_free_2K.hdr")
############################################################
    world_view = camera_info.world_view_transform.cuda().float()
    if world_view.dim() == 2:
        world_view = world_view.unsqueeze(0)  # [1,4,4]

    fov = math.radians(60.0)
    aspect = camera_info.image_width / camera_info.image_height
    near = 0.1
    far = 10.0

    f = 1.0 / math.tan(fov * 0.5)
    proj_matrix = torch.tensor([[
        [f / aspect, 0, 0, 0],
        [0, f, 0, 0],
        [0, 0, -(far + near) / (far - near), -2 * far * near / (far - near)],
        [0, 0, -1, 0]
    ]], dtype=torch.float32, device='cuda')

    mtx_in = proj_matrix @ world_view# [1,4,4] @ [1,4,4] = [1,4,4]


    view_pos = camera_info.camera_center.cuda().float()
    view_pos = view_pos.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # [1,1,1,3]

    print("=== Gaussian to nvdiffrec Conversion ===")
    print(f"FoVx: {camera_info.FoVx:.3f}, FoVy: {camera_info.FoVy:.3f}")

    print(f"View matrix shape: {world_view.shape}")
    print(f"Projection matrix shape: {proj_matrix.shape}")
    print(f"Combined mtx_in shape: {mtx_in.shape}")
    print(f"View pos: {view_pos.squeeze()}")


########################################################
    mtx_in, view_pos = create_test_camera()
    test_background = create_background_from_env_light(env_light, (1024, 1024))
    check_mesh_in_frustum(mesh_obj, mtx_in)
    buffers = render_mesh(
        ctx=ctx,
        mesh=mesh_obj,
        mtx_in=mtx_in,
        view_pos=view_pos,
        lgt=env_light,
        resolution=(1024, 1024),
        num_layers=3,
        background= test_background
    )
    # print(buffers)
    rendered_image = buffers['shaded'][0, ..., :3].clamp(0, 1)
    view_rendered_result(rendered_image,"picture")
    return buffers


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


def create_test_camera():

    fov = 60.0
    aspect = 1.0
    near = 0.1
    far = 100.0


    f = 1.0 / np.tan(np.radians(fov) / 2.0)
    proj_matrix = torch.tensor([
        [f / aspect, 0, 0, 0],
        [0, f, 0, 0],
        [0, 0, -(far + near) / (far - near), -2 * far * near / (far - near)],
        [0, 0, -1, 0]
    ], dtype=torch.float32, device='cuda')

    view_matrix = torch.tensor([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, -3],
        [0, 0, 0, 1]
    ], dtype=torch.float32, device='cuda')

    mtx_in = (proj_matrix @ view_matrix).unsqueeze(0)

    view_pos = torch.tensor([0, 0, 3], device='cuda', dtype=torch.float32).unsqueeze(0).unsqueeze(0).unsqueeze(0)

    return mtx_in, view_pos





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
    plotter.screenshot('picture/name.png', transparent_background=False)
    print("store in : picture/name.png")
    import imageio
    img = (rendered_image.detach().cpu().numpy() * 255).astype('uint8')  # [H,W,3]
    imageio.imwrite('picture/shaded_direct.png', img)
    print("wrote picture/shaded_direct.png")

def test_render_and_view():
    pass


if __name__ == "__main__":
    test_render_and_view()