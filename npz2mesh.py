import torch
import sys
import os


from submodules.nvdiffrec import render as nvdiffrec
from submodules.nvdiffrec.render.render import render_mesh, render_uv, render_layer
from submodules.nvdiffrec.render import  light, texture,material
from submodules.nvdiffrec.render import mesh  as nv_mesh
import nvdiffrast.torch as dr
import numpy as np
from flame_model import flame


def flame_to_mesh(flame_model, verts):
    """
    将FLAME模型输出转换为Mesh类格式
    """
    # 基本网格数据
    v_pos = verts.squeeze(0)  # (V, 3)
    t_pos_idx = flame_model.faces  # (F, 3)

    # 创建基础Mesh
    basic_mesh = nv_mesh.Mesh(v_pos=v_pos, t_pos_idx=t_pos_idx)

    # 自动计算法线
    result_mesh  = nv_mesh.auto_normals(basic_mesh)

    # 如果有纹理坐标，也添加
    if hasattr(flame_model, 'verts_uvs') and hasattr(flame_model, 'textures_idx'):
        v_tex = flame_model.verts_uvs  # (VT, 2)
        t_tex_idx = flame_model.textures_idx  # (F, 3)
        result_mesh .v_tex = v_tex
        result_mesh .t_tex_idx = t_tex_idx

        # 计算切线空间
        result_mesh  = nv_mesh.compute_tangents(result_mesh )


    return result_mesh







def nvdiffrecrender(gaussians, camera_info, timestep=0):
    gaussians.select_mesh_by_timestep(timestep)

    flamemesh = gaussians.flame_model

    print(flamemesh)

    verts = gaussians.verts  # 变形后的顶点
    verts_cano = gaussians.verts_cano  # 规范空间顶点
    faces = gaussians.faces

    # 转换为Mesh格式
    mesh_obj = flame_to_mesh(gaussians.flame_model, verts, verts_cano)
    mesh_obj = nv_mesh.unit_size(mesh_obj)
    if mesh_obj.v_pos is not None and mesh_obj.t_pos_idx is not None:
        print(f"✓ Successfully converted to Mesh:")
        print(f"  Vertices: {mesh_obj.v_pos.shape[0]}")
        print(f"  Faces: {mesh_obj.t_pos_idx.shape[0]}")
        print(f"  Has normals: {mesh_obj.v_nrm is not None}")
        print(f"  Has texcoords: {mesh_obj.v_tex is not None}")
        print(f"  Has tangents: {mesh_obj.v_tng is not None}")
    else:
        print("✗ Mesh conversion failed!")




    color = torch.tensor([1.0, 0.0, 0.0], device='cuda')
    specular_color = torch.tensor([0.04, 0.04, 0.04], device='cuda')

    simple_material = material.Material({
        'bsdf': 'kd',
        'kd': texture.Texture2D(color[None, None, :]) , # 1x1纹理
        'ks': texture.Texture2D(specular_color[None, None, :])
    })

    mesh_obj.material = simple_material
    ctx = dr.RasterizeCudaContext()

    resolution = (512, 512)

    def create_black_light():
        base_res = 16  # 最小分辨率
        # 创建全黑的cubemap
        black_base = torch.zeros(6, base_res, base_res, 3, device='cuda')
        env_light = light.EnvironmentLight(black_base)
        env_light.build_mips()  # 构建mipmaps
        return env_light



    black_light = create_black_light()

    mtx_in = camera_info.full_proj_transform.transpose(0, 1).cuda()  # (4, 4)
    mtx_in = mtx_in.unsqueeze(0)

    view_pos = camera_info.camera_center.cuda()  # (3,)
    view_pos = view_pos.unsqueeze(0).unsqueeze(0).unsqueeze(0)

    buffers = render_mesh(
        ctx=ctx,
        mesh=mesh_obj,
        mtx_in=mtx_in,
        view_pos=view_pos,
        lgt=black_light,
        resolution=resolution,
    )
    print(buffers)
    rendered_image = buffers['shaded'][0, ..., :3].clamp(0, 1)
    view_rendered_result(rendered_image,"picture")
    return buffers







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

def test_render_and_view():
    pass


if __name__ == "__main__":
    test_render_and_view()