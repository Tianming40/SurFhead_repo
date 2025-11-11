#
# Toyota Motor Europe NV/SA and its affiliated companies retain all intellectual
# property and proprietary rights in and to this software and related documentation.
# Any commercial use, reproduction, disclosure or distribution of this software and
# related documentation without an express license agreement from Toyota Motor Europe NV/SA
# is strictly prohibited.
#


from typing import Optional
import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation, get_minimum_axis, flip_align_view
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
# from pytorch3d.transforms import quaternion_multiply
from roma import quat_product, quat_xyzw_to_wxyz, quat_wxyz_to_xyzw, rotvec_to_rotmat, rotmat_to_rotvec
from utils.sh_utils import RGB2SH
from .gaussian_model import GaussianModel
from .flame_gaussian_model import FlameGaussianModel
# from utils.graphics_utils import BasicPointCloud
from utils.graphics_utils import BasicPointCloud, get_dist_weight
from utils.general_utils import strip_symmetric, build_scaling_rotation
import torch.nn.functional as F
from torch.autograd import Function
from roma import rotmat_to_unitquat, quat_xyzw_to_wxyz, unitquat_to_rotmat, special_procrustes
from scene.NVDIFFREC import create_trainable_env_rnd

# from roma import quat_product, quat_xyzw_to_wxyz, quat_wxyz_to_xyzw, rotvec_to_rotmat, rotmat_to_rotvec
class BRDFFlameGaussianModel(FlameGaussianModel):
    def __init__(self, sh_degree: int, sg_degree: int,
                 brdf_dim: int = -1, brdf_mode: str = "none", brdf_envmap_res: int = 64,
                 **flame_kwargs):


        super().__init__(sh_degree, sg_degree, **flame_kwargs)

        if (brdf_dim >= 0 and sh_degree >= 0) or (brdf_dim < 0 and sh_degree < 0):
            raise Exception('Please provide exactly one of either brdf_dim or sh_degree!')
        self.brdf = brdf_dim >= 0


        self.brdf_dim = brdf_dim
        self.brdf_mode = brdf_mode
        self.brdf_envmap_res = brdf_envmap_res

        self._normal = torch.empty(0)
        self._normal2 = torch.empty(0)
        self._specular = torch.empty(0)
        self._roughness = torch.empty(0)

        # BRDF MLP
        if self.brdf:
            self.brdf_mlp = create_trainable_env_rnd(self.brdf_envmap_res, scale=0.0, bias=0.8)
        else:
            self.brdf_mlp = None

        self.diffuse_activation = torch.sigmoid
        self.specular_activation = torch.sigmoid
        self.roughness_activation = torch.sigmoid
        self.roughness_bias = 0.0
        self.default_roughness = 0.6

        self.training_stage = 1

    @property
    def get_diffuse(self):
        if self.brdf:
            return self.diffuse_activation(self._features_dc)
        else:
            return self._features_dc

    @property
    def get_specular(self):
        return self.specular_activation(self._specular)

    @property
    def get_roughness(self):
        return self.roughness_activation(self._roughness + self.roughness_bias)

    @property
    def get_brdf_features(self):
        return self._features_rest


    def set_training_stage(self, stage):

        self.training_stage = stage

        if stage == 1:

            self._freeze_parameters(['_normal', '_normal2', '_specular', '_roughness', 'brdf_mlp'])
            self._unfreeze_parameters(['_xyz', '_scaling', '_rotation', '_opacity',
                                       '_features_dc', '_features_rest', '_features_sg',
                                       'flame_param'])

        elif stage == 2:

            self._freeze_parameters(['_xyz', '_scaling', '_rotation', '_opacity',
                                     '_features_dc', '_features_rest', '_features_sg',
                                     'flame_param'])
            self._unfreeze_parameters(['_normal', '_normal2', '_specular', '_roughness'])

        elif stage == 3:

            self._unfreeze_parameters(['_xyz', '_scaling', '_rotation', '_opacity',
                                       '_features_dc', '_features_rest', '_features_sg',
                                       '_normal', '_normal2', '_specular', '_roughness',
                                       'brdf_mlp', 'flame_param'])

    def _freeze_parameters(self, param_names):
        """冻结指定参数"""
        for name in param_names:
            if hasattr(self, name):
                param = getattr(self, name)
                if isinstance(param, nn.Parameter):
                    param.requires_grad = False
                elif isinstance(param, nn.Module):
                    for p in param.parameters():
                        p.requires_grad = False

    def _unfreeze_parameters(self, param_names):
        """解冻指定参数"""
        for name in param_names:
            if hasattr(self, name):
                param = getattr(self, name)
                if isinstance(param, nn.Parameter):
                    param.requires_grad = True
                elif isinstance(param, nn.Module):
                    for p in param.parameters():
                        p.requires_grad = True


    def create_from_pcd(self, pcd: Optional[BasicPointCloud], spatial_lr_scale: float):
        self.spatial_lr_scale = spatial_lr_scale
        if pcd == None:
            assert self.binding is not None
            num_pts = self.binding.shape[0]
            fused_point_cloud = torch.zeros((num_pts, 3)).float().cuda()
            fused_color = torch.tensor(np.random.random((num_pts, 3)) / 255.0).float().cuda()
        else:
            if not self.brdf:
                fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
                features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
                features[:, :3, 0] = fused_color
                features[:, 3:, 1:] = 0.0
            elif (self.brdf_mode == "envmap" and self.brdf_dim == 0):
                fused_color = torch.tensor(np.asarray(pcd.colors)).float().cuda()
                features = torch.zeros((fused_color.shape[0], self.brdf_dim + 3)).float().cuda()
                features[:, :3] = fused_color
                features[:, 3:] = 0.0
            elif self.brdf_mode == "envmap" and self.brdf_dim > 0:
                fused_color = torch.tensor(np.asarray(pcd.colors)).float().cuda()
                features = torch.zeros((fused_color.shape[0], 3)).float().cuda()
                features[:, :3] = fused_color
                features[:, 3:] = 0.0
                features_rest = torch.zeros((fused_color.shape[0], 3, (self.brdf_dim + 1) ** 2)).float().cuda()
            else:
                raise NotImplementedError


        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        if not self.brdf:
            self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
            self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        else:
            self._features_dc = nn.Parameter(features[:,:3].contiguous().requires_grad_(True))
            if (self.brdf_mode=="envmap" and self.brdf_dim==0):
                self._features_rest = nn.Parameter(features[:,3:].contiguous().requires_grad_(True))
            elif self.brdf_mode=="envmap":
                self._features_rest = nn.Parameter(features_rest.contiguous().requires_grad_(True))

        sg_features = torch.zeros((fused_color.shape[0], self.max_sg_degree)).float().cuda()

        print("Number of points at initialisation: ", self.get_xyz.shape[0])

        if self.binding is None:
            dist2 = torch.clamp_min(distCUDA2(self.get_xyz), 0.0000001)
            scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 2)
        else:

            scales = torch.log(torch.ones((self.get_xyz.shape[0], 2), device="cuda"))
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        self._features_sg = nn.Parameter(sg_features.requires_grad_(True))

        if self.train_kinematic:

            # identity = inverse_sigmoid(torch.ones((self._xyz.shape[0],1), device="cuda")) #!1
            # neighbour = inverse_sigmoid(torch.ones((self._xyz.shape[0],3)*0.01, device="cuda"))  #!3
            identity = inverse_sigmoid(0.9 * torch.ones((self._xyz.shape[0], 1), device="cuda"))  # !1
            neighbour = inverse_sigmoid(0.05 * torch.ones((self._xyz.shape[0], 3), device="cuda"))

            # identity = inverse_sigmoid(torch.ones((self._xyz.shape[0],1), device="cuda")) #!1
            # neighbour = inverse_sigmoid(torch.zeros((self._xyz.shape[0],3), device="cuda"))
            boundary_mask = torch.isin(self.binding, self.flame_model.mask.f.boundary)
            boundary_indices = torch.nonzero(boundary_mask).squeeze(1)

            if self.detach_boundary:
                identity[boundary_indices] = inverse_sigmoid(
                    (1 - (1e-10)) * torch.ones((self._xyz[boundary_indices].shape[0], 1), device="cuda"))
                neighbour[boundary_indices] = inverse_sigmoid(
                    (1e-10) * torch.ones((self._xyz[boundary_indices].shape[0], 3), device="cuda"))

            bw = torch.cat((identity, neighbour), 1)  # .cuda()
            # self.blend_weight = nn.Parameter(.requires_grad_(True))
            # breakpoint()
            self.blend_weight = nn.Parameter(bw.requires_grad_(True))





    # 重写PLY加载方法以支持BRDF参数
    def load_ply(self, path, **kwargs):
        # 先调用父类加载基础参数
        super().load_ply(path, **kwargs)

        # 然后加载BRDF参数（如果PLY文件中存在）
        if self.brdf:
            plydata = PlyData.read(path)

            # 加载法线参数
            if 'nx' in plydata.elements[0] and 'ny' in plydata.elements[0] and 'nz' in plydata.elements[0]:
                normal = np.stack([
                    np.asarray(plydata.elements[0]["nx"]),
                    np.asarray(plydata.elements[0]["ny"]),
                    np.asarray(plydata.elements[0]["nz"])
                ], axis=1)
                self._normal = nn.Parameter(
                    torch.tensor(normal, device="cuda", dtype=torch.float32).requires_grad_(True)
                )

            # 加载其他BRDF参数...