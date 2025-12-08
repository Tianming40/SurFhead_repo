# #
# # Toyota Motor Europe NV/SA and its affiliated companies retain all intellectual
# # property and proprietary rights in and to this software and related documentation.
# # Any commercial use, reproduction, disclosure or distribution of this software and
# # related documentation without an express license agreement from Toyota Motor Europe NV/SA
# # is strictly prohibited.
# #
#
#
# from typing import Optional
# import torch
# import numpy as np
# from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation, get_minimum_axis, flip_align_view
# from torch import nn
# import os
# from utils.system_utils import mkdir_p
# from plyfile import PlyData, PlyElement
# # from pytorch3d.transforms import quaternion_multiply
# from roma import quat_product, quat_xyzw_to_wxyz, quat_wxyz_to_xyzw, rotvec_to_rotmat, rotmat_to_rotvec
# from utils.sh_utils import RGB2SH
# from .gaussian_model import GaussianModel
# from .flame_gaussian_model import FlameGaussianModel
# # from utils.graphics_utils import BasicPointCloud
# from utils.graphics_utils import BasicPointCloud, get_dist_weight
# from utils.general_utils import strip_symmetric, build_scaling_rotation
# import torch.nn.functional as F
# from torch.autograd import Function
# from roma import rotmat_to_unitquat, quat_xyzw_to_wxyz, unitquat_to_rotmat, special_procrustes
# from submodules.nvdiffrec.render.light import create_trainable_env_rnd
#
# # from roma import quat_product, quat_xyzw_to_wxyz, quat_wxyz_to_xyzw, rotvec_to_rotmat, rotmat_to_rotvec
# class BRDFFlameGaussianModel(FlameGaussianModel):
#     def __init__(self, sh_degree: int, sg_degree: int,
#                  brdf_dim: int = -1, brdf_mode: str = "none", brdf_envmap_res: int = 64,
#                  **flame_kwargs):
#
#
#         super().__init__(sh_degree, sg_degree, **flame_kwargs)
#
#         if (brdf_dim >= 0 and sh_degree >= 0) or (brdf_dim < 0 and sh_degree < 0):
#             raise Exception('Please provide exactly one of either brdf_dim or sh_degree!')
#         self.brdf = brdf_dim >= 0
#
#
#         self.brdf_dim = brdf_dim
#         self.brdf_mode = brdf_mode
#         self.brdf_envmap_res = brdf_envmap_res
#
#
#
#         self._specular = torch.empty(0)
#         self._roughness = torch.empty(0)
#
#         # BRDF MLP
#         if self.brdf:
#             self.brdf_mlp = create_trainable_env_rnd(self.brdf_envmap_res, scale=0.0, bias=0.8)
#         else:
#             self.brdf_mlp = None
#
#         self.diffuse_activation = torch.sigmoid
#         self.specular_activation = torch.sigmoid
#         # self.roughness_activation = torch.sigmoid
#         # self.roughness_bias = 0.0
#         # self.default_roughness = 0.6
#
#         self.training_stage = 1
#
#     @property
#     def get_diffuse(self):
#         if self.brdf:
#             return self.diffuse_activation(self._features_dc)
#         else:
#             return self._features_dc
#
#     @property
#     def get_specular(self):
#         return self.specular_activation(self._specular)
#
#     @property
#     def get_roughness(self):
#         return self.roughness_activation(self._roughness + self.roughness_bias)
#
#     @property
#     def get_brdf_features(self):
#         return self._features_rest
#
#     # def create_from_pcd(self, pcd: Optional[BasicPointCloud], spatial_lr_scale: float):
#     #     # TODO
#     #     pass
#
#     # def training_setup(self, training_args):
#     #     self.percent_dense = training_args.percent_dense
#     #     self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
#     #     self.xyz_gradient_accum_abs = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
#     #     self.xyz_gradient_accum_abs_max = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
#     #     # self.tightpruning__mask = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
#     #     self.tight_visibility_mask = torch.zeros((self.get_xyz.shape[0]), device="cuda")
#     #
#     #     self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
#     #
#     #     l = [
#     #         {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
#     #         {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
#     #         {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
#     #         {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
#     #         {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
#     #         {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
#     #         {'params': [self._features_sg], 'lr': training_args.feature_lr, "name": "f_sg"},
#     #     ]
#     #
#     #     if self.train_kinematic:
#     #         l.append({'params': [self.blend_weight], 'lr': training_args.blend_weight_lr, "name": "blend_weight"})
#     #     # breakpoint()
#     #
#     #     if self.brdf:
#     #         l.extend([
#     #             {'params': list(self.brdf_mlp.parameters()), 'lr': training_args.brdf_mlp_lr_init, "name": "brdf_mlp"},
#     #             {'params': [self._roughness], 'lr': training_args.roughness_lr, "name": "roughness"},
#     #             {'params': [self._specular], 'lr': training_args.specular_lr, "name": "specular"},
#     #         ])
#     #
#     #
#     #
#     #     self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
#     #
#     #     self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init * self.spatial_lr_scale,
#     #                                                 lr_final=training_args.position_lr_final * self.spatial_lr_scale,
#     #                                                 lr_delay_mult=training_args.position_lr_delay_mult,
#     #                                                 max_steps=training_args.position_lr_max_steps)
#     #
#     #     self.blend_weight_scheduler_args = get_expon_lr_func(
#     #         lr_init=training_args.blend_weight_lr * self.spatial_lr_scale,
#     #         lr_final=(training_args.blend_weight_lr / 100.0) * self.spatial_lr_scale,
#     #         lr_delay_mult=training_args.position_lr_delay_mult,
#     #         max_steps=training_args.position_lr_max_steps)
#     #
#     #
#     #     self.brdf_mlp_scheduler_args = get_expon_lr_func(lr_init=training_args.brdf_mlp_lr_init,
#     #                                                      lr_final=training_args.brdf_mlp_lr_final,
#     #                                                      lr_delay_mult=training_args.brdf_mlp_lr_delay_mult,
#     #                                                      max_steps=training_args.brdf_mlp_lr_max_steps)
#
#
#     def _update_learning_rate(self, iteration, param):
#         for param_group in self.optimizer.param_groups:
#             if param_group["name"] == param:
#                 try:
#                     lr = getattr(self, f"{param}_scheduler_args", self.brdf_mlp_scheduler_args)(iteration)
#                     param_group['lr'] = lr
#                     return lr
#                 except AttributeError:
#                     pass
#
#     # def update_learning_rate(self, iteration):
#     #     ''' Learning rate scheduling per step '''
#     #     self._update_learning_rate(iteration, "xyz")
#     #     if self.brdf and not self.fix_brdf_lr:
#     #         for param in ["brdf_mlp","roughness","specular","f_dc", "f_rest"]:
#     #             self._update_learning_rate(iteration, param)
#
#     def set_training_stage(self, stage):
#         self.training_stage = stage
#
#         if stage == 1:
#             self.set_requires_grad("specular", False)
#             self.set_requires_grad("roughness", False)
#             self.brdf_mlp.requires_grad_(False)
#
#
#             self.set_requires_grad("xyz", True)
#             self.set_requires_grad("scaling", True)
#             self.set_requires_grad("rotation", True)
#             self.set_requires_grad("opacity", True)
#             self.set_requires_grad("features_dc", True)
#             self.set_requires_grad("features_rest", True)
#             self.set_requires_grad("features_sg", True)
#
#         elif stage == 2:
#
#             self.set_requires_grad("specular", True)
#             self.set_requires_grad("roughness", True)
#             self.set_requires_grad("features_dc", True)
#             self.set_requires_grad("features_rest", True)
#             self.brdf_mlp.requires_grad_(False)
#
#             self.set_requires_grad("xyz", False)
#             self.set_requires_grad("scaling", False)
#             self.set_requires_grad("rotation", False)
#             self.set_requires_grad("opacity", False)
#             self.set_requires_grad("features_sg", False)
#
#         elif stage == 3:
#
#             self.set_requires_grad("specular", True)
#             self.set_requires_grad("roughness", True)
#             self.brdf_mlp.requires_grad_(True)
#
#             self.set_requires_grad("xyz", True)
#             self.set_requires_grad("scaling", True)
#             self.set_requires_grad("rotation", True)
#             self.set_requires_grad("opacity", True)
#             self.set_requires_grad("features_dc", True)
#             self.set_requires_grad("features_rest", True)
#             self.set_requires_grad("features_sg", True)
#
#     def set_requires_grad(self, attrib_name, state: bool):
#         getattr(self, f"_{attrib_name}").requires_grad = state