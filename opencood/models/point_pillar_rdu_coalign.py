# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

from opencood.models.fuse_modules.fusion_in_one import AttFusion, MaxFusion
from opencood.models.fuse_modules.v2xvit_deform_fuse import V2XViTDeformFusion
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.sub_modules.base_bev_backbone_resnet import \
    ResNetBEVBackbone
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.naive_compress import NaiveCompressor
from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import \
    PointPillarScatter
from opencood.models.sub_modules.torch_transformation_utils import \
    warp_affine_simple
from opencood.utils.transformation_utils import normalize_pairwise_tfm


def regroup(x, record_len):
    cum_sum_len = torch.cumsum(record_len, dim=0)
    return torch.tensor_split(x, cum_sum_len[:-1].cpu())


class PointPillarRduCoalign(nn.Module):
    """
    Hybrid CoAlign + V2XViT-Deform with RSU-anchor distance-guided refinement.

    We keep the original multiscale AttFusion backbone and the conservative
    hybrid semantic enhancement branch, then additionally bias the selected
    semantic level toward the RSU feature in far-range regions.
    """
    def __init__(self, args):
        super().__init__()

        self.pillar_vfe = PillarVFE(args['pillar_vfe'],
                                    num_point_features=4,
                                    voxel_size=args['voxel_size'],
                                    point_cloud_range=args['lidar_range'])
        self.scatter = PointPillarScatter(args['point_pillar_scatter'])

        is_resnet = args['base_bev_backbone'].get("resnet", True)
        if is_resnet:
            self.backbone = ResNetBEVBackbone(args['base_bev_backbone'], 64)
        else:
            self.backbone = BaseBEVBackbone(args['base_bev_backbone'], 64)

        self.voxel_size = args['voxel_size']
        self.level_num = len(args['base_bev_backbone']['layer_nums'])

        self.fusion_net = nn.ModuleList()
        for i in range(self.level_num):
            if args['fusion_method'] == "max":
                self.fusion_net.append(MaxFusion())
            else:
                self.fusion_net.append(AttFusion(args['att']['feat_dim'][i]))

        hybrid_cfg = args['hybrid_v2xvit_deform']
        self.hybrid_level = hybrid_cfg.get('level', self.level_num - 1)
        self.hybrid_fusion = V2XViTDeformFusion(hybrid_cfg['fusion'])
        gate_init = hybrid_cfg.get('gate_init', -4.0)
        self.hybrid_gate = nn.Parameter(torch.tensor(float(gate_init)))

        rsu_cfg = args['rsu_distance_guidance']
        self.rsu_index = rsu_cfg.get('rsu_index', -1)
        self.rsu_level = rsu_cfg.get('level', self.hybrid_level)
        self.range_start = float(rsu_cfg.get('range_start', 40.0))
        self.range_end = float(rsu_cfg.get('range_end', 90.0))
        self.center_x = float(rsu_cfg.get('center_x', 0.0))
        self.center_y = float(rsu_cfg.get('center_y', 0.0))
        self.min_alpha = float(rsu_cfg.get('min_alpha', 0.0))
        self.max_alpha = float(rsu_cfg.get('max_alpha', 0.7))
        self.learnable_bias = nn.Parameter(
            torch.tensor(float(rsu_cfg.get('bias_init', -2.0)))
        )
        self.alpha_scale = nn.Parameter(
            torch.tensor(float(rsu_cfg.get('alpha_scale_init', 1.5)))
        )

        self.out_channel = sum(args['base_bev_backbone']['num_upsample_filter'])

        self.shrink_flag = False
        if 'shrink_header' in args:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(args['shrink_header'])
            self.out_channel = args['shrink_header']['dim'][-1]

        self.compression = False
        if "compression" in args:
            self.compression = True
            self.naive_compressor = NaiveCompressor(64, args['compression'])

        self.cls_head = nn.Conv2d(self.out_channel, args['anchor_number'],
                                  kernel_size=1)
        self.reg_head = nn.Conv2d(self.out_channel, 7 * args['anchor_number'],
                                  kernel_size=1)
        self.use_dir = False
        if 'dir_args' in args:
            self.use_dir = True
            self.dir_head = nn.Conv2d(
                self.out_channel,
                args['dir_args']['num_bins'] * args['anchor_number'],
                kernel_size=1
            )

        if args.get('backbone_fix', False):
            self.backbone_fix()

    def backbone_fix(self):
        for p in self.pillar_vfe.parameters():
            p.requires_grad = False
        for p in self.scatter.parameters():
            p.requires_grad = False
        for p in self.backbone.parameters():
            p.requires_grad = False
        if self.compression:
            for p in self.naive_compressor.parameters():
                p.requires_grad = False
        if self.shrink_flag:
            for p in self.shrink_conv.parameters():
                p.requires_grad = False
        for p in self.cls_head.parameters():
            p.requires_grad = False
        for p in self.reg_head.parameters():
            p.requires_grad = False

    def _build_distance_prior(self, feature, level_stride):
        _, _, h, w = feature.shape
        device = feature.device
        dtype = feature.dtype

        x_min = self.center_x - (w / 2.0) * self.voxel_size[0] * level_stride
        y_min = self.center_y - (h / 2.0) * self.voxel_size[1] * level_stride

        x_coords = torch.arange(w, device=device, dtype=dtype)
        y_coords = torch.arange(h, device=device, dtype=dtype)
        x_coords = x_min + (x_coords + 0.5) * self.voxel_size[0] * level_stride
        y_coords = y_min + (y_coords + 0.5) * self.voxel_size[1] * level_stride
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
        distance = torch.sqrt(grid_x ** 2 + grid_y ** 2)

        prior = (distance - self.range_start) / max(
            self.range_end - self.range_start, 1e-6
        )
        prior = prior.clamp(0.0, 1.0)
        prior = self.min_alpha + (self.max_alpha - self.min_alpha) * prior

        scale = torch.sigmoid(self.alpha_scale)
        bias = torch.sigmoid(self.learnable_bias)
        prior = (scale * prior + bias).clamp(0.0, 1.0)
        return prior.unsqueeze(0).unsqueeze(0)

    def _blend_rsu_feature(self, feature_list, record_len, affine_matrix):
        level_feature = feature_list[self.rsu_level]
        split_feature = regroup(level_feature, record_len)
        batch_size = len(split_feature)

        blended_list = []
        level_stride = 2 ** (self.rsu_level + 1)

        for b in range(batch_size):
            node_features = split_feature[b]
            num_agents = node_features.shape[0]
            _, _, h, w = node_features.shape
            rsu_index = self.rsu_index if self.rsu_index >= 0 else num_agents - 1
            if rsu_index < 0 or rsu_index >= num_agents:
                blended_list.append(level_feature[b:b + 1])
                continue

            rsu_feature = node_features[rsu_index:rsu_index + 1]
            rsu_to_ego = affine_matrix[b:b + 1, 0, rsu_index]
            rsu_in_ego = warp_affine_simple(rsu_feature, rsu_to_ego, (h, w))

            fused_feature = level_feature[b:b + 1]
            prior = self._build_distance_prior(fused_feature, level_stride)
            blended = fused_feature + prior * (rsu_in_ego - fused_feature)
            blended_list.append(blended)

        feature_list[self.rsu_level] = torch.cat(blended_list, dim=0)
        return feature_list

    def forward(self, data_dict):
        voxel_features = data_dict['processed_lidar']['voxel_features']
        voxel_coords = data_dict['processed_lidar']['voxel_coords']
        voxel_num_points = data_dict['processed_lidar']['voxel_num_points']
        record_len = data_dict['record_len']

        batch_dict = {
            'voxel_features': voxel_features,
            'voxel_coords': voxel_coords,
            'voxel_num_points': voxel_num_points,
            'record_len': record_len
        }
        batch_dict = self.pillar_vfe(batch_dict)
        batch_dict = self.scatter(batch_dict)

        _, _, h0, w0 = batch_dict['spatial_features'].shape
        normalized_affine_matrix = normalize_pairwise_tfm(
            data_dict['pairwise_t_matrix'], h0, w0, self.voxel_size[0]
        )

        spatial_features = batch_dict['spatial_features']
        if self.compression:
            spatial_features = self.naive_compressor(spatial_features)

        feature_list = self.backbone.get_multiscale_feature(spatial_features)
        fused_feature_list = []

        for i, fuse_module in enumerate(self.fusion_net):
            att_fused = fuse_module(
                feature_list[i], record_len, normalized_affine_matrix
            )
            if i == self.hybrid_level:
                enhanced = self.hybrid_fusion(
                    feature_list[i], record_len, normalized_affine_matrix
                )
                gate = torch.sigmoid(self.hybrid_gate)
                att_fused = att_fused + gate * (enhanced - att_fused)
            fused_feature_list.append(att_fused)

        fused_feature_list = self._blend_rsu_feature(
            fused_feature_list, record_len, normalized_affine_matrix
        )

        fused_feature = self.backbone.decode_multiscale_feature(
            fused_feature_list
        )

        if self.shrink_flag:
            fused_feature = self.shrink_conv(fused_feature)

        output_dict = {
            'cls_preds': self.cls_head(fused_feature),
            'reg_preds': self.reg_head(fused_feature)
        }
        if self.use_dir:
            output_dict.update({'dir_preds': self.dir_head(fused_feature)})

        return output_dict
