# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn

from opencood.models.fuse_modules.fusion_in_one import AttFusion, MaxFusion
from opencood.models.fuse_modules.v2xvit_deform_fuse import V2XViTDeformFusion
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.dsvt_backbone import DSVTBackbone
from opencood.models.sub_modules.mean_vfe import MeanVFE
from opencood.models.sub_modules.torch_transformation_utils import \
    warp_affine_simple
from opencood.utils.model_utils import weight_init
from opencood.utils.transformation_utils import normalize_pairwise_tfm


def regroup(x, record_len):
    cum_sum_len = torch.cumsum(record_len, dim=0)
    return torch.tensor_split(x, cum_sum_len[:-1].cpu())


class DsvtCoalignHybridV2xvitDeformRsuDistance(nn.Module):
    """
    DSVT-based CoAlign model with hybrid V2XViT-Deform enhancement and
    RSU-anchor distance-guided semantic refinement.

    Compared with the PointPillar version, the voxel encoder / sparse backbone
    is replaced by DSVT, while the second-stage cooperative fusion logic stays
    aligned for a fair architecture comparison.
    """
    def __init__(self, args):
        super().__init__()

        lidar_range = np.array(args['lidar_range'])
        voxel_size = np.array(args['voxel_size'])
        grid_size = np.round((lidar_range[3:6] - lidar_range[:3]) /
                             voxel_size).astype(np.int64)

        self.voxel_size = args['voxel_size']
        self.downsample_rate = int(args.get('downsample_rate', 8))

        self.vfe = MeanVFE(args['mean_vfe'],
                           args['mean_vfe']['num_point_features'])
        self.dsvt_backbone = DSVTBackbone(
            args['dsvt'],
            input_channels=args['mean_vfe']['num_point_features'],
            grid_size=grid_size,
        )

        self.bev_backbone = BaseBEVBackbone(
            args['base_bev_backbone'],
            input_channels=args['bev_backbone_input']
        )
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

        self.cls_head = nn.Conv2d(self.out_channel,
                                  args['anchor_number'],
                                  kernel_size=1)
        self.reg_head = nn.Conv2d(self.out_channel,
                                  7 * args['anchor_number'],
                                  kernel_size=1)

        self.use_dir = False
        if 'dir_args' in args:
            self.use_dir = True
            self.dir_head = nn.Conv2d(
                self.out_channel,
                args['dir_args']['num_bins'] * args['anchor_number'],
                kernel_size=1
            )

        self.apply(weight_init)

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
        level_stride = self.downsample_rate * (2 ** self.rsu_level)

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
        # DSVT backbone should build one BEV feature map per valid agent,
        # matching the sum(record_len) convention used by cooperative fusion.
        batch_size = int(voxel_coords[:, 0].max().item()) + 1

        batch_dict = {
            'voxel_features': voxel_features,
            'voxel_coords': voxel_coords,
            'voxel_num_points': voxel_num_points,
            'batch_size': batch_size,
        }

        batch_dict = self.vfe(batch_dict)
        batch_dict = self.dsvt_backbone(batch_dict)

        spatial_features = batch_dict['spatial_features']
        _, _, h0, w0 = spatial_features.shape
        normalized_affine_matrix = normalize_pairwise_tfm(
            data_dict['pairwise_t_matrix'],
            h0,
            w0,
            self.voxel_size[0],
            self.downsample_rate
        )

        feature_list = self.bev_backbone.get_multiscale_feature(spatial_features)
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

        fused_feature = self.bev_backbone.decode_multiscale_feature(
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
