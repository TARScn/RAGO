# -*- coding: utf-8 -*-

import numpy as np
import torch.nn as nn

from opencood.models.sub_modules.mean_vfe import MeanVFE
from opencood.models.sub_modules.dsvt_backbone import DSVTBackbone
from opencood.models.sub_modules.cia_ssd_utils import SSFA
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.utils.model_utils import weight_init


class DsvtUncertainty(nn.Module):
    """DSVT-style stage-1 detector with OpenCOOD uncertainty heads."""

    def __init__(self, args):
        super(DsvtUncertainty, self).__init__()
        lidar_range = np.array(args['lidar_range'])
        voxel_size = np.array(args['voxel_size'])
        grid_size = np.round((lidar_range[3:6] - lidar_range[:3]) /
                             voxel_size).astype(np.int64)

        self.vfe = MeanVFE(args['mean_vfe'],
                           args['mean_vfe']['num_point_features'])
        self.dsvt_backbone = DSVTBackbone(
            args['dsvt'],
            input_channels=args['mean_vfe']['num_point_features'],
            grid_size=grid_size,
        )
        self.ssfa = SSFA(args['ssfa'])
        self.out_channel = args['ssfa']['feature_num']

        self.shrink_flag = False
        if 'shrink_header' in args:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(args['shrink_header'])
            self.out_channel = args['shrink_header']['dim'][-1]

        uncertainty_dim = args['uncertainty_dim']
        self.cls_head = nn.Conv2d(self.out_channel, args['anchor_num'],
                                  kernel_size=1)
        self.reg_head = nn.Conv2d(self.out_channel, 7 * args['anchor_num'],
                                  kernel_size=1)
        self.unc_head = nn.Conv2d(self.out_channel,
                                  uncertainty_dim * args['anchor_num'],
                                  kernel_size=1)

        self.use_dir = False
        if 'dir_args' in args.keys():
            self.use_dir = True
            self.dir_head = nn.Conv2d(
                self.out_channel,
                args['dir_args']['num_bins'] * args['anchor_num'],
                kernel_size=1)

        self.apply(weight_init)

    def forward(self, data_dict):
        voxel_features = data_dict['processed_lidar']['voxel_features']
        voxel_coords = data_dict['processed_lidar']['voxel_coords']
        voxel_num_points = data_dict['processed_lidar']['voxel_num_points']
        batch_size = int(voxel_coords[:, 0].max().item()) + 1

        batch_dict = {
            'voxel_features': voxel_features,
            'voxel_coords': voxel_coords,
            'voxel_num_points': voxel_num_points,
            'batch_size': batch_size,
        }

        batch_dict = self.vfe(batch_dict)
        batch_dict = self.dsvt_backbone(batch_dict)
        out = self.ssfa(batch_dict['spatial_features'])
        if self.shrink_flag:
            out = self.shrink_conv(out)

        output_dict = {
            'cls_preds': self.cls_head(out),
            'reg_preds': self.reg_head(out),
            'unc_preds': self.unc_head(out),
        }

        if self.use_dir:
            output_dict.update({'dir_preds': self.dir_head(out)})

        return output_dict
