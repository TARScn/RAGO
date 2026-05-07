import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from opencood.models.fuse_modules.fuse_utils import regroup as Regroup
from opencood.models.sub_modules.v2xvit_basic import V2XTransformer
from opencood.models.sub_modules.torch_transformation_utils import \
    warp_affine_simple


class DeformableAttentionRefiner(nn.Module):
    """
    A lightweight single-level deformable attention refinement block.

    This keeps the implementation fully in PyTorch so the second-stage
    training can run even when the custom CUDA extension is unavailable.
    """
    def __init__(self, feature_dim, n_heads=4, n_points=4,
                 ffn_dim=None, dropout=0.1):
        super().__init__()
        assert feature_dim % n_heads == 0, \
            "feature_dim should be divisible by n_heads."

        self.feature_dim = feature_dim
        self.n_heads = n_heads
        self.n_points = n_points
        self.head_dim = feature_dim // n_heads

        self.value_proj = nn.Linear(feature_dim, feature_dim)
        self.offset_proj = nn.Linear(feature_dim,
                                     n_heads * n_points * 2)
        self.attn_proj = nn.Linear(feature_dim,
                                   n_heads * n_points)
        self.output_proj = nn.Linear(feature_dim, feature_dim)

        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)
        self.dropout = nn.Dropout(dropout)

        hidden_dim = ffn_dim if ffn_dim is not None else feature_dim * 2
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, feature_dim)
        )

    def _build_reference_points(self, batch_size, height, width, device):
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, height - 0.5, height,
                           dtype=torch.float32, device=device),
            torch.linspace(0.5, width - 0.5, width,
                           dtype=torch.float32, device=device),
            indexing='ij'
        )
        ref = torch.stack((ref_x / width, ref_y / height), dim=-1)
        ref = ref.view(1, height * width, 1, 1, 2).repeat(batch_size, 1, 1, 1, 1)
        return ref

    def forward(self, x):
        """
        Args:
            x: [B, C, H, W]
        Returns:
            [B, C, H, W]
        """
        b, c, h, w = x.shape
        src = x.flatten(2).transpose(1, 2)  # [B, HW, C]

        value = self.value_proj(src).view(b, h * w, self.n_heads, self.head_dim)
        offsets = self.offset_proj(src).view(
            b, h * w, self.n_heads, self.n_points, 2
        )
        attn = self.attn_proj(src).view(
            b, h * w, self.n_heads, self.n_points
        )
        attn = F.softmax(attn, dim=-1)

        ref_points = self._build_reference_points(b, h, w, x.device)
        normalizer = x.new_tensor([w, h]).view(1, 1, 1, 1, 2)
        sampling_locations = ref_points + offsets / normalizer
        sampling_grids = sampling_locations * 2.0 - 1.0

        value_map = value.permute(0, 2, 3, 1).reshape(
            b * self.n_heads, self.head_dim, h, w
        )
        grid = sampling_grids.permute(0, 2, 1, 3, 4).reshape(
            b * self.n_heads, h * w, self.n_points, 2
        )
        sampled = F.grid_sample(
            value_map,
            grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False
        )
        sampled = sampled.view(
            b, self.n_heads, self.head_dim, h * w, self.n_points
        ).permute(0, 3, 1, 4, 2)

        fused = (sampled * attn.unsqueeze(-1)).sum(dim=3)
        fused = fused.reshape(b, h * w, c)
        fused = self.output_proj(fused)

        src = self.norm1(src + self.dropout(fused))
        src = self.norm2(src + self.dropout(self.ffn(src)))
        return src.transpose(1, 2).reshape(b, c, h, w)


class V2XViTDeformFusion(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.fusion_net = V2XTransformer(args['transformer'])

        deform_cfg = copy.deepcopy(args['deformable'])
        deform_dim = deform_cfg.pop('feature_dim', None)
        self.refiner = DeformableAttentionRefiner(
            feature_dim=deform_dim if deform_dim is not None
            else args['transformer']['encoder']['cav_att_config']['dim'],
            **deform_cfg
        )

    def forward(self, x, record_len, normalized_affine_matrix):
        _, _, h, w = x.shape
        b, l = normalized_affine_matrix.shape[:2]

        regroup_feature, mask = Regroup(x, record_len, l)
        prior_encoding = torch.zeros(
            len(record_len), l, 3, 1, 1, device=x.device, dtype=x.dtype
        )
        prior_encoding = prior_encoding.repeat(
            1, 1, 1, regroup_feature.shape[3], regroup_feature.shape[4]
        )

        regroup_feature = torch.cat([regroup_feature, prior_encoding], dim=2)
        regroup_feature = torch.stack([
            warp_affine_simple(regroup_feature[batch_idx],
                               normalized_affine_matrix[batch_idx, 0],
                               (h, w))
            for batch_idx in range(b)
        ])

        regroup_feature = regroup_feature.permute(0, 1, 3, 4, 2)
        spatial_correction_matrix = torch.eye(
            4, device=x.device, dtype=x.dtype
        ).unsqueeze(0).unsqueeze(0).repeat(len(record_len), l, 1, 1)

        fused_feature = self.fusion_net(
            regroup_feature, mask, spatial_correction_matrix
        )
        fused_feature = fused_feature.permute(0, 3, 1, 2)
        fused_feature = self.refiner(fused_feature)
        return fused_feature
