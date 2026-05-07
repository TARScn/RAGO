import math

import torch
import torch.nn as nn


def _as_list(value, length=None):
    if isinstance(value, (list, tuple)):
        out = list(value)
    else:
        out = [value]
    if length is not None and len(out) == 1:
        out = out * length
    return out


def _unique_inverse(x):
    return torch.unique(x, sorted=False, return_inverse=True,
                        return_counts=True)


def get_window_coors(coords, sparse_shape, window_shape, shift=False):
    """Assign each sparse voxel to a shifted/non-shifted 3D window.

    Args:
        coords: [N, 4], [batch, z, y, x].
        sparse_shape: [x, y, z].
        window_shape: [x, y, z].
    """
    device = coords.device
    sparse_shape = torch.as_tensor(sparse_shape, device=device, dtype=torch.long)
    window_shape = torch.as_tensor(window_shape, device=device, dtype=torch.long)
    shift_xyz = window_shape // 2 if shift else torch.zeros_like(window_shape)

    xyz = coords[:, [3, 2, 1]].long()
    shifted = xyz + shift_xyz
    num_win_xyz = torch.div(sparse_shape + window_shape - 1,
                            window_shape,
                            rounding_mode='floor')
    win_xyz = torch.div(shifted, window_shape, rounding_mode='floor')
    win_xyz = torch.minimum(win_xyz, num_win_xyz - 1)
    inner_xyz = shifted - win_xyz * window_shape

    batch = coords[:, 0].long()
    win_inds = (((batch * num_win_xyz[2] + win_xyz[:, 2]) *
                 num_win_xyz[1] + win_xyz[:, 1]) *
                num_win_xyz[0] + win_xyz[:, 0])
    return win_inds, inner_xyz


def build_sets(coords, sparse_shape, window_shape, set_size, shift=False):
    """Build DSVT-style sparse voxel sets inside each local window.

    Two deterministic set partitions are returned:
    - set 0 sorts voxels by x-y-z within each window;
    - set 1 sorts voxels by y-x-z, approximating DSVT's rotated set.
    """
    device = coords.device
    n = coords.shape[0]
    if n == 0:
        empty = torch.zeros((0, set_size), dtype=torch.long, device=device)
        return empty, empty.bool()

    win_inds, inner_xyz = get_window_coors(coords, sparse_shape,
                                           window_shape, shift=shift)
    window_shape_t = torch.as_tensor(window_shape, device=device,
                                     dtype=torch.long)
    key0 = (win_inds * (window_shape_t[0] * window_shape_t[1] *
                        window_shape_t[2]) +
            inner_xyz[:, 2] * (window_shape_t[0] * window_shape_t[1]) +
            inner_xyz[:, 1] * window_shape_t[0] + inner_xyz[:, 0])
    key1 = (win_inds * (window_shape_t[0] * window_shape_t[1] *
                        window_shape_t[2]) +
            inner_xyz[:, 2] * (window_shape_t[0] * window_shape_t[1]) +
            inner_xyz[:, 0] * window_shape_t[1] + inner_xyz[:, 1])

    sets_all = []
    masks_all = []
    for key in (key0, key1):
        order = torch.argsort(key)
        sorted_win = win_inds[order]
        new_group = torch.ones(n, dtype=torch.bool, device=device)
        new_group[1:] = sorted_win[1:] != sorted_win[:-1]
        group_id = torch.cumsum(new_group.long(), dim=0) - 1
        num_groups = int(group_id[-1].item()) + 1
        counts = torch.bincount(group_id, minlength=num_groups)
        group_starts = torch.cumsum(counts, dim=0) - counts
        rank_in_group = torch.arange(n, device=device) - group_starts[group_id]

        set_id_per_sorted = torch.div(rank_in_group,
                                      set_size,
                                      rounding_mode='floor')
        num_sets_per_win = torch.div(counts + set_size - 1, set_size,
                                     rounding_mode='floor')
        set_offsets = torch.cumsum(num_sets_per_win, 0) - num_sets_per_win
        set_ids_sorted = set_offsets[group_id] + set_id_per_sorted
        total_sets = int(num_sets_per_win.sum().item())

        slot = rank_in_group % set_size
        sets = torch.full((total_sets, set_size), -1, dtype=torch.long,
                          device=device)
        sets[set_ids_sorted, slot] = order
        mask = sets < 0
        sets = sets.clamp_min(0)
        sets_all.append(sets.contiguous().long())
        masks_all.append(mask.contiguous().bool())

    return torch.cat(sets_all, dim=0).contiguous(), \
        torch.cat(masks_all, dim=0).contiguous()


class PositionEmbeddingLearned(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(3, channels),
            nn.LayerNorm(channels),
            nn.ReLU(inplace=True),
            nn.Linear(channels, channels),
        )

    def forward(self, coords, sparse_shape):
        xyz = coords[:, [3, 2, 1]].float()
        denom = coords.new_tensor(sparse_shape, dtype=torch.float32).clamp_min(1)
        xyz = xyz / denom
        return self.proj(xyz)


class SetAttention(nn.Module):
    def __init__(self, channels, num_heads, mlp_ratio=2.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, num_heads,
                                          dropout=dropout,
                                          batch_first=True)
        self.norm2 = nn.LayerNorm(channels)
        hidden = int(channels * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, channels),
            nn.Dropout(dropout),
        )

    def forward(self, feats, set_inds, set_masks, pos_embed):
        if set_inds.numel() == 0:
            return feats
        invalid = set_masks | (set_inds < 0) | (set_inds >= feats.shape[0])
        safe_set_inds = set_inds.masked_fill(invalid, 0).long().contiguous()
        safe_set_masks = invalid.bool().contiguous()

        set_feats = feats[safe_set_inds] + pos_embed[safe_set_inds]
        set_feats = self.norm1(set_feats)
        set_feats, _ = self.attn(set_feats, set_feats, set_feats,
                                 key_padding_mask=safe_set_masks,
                                 need_weights=False)
        set_feats = set_feats.masked_fill(safe_set_masks.unsqueeze(-1), 0)

        out = torch.zeros_like(feats)
        counts = feats.new_zeros((feats.shape[0], 1))
        valid_inds = safe_set_inds[~safe_set_masks]
        valid_feats = set_feats[~safe_set_masks]
        if valid_inds.numel() > 0:
            out.index_add_(0, valid_inds, valid_feats)
            counts.index_add_(0, valid_inds,
                              torch.ones_like(valid_feats[:, :1]))
        feats = feats + out / counts.clamp_min(1.0)
        feats = feats + self.mlp(self.norm2(feats))
        return feats


class DSVTBlock(nn.Module):
    """One DSVT block: two set attention layers over rotated set partitions."""

    def __init__(self, channels, num_heads, mlp_ratio=2.0, dropout=0.0):
        super().__init__()
        self.attn0 = SetAttention(channels, num_heads, mlp_ratio, dropout)
        self.attn1 = SetAttention(channels, num_heads, mlp_ratio, dropout)

    def forward(self, feats, sets, masks, pos_embed):
        split = sets.shape[0] // 2
        feats = self.attn0(feats, sets[:split], masks[:split], pos_embed)
        feats = self.attn1(feats, sets[split:], masks[split:], pos_embed)
        return feats


class StagePool(nn.Module):
    """Sparse stage reduction with feature projection."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.LayerNorm(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, feats, coords, stride):
        stride_t = torch.as_tensor(stride, device=coords.device,
                                   dtype=torch.long)
        xyz = coords[:, [3, 2, 1]].long()
        xyz_down = torch.div(xyz, stride_t, rounding_mode='floor')
        new_coords = torch.stack([
            coords[:, 0].long(),
            xyz_down[:, 2],
            xyz_down[:, 1],
            xyz_down[:, 0],
        ], dim=1)
        unique_coords, inverse = torch.unique(new_coords, sorted=False,
                                              return_inverse=True, dim=0)
        pooled = feats.new_zeros((unique_coords.shape[0], feats.shape[1]))
        pooled.index_add_(0, inverse, feats)
        counts = feats.new_zeros((unique_coords.shape[0], 1))
        counts.index_add_(0, inverse, torch.ones_like(feats[:, :1]))
        pooled = pooled / counts.clamp_min(1.0)
        return self.proj(pooled), unique_coords.int()


class DSVTInputLayer:
    """Pure PyTorch replacement for the official DSVT input pre-computation."""

    def __init__(self, sparse_shape, window_shapes, set_sizes,
                 downsample_strides):
        self.sparse_shape = [list(map(int, sparse_shape))]
        self.window_shapes = [list(map(int, s)) for s in window_shapes]
        self.set_sizes = list(map(int, set_sizes))
        self.downsample_strides = [list(map(int, s))
                                   for s in downsample_strides]
        for stride in self.downsample_strides:
            cur = self.sparse_shape[-1]
            self.sparse_shape.append([
                math.ceil(cur[0] / stride[0]),
                math.ceil(cur[1] / stride[1]),
                math.ceil(cur[2] / stride[2]),
            ])

    def __call__(self, coords):
        stage_sets = []
        stage_masks = []
        cur_coords = coords
        for stage_id, sparse_shape in enumerate(self.sparse_shape):
            set_size = self.set_sizes[stage_id]
            window_shape = self.window_shapes[stage_id]
            sets0, masks0 = build_sets(cur_coords, sparse_shape, window_shape,
                                       set_size, shift=False)
            sets1, masks1 = build_sets(cur_coords, sparse_shape, window_shape,
                                       set_size, shift=True)
            stage_sets.append((sets0, sets1))
            stage_masks.append((masks0, masks1))
            if stage_id < len(self.downsample_strides):
                stride = torch.as_tensor(self.downsample_strides[stage_id],
                                         device=coords.device,
                                         dtype=torch.long)
                xyz = cur_coords[:, [3, 2, 1]].long()
                xyz_down = torch.div(xyz, stride, rounding_mode='floor')
                cur_coords = torch.stack([
                    cur_coords[:, 0].long(),
                    xyz_down[:, 2],
                    xyz_down[:, 1],
                    xyz_down[:, 0],
                ], dim=1).int()
                cur_coords = torch.unique(cur_coords, sorted=False, dim=0)
        return stage_sets, stage_masks


class DSVTBackbone(nn.Module):
    """Dynamic Sparse Voxel Transformer backbone for OpenCOOD.

    This is a self-contained PyTorch implementation of the main DSVT ideas:
    dynamic local sparse windows, two rotated set partitions, shifted windows,
    stage-wise sparse pooling, and learned 3D positional embeddings. It avoids
    the official CUDA helper ops so it can run in the existing CoAlign setup.
    """

    def __init__(self, model_cfg, input_channels, grid_size):
        super().__init__()
        self.model_cfg = model_cfg
        self.grid_size = list(map(int, grid_size))
        self.downsample_strides = model_cfg.get(
            'downsample_strides', [[2, 2, 2], [2, 2, 2], [2, 2, 10]])
        self.window_shapes = model_cfg.get(
            'window_shapes', [[12, 12, 8], [12, 12, 8],
                              [12, 12, 5], [12, 12, 1]])
        self.set_sizes = model_cfg.get('set_sizes', [36, 36, 36, 36])
        self.d_model = model_cfg.get('d_model', [128, 128, 128, 128])
        self.depths = model_cfg.get('depths', [2, 2, 2, 2])
        self.num_heads = model_cfg.get('num_heads', [4, 4, 4, 4])
        self.mlp_ratio = float(model_cfg.get('mlp_ratio', 2.0))
        self.dropout = float(model_cfg.get('dropout', 0.0))

        self.input_proj = nn.Sequential(
            nn.Linear(input_channels, self.d_model[0]),
            nn.LayerNorm(self.d_model[0]),
            nn.ReLU(inplace=True),
        )
        self.input_layer = DSVTInputLayer(self.grid_size,
                                          self.window_shapes,
                                          self.set_sizes,
                                          self.downsample_strides)

        sparse_shapes = self.input_layer.sparse_shape
        self.pos_embeds = nn.ModuleList([
            PositionEmbeddingLearned(self.d_model[i])
            for i in range(len(self.d_model))
        ])
        self.stages = nn.ModuleList()
        self.pools = nn.ModuleList()
        for stage_id, channels in enumerate(self.d_model):
            blocks = nn.ModuleList([
                DSVTBlock(channels, self.num_heads[stage_id],
                          self.mlp_ratio, self.dropout)
                for _ in range(self.depths[stage_id])
            ])
            self.stages.append(blocks)
            if stage_id < len(self.d_model) - 1:
                self.pools.append(StagePool(channels,
                                            self.d_model[stage_id + 1]))

        final_shape = sparse_shapes[-1]
        self.bev_w = final_shape[0]
        self.bev_h = final_shape[1]
        self.bev_z = final_shape[2]
        self.num_bev_features = self.d_model[-1]

    def _stage_pool(self, feats, coords, stage_id):
        return self.pools[stage_id](feats, coords,
                                    self.downsample_strides[stage_id])

    def _scatter_to_bev(self, feats, coords, batch_size):
        coords = coords.long()
        bev_y = coords[:, 2]
        bev_x = coords[:, 3]
        valid = (
            (coords[:, 0] >= 0) & (coords[:, 0] < batch_size) &
            (bev_y >= 0) & (bev_y < self.bev_h) &
            (bev_x >= 0) & (bev_x < self.bev_w)
        )
        feats = feats[valid]
        coords = coords[valid]
        bev_y = bev_y[valid]
        bev_x = bev_x[valid]
        linear = coords[:, 0] * (self.bev_h * self.bev_w) + \
            bev_y * self.bev_w + bev_x
        unique, inverse, counts = _unique_inverse(linear)
        pooled = feats.new_zeros((unique.shape[0], feats.shape[1]))
        pooled.index_add_(0, inverse, feats)
        pooled = pooled / counts.to(feats.dtype).view(-1, 1).clamp_min(1.0)

        bev_flat = feats.new_zeros(
            (batch_size * self.bev_h * self.bev_w, feats.shape[1]))
        bev_flat[unique] = pooled
        return bev_flat.view(batch_size, self.bev_h, self.bev_w,
                             feats.shape[1]).permute(0, 3, 1, 2).contiguous()

    def forward(self, batch_dict):
        feats = self.input_proj(batch_dict['voxel_features'])
        coords = batch_dict['voxel_coords'].int()
        batch_size = int(batch_dict['batch_size'])

        for stage_id, blocks in enumerate(self.stages):
            sparse_shape = self.input_layer.sparse_shape[stage_id]
            pos_embed = self.pos_embeds[stage_id](coords, sparse_shape)
            for block_id, block in enumerate(blocks):
                shift = bool(block_id % 2)
                sets, masks = build_sets(
                    coords,
                    sparse_shape,
                    self.window_shapes[stage_id],
                    self.set_sizes[stage_id],
                    shift=shift,
                )
                feats = block(feats, sets, masks, pos_embed)
            if stage_id < len(self.stages) - 1:
                feats, coords = self._stage_pool(feats, coords, stage_id)

        batch_dict['spatial_features'] = self._scatter_to_bev(
            feats, coords, batch_size)
        stride = 1
        for s in self.downsample_strides:
            stride *= int(s[0])
        batch_dict['spatial_features_stride'] = stride
        return batch_dict
