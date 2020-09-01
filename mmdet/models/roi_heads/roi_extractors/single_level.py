import torch
import torch.nn as nn

from mmdet import ops
from mmdet.core import force_fp32
from mmdet.core.utils.misc import topk
from mmdet.models.builder import ROI_EXTRACTORS


@ROI_EXTRACTORS.register_module()
class SingleRoIExtractor(nn.Module):
    """Extract RoI features from a single level feature map.

    If there are mulitple input feature levels, each RoI is mapped to a level
    according to its scale.

    Args:
        roi_layer (dict): Specify RoI layer type and arguments.
        out_channels (int): Output channels of RoI layers.
        featmap_strides (int): Strides of input feature maps.
        finest_scale (int): Scale threshold of mapping to level 0.
    """

    def __init__(self,
                 roi_layer,
                 out_channels,
                 featmap_strides,
                 finest_scale=56,
                 mode='openvino'):
        super(SingleRoIExtractor, self).__init__()
        self.roi_layers = self.build_roi_layers(roi_layer, featmap_strides)
        self.out_channels = out_channels
        self.featmap_strides = featmap_strides
        self.finest_scale = finest_scale
        self.fp16_enabled = False
        assert mode in {'pytorch', 'onnx', 'openvino'}
        self.mode = mode

    @property
    def num_inputs(self):
        """int: Input feature map levels."""
        return len(self.featmap_strides)

    def init_weights(self):
        pass

    def build_roi_layers(self, layer_cfg, featmap_strides):
        cfg = layer_cfg.copy()
        layer_type = cfg.pop('type')
        assert hasattr(ops, layer_type)
        layer_cls = getattr(ops, layer_type)
        roi_layers = nn.ModuleList(
            [layer_cls(spatial_scale=1 / s, **cfg) for s in featmap_strides])
        return roi_layers

    def map_roi_levels(self, rois, num_levels):
        """Map rois to corresponding feature levels by scales.

        - scale < finest_scale * 2: level 0
        - finest_scale * 2 <= scale < finest_scale * 4: level 1
        - finest_scale * 4 <= scale < finest_scale * 8: level 2
        - scale >= finest_scale * 8: level 3

        Args:
            rois (Tensor): Input RoIs, shape (k, 5).
            num_levels (int): Total level number.

        Returns:
            Tensor: Level index (0-based) of each RoI, shape (k, )
        """
        scale = torch.sqrt(
            (rois[:, 3] - rois[:, 1]) * (rois[:, 4] - rois[:, 2]))
        target_lvls = torch.floor(torch.log2(scale / self.finest_scale + 1e-6))
        target_lvls = target_lvls.clamp(min=0, max=num_levels - 1).long()
        return target_lvls

    def roi_rescale(self, rois, scale_factor):
        cx = (rois[:, 1] + rois[:, 3]) * 0.5
        cy = (rois[:, 2] + rois[:, 4]) * 0.5
        w = rois[:, 3] - rois[:, 1]
        h = rois[:, 4] - rois[:, 2]
        new_w = w * scale_factor
        new_h = h * scale_factor
        x1 = cx - new_w * 0.5
        x2 = cx + new_w * 0.5
        y1 = cy - new_h * 0.5
        y2 = cy + new_h * 0.5
        new_rois = torch.stack((rois[:, 0], x1, y1, x2, y2), dim=-1)
        return new_rois

    @force_fp32(apply_to=('feats', ), out_fp16=True)
    def forward(self, feats, rois, roi_scale_factor=None):
        if self.mode == 'pytorch':
            output = self.forward_pytorch(feats, rois, roi_scale_factor)
        elif self.mode == 'onnx':
            output = self.forward_onnx(feats, rois, roi_scale_factor)
        else:
            output = self.forward_openvino(feats, rois, roi_scale_factor)
        return output

    def forward_pytorch(self, feats, rois, roi_scale_factor=None):
        out_size = self.roi_layers[0].out_size
        num_levels = len(feats)
        roi_feats = feats[0].new_zeros(
            rois.size(0), self.out_channels, *out_size)
        # TODO: remove this when parrots supports
        if torch.__version__ == 'parrots':
            roi_feats.requires_grad = True

        if num_levels == 1:
            if len(rois) == 0:
                return roi_feats
            return self.roi_layers[0](feats[0], rois)

        target_lvls = self.map_roi_levels(rois, num_levels)
        if roi_scale_factor is not None:
            rois = self.roi_rescale(rois, roi_scale_factor)
        for i in range(num_levels):
            inds = target_lvls == i
            if inds.any():
                rois_ = rois[inds, :]
                roi_feats_t = self.roi_layers[i](feats[i], rois_)
                roi_feats[inds] = roi_feats_t
            else:
                roi_feats += sum(x.view(-1)[0] for x in self.parameters()) * 0.
        return roi_feats

    def forward_openvino(self, feats, rois, roi_scale_factor=None):
        if len(feats) == 1:
            return self.roi_layers[0](feats[0], rois)

        num_levels = len(feats)
        target_lvls = self.map_roi_levels(rois, num_levels)
        if roi_scale_factor is not None:
            rois = self.roi_rescale(rois, roi_scale_factor)

        roi_feats = None
        for level, (feat, extractor) in enumerate(zip(feats, self.roi_layers)):
            mask = target_lvls == level
            # Set all ROIs from other levels to nought.
            # This could enable optimizations at ROIs' features extractor level.
            level_rois = rois * mask.type_as(rois).unsqueeze(-1)
            level_feats = extractor(feat, level_rois)
            # Zero out features from out of level ROIs.
            level_feats = level_feats * mask.type_as(level_feats).view(-1, 1, 1, 1)
            if roi_feats is None:
                roi_feats = level_feats
            else:
                roi_feats += level_feats
        return roi_feats

    def forward_onnx(self, feats, rois, roi_scale_factor=None):
        from torch.onnx import operators

        if len(feats) == 1:
            return self.roi_layers[0](feats[0], rois)

        num_levels = len(feats)
        target_lvls = self.map_roi_levels(rois, num_levels)
        if roi_scale_factor is not None:
            rois = self.roi_rescale(rois, roi_scale_factor)

        indices = []
        roi_feats = []
        for level, (feat, extractor) in enumerate(zip(feats, self.roi_layers)):
            # Explicit casting to int is required for ONNXRuntime.
            level_indices = torch.nonzero(
                (target_lvls == level).int()).view(-1)
            level_rois = rois[level_indices]
            indices.append(level_indices)

            level_feats = extractor(feat, level_rois)
            roi_feats.append(level_feats)
        # Concatenate roi features from different pyramid levels
        # and rearrange them to match original ROIs order.
        indices = torch.cat(indices, dim=0)
        k = operators.shape_as_tensor(indices)
        _, indices = topk(indices, k, dim=0, largest=False)
        roi_feats = torch.cat(roi_feats, dim=0)[indices]

        return roi_feats
