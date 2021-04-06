from mmcv import ops
from .point_sample import (SimpleRoIAlign, generate_grid,
                          point_sample, rel_roi_point_to_rel_img_point)


ops.SimpleRoIAlign = SimpleRoIAlign

__all__ = [
    'SimpleRoIAlign', 'generate_grid', 'point_sample',
    'rel_roi_point_to_rel_img_point',
]