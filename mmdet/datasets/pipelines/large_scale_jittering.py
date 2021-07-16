import cv2
from copy import deepcopy
import random
import numpy as np

from .transforms import Resize, RandomCrop
from ..builder import PIPELINES


@PIPELINES.register_module()
class LargeScaleJittering:
    def __init__(self, min=0.1, max=2):
        self.min = min
        self.max = max
        self.resize = None
        self.crop = None
        self.colors = []
        for i in range(20):
            c0 = random.randint(0, 255)
            c1 = random.randint(0, 255)
            c2 = random.randint(0, 255)
            self.colors.append(np.array([c0, c1, c2], dtype=np.uint8))

    def _resize(self, results, new_w, new_h):
        if self.resize is None:
            self.resize = Resize(img_scale=(new_w, new_h), keep_ratio=False, multiscale_mode='value')
        else:
            self.resize.img_scale = (new_w, new_h)

        scale_factor = results['scale_factor']
        keep_ratio = results['keep_ratio']
        scale_idx = results['scale_idx']
        pad_shape = results['pad_shape']
        scale = results['scale']

        results['scale'] = (new_w, new_h)
        results.pop('scale_factor')
        results.pop('scale_idx')
        results.pop('keep_ratio')
        results.pop('pad_shape')

        results = self.resize(results, target_size=(new_w, new_h))
        results['scale_factor'] = scale_factor
        results['keep_ratio'] = keep_ratio
        results['scale_idx'] = scale_idx
        results['pad_shape'] = pad_shape
        results['scale'] = scale

        return results

    def _fill(self, results, w, h, value=128):
        h0, w0 = results['img'].shape[:2]
        new_img = np.full((h, w, 3), value, dtype=np.uint8)
        new_img[: h0, : w0, :] = results['img']
        results['img'] = new_img
        _masks = None
        for i, mask in enumerate(results['gt_masks'].masks):
            _mask = np.zeros((h, w), dtype=np.uint8)
            _mask[: h0, : w0] = mask
            if _masks is None:
                _masks = np.expand_dims(_mask, axis=0)
            else:
                _masks = np.concatenate((_masks, np.expand_dims(_mask, axis=0)), axis=0)
        results['gt_masks'].masks = _masks
        results['img_shape'] = (w, h)
        results['gt_masks'].height = h
        results['gt_masks'].width = w
        return results

    def _crop(self, results, w, h):
        if self.crop is None:
            self.crop = RandomCrop((h, w))
        else:
            self.crop.crop_size = (h, w)
        return self.crop(results)

    def  __call__(self, results):
        scale = random.uniform(self.min, self.max)
        h, w = results['img'].shape[:2]
        new_w, new_h = int(scale * w), int(scale * h)
        self._resize(results, new_w, new_h)
        if scale < 1.0:
            results = self._fill(results, w, h)
        elif scale > 1.0:
            results = self._crop(results, w, h)
        return results

    def visualize(self, results, name=''):
        img = deepcopy(results['img'])
        n = max(results['gt_bboxes'].shape[0], results['gt_masks'].masks.shape[0])
        for i in range(n):
            bbox = results['gt_bboxes'][i] if i < results['gt_bboxes'].shape[0] else None
            mask = results['gt_masks'].masks[i] if i < results['gt_masks'].masks.shape[0] else None
            if bbox is not None:
                bbox = [int(x) for x in bbox]
                x0, y0, x1, y1 = bbox
                cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 255), 2)
            if mask is not None:
                m = np.stack([mask, mask, mask], axis=-1)
                c = self.colors[i % len(self.colors)]
                m *= c
                img = cv2.addWeighted(img, 1.0, m, 0.5, 0.0)
        cv2.imshow(name, img)
        cv2.waitKey(0)

