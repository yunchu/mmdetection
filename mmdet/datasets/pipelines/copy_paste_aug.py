import cv2
import random
import numpy as np
from copy import deepcopy
from skimage.filters import gaussian

from .transforms import Resize
from ..builder import PIPELINES

colors = [20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 240, 250]


@PIPELINES.register_module()
class CopyPaste:
    def __init__(
        self,
        blend=False,
        sigma=3,
        p=0.5,
        always_apply=False
    ):
        self.blend = blend
        self.sigma = sigma
        self.p = p
        self.always_apply = always_apply
        self.resize = None

    def image_copy_paste(self, results, alpha):
        if alpha is not None:
            if self.blend:
                alpha = gaussian(alpha, sigma=self.sigma, preserve_range=True)
            img_dtype = results['img'].dtype
            alpha = alpha[..., None]
            results['img'] = results['copy_paste']['img'] * alpha + results['img'] * (1 - alpha)
            results['img'] = results['img'].astype(img_dtype)

    def masks_copy_paste(self, results, alpha):
        if alpha is not None:
            results['gt_masks'].masks = np.array([
                np.where(alpha == 1, 0, mask).astype(np.uint8) for mask in results['gt_masks'].masks
            ])
            self.extract_bboxes(results)
            results['gt_masks'].masks = np.concatenate(
                (results['gt_masks'].masks, results['copy_paste']['gt_masks'].masks), axis=0)
            results['gt_bboxes'] = np.concatenate(
                (results['gt_bboxes'], results['copy_paste']['gt_bboxes']), axis=0)

    @staticmethod
    def concatenate_labels(results):
        results['gt_labels'] = np.concatenate((results['gt_labels'], results['copy_paste']['gt_labels']), axis=0)
        results['gt_bboxes_ignore'] = np.concatenate((results['gt_bboxes_ignore'], results['copy_paste']['gt_bboxes_ignore']), axis=0)

    @staticmethod
    def filter_empty(results):
        inds = np.argwhere(results['gt_masks'].areas > 0).squeeze()
        results['gt_masks'].masks = results['gt_masks'].masks[inds]
        results['gt_bboxes'] = results['gt_bboxes'][inds]
        results['gt_labels'] = results['gt_labels'][inds]

    @staticmethod
    def cast(results, bbox_type, mask_type, label_type):
        results['gt_masks'].masks = results['gt_masks'].masks.astype(mask_type)
        results['gt_bboxes'] = results['gt_bboxes'].astype(bbox_type)
        results['gt_labels'] = results['gt_labels'].astype(label_type)

    @staticmethod
    def extract_bboxes(results):
        masks = results['gt_masks'].masks
        bboxes = []
        for mask in masks:
            xindices = np.where(np.any(mask, axis=0))[0]
            yindices = np.where(np.any(mask, axis=1))[0]
            if yindices.shape[0]:
                y1, y2 = yindices[[0, -1]]
                x1, x2 = xindices[[0, -1]]
            else:
                x1, y1, x2, y2 = 0, 0, 0, 0
            bboxes.append((x1, y1, x2, y2))
        results['gt_bboxes'] = np.array(bboxes)
        return bboxes

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
                img = np.where(m == 1, img * 0.5 + 0.5 * colors[i % len(colors)], img).astype(np.uint8)
        cv2.imshow(name, img)
        cv2.waitKey(0)

    def rescale_paste_target(self, results, target_size):
        h, w = target_size
        image_size = results['img'].shape[:-1]
        if image_size != target_size:
            if self.resize is None:
                self.resize = Resize(img_scale=(w, h), keep_ratio=False, multiscale_mode='value')
            else:
                self.resize.img_scale = (w, h)
            results['keep_ratio'] = False
            results['scale'] = (w, h)
            results.pop('scale_factor')
            results = self.resize(results)
        return results

    def __call__(self, results):
        p = random.uniform(0, 1)
        if p > self.p:
            return results
        bbox_type = results['gt_bboxes'].dtype
        mask_type = results['gt_masks'].masks.dtype
        label_type = results['gt_labels'].dtype
        h, w = results['img'].shape[:-1]
        self.rescale_paste_target(results['copy_paste'], (h, w))
        compose_paste_mask = np.zeros((h, w), dtype=np.uint8)
        for mask in results['copy_paste']['gt_masks'].masks:
            compose_paste_mask = np.logical_or(compose_paste_mask, mask)
        self.image_copy_paste(results, alpha=compose_paste_mask)
        self.masks_copy_paste(results, alpha=compose_paste_mask)
        self.concatenate_labels(results)
        self.filter_empty(results)
        self.cast(results, bbox_type, mask_type, label_type)
        #self.visualize(results)
        return results
