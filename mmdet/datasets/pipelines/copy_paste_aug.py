import cv2
import random
import numpy as np
from copy import deepcopy
from skimage.filters import gaussian

from .auto_augment import Rotate
from .transforms import Resize, RandomFlip
from ..builder import PIPELINES


@PIPELINES.register_module()
class CopyPaste:
    def __init__(
        self,
        blend=False,
        sigma=3,
        horizontal_flip_ratio=0.5,
        rotate=30
    ):
        self.blend = blend
        self.sigma = sigma
        self.resize = None
        self.flip = RandomFlip(flip_ratio=horizontal_flip_ratio)
        if rotate > 0:
            self.rotate = Rotate(level=2, prob=0.5, max_rotate_angle=rotate)
        else:
            self.rotate = None
        self.colors = []
        for i in range(20):
            c0 = random.randint(0, 255)
            c1 = random.randint(0, 255)
            c2 = random.randint(0, 255)
            self.colors.append(np.array([c0, c1, c2], dtype=np.uint8))

    def image_copy_paste(self, results, alpha):
        if alpha is not None:
            if self.blend:
                alpha = gaussian(alpha, sigma=self.sigma, preserve_range=True)
            img_dtype = results['img'].dtype
            alpha = alpha[..., None]
            results['img'] = results['copy_paste']['img'] * alpha + results['img'] * (1 - alpha)
            results['img'] = results['img'].astype(img_dtype)

    def masks_copy_paste(self, results, paste_objects, alpha):
        if alpha is not None:
            results['gt_masks'].masks = np.array([
                np.where(alpha == 1, 0, mask).astype(np.uint8) for mask in results['gt_masks'].masks
            ])
            self.extract_bboxes(results)
            if paste_objects is not None:
                results['gt_masks'].masks = np.concatenate(
                    (results['gt_masks'].masks, results['copy_paste']['gt_masks'].masks[paste_objects]), axis=0)
                results['gt_bboxes'] = np.concatenate(
                    (results['gt_bboxes'], results['copy_paste']['gt_bboxes'][paste_objects]), axis=0)
            else:
                results['gt_masks'].masks = np.concatenate(
                    (results['gt_masks'].masks, results['copy_paste']['gt_masks'].masks), axis=0)
                results['gt_bboxes'] = np.concatenate(
                    (results['gt_bboxes'], results['copy_paste']['gt_bboxes']), axis=0)

    @staticmethod
    def concatenate_labels(results, paste_objects=None):
        if paste_objects is not None:
            results['gt_labels'] = np.concatenate((results['gt_labels'], results['copy_paste']['gt_labels'][paste_objects]), axis=0)
        else:
            results['gt_labels'] = np.concatenate(
                (results['gt_labels'], results['copy_paste']['gt_labels']), axis=0)
        results['gt_bboxes_ignore'] = np.concatenate((results['gt_bboxes_ignore'], results['copy_paste']['gt_bboxes_ignore']), axis=0)

    @staticmethod
    def _filter(results, inds):
        results['gt_masks'].masks = results['gt_masks'].masks[inds]
        results['gt_bboxes'] = results['gt_bboxes'][inds]
        results['gt_labels'] = results['gt_labels'][inds]
        if len(results['gt_masks'].masks.shape) == 2:
            results['gt_masks'].masks = np.expand_dims(results['gt_masks'].masks, axis=0)
        if len(results['gt_bboxes'].shape) == 1:
            results['gt_bboxes'] = np.expand_dims(results['gt_bboxes'], axis=0)
        if len(results['gt_labels'].shape) == 0:
            results['gt_labels'] = np.expand_dims(results['gt_labels'], axis=0)

    def filter_empty(self, results, min_size=5):
        w = results['gt_bboxes'][:, 2] - results['gt_bboxes'][:, 0]
        h = results['gt_bboxes'][:, 3] - results['gt_bboxes'][:, 1]
        inds = np.argwhere((w > min_size) & (h > min_size)).squeeze()
        self._filter(results, inds)
        inds = np.argwhere(results['gt_masks'].areas > 0).squeeze()
        self._filter(results, inds)

    @staticmethod
    def cast(results, bbox_type, mask_type, label_type):
        results['gt_masks'].masks = results['gt_masks'].masks.astype(mask_type)
        results['gt_bboxes'] = results['gt_bboxes'].astype(bbox_type)
        results['gt_labels'] = results['gt_labels'].astype(label_type)

    @staticmethod
    def extract_bboxes(results):
        bboxes = []
        for mask in results['gt_masks'].masks:
            yindices = np.where(np.any(mask, axis=0))[0]
            xindices = np.where(np.any(mask, axis=1))[0]
            if yindices.shape[0]:
                x1, x2 = yindices[[0, -1]]
                y1, y2 = xindices[[0, -1]]
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
                color = self.colors[i % len(self.colors)]
                colored_mask = np.stack([mask, mask, mask], axis=-1) * color
                img = cv2.addWeighted(img, 1.0, colored_mask, 0.5, 0.0)
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

    def get_composed_mask(self, results, h, w, inds=None):
        compose_mask = np.zeros((h, w), dtype=np.uint8)
        for i, mask in enumerate(results['copy_paste']['gt_masks'].masks):
            if inds is None or i in inds:
                compose_mask = np.logical_or(compose_mask, mask)
        return compose_mask

    def __call__(self, results):
        # Get types of modified objects
        bbox_type = results['gt_bboxes'].dtype
        mask_type = results['gt_masks'].masks.dtype
        label_type = results['gt_labels'].dtype
        # Apply augmentations to an image that will share masks
        h, w = results['img'].shape[:-1]
        self.rescale_paste_target(results['copy_paste'], (h, w))
        results['copy_paste'] = self.flip(results['copy_paste'])
        if self.rotate is not None:
            results['copy_paste'] = self.rotate(results['copy_paste'])
        # Get random objects from the image
        objects_num = results['copy_paste']['gt_labels'].shape[0]
        random_num = min(random.randint(0, int(1.1 * objects_num)), objects_num)
        all_nums = [x for x in range(objects_num)]
        paste_objects = random.sample(all_nums, random_num)
        paste_objects = np.array(paste_objects)
        # If list of randomly selected objects is empty do nothing
        if not len(paste_objects):
            return results
        # Get composed mask from all masks on the image
        compose_paste_mask = self.get_composed_mask(results, h, w, paste_objects)
        # Copypaste objects to source image
        self.image_copy_paste(results, alpha=compose_paste_mask)
        self.masks_copy_paste(results, paste_objects, alpha=compose_paste_mask)
        self.concatenate_labels(results, paste_objects)
        self.filter_empty(results)
        self.cast(results, bbox_type, mask_type, label_type)
        #self.visualize(results, 'CopyPaste_result')
        return results
