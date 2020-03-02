import json
import re

import numpy as np

from .custom import CustomDataset
from .registry import DATASETS


@DATASETS.register_module
class BarrierVLPDataset(CustomDataset):

    CLASSES = ('vehicle', 'plate')

    def load_annotations(self, ann_file):
        raw_annotation = json.load(open(ann_file, 'r'))

        cat2label = {
            cat_id: i + 1
            for i, cat_id in enumerate(self.CLASSES)
        }

        def fix_path(path):
            gr = re.match(r'(.+\/annotation_\S+\/data\/)(.*)', path)
            if gr:
                path = gr.group(2)
            return path

        img_infos = []
        for im_annotation in raw_annotation:
            im_bboxes = np.asarray([x['bbox'] for x in im_annotation['objects']
                                    if x['label'] in cat2label], dtype=np.float32)
            im_labels = np.asarray([cat2label[x['label']] for x in im_annotation['objects']
                                    if x['label'] in cat2label], dtype=np.int64)

            assert len(im_bboxes) == len(im_labels)
            if len(im_bboxes) == 0:
                im_bboxes = np.empty((0, 4), dtype=np.float32)
                im_labels = np.empty(0, dtype=np.int64)

            img_infos.append({
                'filename': fix_path(im_annotation['image']),
                'width': im_annotation['image_size'][0],
                'height': im_annotation['image_size'][1],
                'ann': {
                    'bboxes': im_bboxes,
                    'labels': im_labels
                }
            })

        return img_infos
