
import random
import unittest
from copy import deepcopy

import numpy as np

from ote_sdk.entities.annotation import AnnotationSceneKind
from ote_sdk.entities.subset import Subset
from ote_sdk.entities.model_template import parse_model_template
from ote_sdk.tests.test_helpers import generate_random_annotated_image

from sc_sdk.entities.annotation import AnnotationScene, NullAnnotationScene
from sc_sdk.entities.dataset_item import DatasetItem
from sc_sdk.entities.datasets import Dataset, NullDatasetStorage
from sc_sdk.entities.image import Image
from sc_sdk.entities.media_identifier import ImageIdentifier

from e2e_test_system import e2e_pytest_api


from sc_sdk.usecases.repos import ImageRepo

from sc_sdk.tests.test_helpers import generate_random_annotated_image

from sc_sdk.usecases.repos.workspace_repo import WorkspaceRepo
from sc_sdk.utils.project_factory import ProjectFactory

from ote_sdk.entities.annotation import AnnotationSceneKind
from ote_sdk.entities.subset import Subset

from sc_sdk.entities.annotation import AnnotationScene
from sc_sdk.entities.datasets import Dataset, DatasetItem
from sc_sdk.entities.dataset_storage import NullDatasetStorage
from sc_sdk.entities.image import Image


class OTEDataset2:

    def __init__(self, ote_dataset: Dataset, classes=None):
        self.ote_dataset = ote_dataset
        self.CLASSES = classes
        self.keys = ['img']

    def __len__(self):
            return len(self.ote_dataset)

    def __getitem__(self, idx):
        item = self.ote_dataset[idx]
        data_info = dict(dataset_item=item, width=item.width, height=item.height)

        data = deepcopy(data_info)

        dataset_item = data['dataset_item']
        img = dataset_item.numpy

        assert img.shape[0] == data['height']
        assert img.shape[1] == data['width']

        data['img'] = img

        new_data = {}
        new_data['img'] = data['img']

        return new_data

class API(unittest.TestCase):

    def body(self, save_to_repos):
        model_template = parse_model_template('configs/ote/custom-object-detection/mobilenet_v2-2s_ssd-256x256/template.yaml')
        workspace = WorkspaceRepo().get_default_workspace()

        number_of_images = 200
        classes = ['rectangle', 'ellipse', 'triangle']

        project = ProjectFactory.create_project_single_task(
            name="name",
            description="description",
            label_names=classes,
            model_template_id=model_template.model_template_id,
            configurable_parameters=None,
            workspace=workspace,
        )

        items = []

        for i in range(0, number_of_images):
            image_numpy = (np.random.rand(random.randrange(200, 500), random.randrange(200, 500), 3) * 225).astype(np.uint8)
            image = Image(
                name=f"image {i}", dataset_storage=project.dataset_storage, numpy=image_numpy
            )
            items.append(DatasetItem(media=image, annotation_scene=NullAnnotationScene()))

        if save_to_repos:
            for i, item in enumerate(items):
                ImageRepo(project.dataset_storage).save(item.media)

        for i, _ in enumerate(items):
            items[i].subset = Subset.TRAINING

        dataset = Dataset(NullDatasetStorage(), items)
        train_dataset = dataset.get_subset(Subset.TRAINING)

        mm_train_dataset = OTEDataset2(train_dataset, classes)

        items = []
        i = 0
        for a in range(100):
            for item in mm_train_dataset:
                items.append(item)
                print(i)
                i += 1

    @e2e_pytest_api
    def test_save_to_repos_true(self):
        self.body(save_to_repos=True)

    @e2e_pytest_api
    def test_save_to_repos_false(self):
        self.body(save_to_repos=False)
