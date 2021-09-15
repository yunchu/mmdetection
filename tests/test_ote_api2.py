from re import T
from tqdm import tqdm
import random
import unittest
from copy import deepcopy

import numpy as np
from e2e_test_system import e2e_pytest_api
from ote_sdk.entities.model_template import parse_model_template
from ote_sdk.entities.subset import Subset
from sc_sdk.entities.annotation import NullAnnotationScene
from sc_sdk.entities.dataset_item import DatasetItem
from sc_sdk.entities.datasets import Dataset, DatasetItem
from sc_sdk.entities.image import Image, ID
from sc_sdk.usecases.repos import ImageRepo
from sc_sdk.usecases.repos.workspace_repo import WorkspaceRepo
from sc_sdk.utils.project_factory import ProjectFactory


class ObjectDetectionDataset:
    def __init__(self, sc_dataset: Dataset):
        self.sc_dataset = sc_dataset

        self.id_set = set()

    def __len__(self):
            return len(self.sc_dataset)

    def __getitem__(self, idx):
        item = self.sc_dataset[idx]

        data0 = deepcopy(item)
        data1 = deepcopy(item)

        if data0.media.id != ID() and data1.media.id != ID():
            assert data0.media.id == data1.media.id
            assert data0.media.id not in self.id_set
            self.id_set.add(data0.media.id)
        else:
            print('skip checking IDs, it is ok if IDs are empty and images are not saved to repo')

        print(data0.media.image_adapter == data1.media.image_adapter)

        img0 = data0.numpy
        img1 = data1.numpy


        assert np.sum(img0 - img1) == 0


class API(unittest.TestCase):

    def body(self, save_to_repos, non_empty_id):
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
            image_numpy = (np.random.rand(random.randrange(200, 500), random.randrange(200, 500), 3) * 255).astype(np.uint8)
            image = Image(
                name=f"image{i}", dataset_storage=project.dataset_storage, numpy=image_numpy,
                id=f"image{i}" if non_empty_id else None
            )
            items.append(DatasetItem(media=image, annotation_scene=NullAnnotationScene(), subset=Subset.TRAINING))

        if save_to_repos:
            for i, item in enumerate(items):
                ImageRepo(project.dataset_storage).save(item.media)

        dataset = Dataset(project.dataset_storage, items)
        object_detection_dataset = ObjectDetectionDataset(dataset)

        items = []
        for item in tqdm(object_detection_dataset):
            items.append(item)

    @e2e_pytest_api
    def test_save_to_repos_true_empty_id(self):
        """
            def __getitem__(self, idx):
                item = self.sc_dataset[idx]

                data0 = deepcopy(item)
                data1 = deepcopy(item)

                assert data0.media.id == data1.media.id
                assert data0.media.id not in self.id_set
                self.id_set.add(data0.media.id)

                print(data0.media.image_adapter == data1.media.image_adapter)

                img0 = data0.numpy
                img1 = data1.numpy


        >       assert np.sum(img0 - img1) == 0
        E       ValueError: operands could not be broadcast together with shapes (333,468,3) (414,428,3)
        """

        self.body(save_to_repos=True, non_empty_id=False)
