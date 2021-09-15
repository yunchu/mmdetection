
import random
import unittest
from copy import deepcopy

from ote_sdk.entities.annotation import AnnotationSceneKind
from ote_sdk.entities.subset import Subset
from ote_sdk.entities.model_template import parse_model_template
from ote_sdk.tests.test_helpers import generate_random_annotated_image

from sc_sdk.entities.annotation import AnnotationScene
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


class Collect:

    def __init__(self,
                 keys):
        self.keys = keys

    def __call__(self, results):
        data = {}
        for key in self.keys:
            data[key] = results[key]
        return data


class LoadImageFromOTEDataset:

    def __call__(self, results):
        dataset_item = results['dataset_item']
        img = dataset_item.numpy

        assert img.shape[0] == results['height'], f"{img.shape[0]} != {results['height']}"
        assert img.shape[1] == results['width'], f"{img.shape[1]} != {results['width']}"

        results['img'] = img

        return results

class OTEDataset2:

    class _DataInfoProxy:
        """
        This class is intended to be a wrapper to use it in CustomDataset-derived class as `self.data_infos`.
        Instead of using list `data_infos` as in CustomDataset, our implementation of dataset OTEDataset
        uses this proxy class with overriden __len__ and __getitem__; this proxy class
        forwards data access operations to ote_dataset and converts the dataset items to the view
        convenient for mmdetection.
        """
        def __init__(self, ote_dataset, classes):
            self.ote_dataset = ote_dataset
            self.CLASSES = classes

        def __len__(self):
            return len(self.ote_dataset)

        def __getitem__(self, index):
            """
            Prepare a dict 'data_info' that is expected by the mmdet pipeline to handle images and annotations
            :return data_info: dictionary that contains the image and image metadata, as well as the labels of the objects
                in the image
            """

            item = self.ote_dataset[index]
            data_info = dict(dataset_item=item, width=item.width, height=item.height, dataset_id=self.ote_dataset.id, index=index,
                             ann_info=dict(label_list=self.CLASSES))

            return data_info

    def __init__(self, ote_dataset: Dataset, classes=None):
        self.data_infos = OTEDataset2._DataInfoProxy(ote_dataset, classes)
        self.transforms = [LoadImageFromOTEDataset(), Collect(keys=['img'])]

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, idx):
        data = deepcopy(self.data_infos[idx])
        for t in self.transforms:
            data = t(data)
            if data is None:
                return None
        return data

class API(unittest.TestCase):

    def body(self, save_to_repos):
        model_template = parse_model_template('configs/ote/custom-object-detection/mobilenet_v2-2s_ssd-256x256/template.yaml')
        workspace = WorkspaceRepo().get_default_workspace()

        number_of_images = 200
        max_shapes = 10
        min_size_shape = 50
        max_size_shape = 100
        random_seed = 42
        classes = ['rectangle', 'ellipse', 'triangle']

        image_width = lambda : random.randrange(200, 500)
        image_height = lambda : random.randrange(200, 500)

        project = ProjectFactory.create_project_single_task(
            name="name",
            description="description",
            label_names=classes,
            model_template_id=model_template.model_template_id,
            configurable_parameters=None,
            workspace=workspace,
        )

        labels = project.get_labels()

        items = []

        for i in range(0, number_of_images):
            name = f"image {i}"

            image_numpy, annotations = generate_random_annotated_image(
                image_width=image_width(),
                image_height=image_height(),
                labels=labels,
                max_shapes=max_shapes,
                min_size=min_size_shape,
                max_size=max_size_shape,
                random_seed=random_seed,
            )
            image = Image(
                name=name, dataset_storage=project.dataset_storage, numpy=image_numpy
            )
            image_identifier = ImageIdentifier(image.id)

            annotation_scene = AnnotationScene(
                kind=AnnotationSceneKind.ANNOTATION,
                media_identifier=image_identifier,
                annotations=annotations,
            )
            items.append(DatasetItem(media=image, annotation_scene=annotation_scene))

        if save_to_repos:
            for i, item in enumerate(items):
                ImageRepo(project.dataset_storage).save(item.media)

        rng = random.Random()
        rng.shuffle(items)
        for i, _ in enumerate(items):
            subset_region = i / number_of_images
            if subset_region >= 0.8:
                subset = Subset.TESTING
            elif subset_region >= 0.6:
                subset = Subset.VALIDATION
            else:
                subset = Subset.TRAINING
            items[i].subset = subset

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
