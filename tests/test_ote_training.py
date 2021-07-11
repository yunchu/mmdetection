#import functools
#import numpy as np
#import os.path as osp
#import pytest
#import random
#import time
#import warnings
#from concurrent.futures import ThreadPoolExecutor
#
#from flaky import flaky
#from sc_sdk.entities.annotation import Annotation, AnnotationScene, AnnotationSceneKind
#from sc_sdk.entities.dataset_item import DatasetItem
#from sc_sdk.entities.datasets import Dataset, Subset
#from sc_sdk.entities.image import Image
#from sc_sdk.entities.media_identifier import ImageIdentifier
#from sc_sdk.entities.model import NullModel
#from sc_sdk.entities.optimized_model import OptimizedModel
#from sc_sdk.entities.resultset import ResultSet
#from sc_sdk.entities.shapes.box import Box
#from sc_sdk.entities.shapes.ellipse import Ellipse
#from sc_sdk.entities.shapes.polygon import Polygon
#from sc_sdk.entities.task_environment import TaskEnvironment
#from sc_sdk.tests.test_helpers import generate_random_annotated_image, rerun_on_flaky_assert
#from sc_sdk.usecases.tasks.interfaces.model_optimizer import IModelOptimizer
#from sc_sdk.utils.project_factory import ProjectFactory
#
#from mmdet.apis.ote.apis.detection import MMObjectDetectionTask, MMDetectionParameters, configurable_parameters

#######
import importlib
import os.path as osp
import pytest
import sys
import yaml

from collections import namedtuple
from pprint import pformat

from sc_sdk.entities.analyse_parameters import AnalyseParameters
from sc_sdk.entities.dataset_storage import NullDatasetStorage
from sc_sdk.entities.datasets import Subset
from sc_sdk.entities.resultset import ResultSet
from sc_sdk.entities.task_environment import TaskEnvironment
from sc_sdk.logging import logger_factory
from sc_sdk.utils.project_factory import ProjectFactory

from mmdet.apis.ote.extension.datasets.mmdataset import MMDatasetAdapter

from e2e_test_system import e2e_pytest


logger = logger_factory.get_logger('Sample')

def DATASET_PARAMETERS_FIELDS():
    return ['annotations_train',
            'images_train_dir',
            'annotations_val',
            'images_val_dir',
            'annotations_test',
            'images_test_dir',
            ]
DatasetParameters = namedtuple('DatasetParameters', DATASET_PARAMETERS_FIELDS())

@pytest.fixture
def dataset_definitions_fx(request):
    """
    Return dataset definitions read from a YAML file passed as the parameter --dataset-definitions.
    Note that the dataset definitions should store the following structure:
    {
        <dataset_name>: {
            'annotations_train': <annotation_file_path1>
            'images_train_dir': <images_folder_path1>
            'annotations_val': <annotation_file_path2>
            'images_val_dir': <images_folder_path2>
            'annotations_test': <annotation_file_path3>
            'images_test_dir':  <images_folder_path3>
        }
    }
    """
    path = request.config.getoption('--dataset-definitions')
    assert path is not None, (f'The command line parameter "--dataset-definitions" is not set, '
                             f'whereas it is required for the test {request.node.originalname or request.node.name}')
    with open(path) as f:
        data = yaml.safe_load(f)
    return data

@pytest.fixture
def template_paths_fx(request):
    """
    Return mapping model names to template paths, read from YAML file passed as the parameter --template-paths
    Note that the file should store the following structure:
    {
        <model_name>: <template_path>
    }
    """
    path = request.config.getoption('--template-paths')
    assert path is not None, (f'The command line parameter "--template-paths" is not set, '
                             f'whereas it is required for the test {request.node.originalname or request.node.name}')
    with open(path) as f:
        data = yaml.safe_load(f)
    return data

def _load_template(path):
    with open(path) as f:
        template = yaml.full_load(f)
    # Save path to template file, to resolve relative paths later.
    template['hyper_parameters']['params'].setdefault('algo_backend', {})['template'] = path
    return template

def _get_task_class(path):
    module_name, class_name = path.rsplit('.', 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def _create_project_and_connect_to_dataset(dataset):
    project = ProjectFactory().create_project_single_task(
        name='otedet-sample-project',
        description='otedet-sample-project',
        label_names=dataset.get_labels(),
        task_name='otedet-task')
    dataset.set_project_labels(project.get_labels())
    return project



def _get_dataset_params_from_dataset_definitions(dataset_definitions, dataset_name):
    cur_dataset_definition = dataset_definitions[dataset_name]
    training_parameters_fields = {k: v for k, v in cur_dataset_definition.items()
                                  if k in DATASET_PARAMETERS_FIELDS()}

    assert set(DATASET_PARAMETERS_FIELDS()) == set(training_parameters_fields.keys()), \
            f'ERROR: dataset definitions for name={dataset_name} does not contain all required fields'
    assert all(training_parameters_fields.values()), \
            f'ERROR: dataset definitions for name={dataset_name} contains empty values for some required fields'

    params = DatasetParameters(**training_parameters_fields)
    return params

@pytest.mark.parametrize('model_name', ['mobilenet_v2_2s_ssd_256x256'])
@pytest.mark.parametrize('dataset_name',
                         [
#                             'coco_shortened_500',
#                             'vitens_tiled_shortened_500',
                             'vitens_tiled_shortened_500_A'
                         ])
class TestOTETraining:

    def _run_evaluation(self):
        if self.was_training_run and self.stored_exception:
            raise self.stored_exception
        if self.was_training_run:
            return
        try:
            model = run_ote_training(self.dataset_params, self.template_path)
        except Exception as e:
            self.stored_exception = e
            self.was_training_run = True
            raise e
        self.was_training_run = True
        self.model = model

    @staticmethod
    def _create_environment_and_task(project, template):
        task_impl_path = template['task']['impl']
        task_cls = _get_task_class(task_impl_path)

        environment = TaskEnvironment(project=project, task_node=project.tasks[-1])
        params = task_cls.get_configurable_parameters(environment)
        task_cls.apply_template_configurable_parameters(params, template)
        environment.set_configurable_parameters(params)

        task = task_cls(task_environment=environment)
        return environment, task

    def _run_ote_training(self, dataset_params: DatasetParameters, template_file_path: str):
        # check consistency
        print(f'template_file_path = {template_file_path}')
        print(f'Using for train annotation file {dataset_params.annotations_train}')
        print(f'Using for val annotation file {dataset_params.annotations_val}')

        self.dataset = MMDatasetAdapter(
            train_ann_file=dataset_params.annotations_train,
            train_data_root=dataset_params.images_train_dir,
            val_ann_file=dataset_params.annotations_val,
            val_data_root=dataset_params.images_val_dir,
            test_ann_file=dataset_params.annotations_test,
            test_data_root=dataset_params.images_test_dir,
            dataset_storage=NullDatasetStorage)
        print(f'train dataset: {len(self.dataset.get_subset(Subset.TRAINING))} items')
        print(f'validation dataset: {len(self.dataset.get_subset(Subset.VALIDATION))} items')

        self.template = _load_template(template_file_path)

        self.project = _create_project_and_connect_to_dataset(self.dataset)
        self.environment, self.task = self._create_environment_and_task(self.project, self.template)


        # Tweak parameters.
        params = self.task.get_configurable_parameters(self.environment)
        if True: # DEBUG prints
            print(f'params before training=\n{pformat(params.serialize(), width=140)}')
        params.learning_parameters.nncf_quantization.value = False
        params.learning_parameters.num_iters.value = 5
#        params.postprocessing.result_based_confidence_threshold.value = False
#        params.postprocessing.confidence_threshold.value = 0.025
        self.environment.set_configurable_parameters(params)
        self.task.update_configurable_parameters(self.environment)

        logger.info('Start model training... [ROUND 0]')
        self.model = self.task.train(dataset=self.dataset)
        logger.info('Model training finished [ROUND 0]')
        logger.info(f'RES={self.model}')

    def _run_ote_training_once(self, dataset_params: DatasetParameters, template_file_path: str):
        was_training_run = getattr(self, 'was_training_run', False)
        stored_exception = getattr(self, 'stored_exception', None)
        print(f':::DEBUG::: _run_ote_training_once: self={self}, id(self)={id(self)}')
        if was_training_run and stored_exception:
            raise stored_exception
        if was_training_run:
            return
        try:
            self._run_ote_training(dataset_params, template_file_path)
        except Exception as e:
            self.stored_exception = e
            self.was_training_run = True
            raise e
        self.was_training_run = True

    @e2e_pytest
    def test_ote_training(self, dataset_name, model_name, dataset_definitions_fx, template_paths_fx):
        dataset_params = _get_dataset_params_from_dataset_definitions(dataset_definitions_fx, dataset_name)
        template_path = template_paths_fx[model_name]
        self._run_ote_training_once(dataset_params, template_path)

    @e2e_pytest
    def test_ote_training2(self, dataset_name, model_name, dataset_definitions_fx, template_paths_fx):
        dataset_params = _get_dataset_params_from_dataset_definitions(dataset_definitions_fx, dataset_name)
        template_path = template_paths_fx[model_name]
        self._run_ote_training_once(dataset_params, template_path)
#    def test_training(self):
#        self._run_training_once()
#
#    def test_evaluation(self):
#        self._run_training_once()
#        self._run_evaluation_once()


#@e2e_pytest
#@pytest.mark.parametrize('model_name', ['mobilenet_v2_2s_ssd_256x256'])
#@pytest.mark.parametrize('dataset_name',
#                         [
##                             'coco_shortened_500',
##                             'vitens_tiled_shortened_500',
#                             'vitens_tiled_shortened_500_A'
#                         ])
#def test_ote_training(dataset_name, model_name, dataset_definitions_fx, template_paths_fx):
#    dataset_params = _get_dataset_params_from_dataset_definitions(dataset_definitions_fx, dataset_name)
#    template_path = template_paths_fx[model_name]
#    run_ote_training(dataset_params, template_path)
