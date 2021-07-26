import importlib
import logging
import os
import os.path as osp
import pytest
import sys
import yaml

from collections import namedtuple
from copy import deepcopy
from pprint import pformat

#from sc_sdk.entities.analyse_parameters import AnalyseParameters
#from sc_sdk.entities.dataset_storage import NullDatasetStorage
#from sc_sdk.entities.datasets import Subset
#from sc_sdk.entities.resultset import ResultSet
#from sc_sdk.entities.task_environment import TaskEnvironment
#from sc_sdk.logging import logger_factory
#from sc_sdk.utils.project_factory import ProjectFactory
#
#from mmdet.apis.ote.extension.datasets.mmdataset import MMDatasetAdapter

from sc_sdk.entities.dataset_storage import NullDatasetStorage
from sc_sdk.entities.datasets import Subset
from sc_sdk.entities.id import ID
from sc_sdk.entities.inference_parameters import InferenceParameters
from sc_sdk.entities.metrics import Performance, ScoreMetric
from sc_sdk.entities.model import NullModel, Model, ModelStatus
from sc_sdk.entities.model_storage import NullModelStorage
from sc_sdk.entities.optimized_model import ModelOptimizationType, ModelPrecision, OptimizedModel, TargetDevice
from sc_sdk.entities.project import NullProject
from sc_sdk.entities.resultset import ResultSet
from sc_sdk.entities.task_environment import TaskEnvironment
from sc_sdk.logging import logger_factory
from sc_sdk.usecases.tasks.interfaces.export_interface import ExportType

#!!!!!!!!!!!!!!                    from sc_sdk.configuration import cfg_helper
#from sc_sdk.configuration.helper.utils import ids_to_strings

from mmdet.apis.ote.apis.detection.configuration import OTEDetectionConfig
from mmdet.apis.ote.apis.detection.config_utils import apply_template_configurable_parameters
from mmdet.apis.ote.extension.datasets.mmdataset import MMDatasetAdapter
from mmdet.apis.ote.apis.detection.ote_utils import generate_label_schema, load_template, get_task_class

from e2e_test_system import e2e_pytest, DataCollector


logger_name = osp.splitext(osp.basename(__file__))[0]
logger = logger_factory.get_logger(logger_name)
logger.setLevel(logging.DEBUG)

def DATASET_PARAMETERS_FIELDS():
    return ['annotations_train',
            'images_train_dir',
            'annotations_val',
            'images_val_dir',
            'annotations_test',
            'images_test_dir',
            ]

ROOT_PATH_KEY = '_root_path'
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
    data[ROOT_PATH_KEY] = osp.dirname(path)
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
    data[ROOT_PATH_KEY] = osp.dirname(path)
    return data

#def _load_template(path):
#    with open(path) as f:
#        template = yaml.full_load(f)
#    # Save path to template file, to resolve relative paths later.
#    template['hyper_parameters']['params'].setdefault('algo_backend', {})['template'] = path
#    return template
#
#def _get_task_class(path):
#    module_name, class_name = path.rsplit('.', 1)
#    module = importlib.import_module(module_name)
#    return getattr(module, class_name)
#
#
#def _create_project_and_connect_to_dataset(dataset):
#    project = ProjectFactory().create_project_single_task(
#        name='otedet-sample-project',
#        description='otedet-sample-project',
#        label_names=dataset.get_labels(),
#        task_name='otedet-task')
#    dataset.set_project_labels(project.get_labels())
#    return project
#
#

def _make_path_be_abs(some_val, root_path):
    assert isinstance(some_val, (str, dict)), f'Wrong type of value: {some_val}, type={type(some_val)}'
    assert isinstance(root_path, str), f'Wrong type of root_path: {root_path}, type={type(root_path)}'

    if isinstance(some_val, str):
        if not osp.isabs(some_val):
            return osp.join(root_path, some_val)
        return some_val

    some_dict = some_val
    for k in sorted(some_dict.keys()):
        v = some_dict[k]
        if isinstance(v, str) and not osp.isabs(v):
            some_dict[k] = osp.join(root_path, v)
    return some_dict

def _get_dataset_params_from_dataset_definitions(dataset_definitions, dataset_name):
    cur_dataset_definition = dataset_definitions[dataset_name]
    training_parameters_fields = {k: v for k, v in cur_dataset_definition.items()
                                  if k in DATASET_PARAMETERS_FIELDS()}
    _make_path_be_abs(training_parameters_fields, dataset_definitions[ROOT_PATH_KEY])

    assert set(DATASET_PARAMETERS_FIELDS()) == set(training_parameters_fields.keys()), \
            f'ERROR: dataset definitions for name={dataset_name} does not contain all required fields'
    assert all(training_parameters_fields.values()), \
            f'ERROR: dataset definitions for name={dataset_name} contains empty values for some required fields'

    params = DatasetParameters(**training_parameters_fields)
    return params

def to_dict(val):
    """
    General function to convert any class/structure to a serializable dict
    if the class/structure is sufficiently simple; but nested simple structs are possible.
    """
    import numpy as np
    def _to_dict(val):
        PRIMITIVE_TYPES = (bool, int, float, str, complex) #
        if type(val) in PRIMITIVE_TYPES:
            return val
        if isinstance(val, np.number):
            return val
        if type(val) == list:
            return [_to_dict(v) for v in val]
        if type(val) == dict:
            keys = list(val.keys())
            return {k: _to_dict(v) for k, v in val.items()}

        keys = [k for k in dir(val) if not k.startswith('_') and not callable(getattr(val, k))]
        res = {}
        for k in keys:
            res[k] = _to_dict(getattr(val, k))
        return res
    res = _to_dict(val)
    return res

def performance_to_score_name_value(perf: Performance):
    """
    The method is intended to get main score info from Performance class
    """
    if perf is None:
        return None, None
    assert isinstance(perf, Performance)
    score = perf.score
    assert isinstance(score, ScoreMetric)
    name = score.name
    value = score.value
    return name, value

def select_configurable_parameters(json_configurable_parameters):
    selected = {}
    getv = lambda c, n: c[n]['value']
    for section, container in json_configurable_parameters.items():
        for param in container.keys():
            try:
                selected[f'{section}/{param}'] = getv(container, param)
            except TypeError:
                pass
            except KeyError:
                pass
    return selected

class OTETrainingImpl:
    def __init__(self, dataset_params: DatasetParameters, template_file_path: str):
        self.dataset_params = dataset_params
        self.template_file_path = template_file_path

        self.template = None
        self.environment = None
        self.task = None
        self.output_model = None
        self.evaluation_performance = None

        self.was_training_run = False
        self.stored_exception = None

        self.copy_configurable_parameters = None

    @staticmethod
    def _create_environment_and_task(params, labels_schema, template):
        environment = TaskEnvironment(model=NullModel(), configurable_parameters=params, label_schema=labels_schema)
        task_impl_path = template['task']['base']
        task_cls = get_task_class(task_impl_path)
        task = task_cls(task_environment=environment)
        return environment, task

    def _run_ote_training(self):
        print(f'self.template_file_path = {self.template_file_path}')
        print(f'Using for train annotation file {self.dataset_params.annotations_train}')
        print(f'Using for val annotation file {self.dataset_params.annotations_val}')

        logger.debug('Load model template')
        self.template = load_template(self.template_file_path)

        self.dataset = MMDatasetAdapter(
            train_ann_file=self.dataset_params.annotations_train,
            train_data_root=self.dataset_params.images_train_dir,
            val_ann_file=self.dataset_params.annotations_val,
            val_data_root=self.dataset_params.images_val_dir,
            test_ann_file=self.dataset_params.annotations_test,
            test_data_root=self.dataset_params.images_test_dir,
            dataset_storage=NullDatasetStorage)

        self.labels_schema = generate_label_schema(self.dataset.get_labels())
        labels_list = self.labels_schema.get_labels(False)
        self.dataset.set_project_labels(labels_list)
        print(f'train dataset: {len(self.dataset.get_subset(Subset.TRAINING))} items')
        print(f'validation dataset: {len(self.dataset.get_subset(Subset.VALIDATION))} items')


        logger.debug('Setup environment')
        params = OTEDetectionConfig(workspace_id=ID(), project_id=ID(), task_id=ID())
        apply_template_configurable_parameters(params, self.template)
        self.environment, self.task = self._create_environment_and_task(params,
                                                                        self.labels_schema,
                                                                        self.template)



        logger.debug('Set hyperparameters')
        self.task.hyperparams.learning_parameters.num_checkpoints = 2
        self.task.hyperparams.learning_parameters.num_iters = 5

        logger.debug('Train model')
        self.output_model = Model(
            NullProject(),
            NullModelStorage(),
            self.dataset,
            self.environment.get_model_configuration(),
            model_status=ModelStatus.NOT_READY)

        self.copy_configurable_parameters = deepcopy(self.task.hyperparams)
#        p = self.copy_configurable_parameters
#        print(f'p = {p}')
#        print(f'p.__dict__ = {p.__dict__}')
#        print(f'p.learning_parameters.__dict__ = {p.learning_parameters.__dict__}')
#        print(f'vars(p.learning_parameters) = {vars(p.learning_parameters)}')
#        hyperparams_str = ids_to_strings(cfg_helper.convert(p, dict, enum_to_str=True))
#        print(f'type(hyperparams_str) = {type(hyperparams_str)}')
#        print(f'hyperparams_str = {hyperparams_str}')
#        print('=~=~=~')

        self.task.train(self.dataset, self.output_model)
        logger.info(f'performance={self.output_model.performance}')

    def get_training_performance(self):
        return getattr(self.output_model, 'performance', None)

    def run_ote_training_once(self, data_collector):
        if self.was_training_run and self.stored_exception:
            logger.warn('In function run_ote_training_once: found that previous call of the function '
                        'caused exception -- re-raising it')
            raise self.stored_exception

        if not self.was_training_run:
            try:
                self._run_ote_training()
                self.was_training_run = True
            except Exception as e:
                self.stored_exception = e
                self.was_training_run = True
                raise e

        training_performance = self.get_training_performance()

        score_name, score_value = performance_to_score_name_value(training_performance)
        if score_name:
            data_collector.log_final_metric('training_performance/' + score_name, score_value)
        else:
            logger.warning(f'WARNING: Cannot get training performance')

#        logger.info(f'!!!!!!!!!!!!!! self.copy_configurable_parameters = {self.copy_configurable_parameters}')
#        json_configurable_parameters = self.copy_configurable_parameters.to_json()
#        selected_configurable_parameters = select_configurable_parameters(json_configurable_parameters)
#        for k, v in selected_configurable_parameters.items():
#            data_collector.update_metadata(k, v)

        return training_performance

    def run_ote_evaluation(self, data_collector, subset=Subset.VALIDATION):
        logger.debug('Get predictions on the validation set')
        validation_dataset = self.dataset.get_subset(subset)
        self.predicted_validation_dataset = self.task.infer(
            validation_dataset.with_empty_annotations(),
            InferenceParameters(is_evaluation=True))
        self.resultset = ResultSet(
            model=self.output_model,
            ground_truth_dataset=validation_dataset,
            prediction_dataset=self.predicted_validation_dataset,
        )
        logger.debug('Estimate quality on validation set')
        self.evaluation_performance = self.task.evaluate(self.resultset)
        logger.info(f'performance={self.evaluation_performance}')
        score_name, score_value = performance_to_score_name_value(self.evaluation_performance)
        data_collector.log_final_metric('evaluation_performance/' + score_name, score_value)
        return self.evaluation_performance

# pytest magic
def pytest_generate_tests(metafunc):
    if metafunc.cls is None:
        return
    if not issubclass(metafunc.cls, TestOTETraining):
        return
    metafunc.parametrize('model_name', metafunc.cls.parameters['model_name'], scope='class')
    metafunc.parametrize('dataset_name', metafunc.cls.parameters['dataset_name'], scope='class')

class TestOTETraining:
    parameters = {
            'model_name': [
                'mobilenet_v2_2s_ssd_256x256',
                #'model_1'
             ],
            'dataset_name': [
                'vitens_tiled_shortened_500_A',
                #TMPCOMMENT#'vitens_tiled',
#               'coco_shortened_500',
#               'vitens_tiled_shortened_500',
               #'dataset_2',
            ]
    }

    @pytest.fixture(scope='class')
    def cached_from_prev_test_fx(self):
        """
        This fixture is intended for storying the impl class OTETrainingImpl.
        This object should be persistent between tests while the tests use the same parameters
        -- see the method _clean_cache_if_parameters_changed below that is used to clean
        the impl if the parameters are changed.
        """
        return dict()

    @staticmethod
    def _clean_cache_if_parameters_changed(cache, **kwargs):
        is_ok = True
        for k, v in kwargs.items():
            is_ok = is_ok and (cache.get(k) == v)
        if is_ok:
            logger.info('TestOTETraining: parameters were not changed -- cache is kept')
            return

        for k in list(cache.keys()):
            del cache[k]
        for k, v in kwargs.items():
            cache[k] = v
        logger.info('TestOTETraining: parameters were changed -- cache is cleaned')

    @staticmethod
    def _update_impl_in_cache(cache,
                              dataset_name, model_name,
                              dataset_definitions, template_paths):
        TestOTETraining._clean_cache_if_parameters_changed(cache,
                                                           dataset_name=dataset_name,
                                                           model_name=model_name)
        if 'impl' not in cache:
            logger.info('TestOTETraining: creating OTETrainingImpl')
            dataset_params = _get_dataset_params_from_dataset_definitions(dataset_definitions, dataset_name)
            template_path = _make_path_be_abs(template_paths[model_name], template_paths[ROOT_PATH_KEY])
            cache['impl'] = OTETrainingImpl(dataset_params, template_path)

        return cache['impl']

    @pytest.fixture
    def data_collector_fx(self, request):
        setup = deepcopy(request.node.callspec.params)
        setup["environment_name"] = os.environ.get("TT_ENVIRONMENT_NAME", "no-env")
        setup["test_type"] = os.environ.get("TT_TEST_TYPE", "no-env")
        setup["scenario"] = "api"
        setup["test"] = request.node.name
        setup["subject"] = "custom-object-detection"
        setup["project"] = "ote"
        logger.info(f'creating DataCollector: setup=\n{pformat(setup, width=140)}')
        data_collector = DataCollector(name='TestOTETraining',
                                       setup=setup)
        with data_collector:
            logger.info('data_collector is created')
            yield data_collector
        logger.info('data_collector is released')

    @e2e_pytest
    def test_ote_01_training(self, dataset_name, model_name,
                             dataset_definitions_fx, template_paths_fx,
                             cached_from_prev_test_fx,
                             data_collector_fx):
        cache = cached_from_prev_test_fx
        impl = self._update_impl_in_cache(cache,
                                          dataset_name, model_name,
                                          dataset_definitions_fx, template_paths_fx)

        impl.run_ote_training_once(data_collector_fx)

    @e2e_pytest
    def test_ote_02_evaluation(self, dataset_name, model_name,
                               dataset_definitions_fx, template_paths_fx,
                               cached_from_prev_test_fx,
                               data_collector_fx):
        cache = cached_from_prev_test_fx
        impl = self._update_impl_in_cache(cache,
                                          dataset_name, model_name,
                                          dataset_definitions_fx, template_paths_fx)

        impl.run_ote_training_once(data_collector_fx)
        impl.run_ote_evaluation(data_collector_fx)
