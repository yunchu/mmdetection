import glob
import itertools
import logging
import os
import os.path as osp
from collections import namedtuple, OrderedDict
from copy import deepcopy
from pprint import pformat
from typing import Optional, Union

import pytest
import yaml
from e2e_test_system import DataCollector, e2e_pytest_performance
from ote_sdk.configuration.helper import create
from ote_sdk.entities.inference_parameters import InferenceParameters
from ote_sdk.entities.metrics import Performance, ScoreMetric
from ote_sdk.entities.model import (
    ModelEntity,
    ModelPrecision,
    ModelStatus,
    ModelOptimizationType,
    OptimizationMethod,
)
from ote_sdk.entities.model_template import parse_model_template, TargetDevice
from ote_sdk.entities.optimization_parameters import OptimizationParameters
from ote_sdk.entities.resultset import ResultSetEntity
from ote_sdk.entities.subset import Subset
from ote_sdk.usecases.tasks.interfaces.export_interface import ExportType
from ote_sdk.usecases.tasks.interfaces.optimization_interface import OptimizationType
from ote_sdk.entities.task_environment import TaskEnvironment

from sc_sdk.entities.dataset_storage import NullDatasetStorage

from mmdet.apis.ote.apis.detection.config_utils import set_values_as_default
from mmdet.apis.ote.apis.detection.ote_utils import generate_label_schema, get_task_class
from mmdet.apis.ote.extension.datasets.mmdataset import MMDatasetAdapter

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def DATASET_PARAMETERS_FIELDS():
    return ('annotations_train',
            'images_train_dir',
            'annotations_val',
            'images_val_dir',
            'annotations_test',
            'images_test_dir',
            )

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
    if path is None:
        logger.warning(f'The command line parameter "--dataset-definitions" is not set'
                       f'whereas it is required for the test {request.node.originalname or request.node.name}'
                       f' -- ALL THE TESTS THAT REQUIRE THIS PARAMETER ARE SKIPPED')
        return None
    with open(path) as f:
        data = yaml.safe_load(f)
    data[ROOT_PATH_KEY] = osp.dirname(path)
    return data

@pytest.fixture
def template_paths_fx(request):
    """
    Return mapping model names to template paths, received from globbing the folder configs/ote/
    Note that the function searches files with name `template.yaml`, and for each such file
    the model name is the name of the parent folder of the file.
    """
    root = osp.dirname(osp.dirname(osp.realpath(__file__)))
    glb = glob.glob(f'{root}/configs/ote/**/template.yaml', recursive=True)
    data = {}
    for p in glb:
        assert osp.isabs(p), f'Error: not absolute path {p}'
        name = osp.basename(osp.dirname(p))
        if name in data:
            raise RuntimeError(f'Duplication of names in config/ote/ folder: {data[name]} and {p}')
        data[name] = p
    data[ROOT_PATH_KEY] = ''
    return data

def _make_path_be_abs(some_val, root_path):
    assert isinstance(some_val, (str, dict)), f'Wrong type of value: {some_val}, type={type(some_val)}'
    assert isinstance(root_path, str), f'Wrong type of root_path: {root_path}, type={type(root_path)}'

    # Note that os.path.join(a, b) == b if b is an absolute path
    if isinstance(some_val, str):
        return osp.join(root_path, some_val)

    some_dict = some_val
    assert all(isinstance(v, str) for v in some_dict.values()), f'Wrong input dict {some_dict}'
    for k in list(some_dict.keys()):
        some_dict[k] = osp.join(root_path, some_dict[k])
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

def performance_to_score_name_value(perf: Union[Performance, None]):
    """
    The method is intended to get main score info from Performance class
    """
    if perf is None:
        return None, None
    assert isinstance(perf, Performance)
    score = perf.score
    assert isinstance(score, ScoreMetric)
    assert isinstance(score.name, str) and score.name, f'Wrong score name "{score.name}"'
    return score.name, score.value

def convert_hyperparams_to_dict(hyperparams):
    def _convert(p):
        if p is None:
            return None
        d = {}
        groups = getattr(p, 'groups', [])
        parameters = getattr(p, 'parameters', [])
        assert (not groups) or isinstance(groups, list), f'Wrong field "groups" of p={p}'
        assert (not parameters) or isinstance(parameters, list), f'Wrong field "parameters" of p={p}'
        for group_name in groups:
            g = getattr(p, group_name, None)
            d[group_name] = _convert(g)
        for par_name in parameters:
            d[par_name] = getattr(p, par_name, None)
        return d
    return _convert(hyperparams)

class OTETrainingImpl:
    def __init__(self, dataset_params: DatasetParameters, template_file_path: str,
                 num_training_iters: int, batch_size: int):
        self.dataset_params = dataset_params
        self.template_file_path = template_file_path
        self.num_training_iters = num_training_iters
        self.batch_size = batch_size

        self.template = None
        self.environment = None
        self.task = None
        self.output_model = None
        self.environment_for_export = None
        self.exported_model = None
        self.openvino_task = None

        self.was_training_run = False
        self.stored_exception_training = None
        self.was_export_run = False
        self.stored_exception_export = None
        self.was_optimization_pot_run = False
        self.stored_exception_pot = None
        self.was_nncf_optimization_run = False
        self.stored_exception_nncf = None

        self.copy_hyperparams = None

    @staticmethod
    def _create_environment_and_task(params, labels_schema, model_template):
        environment = TaskEnvironment(model=None, hyper_parameters=params, label_schema=labels_schema,
                                      model_template=model_template)
        logger.info('Create base Task')
        task_impl_path = model_template.entrypoints.base
        task_cls = get_task_class(task_impl_path)
        task = task_cls(task_environment=environment)
        return environment, task

    def _run_ote_training(self):
        logger.debug(f'self.template_file_path = {self.template_file_path}')
        logger.debug(f'Using for train annotation file {self.dataset_params.annotations_train}')
        logger.debug(f'Using for val annotation file {self.dataset_params.annotations_val}')

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

        logger.debug('Load model template')
        self.model_template = parse_model_template(self.template_file_path)

        hyper_parameters = self.model_template.hyper_parameters.data
        set_values_as_default(hyper_parameters)

        logger.debug('Setup environment')
        params = create(hyper_parameters)
        logger.debug('Set hyperparameters')
        params.learning_parameters.num_iters = self.num_training_iters
        if self.num_training_iters < 20:
            num_checkpoints = 2
        elif self.num_training_iters < 1000:
            num_checkpoints = 10
        else:
            num_checkpoints = 30

        params.learning_parameters.batch_size = self.batch_size

        params.learning_parameters.num_checkpoints = num_checkpoints

        self.environment, self.task = self._create_environment_and_task(params,
                                                                        self.labels_schema,
                                                                        self.model_template)

        logger.debug('Train model')
        self.output_model = ModelEntity(
            self.dataset,
            self.environment.get_model_configuration(),
            model_status=ModelStatus.NOT_READY)

        self.copy_hyperparams = deepcopy(self.task._hyperparams)

        self.task.train(self.dataset, self.output_model)

        # To check that the training performance is valid
        # and may be converted to score name and value
        self.get_training_performance_as_score_name_value()
        logger.info(f'performance={self.output_model.performance}')

    def get_training_performance_as_score_name_value(self):
        training_performance = getattr(self.output_model, 'performance', None)
        if training_performance is None:
            raise RuntimeError('Cannot get training performance')
        return performance_to_score_name_value(training_performance)

    def run_ote_training_once(self, data_collector):
        if self.was_training_run and self.stored_exception_training:
            logger.warning('In function run_ote_training_once: found that previous call of the function '
                           'caused exception -- re-raising it')
            raise self.stored_exception_training

        if not self.was_training_run:
            try:
                self._run_ote_training()
                self.was_training_run = True
            except Exception as e:
                self.stored_exception_training = e
                self.was_training_run = True
                raise e

        score_name, score_value = self.get_training_performance_as_score_name_value()
        data_collector.log_final_metric('training_performance/' + score_name, score_value)

        hyperparams_dict = convert_hyperparams_to_dict(self.copy_hyperparams)
        for k, v in hyperparams_dict.items():
            data_collector.update_metadata(k, v)

        return score_name, score_value

    def run_ote_evaluation(self, data_collector, subset=Subset.VALIDATION):
        if not self.was_training_run:
            raise RuntimeError('Training was not run for the OTETrainingImpl instance')
        if self.stored_exception_training:
            raise RuntimeError('Training was not successful for the OTETrainingImpl instance')
        logger.debug('Get predictions on the validation set')
        validation_dataset = self.dataset.get_subset(subset)
        predicted_validation_dataset = self.task.infer(
            validation_dataset.with_empty_annotations(),
            InferenceParameters(is_evaluation=True))
        resultset = ResultSetEntity(
            model=self.output_model,
            ground_truth_dataset=validation_dataset,
            prediction_dataset=predicted_validation_dataset,
        )
        logger.debug('Estimate quality on validation set')
        self.task.evaluate(resultset)
        evaluation_performance = resultset.performance
        logger.info(f'performance={evaluation_performance}')
        score_name, score_value = performance_to_score_name_value(evaluation_performance)
        data_collector.log_final_metric('evaluation_performance/' + score_name, score_value)
        return evaluation_performance

    def _run_ote_export(self, data_collector):
        logger.debug('Copy environment for evaluation exported model')

        self.environment_for_export = deepcopy(self.environment)

        logger.debug('Create exported model')
        self.exported_model = ModelEntity(
            self.dataset,
            self.environment_for_export.get_model_configuration(),
            model_status=ModelStatus.NOT_READY)
        logger.debug('Run export')
        self.task.export(ExportType.OPENVINO, self.exported_model)
        logger.debug('Set exported model into environment for export')
        self.environment_for_export.model = self.exported_model

    def run_ote_export_once(self, data_collector):
        if not self.was_training_run:
            raise RuntimeError('Training was not run for the OTETrainingImpl instance')
        if self.stored_exception_training:
            raise RuntimeError('Training was not successful for the OTETrainingImpl instance')
        if self.was_export_run and self.stored_exception_export:
            logger.warning('In function run_ote_export_once: found that previous call of the function '
                           'caused exception -- re-raising it')
            raise self.stored_exception_export

        if not self.was_export_run:
            try:
                self._run_ote_export(data_collector)
                self.was_export_run = True
            except Exception as e:
                self.stored_exception_export = e
                self.was_export_run = True
                raise e

    def _create_openvino_task(self, environment):
        logger.debug('Create OpenVINO Task')
        openvino_task_impl_path = self.model_template.entrypoints.openvino
        openvino_task_cls = get_task_class(openvino_task_impl_path)
        openvino_task = openvino_task_cls(environment)
        return openvino_task

    def run_ote_evaluation_exported(self, data_collector, subset=Subset.VALIDATION):
        if not self.was_training_run:
            raise RuntimeError('Training was not run for the OTETrainingImpl instance')
        if self.stored_exception_training:
            raise RuntimeError('Training was not successful for the OTETrainingImpl instance')
        if not self.was_export_run:
            raise RuntimeError('Export was not run for the OTETrainingImpl instance')
        if self.stored_exception_export:
            raise RuntimeError('Export was not successful for the OTETrainingImpl instance')

        self.openvino_task = self._create_openvino_task(self.environment_for_export)

        logger.debug('Get predictions on the validation set')
        validation_dataset = self.dataset.get_subset(subset)
        predicted_validation_dataset_exp = self.openvino_task.infer(
            validation_dataset.with_empty_annotations(),
            InferenceParameters(is_evaluation=True))
        resultset_exp = ResultSetEntity(
            model=self.exported_model,
            ground_truth_dataset=validation_dataset,
            prediction_dataset=predicted_validation_dataset_exp,
        )
        logger.debug('Estimate quality on validation set')
        self.openvino_task.evaluate(resultset_exp)
        evaluation_performance_exported = resultset_exp.performance

        logger.info(f'performance exported={evaluation_performance_exported}')
        score_name, score_value = performance_to_score_name_value(evaluation_performance_exported)
        data_collector.log_final_metric('evaluation_performance_exported/' + score_name, score_value)
        return evaluation_performance_exported

    def _run_ote_optimization_pot(self, data_collector):
        self.environment_for_pot = deepcopy(self.environment_for_export)
        self.openvino_task_pot = self._create_openvino_task(self.environment_for_pot)

        logger.debug('Create exported model')
        self.optimized_model_pot = ModelEntity(
            self.dataset,
            self.environment_for_pot.get_model_configuration(),
            optimization_type=ModelOptimizationType.POT,
            optimization_methods=OptimizationMethod.QUANTIZATION,
            optimization_objectives={},
            precision=[ModelPrecision.INT8],
            target_device=TargetDevice.CPU,
            performance_improvement={},
            model_size_reduction=1.,
            model_status=ModelStatus.NOT_READY)
        logger.info('Run POT optimization')
        self.openvino_task_pot.optimize(
            OptimizationType.POT,
            self.dataset.get_subset(Subset.TRAINING),
            self.optimized_model_pot,
            OptimizationParameters())

    def run_ote_optimization_pot_once(self, data_collector):
        if not self.was_training_run:
            raise RuntimeError('Training was not run for the OTETrainingImpl instance')
        if self.stored_exception_training:
            raise RuntimeError('Training was not successful for the OTETrainingImpl instance')
        if not self.was_export_run:
            raise RuntimeError('Export was not run for the OTETrainingImpl instance')
        if self.stored_exception_export:
            raise RuntimeError('Export was not successful for the OTETrainingImpl instance')
        if self.was_optimization_pot_run and self.stored_exception_pot:
            logger.warning('In function run_ote_optimization_pot_once: found that previous call of the function '
                           'caused exception -- re-raising it')
            raise self.stored_exception_pot

        if not self.was_optimization_pot_run:
            try:
                self._run_ote_optimization_pot(data_collector)
                self.was_optimization_pot_run = True
            except Exception as e:
                self.stored_exception_pot = e
                self.was_optimization_pot_run = True
                raise e

    def run_ote_evaluation_pot(self, data_collector, subset=Subset.VALIDATION):
        if not self.was_training_run:
            raise RuntimeError('Training was not run for the OTETrainingImpl instance')
        if self.stored_exception_training:
            raise RuntimeError('Training was not successful for the OTETrainingImpl instance')
        if not self.was_export_run:
            raise RuntimeError('Export was not run for the OTETrainingImpl instance')
        if self.stored_exception_export:
            raise RuntimeError('Export was not successful for the OTETrainingImpl instance')
        if not self.was_optimization_pot_run:
            raise RuntimeError('POT optimization was not run for the OTETrainingImpl instance')
        if self.stored_exception_pot:
            raise RuntimeError('POT optimization was not successful for the OTETrainingImpl instance')

        assert self.openvino_task_pot, 'Error: cannot find POT task'

        logger.debug('Get predictions for POT on the validation set')
        validation_dataset_pot = self.dataset.get_subset(subset)
        predicted_validation_dataset_pot = self.openvino_task_pot.infer(
            validation_dataset_pot.with_empty_annotations(),
            InferenceParameters(is_evaluation=True))
        resultset_pot = ResultSetEntity(
            model=self.optimized_model_pot,
            ground_truth_dataset=validation_dataset_pot,
            prediction_dataset=predicted_validation_dataset_pot,
        )
        logger.debug('Estimate quality on validation set')
        self.openvino_task_pot.evaluate(resultset_pot)
        evaluation_performance_pot = resultset_pot.performance

        logger.info(f'performance exported={evaluation_performance_pot}')
        score_name, score_value = performance_to_score_name_value(evaluation_performance_pot)
        data_collector.log_final_metric('evaluation_performance_pot/' + score_name, score_value)
        return evaluation_performance_pot

class BaseOTETestAction:
    _name = None

    @classmethod
    def get_name(cls):
        return cls._name

    def _check_result_prev_stages(self, results_prev_stages, list_required_stages):
        for stage_name in list_required_stages:
            if not results_prev_stages or stage_name not in results_prev_stages:
                raise RuntimeError(f'The action {self.get_name()} requires results of the stage {stage_name}, '
                                   f'but they are absent')

    def __call__(self, data_collector: DataCollector,
                 results_prev_stages: Optional[OrderedDict]=None):
        raise NotImplementedError('The main action method is not implemented')

class OTETestTrainingAction(BaseOTETestAction):
    _name = 'training'
    def __init__(self, dataset_params, template_file_path, num_training_iters, batch_size):
        self.dataset_params = dataset_params
        self.template_file_path = template_file_path
        self.num_training_iters = num_training_iters
        self.batch_size = batch_size

    @staticmethod
    def _create_environment_and_task(params, labels_schema, model_template):
        environment = TaskEnvironment(model=None, hyper_parameters=params, label_schema=labels_schema,
                                      model_template=model_template)
        logger.info('Create base Task')
        task_impl_path = model_template.entrypoints.base
        task_cls = get_task_class(task_impl_path)
        task = task_cls(task_environment=environment)
        return environment, task

    def _get_training_performance_as_score_name_value(self):
        training_performance = getattr(self.output_model, 'performance', None)
        if training_performance is None:
            raise RuntimeError('Cannot get training performance')
        return performance_to_score_name_value(training_performance)

    def _run_ote_training(self, data_collector):
        logger.debug(f'self.template_file_path = {self.template_file_path}')
        logger.debug(f'Using for train annotation file {self.dataset_params.annotations_train}')
        logger.debug(f'Using for val annotation file {self.dataset_params.annotations_val}')

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

        logger.debug('Load model template')
        self.model_template = parse_model_template(self.template_file_path)

        hyper_parameters = self.model_template.hyper_parameters.data
        set_values_as_default(hyper_parameters)

        logger.debug('Setup environment')
        params = create(hyper_parameters)
        logger.debug('Set hyperparameters')
        params.learning_parameters.num_iters = self.num_training_iters
        if self.num_training_iters < 20:
            num_checkpoints = 2
        elif self.num_training_iters < 1000:
            num_checkpoints = 10
        else:
            num_checkpoints = 30

        params.learning_parameters.batch_size = self.batch_size

        params.learning_parameters.num_checkpoints = num_checkpoints

        self.environment, self.task = self._create_environment_and_task(params,
                                                                        self.labels_schema,
                                                                        self.model_template)

        logger.debug('Train model')
        self.output_model = ModelEntity(
            self.dataset,
            self.environment.get_model_configuration(),
            model_status=ModelStatus.NOT_READY)

        self.copy_hyperparams = deepcopy(self.task._hyperparams)

        self.task.train(self.dataset, self.output_model)

        score_name, score_value = self._get_training_performance_as_score_name_value()
        logger.info(f'performance={self.output_model.performance}')
        data_collector.log_final_metric('training_performance/' + score_name, score_value)

        hyperparams_dict = convert_hyperparams_to_dict(self.copy_hyperparams)
        for k, v in hyperparams_dict.items():
            data_collector.update_metadata(k, v)

    def __call__(self, data_collector: DataCollector,
                 results_prev_stages: Optional[OrderedDict]=None):
        self._run_ote_training(data_collector)
        results = {
                'model_template': self.model_template,
                'task': self.task,
                'dataset': self.dataset,
                'environment': self.environment,
                'output_model': self.output_model,
        }
        return results

class OTETestTrainingEvaluationAction(BaseOTETestAction):
    _name = 'training_evaluation'

    def __init__(self, subset=Subset.VALIDATION):
        self.subset = subset

    def _run_ote_evaluation(self, data_collector,
                            dataset, task, trained_model):
        logger.debug('Get predictions on the validation set')
        validation_dataset = dataset.get_subset(self.subset)
        predicted_validation_dataset = task.infer(
            validation_dataset.with_empty_annotations(),
            InferenceParameters(is_evaluation=True))
        resultset = ResultSetEntity(
            model=trained_model,
            ground_truth_dataset=validation_dataset,
            prediction_dataset=predicted_validation_dataset,
        )
        logger.debug('Estimate quality on validation set')
        task.evaluate(resultset)
        evaluation_performance = resultset.performance
        logger.info(f'performance={evaluation_performance}')
        score_name, score_value = performance_to_score_name_value(evaluation_performance)
        data_collector.log_final_metric('evaluation_performance/' + score_name, score_value)

    def __call__(self, data_collector: DataCollector,
                 results_prev_stages: Optional[OrderedDict]=None):
        self._check_result_prev_stages(results_prev_stages, ['training'])

        kwargs = {
                'dataset': results_prev_stages['training']['dataset'],
                'task': results_prev_stages['training']['task'],
                'trained_model': results_prev_stages['training']['output_model'],
        }

        self._run_ote_evaluation(data_collector, **kwargs)
        results = {}
        return results

class OTETestExportAction(BaseOTETestAction):
    _name = 'export'

    def _run_ote_export(self, data_collector,
                        environment, dataset, task):
        logger.debug('Copy environment for evaluation exported model')

        self.environment_for_export = deepcopy(environment)

        logger.debug('Create exported model')
        self.exported_model = ModelEntity(
            dataset,
            self.environment_for_export.get_model_configuration(),
            model_status=ModelStatus.NOT_READY)
        logger.debug('Run export')
        task.export(ExportType.OPENVINO, self.exported_model)
        logger.debug('Set exported model into environment for export')
        self.environment_for_export.model = self.exported_model

    def __call__(self, data_collector: DataCollector,
                 results_prev_stages: Optional[OrderedDict]=None):
        self._check_result_prev_stages(results_prev_stages, ['training'])

        kwargs = {
                'environment': results_prev_stages['training']['environment'],
                'dataset': results_prev_stages['training']['dataset'],
                'task': results_prev_stages['training']['task'],
        }

        self._run_ote_export(data_collector, **kwargs)
        results = {
                'environment': self.environment_for_export,
                'exported_model': self.exported_model,
        }
        return results

def create_openvino_task(model_template, environment):
    logger.debug('Create OpenVINO Task')
    openvino_task_impl_path = model_template.entrypoints.openvino
    openvino_task_cls = get_task_class(openvino_task_impl_path)
    openvino_task = openvino_task_cls(environment)
    return openvino_task

class OTETestExportEvaluationAction(BaseOTETestAction):
    _name = 'export_evaluation'

    def __init__(self, subset=Subset.VALIDATION):
        self.subset = subset

    def _run_ote_export_evaluation(self, data_collector,
                                   model_template, dataset,
                                   environment_for_export, exported_model):
        logger.debug('Get predictions on the validation set for exported model')
        self.openvino_task = create_openvino_task(model_template, environment_for_export)
        validation_dataset = dataset.get_subset(self.subset)
        predicted_validation_dataset = self.openvino_task.infer(
            validation_dataset.with_empty_annotations(),
            InferenceParameters(is_evaluation=True))
        resultset = ResultSetEntity(
            model=exported_model,
            ground_truth_dataset=validation_dataset,
            prediction_dataset=predicted_validation_dataset,
        )
        logger.debug('Estimate quality on validation set')
        self.openvino_task.evaluate(resultset)
        evaluation_performance = resultset.performance
        logger.info(f'performance exported={evaluation_performance}')
        score_name, score_value = performance_to_score_name_value(evaluation_performance)
        data_collector.log_final_metric('evaluation_performance_exported/' + score_name, score_value)

    def __call__(self, data_collector: DataCollector,
                 results_prev_stages: Optional[OrderedDict]=None):
        self._check_result_prev_stages(results_prev_stages, ['training', 'export'])

        kwargs = {
                'model_template': results_prev_stages['training']['model_template'],
                'dataset': results_prev_stages['training']['dataset'],
                'environment_for_export': results_prev_stages['export']['environment'],
                'exported_model': results_prev_stages['export']['exported_model'],
        }

        self._run_ote_export_evaluation(data_collector, **kwargs)
        results = {}
        return results

class OTETestPotAction(BaseOTETestAction):
    _name = 'pot'

    def __init__(self, pot_subset=Subset.TRAINING):
        self.pot_subset = pot_subset

    def _run_ote_pot(self, data_collector,
                     model_template, dataset,
                     environment_for_export):
        logger.debug('Get predictions on the validation set for exported model')
        self.environment_for_pot = deepcopy(environment_for_export)
        self.openvino_task_pot = create_openvino_task(model_template, environment_for_export)

        self.optimized_model_pot = ModelEntity(
            dataset,
            self.environment_for_pot.get_model_configuration(),
            optimization_type=ModelOptimizationType.POT,
            optimization_methods=OptimizationMethod.QUANTIZATION,
            optimization_objectives={},
            precision=[ModelPrecision.INT8],
            target_device=TargetDevice.CPU,
            performance_improvement={},
            model_size_reduction=1.,
            model_status=ModelStatus.NOT_READY)
        logger.info('Run POT optimization')
        self.openvino_task_pot.optimize(
            OptimizationType.POT,
            dataset.get_subset(self.pot_subset),
            self.optimized_model_pot,
            OptimizationParameters())

    def __call__(self, data_collector: DataCollector,
                 results_prev_stages: Optional[OrderedDict]=None):
        self._check_result_prev_stages(results_prev_stages, ['export'])

        kwargs = {
                'model_template': results_prev_stages['training']['model_template'],
                'dataset': results_prev_stages['training']['dataset'],
                'environment_for_export': results_prev_stages['export']['environment'],
        }

        self._run_ote_pot(data_collector, **kwargs)
        results = {
                'openvino_task_pot': self.openvino_task_pot,
                'optimized_model_pot': self.optimized_model_pot,
        }
        return results

class OTETestPotEvaluationAction(BaseOTETestAction):
    _name = 'pot_evaluation'

    def __init__(self, subset=Subset.VALIDATION):
        self.subset = subset

    def _run_ote_pot_evaluation(self, data_collector,
                                dataset,
                                openvino_task_pot,
                                optimized_model_pot,
                                ):
        logger.debug('Get predictions for POT on the validation set')
        validation_dataset_pot = dataset.get_subset(self.subset)
        predicted_validation_dataset_pot = openvino_task_pot.infer(
            validation_dataset_pot.with_empty_annotations(),
            InferenceParameters(is_evaluation=True))
        resultset_pot = ResultSetEntity(
            model=optimized_model_pot,
            ground_truth_dataset=validation_dataset_pot,
            prediction_dataset=predicted_validation_dataset_pot,
        )
        logger.debug('Estimate quality on validation set')
        openvino_task_pot.evaluate(resultset_pot)
        evaluation_performance_pot = resultset_pot.performance

        logger.info(f'performance exported={evaluation_performance_pot}')
        score_name, score_value = performance_to_score_name_value(evaluation_performance_pot)
        data_collector.log_final_metric('evaluation_performance_pot/' + score_name, score_value)

    def __call__(self, data_collector: DataCollector,
                 results_prev_stages: Optional[OrderedDict]=None):
        self._check_result_prev_stages(results_prev_stages, ['training', 'pot'])

        kwargs = {
                'dataset': results_prev_stages['training']['dataset'],
                'openvino_task_pot': results_prev_stages['pot']['openvino_task_pot'],
                'optimized_model_pot': results_prev_stages['pot']['optimized_model_pot'],
        }

        self._run_ote_pot_evaluation(data_collector, **kwargs)
        results = {}
        return results

class OTETestStage:
    """
    OTETestStage -- auxiliary class that
    1. Allows to set up dependency between test stages: before the main action of a test stage is run, all the actions
       for the stages that are pointed in 'depends' list are called beforehand;
    2. Runs for each test stage its action only once: the main action is run inside try-except clause, and
       2.1. if the action is executed without exceptions, a flag
            `was_processed` is set, and the next time the stage is called no action is executed;
       2.2. if the action raises an exception, the exception is stored, the flag `was_processed` is set, and the next
            time the stage is called the exception is re-raised.
    """
    def __init__(self, action: BaseOTETestAction,
                 depends_stages: Optional[list]=None):
        self.was_processed = False
        self.stored_exception = None
        self.action = action
        self.depends_stages = depends_stages if depends_stages else []
        assert isinstance(self.depends_stages, list)
        assert all(isinstance(stage, OTETestStage) for stage in self.depends_stages)
        assert isinstance(self.action, BaseOTETestAction)

    @property
    def name(self):
        return self.action.get_name()

    def was_ok(self):
        return self.was_processed and (self.stored_exception is None)

    def check_is_ok(self):
        if not self.was_processed:
            raise RuntimeError(f'Stage {self.name} was not run yet for this instance of OTETrainingImpl')
        if self.was_ok():
            logger.debug(f'The stage {self.name} was already processed SUCCESSFULLY')
            return

        logger.warning(f'In stage {self.name}: found that previous call of the stage '
                       'caused exception -- re-raising it')
        raise self.stored_exception

    def run_once(self, data_collector: DataCollector, test_results_storage: OrderedDict):
        logger.debug(f'Begin stage {self.name}')
        assert isinstance(test_results_storage, OrderedDict)
        logger.debug(f'For stage "{self.name}": test_results_storage.keys = {list(test_results_storage.keys())}')

        for dep_stage in self.depends_stages:
            logger.debug(f'For stage "{self.name}": Before running dep. stage "{dep_stage.name}"')
            dep_stage.run_once(data_collector, test_results_storage)
            logger.debug(f'For stage "{self.name}": After running dep. stage "{dep_stage.name}"')

        if self.was_processed:
            self.check_is_ok()
            return

        if self.name in test_results_storage:
            raise RuntimeError(f'Error: For stage "{self.name}": another OTETestStage with name {self.name} has been run already')

        try:
            logger.debug(f'For stage "{self.name}": Before running main action')
            result_to_store = self.action(data_collector=data_collector,
                                          results_prev_stages=test_results_storage)
            logger.debug(f'For stage "{self.name}": After running main action')
            self.was_processed = True
            test_results_storage[self.name] = result_to_store
            logger.debug(f'For stage "{self.name}": after addition test_results_storage.keys = '
                         f'{list(test_results_storage.keys())}')
        except Exception as e:
            logger.debug(f'For stage "{self.name}": After running action for stage {self.name} -- CAUGHT EXCEPTION:\n{e}')
            self.stored_exception = e
            self.was_processed = True
            raise e

class OTETrainingImpl2:
    TEST_STAGES = ('training', 'training_evaluation',
                   'export', 'export_evaluation',
                   'pot', 'pot_evaluation')

    def __init__(self, dataset_params: DatasetParameters, template_file_path: str,
                 num_training_iters: int, batch_size: int):
        self.dataset_params = dataset_params
        self.template_file_path = template_file_path
        self.num_training_iters = num_training_iters
        self.batch_size = batch_size

        training_stage = OTETestStage(action=OTETestTrainingAction(self.dataset_params,
                                                                   self.template_file_path,
                                                                   self.num_training_iters,
                                                                   self.batch_size))
        training_evaluation_stage = OTETestStage(action=OTETestTrainingEvaluationAction(),
                                                 depends_stages=[training_stage])
        export_stage = OTETestStage(action=OTETestExportAction(),
                                    depends_stages=[training_stage])
        export_evaluation_stage = OTETestStage(action=OTETestExportEvaluationAction(),
                                               depends_stages=[export_stage])
        pot_stage = OTETestStage(action=OTETestPotAction(),
                                 depends_stages=[export_stage])
        pot_evaluation_stage = OTETestStage(action=OTETestPotEvaluationAction(),
                                            depends_stages=[pot_stage])

        list_all_stages = [training_stage, training_evaluation_stage,
                           export_stage, export_evaluation_stage,
                           pot_stage, pot_evaluation_stage]

        self._stages = OrderedDict((stage.name, stage) for stage in list_all_stages)
        assert list(self._stages.keys()) == list(self.TEST_STAGES)

        # test results should be kept between stages
        self.test_results_storage = OrderedDict()

    def _run_stage(self, stage_name, data_collector):
        assert stage_name in self.TEST_STAGES, f'Wrong stage_name {stage_name}'
        self._stages[stage_name].run_once(data_collector, self.test_results_storage)

    def run_training(self, data_collector):
        self._run_stage('training', data_collector)

    def run_training_evaluation(self, data_collector):
        self._run_stage('training_evaluation', data_collector)

    def run_export(self, data_collector):
        self._run_stage('export', data_collector)

    def run_export_evaluation(self, data_collector):
        self._run_stage('export_evaluation', data_collector)

    def run_pot(self, data_collector):
        self._run_stage('pot', data_collector)

    def run_pot_evaluation(self, data_collector):
        self._run_stage('pot_evaluation', data_collector)

# pytest magic
def pytest_generate_tests(metafunc):
    if metafunc.cls is None:
        return
    if not issubclass(metafunc.cls, TestOTETraining):
        return

    # It allows to filter by usecase
    usecase = metafunc.config.getoption('--test-usecase')

    argnames, argvalues, ids = metafunc.cls.get_list_of_tests(usecase)
    metafunc.parametrize(argnames, argvalues, ids=ids, scope='class')

class TestOTETraining:
    PERFORMANCE_RESULTS = None # it is required for e2e system

    DEFAULT_NUM_ITERS = 1
    DEFAULT_BATCH_SIZE = 2
    SHORT_TEST_PARAMETERS_NAMES_FOR_GENERATING_ID = OrderedDict([
            ('model_name', 'model'),
            ('dataset_name', 'dataset'),
            ('num_training_iters', 'num_iters'),
            ('batch_size', 'batch'),
            ('usecase', 'usecase'),
    ])

    # This tuple TEST_PARAMETERS_DEFINING_IMPL_BEHAVIOR describes test bunches'
    # fields that are important for creating OTETrainingImpl instance.
    #
    # It is supposed that if for the next test these parameters are the same as
    # for the previous one, the result of operations in OTETrainingImpl should
    # be kept and re-used.
    # See the fixture impl_fx and the method _update_impl_in_cache below.
    TEST_PARAMETERS_DEFINING_IMPL_BEHAVIOR = ('model_name',
                                              'dataset_name',
                                              'num_training_iters',
                                              'batch_size')

    # Note that each test bunch describes a group of similar tests
    # If 'model_name' or 'dataset_name' are lists, cartesian product of tests will be run.
    test_bunches = [
#           dict(
#               model_name=[
#                   'face-detection-0200',
#                   'face-detection-0202',
#                   'face-detection-0204',
#                   'face-detection-0205',
#                   'face-detection-0206',
#                   'face-detection-0207',
#               ],
#               dataset_name='airport_faces',
#               usecase='precommit',
#           ),
#           dict(
#               model_name=[
#                   'horizontal-text-detection-0001',
#               ],
#               dataset_name='horizontal_text_detection',
#               usecase='precommit',
#           ),
            dict(
                model_name=[
                   'gen1_mobilenet_v2-2s_ssd-256x256',
                   'gen2_mobilenetV2_SSD',
                   'gen2_mobilenetV2_ATSS',
                   'gen2_resnet50_VFNet',
                ],
                dataset_name='dataset1_tiled_shortened_500_A',
                usecase='precommit',
            ),
            dict(
                model_name=[
                   'gen3_mobilenetV2_SSD',
                   'gen3_mobilenetV2_ATSS',
                   'gen3_resnet50_VFNet',
                ],
                dataset_name='dataset1_tiled_shortened_500_A',
                usecase='precommit',
            ),
#            dict(
#                model_name=[
#                    'person-detection-0200',
#                    'person-detection-0201',
#                    'person-detection-0202',
#                    'person-detection-0203'
#                ],
#                dataset_name='airport_person',
#                usecase='precommit',
#            ),
#            dict(
#                model_name=[
#                    'person-vehicle-bike-detection-2000',
#                    'person-vehicle-bike-detection-2001',
#                    'person-vehicle-bike-detection-2002',
#                    'person-vehicle-bike-detection-2003',
#                    'person-vehicle-bike-detection-2004'
#                ],
#                dataset_name='airport_example',
#                usecase='precommit',
#            ),
#            dict(
#                model_name=[
#                    'vehicle-detection-0200',
#                    'vehicle-detection-0201',
#                    'vehicle-detection-0202',
#                    'vehicle-detection-0203',
#                ],
#                dataset_name='vehicle_detection',
#                usecase='precommit',
#            ),
    ]


    @classmethod
    def _fill_test_parameters_default_values(cls, test_parameters):
        test_parameters['num_training_iters'] = test_parameters.get('num_training_iters', cls.DEFAULT_NUM_ITERS)
        test_parameters['batch_size'] = test_parameters.get('batch_size', cls.DEFAULT_BATCH_SIZE)

    @classmethod
    def _generate_test_id(cls, test_parameters):
        id_parts = (
                f'{short_par_name}={test_parameters[par_name]}'
                for par_name, short_par_name in cls.SHORT_TEST_PARAMETERS_NAMES_FOR_GENERATING_ID.items()
        )
        return ','.join(id_parts)

    @classmethod
    def get_list_of_tests(cls, usecase: Optional[str] = None):
        """
        The functions generates the lists of values for the tests from the field test_bunches of the class.

        The function returns two lists
        * argnames -- a tuple with names of the test parameters, at the moment it is
                      a one-element tuple with the parameter name "test_parameters"
        * argvalues -- list of tuples, each tuple has the same len as argname tuple,
                       at the moment it is a one-element tuple with the dict `test_parameters`
                       that stores the parameters of the test
        * ids -- list of strings with ids corresponding the parameters of the tests
                 each id is a string generated from the corresponding test_parameters
                 value -- see the functions _generate_test_id

        The lists argvalues and ids will have the same length.

        If the parameter `usecase` is set, it makes filtering by usecase field of test bunches.
        """
        test_bunches = cls.test_bunches
        assert all(isinstance(el, dict) for el in test_bunches)

        argnames = ('test_parameters',)
        argvalues = []
        ids = []
        for el in test_bunches:
            el_model_name = el.get('model_name')
            el_dataset_name = el.get('dataset_name')
            el_usecase = el.get('usecase')
            if usecase is not None and el_usecase != usecase:
                continue
            if isinstance(el_model_name, (list, tuple)):
                model_names = el_model_name
            else:
                model_names = [el_model_name]
            if isinstance(el_dataset_name, (list, tuple)):
                dataset_names = el_dataset_name
            else:
                dataset_names = [el_dataset_name]

            model_dataset_pairs = list(itertools.product(model_names, dataset_names))

            for m, d in model_dataset_pairs:
                test_parameters = deepcopy(el)
                test_parameters['model_name'] = m
                test_parameters['dataset_name'] = d
                cls._fill_test_parameters_default_values(test_parameters)
                argvalues.append((test_parameters,))
                ids.append(cls._generate_test_id(test_parameters))

        return argnames, argvalues, ids

    @pytest.fixture(scope='class')
    def cached_from_prev_test_fx(self):
        """
        This fixture is intended for storying the impl class OTETrainingImpl and parameters
        for which the class is created.
        This object should be persistent between tests while the tests use the same parameters
        -- see the method _clean_cache_if_parameters_changed below that is used to clean
        the impl if the parameters are changed.
        """
        return dict()

    @staticmethod
    def _clean_cache_if_parameters_changed(cache, params_defining_cache):
        is_ok = True
        for k, v in params_defining_cache.items():
            is_ok = is_ok and (cache.get(k) == v)
        if is_ok:
            logger.info('TestOTETraining: parameters were not changed -- cache is kept')
            return

        for k in list(cache.keys()):
            del cache[k]
        for k, v in params_defining_cache.items():
            cache[k] = v
        logger.info('TestOTETraining: parameters were changed -- cache is cleaned')

    @classmethod
    def _update_impl_in_cache(cls, cache,
                              test_parameters,
                              dataset_definitions, template_paths):
        """
        If the main parameters of the test differs w.r.t. the previous test,
        the cache will be cleared and new instance of OTETrainingImpl will be created.
        Otherwise the previous instance of OTETrainingImpl will be re-used
        """
        if dataset_definitions is None:
            pytest.skip('The parameter "--dataset-definitions" is not set')
        params_defining_cache = {k: test_parameters[k] for k in cls.TEST_PARAMETERS_DEFINING_IMPL_BEHAVIOR}

        assert '_impl_' not in params_defining_cache, \
                'ERROR: parameters defining test behavior should not contain special key "_impl_"'

        cls._clean_cache_if_parameters_changed(cache, params_defining_cache)

        if '_impl_' not in cache:
            logger.info('TestOTETraining: creating OTETrainingImpl')

            model_name = test_parameters['model_name']
            dataset_name = test_parameters['dataset_name']
            num_training_iters = int(test_parameters['num_training_iters'])
            batch_size = int(test_parameters['batch_size'])

            dataset_params = _get_dataset_params_from_dataset_definitions(dataset_definitions, dataset_name)
            template_path = _make_path_be_abs(template_paths[model_name], template_paths[ROOT_PATH_KEY])

            cache['_impl_'] = OTETrainingImpl2(dataset_params, template_path, num_training_iters, batch_size)

        return cache['_impl_']

    @pytest.fixture
    def impl_fx(self, request, dataset_definitions_fx, template_paths_fx,
                cached_from_prev_test_fx):
        """
        This fixture returns the impl class OTETrainingImpl that should be used for the current test.
        Note that the cache from the fixture cached_from_prev_test_fx allows to store the instance of the class
        between the tests.
        If the main parameters used for this test are the same as the main parameters used for the previous test,
        the instance of the implementation class will be kept and re-used. It is helpful for tests that can
        re-use the result of operations (model training, model optimization, etc) made for the previous tests,
        if these operations are time-consuming.
        If the main parameters used for this test differs w.r.t. the previous test, a new instance of TestOTETraining
        class will be created.
        """
        cur_request_parameters = deepcopy(request.node.callspec.params)
        if 'test_parameters' not in cur_request_parameters:
            raise RuntimeError(f'Test {request.node.name} should be parametrized by parameter "test_parameters"')

        test_parameters = cur_request_parameters['test_parameters']
        impl = self._update_impl_in_cache(cached_from_prev_test_fx,
                                          test_parameters,
                                          dataset_definitions_fx, template_paths_fx)
        return impl

    @pytest.fixture
    def data_collector_fx(self, request):
        setup = deepcopy(request.node.callspec.params)
        setup["environment_name"] = os.environ.get("TT_ENVIRONMENT_NAME", "no-env")
        setup["test_type"] = os.environ.get("TT_TEST_TYPE", "no-test-type")
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

    @e2e_pytest_performance
    def test_ote_01_training(self,
                             test_parameters, # is required for impl_fx magic
                             impl_fx, data_collector_fx):
        impl_fx.run_training(data_collector_fx)

    @e2e_pytest_performance
    def test_ote_02_evaluation(self,
                               test_parameters, # is required for impl_fx magic
                               impl_fx, data_collector_fx):
        impl_fx.run_training_evaluation(data_collector_fx)

    @e2e_pytest_performance
    def test_ote_03_export(self,
                           test_parameters, # is required for impl_fx magic
                           impl_fx, data_collector_fx):
        impl_fx.run_export(data_collector_fx)

    @e2e_pytest_performance
    def test_ote_04_evaluation_exported(self,
                                        test_parameters, # is required for impl_fx magic
                                        impl_fx, data_collector_fx):
        impl_fx.run_export_evaluation(data_collector_fx)

    @e2e_pytest_performance
    def test_ote_05_optimize_pot(self,
                                 test_parameters, # is required for impl_fx magic
                                 impl_fx, data_collector_fx):
        # TODO: check that this test does all what is required
        impl_fx.run_pot(data_collector_fx)

    @e2e_pytest_performance
    def test_ote_06_evaluation_optimized_pot(self,
                                             test_parameters, # is required for impl_fx magic
                                             impl_fx, data_collector_fx):
        # TODO: check that this test does all what is required
        impl_fx.run_pot_evaluation(data_collector_fx)
