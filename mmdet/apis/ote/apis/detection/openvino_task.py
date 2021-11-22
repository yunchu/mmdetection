# Copyright (C) 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

import attr
import inspect
import json
import os
from pathlib import Path
from shutil import Error, copyfile, copytree
import sys
import subprocess
import tempfile
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from addict import Dict as ADDict
from compression.api import DataLoader
from compression.engines.ie_engine import IEEngine
from compression.graph import load_model, save_model
from compression.graph.model_utils import compress_model_weights, get_nodes_by_type
from compression.pipeline.initializer import create_pipeline
from ote_sdk.entities.annotation import Annotation, AnnotationSceneEntity, AnnotationSceneKind
from ote_sdk.entities.datasets import DatasetEntity
from ote_sdk.entities.inference_parameters import InferenceParameters, default_progress_callback
from ote_sdk.entities.label import LabelEntity
from ote_sdk.entities.model import (
    ModelStatus,
    ModelEntity,
    ModelFormat,
    OptimizationMethod,
    ModelPrecision,
)
from ote_sdk.entities.optimization_parameters import OptimizationParameters
from ote_sdk.entities.resultset import ResultSetEntity
from ote_sdk.entities.scored_label import ScoredLabel
from ote_sdk.entities.shapes.rectangle import Rectangle
from ote_sdk.entities.task_environment import TaskEnvironment
from ote_sdk.usecases.evaluation.metrics_helper import MetricsHelper
from ote_sdk.usecases.exportable_code.inference import BaseInferencer
from ote_sdk.usecases.exportable_code.prediction_to_annotation_converter import DetectionBoxToAnnotationConverter
import ote_sdk.usecases.exportable_code.demo as demo
from ote_sdk.usecases.tasks.interfaces.evaluate_interface import IEvaluationTask
from ote_sdk.usecases.tasks.interfaces.inference_interface import IInferenceTask
from ote_sdk.usecases.tasks.interfaces.optimization_interface import IOptimizationTask, OptimizationType

from openvino.inference_engine import ExecutableNetwork, IECore, InferRequest
from openvino.model_zoo.model_api.models import Model
from openvino.model_zoo.model_api.adapters import create_core, OpenvinoAdapter
from .configuration import OTEDetectionConfig
from mmdet.utils.logger import get_root_logger

from . import model_wrapers

logger = get_root_logger()


class OpenVINODetectionInferencer(BaseInferencer):
    def __init__(
        self,
        hparams: OTEDetectionConfig,
        labels: List[LabelEntity],
        model_file: Union[str, bytes],
        weight_file: Union[str, bytes, None] = None,
        device: str = "CPU",
        num_requests: int = 1,
    ):
        """
        Inferencer implementation for OTEDetection using OpenVINO backend.

        :param hparams: Hyper parameters that the model should use.
        :param labels: List of labels that was used during model training.
        :param model_file: Path OpenVINO IR model definition file.
        :param weight_file: Path OpenVINO IR model weights file.
        :param device: Device to run inference on, such as CPU, GPU or MYRIAD. Defaults to "CPU".
        :param num_requests: Maximum number of requests that the inferencer can make. Defaults to 1.

        """
        self.labels = labels
        try:
            model_adapter = OpenvinoAdapter(create_core(), model_file, weight_file, device=device, max_num_requests=num_requests)
            label_names = [label.name for label in self.labels]
            self.configuration = {**attr.asdict(hparams.inference_parameters.postprocessing,
                                  filter=lambda attr, value: attr.name not in ['header', 'description', 'type', 'visible_in_ui']),
                                  'labels': label_names}
            self.model = Model.create_model(hparams.inference_parameters.class_name.value, model_adapter, self.configuration)
        except ValueError as e:
            print(e)
        self.exec_net = self.ie.load_network(self.model.net, device_name=device)
        self.converter = DetectionBoxToAnnotationConverter(self.labels)

    def pre_process(self, image: np.ndarray) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        return self.model.preprocess(image)

    def post_process(self, prediction: Dict[str, np.ndarray], metadata: Dict[str, Any]) -> AnnotationSceneEntity:
        detections = self.model.postprocess(prediction, metadata)

        return self.converter.convert_to_annotation(detections, metadata)

    def forward(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        return self.exec_net.infer(inputs)


class OTEOpenVinoDataLoader(DataLoader):
    def __init__(self, dataset: DatasetEntity, inferencer: BaseInferencer):
        self.dataset = dataset
        self.inferencer = inferencer

    def __getitem__(self, index):
        image = self.dataset[index].numpy
        annotation = self.dataset[index].annotation_scene
        inputs, metadata = self.inferencer.pre_process(image)

        return (index, annotation), inputs, metadata

    def __len__(self):
        return len(self.dataset)


class OpenVINODetectionTask(IInferenceTask, IEvaluationTask, IOptimizationTask):
    def __init__(self, task_environment: TaskEnvironment):
        logger.info('Loading OpenVINO OTEDetectionTask')
        self.task_environment = task_environment
        self.model = self.task_environment.model
        self.confidence_threshold: float = 0.0
        self.model_name = task_environment.model_template.name.replace(" ", "_")
        self.inferencer = self.load_inferencer()
        logger.info('OpenVINO task initialization completed')

    @property
    def hparams(self):
        return self.task_environment.get_hyper_parameters(OTEDetectionConfig)

    def load_inferencer(self) -> OpenVINODetectionInferencer:
        labels = self.task_environment.label_schema.get_labels(include_empty=False)
        _hparams = self.hparams
        try:
            self.confidence_threshold = np.frombuffer(self.model.get_data("confidence_threshold"), dtype=np.float32)[0]
            _hparams.inference_parameters.postprocessing.confidence_threshold = self.confidence_threshold
        except KeyError:
            self.confidence_threshold = _hparams.inference_parameters.postprocessing.confidence_threshold
        return OpenVINODetectionInferencer(_hparams,
                                           labels,
                                           self.model.get_data("openvino.xml"),
                                           self.model.get_data("openvino.bin"))

    def infer(self, dataset: DatasetEntity, inference_parameters: Optional[InferenceParameters] = None) -> DatasetEntity:
        logger.info('Start OpenVINO inference')
        update_progress_callback = default_progress_callback
        if inference_parameters is not None:
            update_progress_callback = inference_parameters.update_progress
        dataset_size = len(dataset)
        for i, dataset_item in enumerate(dataset, 1):
            predicted_scene = self.inferencer.predict(dataset_item.numpy)
            dataset_item.append_annotations(predicted_scene.annotations)
            update_progress_callback(int(i / dataset_size * 100))
        logger.info('OpenVINO inference completed')
        return dataset

    def evaluate(self,
                 output_result_set: ResultSetEntity,
                 evaluation_metric: Optional[str] = None):
        logger.info('Start OpenVINO metric evaluation')
        if evaluation_metric is not None:
            logger.warning(f'Requested to use {evaluation_metric} metric, but parameter is ignored. Use F-measure instead.')
        output_result_set.performance = MetricsHelper.compute_f_measure(output_result_set).get_performance()
        logger.info('OpenVINO metric evaluation completed')

    def deploy(self,
               output_path: str):
        work_dir = os.path.dirname(demo.__file__)
        model_file = inspect.getfile(type(self.inferencer.model))
        parameters = {}
        parameters['name_of_model'] = self.model_name
        parameters['type_of_model'] = self.hparams.inference_parameters.class_name.value
        parameters['converter_type'] = 'DETECTION'
        parameters['model_parameters'] = self.inferencer.configuration
        name_of_package = parameters['name_of_model'].lower()
        with tempfile.TemporaryDirectory() as tempdir:
            copyfile(os.path.join(work_dir, "setup.py"), os.path.join(tempdir, "setup.py"))
            copyfile(os.path.join(work_dir, "requirements.txt"), os.path.join(tempdir, "requirements.txt"))
            copytree(os.path.join(work_dir, "demo_package"), os.path.join(tempdir, name_of_package))
            xml_path = os.path.join(tempdir, name_of_package, "model.xml")
            bin_path = os.path.join(tempdir, name_of_package, "model.bin")
            config_path = os.path.join(tempdir, name_of_package, "config.json")
            with open(xml_path, "wb") as f:
                f.write(self.model.get_data("openvino.xml"))
            with open(bin_path, "wb") as f:
                f.write(self.model.get_data("openvino.bin"))
            with open(config_path, "w") as f:
                json.dump(parameters, f)
            # generate model.py
            if (inspect.getmodule(self.inferencer.model) in
                [module[1] for module in inspect.getmembers(model_wrapers, inspect.ismodule)]):
                copyfile(model_file, os.path.join(tempdir, name_of_package, "model.py"))
            # create wheel package
            subprocess.run([sys.executable, os.path.join(tempdir, "setup.py"), 'bdist_wheel',
                            '--dist-dir', output_path, 'clean', '--all'])

    def optimize(self,
                 optimization_type: OptimizationType,
                 dataset: DatasetEntity,
                 output_model: ModelEntity,
                 optimization_parameters: Optional[OptimizationParameters]):
        logger.info('Start POT optimization')

        if optimization_type is not OptimizationType.POT:
            raise ValueError('POT is the only supported optimization type for OpenVino models')

        data_loader = OTEOpenVinoDataLoader(dataset, self.inferencer)

        with tempfile.TemporaryDirectory() as tempdir:
            xml_path = os.path.join(tempdir, "model.xml")
            bin_path = os.path.join(tempdir, "model.bin")
            with open(xml_path, "wb") as f:
                f.write(self.model.get_data("openvino.xml"))
            with open(bin_path, "wb") as f:
                f.write(self.model.get_data("openvino.bin"))

            model_config = ADDict({
                'model_name': 'openvino_model',
                'model': xml_path,
                'weights': bin_path
            })

            model = load_model(model_config)

            if get_nodes_by_type(model, ['FakeQuantize']):
                logger.warning("Model is already optimized by POT")
                output_model.model_status = ModelStatus.FAILED
                return

        engine_config = ADDict({
            'device': 'CPU'
        })

        stat_subset_size = self.hparams.pot_parameters.stat_subset_size
        preset = self.hparams.pot_parameters.preset.name.lower()

        algorithms = [
            {
                'name': 'DefaultQuantization',
                'params': {
                    'target_device': 'ANY',
                    'preset': preset,
                    'stat_subset_size': min(stat_subset_size, len(data_loader))
                }
            }
        ]

        engine = IEEngine(config=engine_config, data_loader=data_loader, metric=None)

        pipeline = create_pipeline(algorithms, engine)

        compressed_model = pipeline.run(model)

        compress_model_weights(compressed_model)

        with tempfile.TemporaryDirectory() as tempdir:
            save_model(compressed_model, tempdir, model_name="model")
            with open(os.path.join(tempdir, "model.xml"), "rb") as f:
                output_model.set_data("openvino.xml", f.read())
            with open(os.path.join(tempdir, "model.bin"), "rb") as f:
                output_model.set_data("openvino.bin", f.read())
            output_model.set_data("confidence_threshold", np.array([self.confidence_threshold], dtype=np.float32).tobytes())

        # set model attributes for quantized model
        output_model.model_status = ModelStatus.SUCCESS
        output_model.model_format = ModelFormat.OPENVINO
        output_model.optimization_type = OptimizationType.POT
        output_model.optimization_methods = [OptimizationMethod.QUANTIZATION]
        output_model.precision = [ModelPrecision.INT8]

        self.model = output_model
        self.inferencer = self.load_inferencer()
        logger.info('POT optimization completed')
