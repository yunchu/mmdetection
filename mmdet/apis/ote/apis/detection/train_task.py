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

import copy
import io
import logging
import os
from collections import defaultdict
from typing import List, Optional

import torch
from ote_sdk.configuration import cfg_helper
from ote_sdk.configuration.helper.utils import ids_to_strings
from ote_sdk.entities.datasets import DatasetEntity
from ote_sdk.entities.metrics import (BarChartInfo, BarMetricsGroup, CurveMetric, LineChartInfo, LineMetricsGroup, MetricsGroup,
                                      ScoreMetric, VisualizationType)
from ote_sdk.entities.model import ModelEntity, ModelPrecision, ModelStatus
from ote_sdk.entities.resultset import ResultSetEntity
from ote_sdk.entities.subset import Subset
from ote_sdk.entities.train_parameters import TrainParameters, default_progress_callback
from ote_sdk.usecases.evaluation.metrics_helper import MetricsHelper
from ote_sdk.usecases.tasks.interfaces.training_interface import ITrainingTask

from mmdet.apis import train_detector
from mmdet.apis.ote.apis.detection.config_utils import prepare_for_training, set_hyperparams
from mmdet.apis.ote.apis.detection.inference_task import OTEDetectionInferenceTask
from mmdet.apis.ote.apis.detection.ote_utils import TrainingProgressCallback
from mmdet.apis.ote.extension.utils.hooks import OTELoggerHook
from mmdet.datasets import build_dataset

logger = logging.getLogger(__name__)


class OTEDetectionTrainingTask(OTEDetectionInferenceTask, ITrainingTask):

    def _generate_training_metrics(self, learning_curves, map) -> Optional[List[MetricsGroup]]:
        """
        Parses the mmdetection logs to get metrics from the latest training run

        :return output List[MetricsGroup]
        """
        output: List[MetricsGroup] = []

        # Learning curves.
        for key, curve in learning_curves.items():
            metric_curve = CurveMetric(xs=curve.x, ys=curve.y, name=key)
            visualization_info = LineChartInfo(name=key, x_axis_label="Epoch", y_axis_label=key)
            output.append(LineMetricsGroup(metrics=[metric_curve], visualization_info=visualization_info))

        # Final mAP value on the validation set.
        output.append(
            BarMetricsGroup(
                metrics=[ScoreMetric(value=map, name="mAP")],
                visualization_info=BarChartInfo("Validation score", visualization_type=VisualizationType.RADIAL_BAR)
            )
        )

        return output


    def train(self, dataset: DatasetEntity, output_model: ModelEntity, train_parameters: Optional[TrainParameters] = None):
        """ Trains a model on a dataset """

        set_hyperparams(self._config, self._hyperparams)

        train_dataset = dataset.get_subset(Subset.TRAINING)
        val_dataset = dataset.get_subset(Subset.VALIDATION)
        config = self._config

        # Create a copy of the network.
        old_model = copy.deepcopy(self._model)

        # Check for stop signal between pre-eval and training. If training is cancelled at this point,
        # old_model should be restored.
        if self._should_stop:
            logger.info('Training cancelled.')
            self._model = old_model
            self._should_stop = False
            self._is_training = False
            self._training_work_dir = None
            return

        # Run training.
        update_progress_callback = default_progress_callback
        if train_parameters is not None:
            update_progress_callback = train_parameters.update_progress
        time_monitor = TrainingProgressCallback(update_progress_callback)
        learning_curves = defaultdict(OTELoggerHook.Curve)
        training_config = prepare_for_training(config, train_dataset, val_dataset, time_monitor, learning_curves)
        self._training_work_dir = training_config.work_dir
        mm_train_dataset = build_dataset(training_config.data.train)
        self._is_training = True
        self._model.train()
        train_detector(model=self._model, dataset=mm_train_dataset, cfg=training_config, validate=True)
        logger.info("Training finished.")

        # Check for stop signal when training has stopped. If should_stop is true, training was cancelled and no new
        # model should be returned. Old train model is restored.
        if self._should_stop:
            logger.info('Training cancelled.')
            self._model = old_model
            self._should_stop = False
            self._is_training = False
            return

        # Load best weights.
        checkpoint_file_path = os.path.join(training_config.work_dir, 'best.pth')
        if not os.path.isfile(checkpoint_file_path):
            checkpoint_file_path = os.path.join(training_config.work_dir, 'latest.pth')
        checkpoint = torch.load(checkpoint_file_path)
        self._model.load_state_dict(checkpoint['state_dict'])

        # Get predictions on the validation set.
        val_preds, val_map = self._infer_detector(self._model, config, val_dataset, dump_features=False, eval=True)
        preds_val_dataset = val_dataset.with_empty_annotations()
        self._add_predictions_to_dataset(val_preds, preds_val_dataset, 0.0)
        resultset = ResultSetEntity(
            model=output_model,
            ground_truth_dataset=val_dataset,
            prediction_dataset=preds_val_dataset,
        )

        # Adjust confidence threshold.
        adaptive_threshold = self._hyperparams.postprocessing.result_based_confidence_threshold
        if adaptive_threshold:
            logger.info('Adjusting the confidence threshold')
            metric = MetricsHelper.compute_f_measure(resultset, vary_confidence_threshold=True)
            best_confidence_threshold = metric.best_confidence_threshold.value
            if best_confidence_threshold is None:
                raise ValueError(f"Cannot compute metrics: Invalid confidence threshold!")
            logger.info(f"Setting confidence threshold to {best_confidence_threshold} based on results")
            self.confidence_threshold = best_confidence_threshold
        else:
            metric = MetricsHelper.compute_f_measure(resultset, vary_confidence_threshold=False)

        if self.confidence_threshold is None:
            logger.error('Confidence threshold is set to None. Falling back to the user defined value.')
            self.confidence_threshold = self._hyperparams.postprocessing.confidence_threshold

        # Compose performance statistics.
        performance = metric.get_performance()
        performance.dashboard_metrics.extend(self._generate_training_metrics(learning_curves, val_map))
        logger.info(f'Final model performance: {str(performance)}')

        # Save resulting model.
        self.save_model(output_model)
        output_model.performance = performance
        output_model.model_status = ModelStatus.SUCCESS

        self._is_training = False


    def save_model(self, output_model: ModelEntity):
        buffer = io.BytesIO()
        hyperparams_str = ids_to_strings(cfg_helper.convert(self._hyperparams, dict, enum_to_str=True))
        labels = {label.name: label.color.rgb_tuple for label in self._labels}
        modelinfo = {'model': self._model.state_dict(), 'config': hyperparams_str, 'labels': labels,
            'confidence_threshold': self.confidence_threshold, 'VERSION': 1}
        torch.save(modelinfo, buffer)
        output_model.set_data("weights.pth", buffer.getvalue())
        output_model.precision = [ModelPrecision.FP32]


    def cancel_training(self):
        """
        Sends a cancel training signal to gracefully stop the optimizer. The signal consists of creating a
        '.stop_training' file in the current work_dir. The runner checks for this file periodically.
        The stopping mechanism allows stopping after each iteration, but validation will still be carried out. Stopping
        will therefore take some time.
        """
        logger.info("Cancel training requested.")
        self._should_stop = True
        stop_training_filepath = os.path.join(self._training_work_dir, '.stop_training')
        open(stop_training_filepath, 'a').close()
