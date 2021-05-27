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

import abc
import argparse
import yaml
import importlib
import os
import os.path as osp
import sys
import json
from pathlib import Path
import shutil
import random
from typing import Tuple, Union, List

import cv2 as cv
from tqdm import tqdm
from zipfile import ZipFile

from mmdet.datasets import CocoDataset
from sc_sdk.entities.analyse_parameters import AnalyseParameters
from sc_sdk.entities.annotation import Annotation, AnnotationKind
from sc_sdk.entities.datasets import Dataset, Subset, NullDataset
from sc_sdk.entities.id import ID
from sc_sdk.entities.image import Image
from sc_sdk.entities.label import ScoredLabel, Label
from sc_sdk.entities.project import Project, NullProject
from sc_sdk.entities.resultset import ResultSet
from sc_sdk.entities.shapes.box import Box
from sc_sdk.entities.task_environment import TaskEnvironment
from sc_sdk.entities.url import URL
from sc_sdk.logging import logger_factory
from sc_sdk.tests.test_helpers import generate_training_dataset_of_all_annotated_media_in_project
from sc_sdk.usecases.repos import *
from sc_sdk.usecases.adapters.binary_interpreters import RAWBinaryInterpreter
from sc_sdk.utils.project_factory import ProjectFactory
from sc_sdk.communication.mappers.mongodb_mapper import LabelToMongo
from sc_sdk.entities.optimized_model import OpenVINOModel, OptimizedModel, Precision

from mmdet.apis.ote.apis.detection import MMObjectDetectionTask


logger = logger_factory.get_logger("Sample")


def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('template_file_path', help='path to template file')
    args = parser.parse_args()
    return args


def load_template(path):
    with open(path) as f:
        template = yaml.full_load(f)
    return template


def create_project(projectname, taskname, classes):
    project = ProjectFactory().create_project_single_task(name=projectname, description="",
        label_names=classes, task_name=taskname)
    ProjectRepo().save(project)
    logger.info(f'New project created {str(project)}')
    return project

def load_project(projectname, taskname, classes):
    project = ProjectRepo().get_latest_by_name(projectname)
    if isinstance(project, NullProject):
        project = create_project(projectname, taskname, classes)
    else:
        logger.info(f'Existing project loaded {str(project)}')
    return project

def get_label(x, all_labels):
    label_name = CocoDataset.CLASSES[x]
    return [label for label in all_labels if label.name == label_name][0]

def create_coco_dataset(project, cfg=None):
    pipeline = [dict(type='LoadImageFromFile'), dict(type='LoadAnnotations', with_bbox=True)]
    coco_dataset = CocoDataset(ann_file='data/coco/annotations/instances_val2017.json',
        img_prefix='data/coco/val2017/', pipeline=pipeline)

    logger.info(f'Loading images and annotation from {str(coco_dataset)} to repos')

    for datum in tqdm(coco_dataset):
        imdata = datum['img']
        imshape = imdata.shape
        image = Image(name=datum['ori_filename'], project=project, numpy=imdata)
        ImageRepo(project).save(image)

        gt_bboxes = datum['gt_bboxes']
        gt_labels = datum['gt_labels']

        shapes = []
        for label, bbox in zip(gt_labels, gt_bboxes):
            project_label = get_label(label, project.get_labels())
            shapes.append(
                Box(x1=float(bbox[0] / imshape[1]),
                    y1=float(bbox[1] / imshape[0]),
                    x2=float(bbox[2] / imshape[1]),
                    y2=float(bbox[3] / imshape[0]),
                    labels=[ScoredLabel(project_label)]))
        annotation = Annotation(kind=AnnotationKind.ANNOTATION, media_identifier=image.media_identifier, shapes=shapes)
        AnnotationRepo(project).save(annotation)

    dataset = generate_training_dataset_of_all_annotated_media_in_project(project)
    DatasetRepo(project).save(dataset)
    logger.info(f'New dataset created {dataset}')
    return dataset

def load_dataset(project, dataset_id=None):
    dataset = NullDataset()
    if dataset_id is not None:
        dataset = DatasetRepo(project).get_by_id(dataset_id)
    if isinstance(dataset, NullDataset):
        dataset = create_coco_dataset(project)
    else:
        logger.info(f'Existing dataset loaded {str(dataset)}')
    return dataset

def get_task_class(path):
    module_name, class_name = path.rsplit('.', 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)

def zip_folder_or_file(zip_file: ZipFile, path: str, folder_prefix: str = "",
                       skip_parent: bool = False):
    """
    Recursively (in case path is a folder) put the content of file_path into zip_file.

    :param zip_file: the zip file object
    :param path: path to the folder/file to be added to the zip file
    :param folder_prefix: the directory inside the zip file in which the data will be stored
    :param skip_parent: if this is set to True, only the content of the path will be stored inside folder_prefix,
        not the directory.
    """
    dir_, original_filename = os.path.split(path)
    if not os.path.isdir(path):
        zip_file.write(path, os.path.join(folder_prefix, original_filename))
    else:
        if not skip_parent:
            # only update folder prefix if skip_parent is False
            folder_prefix = os.path.join(folder_prefix, original_filename)

        for filename in os.listdir(path):
            zip_folder_or_file(zip_file, os.path.join(path, filename),
                               folder_prefix, skip_parent=False)

class IZippedObjectEntry:
    """
    Abstract class representing objects which can be written to a zip file.
    """

    @abc.abstractmethod
    def __repr__(self):
        pass

    @abc.abstractmethod
    def write_to_zip_file(self, zip_file: ZipFile):
        """
        Write the object to a given zip file.

        :param zip_file: Zip file
        """
        pass

class ZippedFileEntry(IZippedObjectEntry):
    """
    This class represents one file which will be zipped and exported.

    :param filepath: the filepath inside the zip file where data can be found.
    :param data: this data will be stored inside a file whose path as defined in filepath.
    """

    def __init__(self, filepath: str, data: Union[str, bytes]):
        self.filepath: str = filepath
        self.data: Union[str, bytes] = data

    def __repr__(self):
        return f"ZippedFileEntry({self.filepath}, {len(self.data)} bytes)"

    def write_to_zip_file(self, zip_file: ZipFile):
        zip_file.writestr(self.filepath, self.data)

class ZippedFolderEntry(IZippedObjectEntry):
    """
    This class represents a folder which will be zipped and exported.

    :param filepath: the filepath inside the zip file where data can be found.
    :param data: the URL pointing to a folder in filesystem
    :param content_only: set to True to only copy the content of the data inside URL,
        set to False to also copy the folder the URL points to.
    """

    def __init__(self, filepath: str, data: URL, content_only: bool = False):
        self.filepath: str = filepath
        self.data: URL = data
        self.content_only: bool = content_only

    def __repr__(self):
        return f"ZippedFolderEntry({self.filepath}, folder={self.data.path})"

    def write_to_zip_file(self, zip_file: ZipFile):
        zip_folder_or_file(zip_file=zip_file, path=self.data.path, folder_prefix=self.filepath,
                           skip_parent=self.content_only)

class OptimizedModelExporter:
    @staticmethod
    def export_optimized_model(root_project: Project, optimized_model: OptimizedModel):
        try:
            binary_interpreter = RAWBinaryInterpreter()
            openvino_xml_data = BinaryRepo(root_project).get_by_url(optimized_model.openvino_xml_url,
                                                                    binary_interpreter=binary_interpreter)
            openvino_bin_data = BinaryRepo(root_project).get_by_url(optimized_model.openvino_bin_url,
                                                                    binary_interpreter=binary_interpreter)
            yield ZippedFileEntry(f"optimized models/{optimized_model.precision.name}/inference_model.xml",
                                    openvino_xml_data)
            yield ZippedFileEntry(f"optimized models/{optimized_model.precision.name}/inference_model.bin",
                                    openvino_bin_data)
            label_data = OptimizedModelExporter.generate_label_data(optimized_model.model)
            yield ZippedFileEntry(f"optimized models/{optimized_model.precision.name}/labels.json", label_data)
        except FileNotFoundError:
            logger.warning(f"Failed to export the optimized model {optimized_model.name} "
                            f"because the file is no longer available.")

    @staticmethod
    def generate_label_data(model) -> str:
        labels = model.configuration.labels
        mapped_labels = []
        for label in labels:
            # FIXME.
            mapped_labels.append(LabelToMongo().forward(label))
        for label in mapped_labels:
            label["_id"] = str(label["_id"])
            label["task_id"] = str(label["task_id"])
            label["creation_date"] = label["creation_date"].isoformat()
        return json.dumps(mapped_labels)


def main(args):
    template = load_template(args.template_file_path)
    template['hyper_parameters']['params'].setdefault('algo_backend', {})['template'] = args.template_file_path
    task_impl_path = template['task']['impl']
    task_cls = get_task_class(task_impl_path)

    projectname = 'otedet-sample-project'
    taskname = 'otedet-task'
    project = load_project(projectname, taskname, CocoDataset.CLASSES)
    print('Tasks:', [task.task_name for task in project.tasks])

    dataset = load_dataset(project, dataset_id=ID('60ac24a07f5af5273658a814'))
    print(dataset)
    # dataset = create_coco_dataset(project)
    print(f"train dataset: {len(dataset.get_subset(Subset.TRAINING))} items")
    print(f"validation dataset: {len(dataset.get_subset(Subset.VALIDATION))} items")

    environment = TaskEnvironment(project=project, task_node=project.tasks[-1])
    params = task_cls.get_configurable_parameters(environment)
    task_cls.apply_template_configurable_parameters(params, template)
    params.algo_backend.template.value = args.template_file_path
    environment.set_configurable_parameters(params)

    task = task_cls(task_environment=environment)

    # Tweak parameters.
    params = task.get_configurable_parameters(environment)
    # params.learning_parameters.learning_rate.value = 1e-3
    params.learning_parameters.learning_rate_schedule.value = 'cyclic'
    # params.learning_parameters.learning_rate_warmup_iters.value = 0
    params.learning_parameters.batch_size.value = 32
    params.learning_parameters.num_epochs.value = 1
    environment.set_configurable_parameters(params)
    task.update_configurable_parameters(environment)

    logger.info('Start model training... [ROUND 0]')
    model = task.train(dataset=dataset)
    logger.info('Model training finished [ROUND 0]')

    validation_dataset = dataset.get_subset(Subset.VALIDATION)
    predicted_validation_dataset = task.analyse(
        validation_dataset.with_empty_annotations(),
        AnalyseParameters(is_evaluation=True))
    resultset = ResultSet(
        model=model,
        ground_truth_dataset=validation_dataset,
        prediction_dataset=predicted_validation_dataset,
    )

    performance = task.compute_performance(resultset)
    resultset.performance = performance

    print(resultset.performance)

    optimized_model = task.optimize_loaded_model()[0]
    OptimizedModelExporter.export_optimized_model(project, optimized_model)


if __name__ == '__main__':
    args = parse_args()
    sys.exit(main(args) or 0)
