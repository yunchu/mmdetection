import logging
import pickle
from functools import wraps
from typing import Dict, Any

from ote_sdk.entities.dataset_item import DatasetItemEntity
from ote_sdk.entities.datasets import DatasetEntity
from ote_sdk.entities.image import Image

logger = logging.getLogger(__name__)


def debug_trace(func):
    @wraps(func)
    def wrapped_function(self, *args, **kwargs):
        class_name = self.__class__.__name__
        func_name = func.__name__
        if self._hyperparams.debug_parameters.enable_debug_dump:
            dump_dict = {
                'class_name': class_name,
                'entrypoint': func_name,
                'task': self,
            }
            if func_name not in debug_trace_registry:
                raise ValueError(f'Debug tracing is not implemented for {func_name} method.')
            dump_dict['arguments'] = debug_trace_registry[func_name](self, *args, **kwargs)
            logger.warning(f'Saving debug dump for {class_name}.{func_name} call to {self._debug_dump_file_path}')
            with open(self._debug_dump_file_path, 'ab') as fp:
                pickle.dump(dump_dict, fp)
        return func(self, *args, **kwargs)
    return wrapped_function


def infer_debug_trace(self, dataset, inference_parameters=None):
    return {'dataset': dump_dataset(dataset)}


def evaluate_debug_trace(self, output_resultset, evaluation_metric=None):
    return {
        'output_resultset': {
            'purpose': output_resultset.purpose,
            'ground_truth_dataset' : dump_dataset(output_resultset.ground_truth_dataset),
            'prediction_dataset' : dump_dataset(output_resultset.prediction_dataset)
        },
        'evaluation_metric': evaluation_metric,
    }

def export_debug_trace(self, export_type, output_model):
    return {
        'export_type': export_type
    }

def train_debug_trace(self, dataset, output_model, train_parameters=None):
    return {
        'dataset': dump_dataset(dataset),
        'train_parameters': None if train_parameters is None else {'resume': train_parameters.resume}
    }

debug_trace_registry = {
    'infer': infer_debug_trace,
    'train': train_debug_trace,
    'evaluate': evaluate_debug_trace,
    'export': export_debug_trace,
}


def dump_dataset_item(item: DatasetItemEntity):
    dump = {
        'subset': item.subset,
        'numpy': item.numpy,
        'roi': item.roi,
        'annotation_scene': item.annotation_scene
    }
    return dump


def load_dataset_item(dump: Dict[str, Any]):
    return DatasetItemEntity(
        media=Image(dump['numpy']),
        annotation_scene=dump['annotation_scene'],
        roi=dump['roi'],
        subset=dump['subset'])


def dump_dataset(dataset: DatasetEntity):
    dump = {
        'purpose': dataset.purpose,
        'items': list(dump_dataset_item(item) for item in dataset)
    }
    return dump


def load_dataset(dump: Dict[str, Any]):
    return DatasetEntity(
        items=[load_dataset_item(i) for i in dump['items']],
        purpose=dump['purpose'])

