import datetime
import logging
import os
import pickle
import socket
from functools import wraps
from typing import Dict, Any

from ote_sdk.entities.dataset_item import DatasetItemEntity
from ote_sdk.entities.datasets import DatasetEntity
from ote_sdk.entities.image import Image

from mmdet.utils.logger import get_root_logger


logger = get_root_logger()


def get_dump_file_path():
    full_path = os.path.join(
        '/NOUS' if os.path.exists('/NOUS') else '/tmp',
        'debug_dumps',
        socket.gethostname(),
        datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + '.pkl')
    return full_path


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
            os.makedirs(os.path.dirname(self._debug_dump_file_path), exist_ok=True)
            with open(self._debug_dump_file_path, 'ab') as fp:
                pickle.dump(dump_dict, fp)
        return func(self, *args, **kwargs)
    return wrapped_function


def infer_debug_trace(self, dataset, inference_parameters=None):
    return {'dataset': dump_dataset(dataset)}


def evaluate_debug_trace(self, output_result_set, evaluation_metric=None):
    return {
        'output_resultset': {
            'purpose': output_result_set.purpose,
            'ground_truth_dataset' : dump_dataset(output_result_set.ground_truth_dataset),
            'prediction_dataset' : dump_dataset(output_result_set.prediction_dataset)
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


if __name__ == '__main__':
    import argparse
    from ote_sdk.entities.model import ModelEntity, ModelStatus
    from ote_sdk.entities.resultset import ResultSetEntity


    def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument('dump_path')
        return parser.parse_args()


    def main():
        args = parse_args()
        assert os.path.exists(args.dump_path)

        output_model = None
        train_dataset = None

        with open(args.dump_path, 'rb') as f:
            while True:
                print('reading dump record...')
                logger = get_root_logger()
                logger.setLevel(logging.ERROR)
                try:
                    dump = pickle.load(f)
                except EOFError:
                    print('no more records found in the dump file')
                    break
                logger.setLevel(logging.INFO)

                task = dump['task']
                # Disable debug dump when replay another debug dump
                task._task_environment.get_hyper_parameters().debug_parameters.enable_debug_dump = False
                method_args = {}

                entrypoint = dump['entrypoint']
                print('*' * 80)

                print(f'{type(task)=}, {entrypoint=}')
                print('=' * 80)

                while True:
                    action = input('[r]eplay, [s]kip or [q]uit : [r] ')
                    action = action.lower()
                    if action == '':
                        action = 'r'
                    if action not in {'r', 's', 'q'}:
                        continue
                    else:
                        break

                if action == 's':
                    print('skipping the step replay')
                    continue
                if action == 'q':
                    print('quiting dump replay session')
                    exit(0)

                print('replaying the step')

                if entrypoint == 'train':
                    method_args['dataset'] = load_dataset(dump['arguments']['dataset'])
                    train_dataset = method_args['dataset']
                    method_args['output_model'] = ModelEntity(
                        method_args['dataset'],
                        task._task_environment,
                        model_status=ModelStatus.NOT_READY)
                    output_model = method_args['output_model']
                    method_args['train_parameters'] = None
                elif entrypoint == 'infer':
                    method_args['dataset'] = load_dataset(dump['arguments']['dataset'])
                    method_args['inference_parameters'] = None
                elif entrypoint == 'export':
                    method_args['output_model'] = ModelEntity(
                        train_dataset,
                        task._task_environment,
                        model_status=ModelStatus.NOT_READY)
                    output_model = method_args['output_model']
                    method_args['export_type'] = dump['arguments']['export_type']
                elif entrypoint == 'evaluate':
                    output_model = ModelEntity(
                        DatasetEntity(),
                        task._task_environment,
                        model_status=ModelStatus.SUCCESS)
                    output_model.configuration.label_schema = task._task_environment.label_schema
                    method_args['output_result_set'] = ResultSetEntity(
                        model=output_model,
                        ground_truth_dataset=load_dataset(dump['arguments']['output_resultset']['ground_truth_dataset']),
                        prediction_dataset=load_dataset(dump['arguments']['output_resultset']['prediction_dataset'])
                    )
                    method_args['evaluation_metric'] = dump['arguments']['evaluation_metric']
                else:
                    raise RuntimeError(f'Unknown {entrypoint=}')

                output = getattr(task, entrypoint)(**method_args)
                print(f'\nOutput {type(output)=}\n\n\n\n')

    main()
