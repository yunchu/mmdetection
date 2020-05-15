import argparse

import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import collate, scatter

from mmdet.apis.inference import LoadImage
from mmdet.datasets.pipelines import Compose
from mmdet.models import build_detector
from mmdet.utils import get_model_complexity_info


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[1280, 800],
        help='input image size')
    parser.add_argument('--ptflops', action='store_true')
    args = parser.parse_args()
    return args


def get_fake_input(cfg, orig_img_shape=(128, 128, 3), device='cuda'):
    test_pipeline = [LoadImage()] + cfg.data.test.pipeline[1:]
    test_pipeline = Compose(test_pipeline)
    data = dict(img=np.zeros(orig_img_shape, dtype=np.uint8))
    data = test_pipeline(data)
    data = scatter(collate([data], samples_per_gpu=1), [device])[0]
    data['return_loss'] = False
    return data


def main():

    args = parse_args()

    if len(args.shape) == 1:
        input_shape = (3, args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        input_shape = (3, ) + tuple(args.shape)
    else:
        input_shape = (3, 256, 256)

    cfg = Config.fromfile(args.config)
    model = build_detector(cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg).cuda()
    model.eval()

    if not args.ptflops:
        if hasattr(model, 'forward_dummy'):
            model.forward = model.forward_dummy
        else:
            raise NotImplementedError(
                'FLOPs counter is currently not currently supported with {}'.
                format(model.__class__.__name__))

    if args.ptflops:
        device = next(model.parameters()).device
        with torch.no_grad():
            flops, params = get_model_complexity_info(
                model, input_shape, as_strings=True, print_per_layer_stat=True,
                input_constructor=lambda sh: get_fake_input(cfg, [sh[1], sh[2], sh[0]], device))
            print('Flops:  ' + flops)
            print('Params: ' + params)
    else:
        flops, params = get_model_complexity_info(model, input_shape)
        split_line = '=' * 30
        print('{0}\nInput shape: {1}\nFlops: {2}\nParams: {3}\n{0}'.format(
            split_line, input_shape, flops, params))
        print('!!!Please be cautious if you use the results in papers. '
            'You may need to check if all ops are supported and verify that the '
            'flops computation is correct.')


if __name__ == '__main__':
    main()
