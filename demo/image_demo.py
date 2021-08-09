from argparse import ArgumentParser

from mmdet.apis import inference_detector, init_detector, show_result_pyplot


def main():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    parser.add_argument(
        '--output-file', default=None, help='Output file path')
    parser.add_argument(
        '--self-training', type=str, default='',
        help='Path to images for creating new annotation in COCO representation')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)

    if args.self_training:
        import cv2
        import json
        from os import listdir
        from os import path as osp
        from tqdm import tqdm
        from mmdet.converter import mask_to_polygon, insert_image_data, insert_annotation_data
        dataset_path = '../data/coco/annotations/instances_val2017.json'
        with open(dataset_path, 'r') as f:
            output = json.load(f)
        for k, v in output['info'].items():
            output['info'][k] = '' if isinstance(v, str) else 0
        output['licenses'] = []
        images = [osp.join(args.self_training, img) for img in listdir(args.self_training) if img.endswith('jpg')]
        output['images'] = []
        output['annotations'] = []
        segm_id = 0
        for img_id, image in enumerate(tqdm(images)):
            img = cv2.imread(image)
            h, w = img.shape[:2]
            image_data = {'id': img_id, 'name': image, 'height': h, 'width': w}
            result = inference_detector(model, img)
            #show_result_pyplot(model, img, result, score_thr=args.score_thr, output_file=args.output_file)
            is_anno = False
            for label, masks in enumerate(result[1]):
                if len(masks):
                    for i, mask in enumerate(masks):
                        polygon = mask_to_polygon(mask)
                        if not len(polygon) or result[0][label][i][-1] < 0.75:
                            continue
                        object = {'label': label + 1, 'points': polygon}
                        insert_annotation_data(img_id, segm_id, object, (h, w), output)
                        segm_id += 1
                        is_anno = True
            if is_anno:
                insert_image_data(image_data, output)
            #if len(output['images']) == coco_len:
            #    break
        with open(args.self_training + '_0089.json', 'w') as f:
            json.dump(output, f)


    # test a single image
    result = inference_detector(model, args.img)
    # show the results
    show_result_pyplot(model, args.img, result, score_thr=args.score_thr, output_file=args.output_file)


if __name__ == '__main__':
    main()
