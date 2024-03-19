# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import warnings
from argparse import ArgumentParser

import mmcv

from mmpose.apis import (inference_bottom_up_pose_model, init_pose_model,
                         vis_pose_result)
from mmpose.datasets import DatasetInfo
import json

def main():
    """Visualize the demo images."""
    parser = ArgumentParser()
    parser.add_argument('pose_config', help='Config file for detection')
    parser.add_argument('pose_checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--img-path',
        type=str,
        help='Path to an image file or a image folder.')
    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='whether to show img')
    parser.add_argument(
        '--out-img-root',
        type=str,
        default='',
        help='Root of the output img file. '
        'Default not saving the visualization images.')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--kpt-thr', type=float, default=0.3, help='Keypoint score threshold')
    parser.add_argument(
        '--pose-nms-thr',
        type=float,
        default=0.9,
        help='OKS threshold for pose NMS')
    parser.add_argument(
        '--radius',
        type=int,
        default=4,
        help='Keypoint radius for visualization')
    parser.add_argument(
        '--thickness',
        type=int,
        default=1,
        help='Link thickness for visualization')

    args = parser.parse_args()

    assert args.show or (args.out_img_root != '')

    # prepare image list
    if osp.isfile(args.img_path):
        image_list = [args.img_path]
    elif osp.isdir(args.img_path):
        image_list = [
            osp.join(args.img_path, fn) for fn in os.listdir(args.img_path)
            if fn.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp'))
        ]
    else:
        raise ValueError('Image path should be an image or image folder.'
                         f'Got invalid image path: {args.img_path}')

    # build the pose model from a config file and a checkpoint file
    pose_model = init_pose_model(
        args.pose_config, args.pose_checkpoint, device=args.device.lower())

    dataset = pose_model.cfg.data['test']['type']
    dataset_info = pose_model.cfg.data['test'].get('dataset_info', None)
    if dataset_info is None:
        warnings.warn(
            'Please set `dataset_info` in the config.'
            'Check https://github.com/open-mmlab/mmpose/pull/663 for details.',
            DeprecationWarning)
        assert (dataset == 'BottomUpCocoDataset')
    else:
        dataset_info = DatasetInfo(dataset_info)

    # optional
    return_heatmap = False

    # e.g. use ('backbone', ) to return backbone feature
    output_layer_names = None


    # process each image
    for image_name in mmcv.track_iter_progress(image_list):

        # test a single image, with a list of bboxes.
        pose_results, returned_outputs = inference_bottom_up_pose_model(
            pose_model,
            image_name,
            dataset=dataset,
            dataset_info=dataset_info,
            pose_nms_thr=args.pose_nms_thr,
            return_heatmap=return_heatmap,
            outputs=output_layer_names)
        
        # Prepare keypoints for the current image
        keypoints_json = []
        person_id = 1  # Initialize person ID for each image
        for person in pose_results:
            keypoints = []
            for keypoint in person['keypoints']:
                x, y, score = keypoint
                visibility = 2 if score > args.kpt_thr else 0
                keypoints.extend([float(x), float(y), visibility])

            person_entry = {
                "person_id": person_id,  # Assign unique person ID
                "keypoints": keypoints,
                "score": float(person['score']),
            }
            keypoints_json.append(person_entry)
            person_id += 1  # Increment person ID for the next person

        if args.out_img_root == '':
            out_file = None
        else:
            os.makedirs(args.out_img_root, exist_ok=True)
            out_file = os.path.join(
                args.out_img_root,
                f'vis_{osp.splitext(osp.basename(image_name))[0]}.jpg')

        # show the results
        vis_pose_result(
            pose_model,
            image_name,
            pose_results,
            radius=args.radius,
            thickness=args.thickness,
            dataset=dataset,
            dataset_info=dataset_info,
            kpt_score_thr=args.kpt_thr,
            show=args.show,
            out_file=out_file)
        
        # Place this after your image processing loop
        # Generate the JSON file name from the image name
        json_file_name = osp.splitext(osp.basename(image_name))[0] + '_keypoints.json'
        output_json_path = osp.join(args.out_img_root, json_file_name)
        
        # Save the keypoints for the current image
        with open(output_json_path, 'w') as f:
            json.dump(keypoints_json, f)
        
        print(f"Saved keypoints for {osp.basename(image_name)} to {output_json_path}")


if __name__ == '__main__':
    main()
