# -*- coding: utf-8 -*-

import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset as build_default_dataset
from opencood.data_utils.datasets.rsu_anchor_builder import \
    build_dataset as build_rsu_anchor_dataset
from opencood.tools import inference_utils, train_utils
from opencood.utils import eval_utils
from opencood.visualization import simple_vis

torch.multiprocessing.set_sharing_strategy('file_system')


def test_parser():
    parser = argparse.ArgumentParser(
        description="Inference with distance bucket evaluation"
    )
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Continued training path')
    parser.add_argument('--fusion_method', type=str,
                        default='intermediate',
                        help='no, no_w_uncertainty, late, early or intermediate')
    parser.add_argument('--save_vis_interval', type=int, default=40,
                        help='interval of saving visualization')
    parser.add_argument('--save_npy', action='store_true',
                        help='whether to save prediction and gt result in npy')
    parser.add_argument('--no_score', action='store_true',
                        help='whether print the score of prediction')
    parser.add_argument('--note', default="", type=str,
                        help='any other thing?')
    parser.add_argument('--rsu_anchor', action='store_true',
                        help='use the rsu-anchor dataset builder')
    parser.add_argument('--dist_bins', type=str, default='0,30,60,1e8',
                        help='comma-separated distance bin edges in meters')
    parser.add_argument('--bucket_names', type=str,
                        default='0_30m,30_60m,60m_plus',
                        help='comma-separated bucket names')
    opt = parser.parse_args()
    return opt


def build_result_stat():
    return {
        0.3: {'tp': [], 'fp': [], 'gt': 0, 'score': []},
        0.5: {'tp': [], 'fp': [], 'gt': 0, 'score': []},
        0.7: {'tp': [], 'fp': [], 'gt': 0, 'score': []}
    }


def parse_distance_buckets(dist_bins, bucket_names):
    bin_edges = [float(x.strip()) for x in dist_bins.split(',') if x.strip()]
    names = [x.strip() for x in bucket_names.split(',') if x.strip()]

    if len(bin_edges) < 2:
        raise ValueError("dist_bins must contain at least two edges")
    if len(names) != len(bin_edges) - 1:
        raise ValueError("bucket_names count must equal len(dist_bins) - 1")

    bucket_specs = []
    for i, name in enumerate(names):
        bucket_specs.append({
            'name': name,
            'min_dist': bin_edges[i],
            'max_dist': bin_edges[i + 1]
        })
    return bucket_specs


def filter_boxes_by_distance(box_tensor, score_tensor, min_dist, max_dist):
    if box_tensor is None or box_tensor.shape[0] == 0:
        return None, None

    centers = box_tensor.mean(dim=1)[:, :2]
    distances = torch.linalg.norm(centers, dim=1)

    if np.isinf(max_dist):
        keep_mask = distances >= min_dist
    else:
        keep_mask = (distances >= min_dist) & (distances < max_dist)

    if keep_mask.sum() == 0:
        return None, None

    filtered_boxes = box_tensor[keep_mask]
    filtered_scores = None if score_tensor is None else score_tensor[keep_mask]
    return filtered_boxes, filtered_scores


def eval_final_results_safe(result_stat, save_path, infer_info):
    dump_dict = {}
    metrics = {}

    for iou in [0.30, 0.50, 0.70]:
        if result_stat[iou]['gt'] == 0:
            ap = 0.0
            mrec = [0.0, 1.0]
            mpre = [0.0, 0.0]
        else:
            ap, mrec, mpre = eval_utils.calculate_ap(result_stat, iou)

        metrics[iou] = ap
        dump_dict.update({
            f'ap_{int(iou * 100):02d}': ap,
            f'mpre_{int(iou * 100):02d}': mpre,
            f'mrec_{int(iou * 100):02d}': mrec
        })

    dump_dict['gt'] = int(result_stat[0.5]['gt'])
    yaml_utils.save_yaml(
        dump_dict,
        os.path.join(save_path, f'eval_{infer_info}.yaml')
    )

    print(
        '[%s] gt=%d, AP@0.3=%.4f, AP@0.5=%.4f, AP@0.7=%.4f' %
        (
            infer_info,
            result_stat[0.5]['gt'],
            metrics[0.30],
            metrics[0.50],
            metrics[0.70]
        )
    )

    return metrics[0.30], metrics[0.50], metrics[0.70]


def main():
    opt = test_parser()
    if opt.rsu_anchor:
        opt.note += "_rsu_anchor"

    assert opt.fusion_method in ['late', 'early', 'intermediate',
                                 'no', 'no_w_uncertainty', 'single']

    hypes = yaml_utils.load_yaml(None, opt)
    hypes['validate_dir'] = hypes['test_dir']
    if "OPV2V" in hypes['test_dir'] or "v2xsim" in hypes['test_dir']:
        assert "test" in hypes['validate_dir']

    left_hand = True if ("OPV2V" in hypes['test_dir'] or
                         "V2XSET" in hypes['test_dir']) else False
    print(f"Left hand visualizing: {left_hand}")

    if 'box_align' in hypes.keys():
        hypes['box_align']['val_result'] = hypes['box_align']['test_result']

    print('Creating Model')
    model = train_utils.create_model(hypes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Loading Model from checkpoint')
    saved_path = opt.model_dir
    resume_epoch, model = train_utils.load_saved_model(saved_path, model)
    print(f"resume from {resume_epoch} epoch.")
    opt.note += f"_epoch{resume_epoch}"

    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    np.random.seed(303)

    dataset_builder = (build_rsu_anchor_dataset
                       if opt.rsu_anchor else build_default_dataset)
    print('Dataset Building')
    opencood_dataset = dataset_builder(hypes, visualize=True, train=False)
    data_loader = DataLoader(opencood_dataset,
                             batch_size=1,
                             num_workers=4,
                             collate_fn=opencood_dataset.collate_batch_test,
                             shuffle=False,
                             pin_memory=False,
                             drop_last=False)

    bucket_specs = parse_distance_buckets(opt.dist_bins, opt.bucket_names)
    bucket_result_stat = {
        bucket['name']: build_result_stat() for bucket in bucket_specs
    }

    infer_info = opt.fusion_method + opt.note

    for i, batch_data in enumerate(data_loader):
        print(f"{infer_info}_{i}")
        if batch_data is None:
            continue

        with torch.no_grad():
            batch_data = train_utils.to_device(batch_data, device)

            if opt.fusion_method == 'late':
                infer_result = inference_utils.inference_late_fusion(
                    batch_data, model, opencood_dataset
                )
            elif opt.fusion_method == 'early':
                infer_result = inference_utils.inference_early_fusion(
                    batch_data, model, opencood_dataset
                )
            elif opt.fusion_method == 'intermediate':
                infer_result = inference_utils.inference_intermediate_fusion(
                    batch_data, model, opencood_dataset
                )
            elif opt.fusion_method == 'no':
                infer_result = inference_utils.inference_no_fusion(
                    batch_data, model, opencood_dataset
                )
            elif opt.fusion_method == 'no_w_uncertainty':
                infer_result = inference_utils.inference_no_fusion_w_uncertainty(
                    batch_data, model, opencood_dataset
                )
            elif opt.fusion_method == 'single':
                infer_result = inference_utils.inference_no_fusion(
                    batch_data, model, opencood_dataset, single_gt=True
                )
            else:
                raise NotImplementedError(
                    'Only single, no, no_w_uncertainty, early, late and '
                    'intermediate fusion is supported.'
                )

            pred_box_tensor = infer_result['pred_box_tensor']
            gt_box_tensor = infer_result['gt_box_tensor']
            pred_score = infer_result['pred_score']

            for bucket in bucket_specs:
                pred_bucket, score_bucket = filter_boxes_by_distance(
                    pred_box_tensor,
                    pred_score,
                    bucket['min_dist'],
                    bucket['max_dist']
                )
                gt_bucket, _ = filter_boxes_by_distance(
                    gt_box_tensor,
                    None,
                    bucket['min_dist'],
                    bucket['max_dist']
                )

                gt_bucket = gt_bucket if gt_bucket is not None else \
                    torch.zeros((0, 8, 3), device=device)

                eval_utils.caluclate_tp_fp(
                    pred_bucket, score_bucket, gt_bucket,
                    bucket_result_stat[bucket['name']], 0.3
                )
                eval_utils.caluclate_tp_fp(
                    pred_bucket, score_bucket, gt_bucket,
                    bucket_result_stat[bucket['name']], 0.5
                )
                eval_utils.caluclate_tp_fp(
                    pred_bucket, score_bucket, gt_bucket,
                    bucket_result_stat[bucket['name']], 0.7
                )

            if opt.save_npy:
                npy_save_path = os.path.join(opt.model_dir, 'npy')
                if not os.path.exists(npy_save_path):
                    os.makedirs(npy_save_path)
                inference_utils.save_prediction_gt(
                    pred_box_tensor,
                    gt_box_tensor,
                    batch_data['ego']['origin_lidar'][0],
                    i,
                    npy_save_path
                )

            if not opt.no_score:
                infer_result.update({'score_tensor': pred_score})

            if getattr(opencood_dataset, "heterogeneous", False):
                cav_box_np, lidar_agent_record = \
                    inference_utils.get_cav_box(batch_data)
                infer_result.update({"cav_box_np": cav_box_np,
                                     "lidar_agent_record": lidar_agent_record})

            if (i % opt.save_vis_interval == 0) and (pred_box_tensor is not None):
                vis_save_path_root = os.path.join(
                    opt.model_dir, f'vis_{infer_info}_distance_buckets'
                )
                if not os.path.exists(vis_save_path_root):
                    os.makedirs(vis_save_path_root)

                vis_save_path = os.path.join(
                    vis_save_path_root, 'bev_%05d.png' % i
                )
                simple_vis.visualize(infer_result,
                                     batch_data['ego']['origin_lidar'][0],
                                     hypes['postprocess']['gt_range'],
                                     vis_save_path,
                                     method='bev',
                                     left_hand=left_hand)

        torch.cuda.empty_cache()

    summary_dict = {}
    for bucket in bucket_specs:
        bucket_name = bucket['name']
        bucket_infer_info = f"{infer_info}_{bucket_name}"
        ap30, ap50, ap70 = eval_final_results_safe(
            bucket_result_stat[bucket_name],
            opt.model_dir,
            bucket_infer_info
        )
        summary_dict[bucket_name] = {
            'range': [
                bucket['min_dist'],
                'inf' if np.isinf(bucket['max_dist']) else bucket['max_dist']
            ],
            'gt': int(bucket_result_stat[bucket_name][0.5]['gt']),
            'ap30': ap30,
            'ap50': ap50,
            'ap70': ap70
        }

    yaml_utils.save_yaml(
        summary_dict,
        os.path.join(opt.model_dir,
                     f'eval_{infer_info}_distance_buckets.yaml')
    )


if __name__ == '__main__':
    main()
