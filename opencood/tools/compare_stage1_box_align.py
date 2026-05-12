import argparse
import copy
import json
import os
from collections import OrderedDict

import numpy as np

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.models.sub_modules.box_align_v2 import \
    box_alignment_relative_sample_np


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare SECOND and DSVT stage1_boxes.json for "
                    "second-stage box alignment quality.")
    parser.add_argument(
        "--second-json",
        type=str,
        required=True,
        help="Path to stage1_boxes.json generated from SECOND.")
    parser.add_argument(
        "--dsvt-json",
        type=str,
        required=True,
        help="Path to stage1_boxes.json generated from DSVT.")
    parser.add_argument(
        "--coalign-yaml",
        type=str,
        default="opencood/hypes_yaml/v2xset/lidar_only_with_noise/"
                "coalign/pointpillar_coalign.yaml",
        help="Second-stage CoAlign yaml used to read box_align args and "
             "default noise setting.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="opencood/logs/stage1_align_compare",
        help="Directory to save summaries.")
    parser.add_argument(
        "--tag",
        type=str,
        default="compare",
        help="Prefix for output files.")
    parser.add_argument(
        "--num-trials",
        type=int,
        default=5,
        help="Number of random noise trials per sample.")
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Base random seed.")
    parser.add_argument(
        "--pos-std",
        type=float,
        default=None,
        help="Override translation std in meters. Defaults to yaml noise.")
    parser.add_argument(
        "--rot-std",
        type=float,
        default=None,
        help="Override yaw std in degrees. Defaults to yaml noise.")
    parser.add_argument(
        "--pos-mean",
        type=float,
        default=None,
        help="Override translation mean in meters. Defaults to yaml noise.")
    parser.add_argument(
        "--rot-mean",
        type=float,
        default=None,
        help="Override yaw mean in degrees. Defaults to yaml noise.")
    parser.add_argument(
        "--limit-samples",
        type=int,
        default=0,
        help="Only evaluate the first N common samples. 0 means all.")
    return parser.parse_args()


def load_eval_config(args):
    hypes = yaml_utils.load_yaml(args.coalign_yaml, None)
    noise_args = hypes.get("noise_setting", {}).get("args", {})
    box_align_args = copy.deepcopy(hypes.get("box_align", {}).get("args", {}))

    config = {
        "pos_std": args.pos_std if args.pos_std is not None
        else noise_args.get("pos_std", 0.2),
        "rot_std": args.rot_std if args.rot_std is not None
        else noise_args.get("rot_std", 0.2),
        "pos_mean": args.pos_mean if args.pos_mean is not None
        else noise_args.get("pos_mean", 0.0),
        "rot_mean": args.rot_mean if args.rot_mean is not None
        else noise_args.get("rot_mean", 0.0),
        "box_align_args": box_align_args,
    }
    return config


def normalize_angle_deg(angle):
    return np.minimum(np.abs(angle), 360 - np.abs(angle))


def calc_pose_error(clean_pose_dof3, pose_dof3):
    diff = np.abs(clean_pose_dof3 - pose_dof3)
    diff[:, 2] = normalize_angle_deg(diff[:, 2])
    trans = np.linalg.norm(diff[:, :2], axis=1)
    rot = diff[:, 2]
    return trans, rot


def build_noise(clean_pose, rng, pos_std, rot_std, pos_mean, rot_mean):
    noisy_pose = clean_pose.copy()
    agent_num = clean_pose.shape[0]
    if agent_num <= 1:
        return noisy_pose

    noisy_pose[1:, 0] += rng.normal(pos_mean, pos_std, size=agent_num - 1)
    noisy_pose[1:, 1] += rng.normal(pos_mean, pos_std, size=agent_num - 1)
    noisy_pose[1:, 4] += rng.normal(rot_mean, rot_std, size=agent_num - 1)
    return noisy_pose


def prepare_sample_content(content):
    return {
        "pred_corners_list": [
            np.array(corners, dtype=np.float64)
            for corners in content["pred_corner3d_np_list"]
        ],
        "uncertainty_list": [
            np.array(uncertainty, dtype=np.float64)
            for uncertainty in content["uncertainty_np_list"]
        ],
        "lidar_pose_clean": np.array(
            content["lidar_pose_clean_np"], dtype=np.float64),
        "cav_id_list": [str(cav_id) for cav_id in content["cav_id_list"]],
    }


def sample_stats_template():
    return {
        "sample_count": 0,
        "trial_count": 0,
        "agent_count": 0,
        "non_ego_agent_count": 0,
        "box_count_total": 0,
        "nonempty_agent_count": 0,
        "trans_before": [],
        "trans_after": [],
        "rot_before": [],
        "rot_after": [],
        "trans_improve": [],
        "rot_improve": [],
        "trans_better_trials": 0,
        "rot_better_trials": 0,
        "unchanged_trials": 0,
    }


def summarize_metric(values):
    arr = np.array(values, dtype=np.float64)
    if arr.size == 0:
        return OrderedDict([
            ("mean", None),
            ("median", None),
            ("p90", None),
            ("p95", None),
        ])
    return OrderedDict([
        ("mean", float(arr.mean())),
        ("median", float(np.median(arr))),
        ("p90", float(np.quantile(arr, 0.9))),
        ("p95", float(np.quantile(arr, 0.95))),
    ])


def finalize_stats(stats):
    summary = OrderedDict()
    summary["sample_count"] = stats["sample_count"]
    summary["trial_count"] = stats["trial_count"]
    summary["agent_count"] = stats["agent_count"]
    summary["non_ego_agent_count"] = stats["non_ego_agent_count"]
    summary["box_count_total"] = stats["box_count_total"]
    summary["avg_boxes_per_sample"] = (
        stats["box_count_total"] / stats["sample_count"]
        if stats["sample_count"] > 0 else None
    )
    summary["avg_nonempty_agents_per_sample"] = (
        stats["nonempty_agent_count"] / stats["sample_count"]
        if stats["sample_count"] > 0 else None
    )
    summary["trans_before"] = summarize_metric(stats["trans_before"])
    summary["trans_after"] = summarize_metric(stats["trans_after"])
    summary["trans_improve"] = summarize_metric(stats["trans_improve"])
    summary["rot_before_deg"] = summarize_metric(stats["rot_before"])
    summary["rot_after_deg"] = summarize_metric(stats["rot_after"])
    summary["rot_improve_deg"] = summarize_metric(stats["rot_improve"])
    summary["trans_better_ratio"] = (
        stats["trans_better_trials"] / stats["trial_count"]
        if stats["trial_count"] > 0 else None
    )
    summary["rot_better_ratio"] = (
        stats["rot_better_trials"] / stats["trial_count"]
        if stats["trial_count"] > 0 else None
    )
    summary["unchanged_ratio"] = (
        stats["unchanged_trials"] / stats["trial_count"]
        if stats["trial_count"] > 0 else None
    )
    return summary


def evaluate_model(model_name, data_dict, sample_ids, config, num_trials, seed):
    stats = sample_stats_template()
    per_sample_rows = []
    align_args = copy.deepcopy(config["box_align_args"])

    for sample_idx in sample_ids:
        content = prepare_sample_content(data_dict[sample_idx])
        pred_corners_list = content["pred_corners_list"]
        uncertainty_list = content["uncertainty_list"]
        lidar_pose_clean = content["lidar_pose_clean"]
        lidar_pose_clean_dof3 = lidar_pose_clean[:, [0, 1, 4]]

        stats["sample_count"] += 1
        stats["agent_count"] += int(lidar_pose_clean.shape[0])
        stats["non_ego_agent_count"] += max(int(lidar_pose_clean.shape[0]) - 1, 0)
        stats["box_count_total"] += int(sum(len(x) for x in pred_corners_list))
        stats["nonempty_agent_count"] += int(sum(len(x) > 0 for x in pred_corners_list))

        sample_trans_before = []
        sample_trans_after = []
        sample_rot_before = []
        sample_rot_after = []

        for trial_idx in range(num_trials):
            stats["trial_count"] += 1
            rng = np.random.default_rng(seed + trial_idx * 1000003 + int(sample_idx))
            noisy_pose = build_noise(
                lidar_pose_clean,
                rng,
                config["pos_std"],
                config["rot_std"],
                config["pos_mean"],
                config["rot_mean"],
            )
            refined_pose_dof3 = box_alignment_relative_sample_np(
                pred_corners_list,
                noisy_pose,
                uncertainty_list=uncertainty_list,
                **align_args
            )
            noisy_pose_dof3 = noisy_pose[:, [0, 1, 4]]

            before_trans, before_rot = calc_pose_error(
                lidar_pose_clean_dof3[1:], noisy_pose_dof3[1:])
            after_trans, after_rot = calc_pose_error(
                lidar_pose_clean_dof3[1:], refined_pose_dof3[1:])

            if before_trans.size == 0:
                continue

            before_trans_mean = float(before_trans.mean())
            after_trans_mean = float(after_trans.mean())
            before_rot_mean = float(before_rot.mean())
            after_rot_mean = float(after_rot.mean())

            stats["trans_before"].append(before_trans_mean)
            stats["trans_after"].append(after_trans_mean)
            stats["rot_before"].append(before_rot_mean)
            stats["rot_after"].append(after_rot_mean)
            stats["trans_improve"].append(before_trans_mean - after_trans_mean)
            stats["rot_improve"].append(before_rot_mean - after_rot_mean)

            sample_trans_before.append(before_trans_mean)
            sample_trans_after.append(after_trans_mean)
            sample_rot_before.append(before_rot_mean)
            sample_rot_after.append(after_rot_mean)

            if after_trans_mean < before_trans_mean:
                stats["trans_better_trials"] += 1
            if after_rot_mean < before_rot_mean:
                stats["rot_better_trials"] += 1
            if np.allclose(refined_pose_dof3, noisy_pose_dof3, atol=1e-6):
                stats["unchanged_trials"] += 1

        per_sample_rows.append(OrderedDict([
            ("sample_idx", int(sample_idx)),
            ("model", model_name),
            ("agent_num", int(lidar_pose_clean.shape[0])),
            ("nonempty_agent_num", int(sum(len(x) > 0 for x in pred_corners_list))),
            ("box_num", int(sum(len(x) for x in pred_corners_list))),
            ("trans_before_mean", float(np.mean(sample_trans_before))
             if sample_trans_before else None),
            ("trans_after_mean", float(np.mean(sample_trans_after))
             if sample_trans_after else None),
            ("trans_improve_mean", float(np.mean(np.array(sample_trans_before) -
                                                 np.array(sample_trans_after)))
             if sample_trans_before else None),
            ("rot_before_mean_deg", float(np.mean(sample_rot_before))
             if sample_rot_before else None),
            ("rot_after_mean_deg", float(np.mean(sample_rot_after))
             if sample_rot_after else None),
            ("rot_improve_mean_deg", float(np.mean(np.array(sample_rot_before) -
                                                   np.array(sample_rot_after)))
             if sample_rot_before else None),
        ]))

    return finalize_stats(stats), per_sample_rows


def validate_common_samples(second_data, dsvt_data):
    common = sorted(set(second_data.keys()) & set(dsvt_data.keys()), key=lambda x: int(x))
    usable = []
    skipped = []
    for sample_idx in common:
        second_content = second_data[sample_idx]
        dsvt_content = dsvt_data[sample_idx]
        if second_content is None or dsvt_content is None:
            skipped.append((sample_idx, "sample content is None"))
            continue

        second_cavs = [str(x) for x in second_content["cav_id_list"]]
        dsvt_cavs = [str(x) for x in dsvt_content["cav_id_list"]]
        if second_cavs != dsvt_cavs:
            skipped.append((sample_idx, "cav_id_list mismatch"))
            continue

        second_pose = np.array(second_content["lidar_pose_clean_np"], dtype=np.float64)
        dsvt_pose = np.array(dsvt_content["lidar_pose_clean_np"], dtype=np.float64)
        if second_pose.shape != dsvt_pose.shape or not np.allclose(second_pose, dsvt_pose):
            skipped.append((sample_idx, "lidar_pose_clean mismatch"))
            continue

        usable.append(sample_idx)
    return usable, skipped


def write_json(path, content):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(content, f, indent=2)


def write_csv(path, rows):
    if not rows:
        with open(path, "w", encoding="utf-8") as f:
            f.write("")
        return

    headers = list(rows[0].keys())
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(headers) + "\n")
        for row in rows:
            values = []
            for key in headers:
                value = row[key]
                if value is None:
                    values.append("")
                else:
                    values.append(str(value))
            f.write(",".join(values) + "\n")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    config = load_eval_config(args)
    second_data = read_json(args.second_json)
    dsvt_data = read_json(args.dsvt_json)
    sample_ids, skipped = validate_common_samples(second_data, dsvt_data)

    if args.limit_samples > 0:
        sample_ids = sample_ids[:args.limit_samples]

    second_summary, second_rows = evaluate_model(
        "SECOND", second_data, sample_ids, config, args.num_trials, args.seed)
    dsvt_summary, dsvt_rows = evaluate_model(
        "DSVT", dsvt_data, sample_ids, config, args.num_trials, args.seed)

    diff_summary = OrderedDict([
        ("trans_after_mean_gap_m",
         None if second_summary["trans_after"]["mean"] is None
         or dsvt_summary["trans_after"]["mean"] is None
         else dsvt_summary["trans_after"]["mean"] -
         second_summary["trans_after"]["mean"]),
        ("trans_improve_mean_gap_m",
         None if second_summary["trans_improve"]["mean"] is None
         or dsvt_summary["trans_improve"]["mean"] is None
         else dsvt_summary["trans_improve"]["mean"] -
         second_summary["trans_improve"]["mean"]),
        ("rot_after_mean_gap_deg",
         None if second_summary["rot_after_deg"]["mean"] is None
         or dsvt_summary["rot_after_deg"]["mean"] is None
         else dsvt_summary["rot_after_deg"]["mean"] -
         second_summary["rot_after_deg"]["mean"]),
        ("rot_improve_mean_gap_deg",
         None if second_summary["rot_improve_deg"]["mean"] is None
         or dsvt_summary["rot_improve_deg"]["mean"] is None
         else dsvt_summary["rot_improve_deg"]["mean"] -
         second_summary["rot_improve_deg"]["mean"]),
    ])

    report = OrderedDict([
        ("config", OrderedDict([
            ("second_json", args.second_json),
            ("dsvt_json", args.dsvt_json),
            ("coalign_yaml", args.coalign_yaml),
            ("num_trials", args.num_trials),
            ("seed", args.seed),
            ("pos_std", config["pos_std"]),
            ("rot_std", config["rot_std"]),
            ("pos_mean", config["pos_mean"]),
            ("rot_mean", config["rot_mean"]),
            ("box_align_args", config["box_align_args"]),
            ("sample_count", len(sample_ids)),
        ])),
        ("skipped_samples", [
            OrderedDict([("sample_idx", sample_idx), ("reason", reason)])
            for sample_idx, reason in skipped
        ]),
        ("SECOND", second_summary),
        ("DSVT", dsvt_summary),
        ("difference", diff_summary),
    ])

    json_path = os.path.join(args.output_dir, f"{args.tag}_summary.json")
    csv_path = os.path.join(args.output_dir, f"{args.tag}_per_sample.csv")
    write_json(json_path, report)
    write_csv(csv_path, second_rows + dsvt_rows)

    print("=== Stage1 Box Alignment Comparison ===")
    print(f"Common usable samples: {len(sample_ids)}")
    print(f"Skipped common samples: {len(skipped)}")
    print(f"Noise setting: pos_std={config['pos_std']}, rot_std={config['rot_std']}, "
          f"pos_mean={config['pos_mean']}, rot_mean={config['rot_mean']}")
    print("")
    print("[SECOND]")
    print(json.dumps(second_summary, indent=2))
    print("")
    print("[DSVT]")
    print(json.dumps(dsvt_summary, indent=2))
    print("")
    print("[DSVT - SECOND]")
    print(json.dumps(diff_summary, indent=2))
    print("")
    print(f"Saved summary to: {json_path}")
    print(f"Saved per-sample csv to: {csv_path}")


if __name__ == "__main__":
    main()
