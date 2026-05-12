import argparse
import copy
import itertools
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
        description="Ablate CoAlign box alignment settings on SECOND and "
                    "DSVT stage1_boxes.json to diagnose why DSVT hurts "
                    "second-stage alignment.")
    parser.add_argument(
        "--second-json",
        type=str,
        required=True,
        help="Path to SECOND stage1_boxes.json.")
    parser.add_argument(
        "--dsvt-json",
        type=str,
        required=True,
        help="Path to DSVT stage1_boxes.json.")
    parser.add_argument(
        "--coalign-yaml",
        type=str,
        default="opencood/hypes_yaml/v2xset/lidar_only_with_noise/"
                "coalign/pointpillar_coalign.yaml",
        help="Yaml used to load default noise and box_align args.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="opencood/logs/stage1_align_ablation",
        help="Directory to save ablation results.")
    parser.add_argument(
        "--tag",
        type=str,
        default="ablation",
        help="Prefix for output files.")
    parser.add_argument(
        "--num-trials",
        type=int,
        default=10,
        help="Number of random noise trials per sample.")
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Base random seed.")
    parser.add_argument(
        "--limit-samples",
        type=int,
        default=0,
        help="Only evaluate the first N common samples. 0 means all.")
    parser.add_argument(
        "--pos-std",
        type=float,
        default=None,
        help="Override translation std in meters.")
    parser.add_argument(
        "--rot-std",
        type=float,
        default=None,
        help="Override yaw std in degrees.")
    parser.add_argument(
        "--pos-mean",
        type=float,
        default=None,
        help="Override translation mean in meters.")
    parser.add_argument(
        "--rot-mean",
        type=float,
        default=None,
        help="Override yaw mean in degrees.")
    parser.add_argument(
        "--scenario-set",
        type=str,
        default="default",
        choices=["default", "focused", "full"],
        help="Predefined ablation group.")
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


def sample_stats_template():
    return {
        "sample_count": 0,
        "trial_count": 0,
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


def finalize_stats(stats):
    summary = OrderedDict()
    summary["sample_count"] = stats["sample_count"]
    summary["trial_count"] = stats["trial_count"]
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


def evaluate_model(data_dict, sample_ids, align_args, noise_cfg, num_trials, seed):
    stats = sample_stats_template()

    for sample_idx in sample_ids:
        content = prepare_sample_content(data_dict[sample_idx])
        pred_corners_list = content["pred_corners_list"]
        uncertainty_list = content["uncertainty_list"]
        lidar_pose_clean = content["lidar_pose_clean"]
        lidar_pose_clean_dof3 = lidar_pose_clean[:, [0, 1, 4]]
        stats["sample_count"] += 1

        for trial_idx in range(num_trials):
            stats["trial_count"] += 1
            rng = np.random.default_rng(seed + trial_idx * 1000003 + int(sample_idx))
            noisy_pose = build_noise(
                lidar_pose_clean,
                rng,
                noise_cfg["pos_std"],
                noise_cfg["rot_std"],
                noise_cfg["pos_mean"],
                noise_cfg["rot_mean"],
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

            if after_trans_mean < before_trans_mean:
                stats["trans_better_trials"] += 1
            if after_rot_mean < before_rot_mean:
                stats["rot_better_trials"] += 1
            if np.allclose(refined_pose_dof3, noisy_pose_dof3, atol=1e-6):
                stats["unchanged_trials"] += 1

    return finalize_stats(stats)


def make_scenarios(base_args, scenario_set):
    base = copy.deepcopy(base_args)
    base.setdefault("landmark_SE2", True)
    base.setdefault("adaptive_landmark", False)
    base.setdefault("normalize_uncertainty", False)
    base.setdefault("abandon_hard_cases", True)
    base.setdefault("drop_hard_boxes", True)
    base.setdefault("use_uncertainty", True)

    scenarios = [
        OrderedDict([
            ("name", "baseline"),
            ("description", "Current CoAlign box_align args from yaml."),
            ("align_args", copy.deepcopy(base)),
        ]),
        OrderedDict([
            ("name", "no_uncertainty"),
            ("description", "Disable uncertainty weighting only."),
            ("align_args", dict(base, use_uncertainty=False)),
        ]),
        OrderedDict([
            ("name", "normalize_uncertainty"),
            ("description", "Keep uncertainty but normalize certainty weights."),
            ("align_args", dict(base, normalize_uncertainty=True)),
        ]),
        OrderedDict([
            ("name", "no_uncertainty_no_hard_filters"),
            ("description", "Disable uncertainty and hard-case dropping."),
            ("align_args", dict(base,
                                use_uncertainty=False,
                                abandon_hard_cases=False,
                                drop_hard_boxes=False)),
        ]),
        OrderedDict([
            ("name", "uncertainty_no_hard_filters"),
            ("description", "Keep uncertainty but disable hard-case dropping."),
            ("align_args", dict(base,
                                abandon_hard_cases=False,
                                drop_hard_boxes=False)),
        ]),
    ]

    thres_values = [1.0, 1.5, 2.0, 2.5]
    yaw_var_values = [0.1, 0.2, 0.3, 0.4]

    for thres in thres_values:
        scenarios.append(OrderedDict([
            ("name", f"thres_{thres:.1f}"),
            ("description", f"Change clustering threshold to {thres:.1f} m."),
            ("align_args", dict(base, thres=thres)),
        ]))

    for yaw_var_thres in yaw_var_values:
        scenarios.append(OrderedDict([
            ("name", f"yaw_var_{yaw_var_thres:.1f}"),
            ("description", f"Change yaw variance threshold to {yaw_var_thres:.1f}."),
            ("align_args", dict(base, yaw_var_thres=yaw_var_thres)),
        ]))

    scenarios.extend([
        OrderedDict([
            ("name", "adaptive_landmark"),
            ("description", "Enable adaptive landmark fallback on yaw-varying clusters."),
            ("align_args", dict(base, adaptive_landmark=True)),
        ]),
        OrderedDict([
            ("name", "drop_unsure_edge"),
            ("description", "Drop low-certainty edges during graph construction."),
            ("align_args", dict(base, drop_unsure_edge=True)),
        ]),
        OrderedDict([
            ("name", "no_hard_filters"),
            ("description", "Disable abandon_hard_cases and drop_hard_boxes."),
            ("align_args", dict(base,
                                abandon_hard_cases=False,
                                drop_hard_boxes=False)),
        ]),
    ])

    if scenario_set == "focused":
        keep = {
            "baseline",
            "no_uncertainty",
            "normalize_uncertainty",
            "uncertainty_no_hard_filters",
            "no_uncertainty_no_hard_filters",
            "thres_1.0",
            "thres_2.0",
            "thres_2.5",
            "adaptive_landmark",
            "drop_unsure_edge",
        }
        scenarios = [s for s in scenarios if s["name"] in keep]
    elif scenario_set == "full":
        combo_values = list(itertools.product(
            [False, True],
            [False, True],
            [1.0, 1.5, 2.0],
            [0.2, 0.3],
        ))
        for use_uncertainty, normalize_uncertainty, thres, yaw_var_thres in combo_values:
            name = (
                f"grid_u{int(use_uncertainty)}_n{int(normalize_uncertainty)}"
                f"_t{str(thres).replace('.', '')}_y{str(yaw_var_thres).replace('.', '')}"
            )
            scenarios.append(OrderedDict([
                ("name", name),
                ("description", "Grid search over uncertainty normalization, "
                                 "threshold, and yaw variance threshold."),
                ("align_args", dict(base,
                                    use_uncertainty=use_uncertainty,
                                    normalize_uncertainty=normalize_uncertainty,
                                    thres=thres,
                                    yaw_var_thres=yaw_var_thres)),
            ]))

    unique = OrderedDict()
    for scenario in scenarios:
        unique[scenario["name"]] = scenario
    return list(unique.values())


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
                values.append("" if value is None else str(value))
            f.write(",".join(values) + "\n")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    cfg = load_eval_config(args)
    noise_cfg = {
        "pos_std": cfg["pos_std"],
        "rot_std": cfg["rot_std"],
        "pos_mean": cfg["pos_mean"],
        "rot_mean": cfg["rot_mean"],
    }
    second_data = read_json(args.second_json)
    dsvt_data = read_json(args.dsvt_json)
    sample_ids, skipped = validate_common_samples(second_data, dsvt_data)
    if args.limit_samples > 0:
        sample_ids = sample_ids[:args.limit_samples]

    scenarios = make_scenarios(cfg["box_align_args"], args.scenario_set)
    rows = []
    scenario_reports = []

    for scenario in scenarios:
        align_args = copy.deepcopy(scenario["align_args"])
        second_summary = evaluate_model(
            second_data, sample_ids, align_args, noise_cfg, args.num_trials, args.seed)
        dsvt_summary = evaluate_model(
            dsvt_data, sample_ids, align_args, noise_cfg, args.num_trials, args.seed)

        diff = OrderedDict([
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

        scenario_reports.append(OrderedDict([
            ("name", scenario["name"]),
            ("description", scenario["description"]),
            ("align_args", align_args),
            ("SECOND", second_summary),
            ("DSVT", dsvt_summary),
            ("difference", diff),
        ]))

        rows.append(OrderedDict([
            ("scenario", scenario["name"]),
            ("trans_after_second", second_summary["trans_after"]["mean"]),
            ("trans_after_dsvt", dsvt_summary["trans_after"]["mean"]),
            ("trans_after_gap_dsvt_minus_second", diff["trans_after_mean_gap_m"]),
            ("trans_improve_second", second_summary["trans_improve"]["mean"]),
            ("trans_improve_dsvt", dsvt_summary["trans_improve"]["mean"]),
            ("trans_improve_gap_dsvt_minus_second", diff["trans_improve_mean_gap_m"]),
            ("rot_after_second_deg", second_summary["rot_after_deg"]["mean"]),
            ("rot_after_dsvt_deg", dsvt_summary["rot_after_deg"]["mean"]),
            ("rot_after_gap_dsvt_minus_second_deg", diff["rot_after_mean_gap_deg"]),
            ("rot_improve_second_deg", second_summary["rot_improve_deg"]["mean"]),
            ("rot_improve_dsvt_deg", dsvt_summary["rot_improve_deg"]["mean"]),
            ("rot_improve_gap_dsvt_minus_second_deg", diff["rot_improve_mean_gap_deg"]),
            ("second_unchanged_ratio", second_summary["unchanged_ratio"]),
            ("dsvt_unchanged_ratio", dsvt_summary["unchanged_ratio"]),
        ]))

    rows_sorted = sorted(
        rows,
        key=lambda row: (
            float("inf") if row["trans_after_gap_dsvt_minus_second"] is None
            else row["trans_after_gap_dsvt_minus_second"],
            float("inf") if row["rot_after_gap_dsvt_minus_second_deg"] is None
            else row["rot_after_gap_dsvt_minus_second_deg"],
        )
    )

    report = OrderedDict([
        ("config", OrderedDict([
            ("second_json", args.second_json),
            ("dsvt_json", args.dsvt_json),
            ("coalign_yaml", args.coalign_yaml),
            ("num_trials", args.num_trials),
            ("seed", args.seed),
            ("scenario_set", args.scenario_set),
            ("sample_count", len(sample_ids)),
            ("noise_cfg", noise_cfg),
            ("base_box_align_args", cfg["box_align_args"]),
        ])),
        ("skipped_samples", [
            OrderedDict([("sample_idx", sample_idx), ("reason", reason)])
            for sample_idx, reason in skipped
        ]),
        ("ranked_rows", rows_sorted),
        ("scenarios", scenario_reports),
    ])

    json_path = os.path.join(args.output_dir, f"{args.tag}_ablation.json")
    csv_path = os.path.join(args.output_dir, f"{args.tag}_ablation.csv")
    write_json(json_path, report)
    write_csv(csv_path, rows_sorted)

    print("=== Stage1 Box Alignment Ablation ===")
    print(f"Usable common samples: {len(sample_ids)}")
    print(f"Scenarios: {len(scenarios)}")
    print(f"Saved ablation json to: {json_path}")
    print(f"Saved ablation csv to: {csv_path}")
    print("")
    print("Top 10 scenarios by smallest DSVT-SECOND translation gap:")
    for row in rows_sorted[:10]:
        print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
