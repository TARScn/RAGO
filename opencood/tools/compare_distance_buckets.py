# -*- coding: utf-8 -*-

import argparse
import csv
import glob
import json
import os
from collections import OrderedDict

import yaml


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare distance-bucket detection results across runs."
    )
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Run spec in the form label=path, where path is an experiment "
             "directory or a distance_buckets yaml file. Repeat this flag "
             "for multiple runs. The first run is treated as baseline."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="opencood/logs/distance_bucket_compare",
        help="Directory to save comparison outputs."
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="compare",
        help="Prefix for output files."
    )
    return parser.parse_args()


def parse_run_spec(run_spec):
    if "=" not in run_spec:
        raise ValueError(
            f"Invalid --run spec '{run_spec}'. Expected label=path."
        )
    label, path = run_spec.split("=", 1)
    label = label.strip()
    path = path.strip()
    if not label or not path:
        raise ValueError(
            f"Invalid --run spec '{run_spec}'. Label and path must be non-empty."
        )
    return label, path


def resolve_eval_yaml(path):
    if os.path.isfile(path):
        return path

    if not os.path.isdir(path):
        raise FileNotFoundError(f"Path not found: {path}")

    candidates = glob.glob(os.path.join(path, "*distance_buckets.yaml"))
    if not candidates:
        raise FileNotFoundError(
            f"No *distance_buckets.yaml found under: {path}"
        )

    candidates.sort(key=os.path.getmtime)
    return candidates[-1]


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def canonicalize_bucket_data(bucket_data):
    canonical = OrderedDict()
    for bucket_name, info in bucket_data.items():
        canonical[bucket_name] = OrderedDict([
            ("ap30", float(info["ap30"])),
            ("ap50", float(info["ap50"])),
            ("ap70", float(info["ap70"])),
            ("gt", int(info["gt"])),
            ("range", info["range"]),
        ])
    return canonical


def format_float(value):
    if value is None:
        return ""
    return f"{value:.4f}"


def build_bucket_table(bucket_name, run_results, baseline_label):
    baseline_info = run_results[baseline_label]["buckets"][bucket_name]
    rows = []
    for label, result in run_results.items():
        info = result["buckets"][bucket_name]
        rows.append(OrderedDict([
            ("bucket", bucket_name),
            ("range_min", info["range"][0]),
            ("range_max", info["range"][1]),
            ("gt", info["gt"]),
            ("label", label),
            ("ap30", info["ap30"]),
            ("ap50", info["ap50"]),
            ("ap70", info["ap70"]),
            ("delta_ap30_vs_baseline", info["ap30"] - baseline_info["ap30"]),
            ("delta_ap50_vs_baseline", info["ap50"] - baseline_info["ap50"]),
            ("delta_ap70_vs_baseline", info["ap70"] - baseline_info["ap70"]),
        ]))
    return rows


def build_summary_rows(bucket_names, run_results, baseline_label):
    rows = []
    for bucket_name in bucket_names:
        rows.extend(build_bucket_table(bucket_name, run_results, baseline_label))
    return rows


def write_json(path, content):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(content, f, indent=2, ensure_ascii=False)


def write_csv(path, rows):
    if not rows:
        with open(path, "w", encoding="utf-8") as f:
            f.write("")
        return

    headers = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_markdown(path, bucket_names, run_results, baseline_label):
    lines = []
    lines.append(f"# Distance Bucket Comparison")
    lines.append("")
    lines.append(f"Baseline: `{baseline_label}`")
    lines.append("")

    for bucket_name in bucket_names:
        baseline_info = run_results[baseline_label]["buckets"][bucket_name]
        lines.append(f"## {bucket_name}")
        lines.append("")
        lines.append(
            f"Range: `{baseline_info['range'][0]}` to `{baseline_info['range'][1]}`, "
            f"GT: `{baseline_info['gt']}`"
        )
        lines.append("")
        lines.append(
            "| Label | AP30 | dAP30 | AP50 | dAP50 | AP70 | dAP70 |"
        )
        lines.append(
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"
        )
        for label, result in run_results.items():
            info = result["buckets"][bucket_name]
            lines.append(
                f"| {label} | "
                f"{format_float(info['ap30'])} | "
                f"{format_float(info['ap30'] - baseline_info['ap30'])} | "
                f"{format_float(info['ap50'])} | "
                f"{format_float(info['ap50'] - baseline_info['ap50'])} | "
                f"{format_float(info['ap70'])} | "
                f"{format_float(info['ap70'] - baseline_info['ap70'])} |"
            )
        lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def print_console_report(bucket_names, run_results, baseline_label):
    print("=== Distance Bucket Comparison ===")
    print(f"Baseline: {baseline_label}")
    print("")

    for bucket_name in bucket_names:
        baseline_info = run_results[baseline_label]["buckets"][bucket_name]
        print(
            f"[{bucket_name}] range={baseline_info['range']}, gt={baseline_info['gt']}"
        )
        header = (
            f"{'label':<24}"
            f"{'ap30':>10}{'d30':>10}"
            f"{'ap50':>10}{'d50':>10}"
            f"{'ap70':>10}{'d70':>10}"
        )
        print(header)
        print("-" * len(header))
        for label, result in run_results.items():
            info = result["buckets"][bucket_name]
            print(
                f"{label:<24}"
                f"{info['ap30']:>10.4f}{info['ap30'] - baseline_info['ap30']:>10.4f}"
                f"{info['ap50']:>10.4f}{info['ap50'] - baseline_info['ap50']:>10.4f}"
                f"{info['ap70']:>10.4f}{info['ap70'] - baseline_info['ap70']:>10.4f}"
            )
        print("")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    run_results = OrderedDict()
    for run_spec in args.run:
        label, raw_path = parse_run_spec(run_spec)
        eval_yaml_path = resolve_eval_yaml(raw_path)
        bucket_data = canonicalize_bucket_data(load_yaml(eval_yaml_path))
        run_results[label] = {
            "input_path": raw_path,
            "eval_yaml_path": eval_yaml_path,
            "buckets": bucket_data,
        }

    if len(run_results) < 2:
        raise ValueError("Please provide at least two --run entries.")

    baseline_label = next(iter(run_results.keys()))
    baseline_buckets = run_results[baseline_label]["buckets"]
    bucket_names = list(baseline_buckets.keys())

    warnings = []
    for label, result in run_results.items():
        current_bucket_names = list(result["buckets"].keys())
        if current_bucket_names != bucket_names:
            raise ValueError(
                f"Bucket names mismatch for run '{label}'. "
                f"Expected {bucket_names}, got {current_bucket_names}."
            )
        for bucket_name in bucket_names:
            gt_baseline = baseline_buckets[bucket_name]["gt"]
            gt_current = result["buckets"][bucket_name]["gt"]
            if gt_baseline != gt_current:
                warnings.append(
                    f"GT mismatch at bucket '{bucket_name}' for run '{label}': "
                    f"baseline={gt_baseline}, current={gt_current}"
                )

    summary_rows = build_summary_rows(bucket_names, run_results, baseline_label)
    report = OrderedDict([
        ("baseline_label", baseline_label),
        ("bucket_names", bucket_names),
        ("warnings", warnings),
        ("runs", run_results),
        ("summary_rows", summary_rows),
    ])

    json_path = os.path.join(args.output_dir, f"{args.tag}_distance_compare.json")
    csv_path = os.path.join(args.output_dir, f"{args.tag}_distance_compare.csv")
    md_path = os.path.join(args.output_dir, f"{args.tag}_distance_compare.md")

    write_json(json_path, report)
    write_csv(csv_path, summary_rows)
    write_markdown(md_path, bucket_names, run_results, baseline_label)

    print_console_report(bucket_names, run_results, baseline_label)
    if warnings:
        print("Warnings:")
        for warning in warnings:
            print(f"- {warning}")
        print("")

    print(f"Saved json to: {json_path}")
    print(f"Saved csv to: {csv_path}")
    print(f"Saved markdown to: {md_path}")


if __name__ == "__main__":
    main()
