# V2XSet RDU-CoAlign Ablation YAMLs

This directory contains the V2XSet yaml files for the compact RDU-CoAlign ablation suite.

| Experiment | YAML | Used modules | Unused modules |
|---|---|---|---|
| E0 baseline w/o box alignment | `e0_baseline_woba.yaml` | multiscale AttFusion | box alignment, semantic enhancement, RSU distance guidance |
| E1 CoAlign baseline | `e1_coalign_box_align.yaml` | multiscale AttFusion, box alignment | semantic enhancement, RSU distance guidance |
| E2 semantic enhancement | `e2_coalign_semantic_enhancement.yaml` | multiscale AttFusion, box alignment, hybrid V2XViT-Deform | RSU distance guidance |
| E3 RSU distance guidance | `e3_coalign_rsu_distance.yaml` | multiscale AttFusion, box alignment, RSU distance guidance | semantic enhancement |
| E4 full RDU-CoAlign | `e4_rdu_coalign_full.yaml` | multiscale AttFusion, box alignment, hybrid V2XViT-Deform, RSU distance guidance | none |
| E5 RDU-CoAlign w/o box alignment | `e5_rdu_coalign_woba.yaml` | multiscale AttFusion, hybrid V2XViT-Deform, RSU distance guidance | box alignment |

Train the full RDU-CoAlign configuration with:

```bash
python opencood/tools/train.py \
  -y opencood/hypes_yaml/v2xset/lidar_only_with_noise/rdu_coalign/e4_rdu_coalign_full.yaml
```

Run inference from the produced log directory with:

```bash
python opencood/tools/inference.py \
  --model_dir opencood/logs/<experiment_dir> \
  --fusion_method intermediate
```
