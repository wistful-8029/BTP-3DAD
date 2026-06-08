# Back to Point: Exploring Point-Language Models for Zero-Shot 3D Anomaly Detection

 **[CVPR 2026]**  [Back to Point: Exploring Point-Language Models for Zero-Shot 3D Anomaly Detection](https://arxiv.org/abs/2603.21511)


## Update

- **2026-05-14:** We corrected the AU-PRO evaluation. The original implementation reported a point-coverage-style AUC rather than the standard thresholded per-region overlap curve using FPR as the x-axis. We have updated the arXiv results and provide the corrected implementation in `test_standard_aupro.py`. Please use this script for standard AU-PRO reporting; `test.py` is kept only for compatibility with the previous metric. For details about the AU-PRO correction, please see [docs/metric_update.md](docs/metric_update.md).

- **2026-05-14:** We updated the object-level score aggregation in the current GitHub evaluation code, replacing max-based local aggregation with a top-ratio mean over high-scoring patches. See [Object-Level Score Aggregation](#object-level-score-aggregation).

- **2026-05-11:** We released the complete code.

- **2026-02-26:** We added a minimal baseline to help developers quickly build new methods on top of this codebase.

## Object-Level Score Aggregation

For object-level AUROC/AP, each test object is assigned one anomaly score and then evaluated with standard binary ranking metrics. The CVPR paper describes object-level supervision as a fusion of global and local predictions, and the corresponding reference training code uses `global_alpha * global_score + (1 - global_alpha) * max(local_patch_scores)`. During code cleanup and additional evaluation, we found that replacing the single maximum local score with the mean of the highest patch anomaly probabilities is more effective and stable in practice. Therefore, `test_standard_aupro.py` uses this recommended aggregation by default:

```text
object_score = global_alpha * global_prob
             + (1 - global_alpha) * mean(top topk_ratio patch_probs)
```

By default, `global_alpha = 0.5` and `topk_ratio = 0.2`.

| Dataset | Version | Object-level score aggregation | Object AUROC | Object AP | Object F1max |
| --- | --- | --- | --- | --- | --- |
| Real3D-AD | CVPR paper | `global_alpha * global_score + (1 - global_alpha) * max(local_patch_scores)` | 61.4 ± 2.0 | 65.1 ± 1.5 | 71.3 ± 0.7 |
| Real3D-AD | Current GitHub implementation | `global_alpha * global_prob + (1 - global_alpha) * mean(top topk_ratio patch_probs)` | 66.60 ± 2.0 | 69.29 ± 1.8 | 72.84 ± 0.7 |
| Anomaly-ShapeNet | CVPR paper | `global_alpha * global_score + (1 - global_alpha) * max(local_patch_scores)` | 65.2 ± 1.1 | 71.4 ± 1.1 | 76.3 ± 0.1 |
| Anomaly-ShapeNet | Current GitHub implementation | `global_alpha * global_prob + (1 - global_alpha) * mean(top topk_ratio patch_probs)` | 68.37 ± 1.7 | 75.10 ± 1.6 | 76.75 ± 1.0 |

## Installation

```bash
conda create -n BTP python=3.10 -y
conda activate BTP
git clone https://github.com/wistful-8029/BTP-3DAD.git
cd BTP-3DAD
pip install -r requirements.txt
```

> Note: building CUDA extensions such as `pointnet2_ops` and `KNN-CUDA` can be environment-dependent and may require extra steps. Please refer to [Pointnet2_PyTorch](https://github.com/erikwijmans/Pointnet2_PyTorch) and [KNN_CUDA](https://github.com/unlimblue/KNN_CUDA), then choose the build method that matches your PyTorch/CUDA versions.

Download the pretrained weights below and place them under `pretrained/`:

- [open_clip_pytorch_model.bin](https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k/blob/main/open_clip_pytorch_model.bin)
- [ULIP-2-PointBERT-10k-xyzrgb-pc-vit_g-objaverse_shapenet-pretrained.pt](https://huggingface.co/datasets/SFXX/ulip/blob/main/ULIP-2/pretrained_models/ULIP-2-PointBERT-10k-xyzrgb-pc-vit_g-objaverse_shapenet-pretrained.pt)

Expected layout:

```text
BTP-3DAD/
└── pretrained/
    ├── open_clip_pytorch_model.bin
    └── ULIP-2-PointBERT-10k-xyzrgb-pc-vit_g-objaverse_shapenet-pretrained.pt
```

## Data Preparation

Download the datasets from [Real3D-AD](https://drive.google.com/file/d/1oM4qjhlIMsQc_wiFIFIVBvuuR8nyk2k0/view?usp=sharing) and [Anomaly-ShapeNet](https://huggingface.co/datasets/Chopper233/Anomaly-ShapeNet).

Example raw Real3D-AD layout:

```text
data/
└── real3dad/
    ├── airplane/
    │   ├── train/
    │   ├── test/
    │   └── gt/
    ├── car/
    └── ...
```

Preprocess Real3D-AD into sampled point-cloud npz files:

```bash
python utils/processing_real3d.py \
  --raw_root /path/to/data/real3dad \
  --out_root ./data/Real3D-AD-2048-npz \
  --num_samples 2048
```

The processed layout should look like:

```text
data/
└── Real3D-AD-2048-npz/
    ├── airplane/
    │   ├── train/
    │   │   ├── 60_template.npz
    │   │   └── 128_template.npz
    │   └── test/
    │       ├── 142_good.npz
    │       └── 142_good_cut.npz
    ├── car/
    └── ...
```

The npz files can store both sampled 2048-point data and original-resolution metadata.

## Training

Train one one-vs-rest model for a single category:

```bash
python train.py \
  --train_class airplane \
  --train_split test \
  --data_root ./data/Real3D-AD-2048-npz \
  --model_path ./pretrained/ULIP-2-PointBERT-10k-xyzrgb-pc-vit_g-objaverse_shapenet-pretrained.pt \
  --output_root ./outputs/train
```

Train all Real3D-AD categories:

```bash
python train.py \
  --train_split test \
  --repeats 1 \
  --data_root ./data/Real3D-AD-2048-npz \
  --model_path ./pretrained/ULIP-2-PointBERT-10k-xyzrgb-pc-vit_g-objaverse_shapenet-pretrained.pt \
  --output_root ./outputs/train
```


## Evaluation With Standard AU-PRO

Use `test_standard_aupro.py` for standard AU-PRO reporting:

```bash
python test_standard_aupro.py \
  --ckpt_root ./outputs/train/Real3D/airplane/run_00_seed111/best.pth \
  --train_class airplane \
  --dataset_name Real3D \
  --data_root ./data/Real3D-AD-2048-npz \
  --model_path ./pretrained/ULIP-2-PointBERT-10k-xyzrgb-pc-vit_g-objaverse_shapenet-pretrained.pt \
  --output_dir ./outputs/eval/airplane_standard_aupro
```




## Legacy Evaluation

`test.py` is retained for compatibility with the earlier metric implementation. Its AU-PRO value is actually a point-ranking anomalous-point coverage AUC and should not be cited as standard AU-PRO.




## Acknowledgements

We sincerely thank the authors and contributors of [ulip2_encoder](https://github.com/SanBingYouYong/ulip2_encoder), [ULIP](https://github.com/salesforce/ULIP), [AnomalyCLIP](https://github.com/zqhang/AnomalyCLIP), and [PointAD](https://github.com/zqhang/PointAD) for their valuable open-source contributions.

## BibTeX Citation

If you find this paper and repository useful, please cite our paper.

```bibtex
@inproceedings{li2026back,
  title={Back to Point: Exploring Point-Language Models for Zero-Shot 3D Anomaly Detection},
  author={Li, Kaiqiang and Li, Gang and Zhou, Mingle and Li, Min and Han, Delong and Wan, Jin},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={14167--14177},
  year={2026}
}
```
