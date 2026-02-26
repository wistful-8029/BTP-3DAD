
# Back to Point: Exploring Point-Language Models for Zero-Shot 3D Anomaly Detection

# Notes
We are still extending our experiments, so we are not able to release the full codebase at this stage. Instead, we provide a minimal working example that applies a simple linear projection to map ULIP patch features into a channel space aligned with the text embeddings. This release is intended to help researchers quickly build upon ULIP-based 3D anomaly detection frameworks and explore more pioneering directions.
# ULIP权重
https://huggingface.co/datasets/SFXX/ulip/tree/main/ULIP-2/pretrained_models
# Quick Start
## Installation
写一下环境配置
## Data Preparation

Download the datasets from [Real3D-AD](https://drive.google.com/file/d/1oM4qjhlIMsQc_wiFIFIVBvuuR8nyk2k0/view?usp=sharing) and [Anomaly-ShapeNet](https://huggingface.co/datasets/Chopper233/Anomaly-ShapeNet).

After downloading, organize the raw data as follows (example for Real3D-AD):

```text
data/
└── real3dad/
    ├── airplane/
    │   ├── train/
    │   │   ├── 1_prototype.pcd
    │   │   ├── 2_prototype.pcd
    │   │   └── ...
    │   ├── test/
    │   │   ├── 1_bulge.pcd
    │   │   ├── 2_sink.pcd
    │   │   └── ...
    │   └── gt/
    │       ├── 1_bulge.txt
    │       ├── 2_sink.txt
    │       └── ...
    ├── car/
    └── ...
```
### Data Preprocessing
We preprocess the raw point clouds by uniformly sampling each shape to 2048 points.
```
python utils/processing_real3d.py \
  --raw_root /path/to/data/real3dad \
  --out_root /path/to/data/Real3D-AD-2048 \
  --num_samples 2048
```
The processed dataset will be saved as:
```
data/
└── Real3D-AD-2048/
    ├── airplane/
    │   ├── train/
    │   │   ├── 60_template.npy
    │   │   ├── 128_template.npy
    │   │   └── ...
    │   ├── test/
    │   │   ├── 67_good.npy
    │   │   ├── 67_good_cut.npy
    │   │   └── ...
    ├── car/
    └── ...
```
## Pretrained Models

Download the pretrained weights below and place them under `pretrained/`:

- [open_clip_pytorch_model.bin](https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k/blob/main/open_clip_pytorch_model.bin)
- [ULIP-2-PointBERT-10k-xyzrgb-pc-vit_g-objaverse_shapenet-pretrained.pt](https://huggingface.co/datasets/SFXX/ulip/blob/main/ULIP-2/pretrained_models/ULIP-2-PointBERT-10k-xyzrgb-pc-vit_g-objaverse_shapenet-pretrained.pt)
```
BTP-3DAD/
└── pretrained/
    ├── open_clip_pytorch_model.bin
    └── ULIP-2-PointBERT-10k-xyzrgb-pc-vit_g-objaverse_shapenet-pretrained.pt
```
## Run

# Citation
