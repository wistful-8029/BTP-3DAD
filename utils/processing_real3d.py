#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
processing_real3d.py

Purpose
-------
Preprocess Real3D-AD point clouds by downsampling each sample to a fixed number of points using
Farthest Point Sampling (FPS), and saving the results in NumPy `.npy` format.

Input Dataset Layout (expected)
------------------------------
raw_root/
  <class_name>/
    train/*.pcd
    test/*.pcd
    gt/*.txt

Notes on labels:
- Train split:
  - Reads `train/*.pcd` and saves XYZ only: shape [S, 3].
- Test split:
  - If filename contains substring "good": treated as normal sample.
    Reads `test/*.pcd`, downsamples XYZ, appends an all-zero label column -> [S, 4].
  - Otherwise: treated as anomalous sample.
    Reads the corresponding `gt/<same_stem>.txt` (each row: x y z label), downsamples using FPS indices,
    and saves XYZ + sampled labels -> [S, 4].

Outputs
-------
out_root/
  <class_name>/
    train/*.npy   # [S, 3]
    test/*.npy    # [S, 4]

Dependencies
------------
- numpy
- torch
- open3d
- tqdm

How to Run
----------
python processing_real3d.py \
  --raw_root "D:\\WorkSpace\\code\\datasets\\Real3D-AD-PCD" \
  --out_root "D:\\WorkSpace\\code\\datasets\\Real3D-AD-2048" \
  --num_samples 2048 \
  --device cpu \
  --overwrite

Arguments
---------
--raw_root     Path to the original Real3D-AD root directory.
--out_root     Path to the output root directory.
--num_samples  Number of points after FPS downsampling (default: 2048).
--device       "cpu" or "cuda" (default: cpu).
--seed         Random seed for FPS initialization (default: 0).
--overwrite    Overwrite existing outputs if set.

Reproducibility
---------------
FPS uses a random initial point per sample; use `--seed` to make results deterministic.
For strict determinism on GPU, you may also need extra PyTorch determinism flags, depending on your setup.
"""

import argparse
import os
import random

import numpy as np
import torch
import open3d as o3d
from tqdm import tqdm


def set_seed(seed: int) -> None:
    """Set RNG seeds for Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    """Create a directory if it does not exist."""
    os.makedirs(path, exist_ok=True)


def list_classes(raw_root: str):
    """List class subdirectories under `raw_root`."""
    classes = []
    for name in os.listdir(raw_root):
        p = os.path.join(raw_root, name)
        if os.path.isdir(p):
            classes.append(name)
    classes.sort()
    return classes


def stem(filename: str) -> str:
    """Return file stem without extension."""
    return os.path.splitext(filename)[0]


def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    Gather points by indices.

    Parameters
    ----------
    points : torch.Tensor
        Point tensor of shape [B, N, C].
    idx : torch.Tensor
        Index tensor of shape [B, S].

    Returns
    -------
    torch.Tensor
        Gathered tensor of shape [B, S, C].
    """
    device = points.device
    bsz = points.shape[0]

    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)

    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1

    batch_indices = torch.arange(bsz, dtype=torch.long, device=device).view(view_shape).repeat(repeat_shape)
    return points[batch_indices, idx, :]


def farthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """
    Farthest Point Sampling (FPS).

    Parameters
    ----------
    xyz : torch.Tensor
        Input point cloud of shape [B, N, 3].
    npoint : int
        Number of sampled points.

    Returns
    -------
    torch.Tensor
        Sampled indices of shape [B, npoint].
    """
    device = xyz.device
    bsz, n, _ = xyz.shape
    if npoint > n:
        raise ValueError(f"npoint={npoint} > N={n}, cannot FPS sample more points than exist.")

    centroids = torch.zeros(bsz, npoint, dtype=torch.long, device=device)
    distance = torch.full((bsz, n), 1e10, device=device)
    farthest = torch.randint(0, n, (bsz,), dtype=torch.long, device=device)
    batch_indices = torch.arange(bsz, dtype=torch.long, device=device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(bsz, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, dim=-1)[1]

    return centroids


def read_pcd_xyz(pcd_path: str) -> np.ndarray:
    """
    Read XYZ coordinates from a .pcd file using Open3D.

    Returns
    -------
    np.ndarray
        Array of shape [N, 3], dtype float32.
    """
    pcd = o3d.io.read_point_cloud(pcd_path)
    xyz = np.asarray(pcd.points, dtype=np.float32)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"Invalid PCD points shape: {xyz.shape} from {pcd_path}")
    return xyz


def read_gt_txt_xyzl(txt_path: str):
    """
    Read XYZ + label from a ground-truth .txt file.

    Expected format per row: x y z label

    Returns
    -------
    xyz : np.ndarray
        Array of shape [N, 3], dtype float32.
    label : np.ndarray
        Array of shape [N], dtype float32.
    """
    data = np.loadtxt(txt_path, dtype=np.float32)
    if data.ndim != 2 or data.shape[1] < 4:
        raise ValueError(f"Invalid GT txt shape: {data.shape} from {txt_path}; expected [N,4].")
    xyz = data[:, :3].astype(np.float32)
    label = data[:, 3].astype(np.float32)
    return xyz, label


def fps_sample_xyz(xyz: np.ndarray, num_samples: int, device: torch.device):
    """
    FPS-downsample XYZ coordinates.

    Parameters
    ----------
    xyz : np.ndarray
        Input array of shape [N, 3].
    num_samples : int
        Number of points after downsampling.
    device : torch.device
        Torch device for FPS computation.

    Returns
    -------
    sampled_xyz : np.ndarray
        Downsampled array of shape [S, 3].
    idx_np : np.ndarray
        Sampled indices (w.r.t. original xyz), shape [S].
    """
    pts = torch.from_numpy(xyz).to(device=device, dtype=torch.float32).unsqueeze(0)  # [1, N, 3]
    idx = farthest_point_sample(pts, num_samples)  # [1, S]
    sampled = index_points(pts, idx).squeeze(0).detach().cpu().numpy()  # [S, 3]
    idx_np = idx.squeeze(0).detach().cpu().numpy().astype(np.int64)  # [S]
    return sampled, idx_np


def process_dataset(raw_root: str, out_root: str, num_samples: int, device: torch.device, overwrite: bool) -> None:
    """
    Process the full dataset under `raw_root` and write outputs to `out_root`.
    """
    ensure_dir(out_root)
    classes = list_classes(raw_root)

    for cls in tqdm(classes, desc="Processing classes"):
        class_dir = os.path.join(raw_root, cls)
        out_class_dir = os.path.join(out_root, cls)
        ensure_dir(out_class_dir)

        # -------------------------
        # Train split: save XYZ only
        # -------------------------
        train_dir = os.path.join(class_dir, "train")
        if os.path.isdir(train_dir):
            out_train_dir = os.path.join(out_class_dir, "train")
            ensure_dir(out_train_dir)

            for fn in os.listdir(train_dir):
                if not fn.lower().endswith(".pcd"):
                    continue
                src = os.path.join(train_dir, fn)
                dst = os.path.join(out_train_dir, stem(fn) + ".npy")
                if (not overwrite) and os.path.exists(dst):
                    continue

                xyz = read_pcd_xyz(src)
                sampled_xyz, _ = fps_sample_xyz(xyz, num_samples, device)
                np.save(dst, sampled_xyz)

        # -------------------------
        # Test split: save XYZ + label
        # -------------------------
        test_dir = os.path.join(class_dir, "test")
        gt_dir = os.path.join(class_dir, "gt")
        if os.path.isdir(test_dir):
            out_test_dir = os.path.join(out_class_dir, "test")
            ensure_dir(out_test_dir)

            for fn in os.listdir(test_dir):
                if not fn.lower().endswith(".pcd"):
                    continue

                src_pcd = os.path.join(test_dir, fn)
                dst = os.path.join(out_test_dir, stem(fn) + ".npy")
                if (not overwrite) and os.path.exists(dst):
                    continue

                # Normal sample: "good" in filename -> label all zeros
                if "good" in fn:
                    xyz = read_pcd_xyz(src_pcd)
                    sampled_xyz, _ = fps_sample_xyz(xyz, num_samples, device)
                    label0 = np.zeros((num_samples, 1), dtype=np.float32)
                    out = np.concatenate([sampled_xyz, label0], axis=1)  # [S, 4]
                    np.save(dst, out)
                else:
                    # Anomalous sample: load GT txt and sample labels with the same FPS indices
                    txt_path = os.path.join(gt_dir, stem(fn) + ".txt")
                    if not os.path.exists(txt_path):
                        raise FileNotFoundError(f"Missing GT txt for {src_pcd}: expected {txt_path}")

                    xyz, labels = read_gt_txt_xyzl(txt_path)
                    sampled_xyz, idx_np = fps_sample_xyz(xyz, num_samples, device)
                    sampled_labels = labels[idx_np].reshape(-1, 1).astype(np.float32)
                    out = np.concatenate([sampled_xyz, sampled_labels], axis=1)  # [S, 4]
                    np.save(dst, out)


def build_argparser():
    """Build a minimal CLI for public release."""
    p = argparse.ArgumentParser(
        description="Preprocess Real3D-AD point clouds: FPS downsample and save as .npy."
    )
    p.add_argument("--raw_root", type=str, required=True, help="Path to the Real3D-AD raw root directory.")
    p.add_argument("--out_root", type=str, required=True, help="Path to the output root directory.")
    p.add_argument("--num_samples", type=int, default=2048, help="Number of points after FPS downsampling.")
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help='Compute device: "cpu" or "cuda".')
    p.add_argument("--seed", type=int, default=0, help="Random seed for FPS initialization.")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing output files if set.")
    return p


def main():
    args = build_argparser().parse_args()
    set_seed(args.seed)

    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is not available. Please use "--device cpu".')
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    process_dataset(
        raw_root=args.raw_root,
        out_root=args.out_root,
        num_samples=args.num_samples,
        device=device,
        overwrite=args.overwrite,
    )
    print(f"[OK] Done. Output root: {args.out_root}")


if __name__ == "__main__":
    main()
