                      
                       
"""
Preprocess Real3D-AD and Anomaly-ShapeNet point clouds.

The script reads raw .pcd samples, downsamples each point cloud to a fixed
number of points with Farthest Point Sampling (FPS), and stores the sampled
points together with labels and the mapping back to the original point cloud.

Supported raw layouts
---------------------
Real3D-AD:
  raw_root/
    <class_name>/
      train/*.pcd
      test/*.pcd
      gt/*.txt

Anomaly-ShapeNet:
  raw_root/
    <class_name>/
      train/*.pcd
      test/*.pcd
      GT/*.txt

Output layout
-------------
out_root/
  <class_name>/
    train/*.npz
    test/*.npz

Each .npz contains:
  points              float32 [S, 3], sampled XYZ
  labels              int64   [S], sampled point labels, 0 for normal samples
  sample_indices      int64   [S], sampled point row indices in the original cloud
  original_points     float32 [N, 3], raw XYZ used as the sampling source
  normalized_points   float32 [N, 3], full normalized XYZ before FPS
  original_num_points int64   scalar, original point count
  normalize_center    float32 [3], centroid used for normalization
  normalize_scale     float32 scalar, radius used for normalization
  source_path         string, original .pcd path
  sampling_source     string, "pcd" for normal samples or "gt" for supervised anomaly samples
  class_name          string
  split               string
  is_anomaly          bool

`sample_indices` is the key needed to scatter sampled-point predictions back
to the original point cloud. For example:

  original_scores = np.full(original_num_points, np.nan, dtype=np.float32)
  original_scores[sample_indices] = sampled_scores

For backward compatibility, `--output_format npy` keeps the old array output
without metadata. Prefer the default `.npz` format for new experiments.
"""

import argparse
import os
import random
from dataclasses import dataclass
from typing import Callable, Iterable, Optional

import numpy as np
import open3d as o3d
import torch
from tqdm import tqdm


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    gt_dir_name: str
    is_normal_test: Callable[[str], bool]


DATASET_SPECS = {
    "real3d": DatasetSpec(
        name="real3d",
        gt_dir_name="gt",
        is_normal_test=lambda filename: "good" in filename.lower(),
    ),
    "anomaly_shapenet": DatasetSpec(
        name="anomaly_shapenet",
        gt_dir_name="GT",
        is_normal_test=lambda filename: "positive" in filename.lower(),
    ),
}


def set_seed(seed: int) -> None:
    """Set RNG seeds for Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def list_classes(raw_root: str) -> list[str]:
    classes = []
    for name in os.listdir(raw_root):
        p = os.path.join(raw_root, name)
        if os.path.isdir(p):
            classes.append(name)
    classes.sort()
    return classes


def iter_pcd_files(path: str) -> Iterable[str]:
    for filename in sorted(os.listdir(path)):
        if filename.lower().endswith(".pcd"):
            yield filename


def stem(filename: str) -> str:
    return os.path.splitext(filename)[0]


def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    Gather points by indices.

    points: [B, N, C]
    idx: [B, S]
    returns: [B, S, C]
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
    Farthest Point Sampling.

    xyz: [B, N, 3]
    returns sampled indices: [B, npoint]
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
    """Read XYZ coordinates from a .pcd file using Open3D."""
    pcd = o3d.io.read_point_cloud(pcd_path)
    xyz = np.asarray(pcd.points, dtype=np.float32)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"Invalid PCD points shape: {xyz.shape} from {pcd_path}")
    return xyz


def loadtxt_auto_delimiter(txt_path: str) -> np.ndarray:
    """Read whitespace- or comma-delimited numeric txt files."""
    try:
        return np.loadtxt(txt_path, dtype=np.float32)
    except ValueError:
        return np.loadtxt(txt_path, dtype=np.float32, delimiter=",")


def read_gt_txt_xyzl(txt_path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Read XYZ + label from a ground-truth .txt file.

    Expected format per row: x y z label, whitespace- or comma-delimited.
    """
    data = loadtxt_auto_delimiter(txt_path)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.ndim != 2 or data.shape[1] < 4:
        raise ValueError(f"Invalid GT txt shape: {data.shape} from {txt_path}; expected [N,4].")
    xyz = data[:, :3].astype(np.float32)
    label = data[:, 3].astype(np.int64)
    return xyz, label


def normalize_points_np(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.float32]:
    """Unit-sphere normalize a point cloud, matching the original ref preprocessing."""
    center = points.mean(axis=0, keepdims=True).astype(np.float32)
    centered = points - center
    scale = np.max(np.sqrt(np.sum(centered ** 2, axis=-1))).astype(np.float32)
    if float(scale) <= 0:
        scale = np.float32(1.0)
    normalized = (centered / scale).astype(np.float32)
    return normalized, center.reshape(3), scale


def fps_sample_xyz(xyz: np.ndarray, num_samples: int, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """
    FPS-downsample XYZ coordinates.

    Returns sampled_xyz [S, 3] and sample indices [S] relative to the original
    XYZ array.
    """
    pts = torch.from_numpy(xyz).to(device=device, dtype=torch.float32).unsqueeze(0)
    idx = farthest_point_sample(pts, num_samples)
    sampled = index_points(pts, idx).squeeze(0).detach().cpu().numpy()
    idx_np = idx.squeeze(0).detach().cpu().numpy().astype(np.int64)
    return sampled.astype(np.float32), idx_np


def infer_dataset_spec(raw_root: str) -> DatasetSpec:
    classes = list_classes(raw_root)
    if not classes:
        raise FileNotFoundError(f"No class directories found under: {raw_root}")

    has_upper_gt = any(os.path.isdir(os.path.join(raw_root, cls, "GT")) for cls in classes)
    has_lower_gt = any(os.path.isdir(os.path.join(raw_root, cls, "gt")) for cls in classes)
    if has_upper_gt and not has_lower_gt:
        return DATASET_SPECS["anomaly_shapenet"]
    if has_lower_gt and not has_upper_gt:
        return DATASET_SPECS["real3d"]

    raise ValueError(
        "Cannot infer dataset type. Please pass --dataset real3d or --dataset anomaly_shapenet."
    )


def make_output_path(out_split_dir: str, filename: str, output_format: str) -> str:
    return os.path.join(out_split_dir, stem(filename) + f".{output_format}")


def save_sample(
    dst: str,
    output_format: str,
    points: np.ndarray,
    labels: np.ndarray,
    sample_indices: np.ndarray,
    original_points: np.ndarray,
    normalized_points: np.ndarray,
    normalize_center: np.ndarray,
    normalize_scale: np.float32,
    original_num_points: int,
    source_path: str,
    sampling_source: str,
    class_name: str,
    split: str,
    is_anomaly: bool,
) -> None:
    if output_format == "npz":
        np.savez_compressed(
            dst,
            points=points.astype(np.float32),
            labels=labels.astype(np.int64),
            sample_indices=sample_indices.astype(np.int64),
            original_points=original_points.astype(np.float32),
            normalized_points=normalized_points.astype(np.float32),
            original_num_points=np.array(original_num_points, dtype=np.int64),
            normalize_center=normalize_center.astype(np.float32),
            normalize_scale=np.array(normalize_scale, dtype=np.float32),
            source_path=np.array(source_path),
            sampling_source=np.array(sampling_source),
            class_name=np.array(class_name),
            split=np.array(split),
            is_anomaly=np.array(is_anomaly, dtype=bool),
        )
        return

    if output_format == "npy":
        out = np.concatenate([points, labels.reshape(-1, 1).astype(np.float32)], axis=1)
        np.save(dst, out)
        return

    raise ValueError(f"Unsupported output format: {output_format}")


def process_one_pcd(
    src_pcd: str,
    dst: str,
    output_format: str,
    num_samples: int,
    device: torch.device,
    class_name: str,
    split: str,
    is_anomaly: bool,
    gt_txt: Optional[str] = None,
) -> None:
    xyz = read_pcd_xyz(src_pcd)

    if gt_txt is None:
        sampling_source = "pcd"
        original_xyz = xyz
        normalized_xyz, normalize_center, normalize_scale = normalize_points_np(original_xyz)
        original_num_points = normalized_xyz.shape[0]
        sampled_xyz, sample_indices = fps_sample_xyz(normalized_xyz, num_samples, device)
        sampled_labels = np.zeros(num_samples, dtype=np.int64)
    else:
        gt_xyz, labels = read_gt_txt_xyzl(gt_txt)
        if gt_xyz.shape[0] != xyz.shape[0]:
            print(
                "[Warn] PCD/GT point count mismatch; using GT coordinates as sampling source: "
                f"{src_pcd} pcd={xyz.shape[0]}, gt={gt_xyz.shape[0]}"
            )

                                                                             
                                            
        sampling_source = "gt"
        original_xyz = gt_xyz
        normalized_xyz, normalize_center, normalize_scale = normalize_points_np(original_xyz)
        original_num_points = normalized_xyz.shape[0]
        sampled_xyz, sample_indices = fps_sample_xyz(normalized_xyz, num_samples, device)
        sampled_labels = labels[sample_indices]

    save_sample(
        dst=dst,
        output_format=output_format,
        points=sampled_xyz,
        labels=sampled_labels,
        sample_indices=sample_indices,
        original_points=original_xyz,
        normalized_points=normalized_xyz,
        normalize_center=normalize_center,
        normalize_scale=normalize_scale,
        original_num_points=original_num_points,
        source_path=src_pcd,
        sampling_source=sampling_source,
        class_name=class_name,
        split=split,
        is_anomaly=is_anomaly,
    )


def process_dataset(
    raw_root: str,
    out_root: Optional[str],
    dataset: str,
    num_samples: int,
    device: torch.device,
    overwrite: bool,
    output_format: str,
) -> None:
    spec = infer_dataset_spec(raw_root) if dataset == "auto" else DATASET_SPECS[dataset]
    if out_root is None:
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        default_name = {
            "real3d": "Real3D-AD-2048-npz",
            "anomaly_shapenet": "AnomalyShapeNet-2048-npz",
        }[spec.name]
        out_root = os.path.join(repo_root, "data", default_name)

    ensure_dir(out_root)
    classes = list_classes(raw_root)

    print(f"[Info] dataset={spec.name} raw_root={raw_root}")
    print(f"[Info] output_format={output_format} out_root={out_root}")

    for cls in tqdm(classes, desc="Processing classes"):
        class_dir = os.path.join(raw_root, cls)
        out_class_dir = os.path.join(out_root, cls)
        ensure_dir(out_class_dir)

        train_dir = os.path.join(class_dir, "train")
        if os.path.isdir(train_dir):
            out_train_dir = os.path.join(out_class_dir, "train")
            ensure_dir(out_train_dir)

            for fn in iter_pcd_files(train_dir):
                src = os.path.join(train_dir, fn)
                dst = make_output_path(out_train_dir, fn, output_format)
                if (not overwrite) and os.path.exists(dst):
                    continue

                process_one_pcd(
                    src_pcd=src,
                    dst=dst,
                    output_format=output_format,
                    num_samples=num_samples,
                    device=device,
                    class_name=cls,
                    split="train",
                    is_anomaly=False,
                )

        test_dir = os.path.join(class_dir, "test")
        gt_dir = os.path.join(class_dir, spec.gt_dir_name)
        if os.path.isdir(test_dir):
            out_test_dir = os.path.join(out_class_dir, "test")
            ensure_dir(out_test_dir)

            for fn in iter_pcd_files(test_dir):
                src_pcd = os.path.join(test_dir, fn)
                dst = make_output_path(out_test_dir, fn, output_format)
                if (not overwrite) and os.path.exists(dst):
                    continue

                is_normal = spec.is_normal_test(fn)
                gt_txt = None
                if not is_normal:
                    gt_txt = os.path.join(gt_dir, stem(fn) + ".txt")
                    if not os.path.exists(gt_txt):
                        raise FileNotFoundError(f"Missing GT txt for {src_pcd}: expected {gt_txt}")

                process_one_pcd(
                    src_pcd=src_pcd,
                    dst=dst,
                    output_format=output_format,
                    num_samples=num_samples,
                    device=device,
                    class_name=cls,
                    split="test",
                    is_anomaly=not is_normal,
                    gt_txt=gt_txt,
                )


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Preprocess Real3D-AD or Anomaly-ShapeNet point clouds with FPS."
    )
    p.add_argument("--raw_root", type=str, required=True, help="Path to the raw dataset root directory.")
    p.add_argument(
        "--out_root",
        type=str,
        default=None,
        help="Path to the output root directory. Defaults to ./data/<dataset>-2048-npz.",
    )
    p.add_argument(
        "--dataset",
        type=str,
        default="auto",
        choices=["auto", "real3d", "anomaly_shapenet"],
        help="Dataset type. 'auto' infers it from gt/GT directories.",
    )
    p.add_argument("--num_samples", type=int, default=2048, help="Number of points after FPS downsampling.")
    p.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"], help='Compute device: "cpu" or "cuda".')
    p.add_argument("--seed", type=int, default=0, help="Random seed for FPS initialization.")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing output files if set.")
    p.add_argument(
        "--output_format",
        type=str,
        default="npz",
        choices=["npz", "npy"],
        help="Use npz to keep sample_indices metadata; npy is legacy array output.",
    )
    return p


def main() -> None:
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
        dataset=args.dataset,
        num_samples=args.num_samples,
        device=device,
        overwrite=args.overwrite,
        output_format=args.output_format,
    )
    print(f"[OK] Done. Output root: {args.out_root}")


if __name__ == "__main__":
    main()
