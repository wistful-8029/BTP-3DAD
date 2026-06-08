import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, auc, roc_auc_score
from tqdm import tqdm

from data.anomaly_datasets import PointCloudDataset
from models.customized_network.GFCM import GeometricFeatureCreationModule
from models.customized_network.HybridPromptLearner import HybridPromptLearner
from models.customized_network.MultiLayerFeatureEmbedding import MultiLayerFeatureEncoder
from models.ulip2_encoder import ULIP2Encoder
from utils.utils import compute_global_anomaly_score, compute_patch_scores, patch_scores_to_point_scores


def f1_max_from_scores(y_true, y_score, eps=1e-12):
    y_true = np.asarray(y_true).astype(int).ravel()
    y_score = np.asarray(y_score).ravel()
    if y_true.min() == y_true.max():
        return float("nan")

    order = np.argsort(-y_score, kind="mergesort")
    y_sorted = y_true[order]
    positives = y_true.sum()
    tp_cum = np.cumsum(y_sorted)
    predicted_positive = np.arange(1, len(y_true) + 1)
    precision = tp_cum / (predicted_positive + eps)
    recall = tp_cum / (positives + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    return float(np.nanmax(f1))


def compute_point_level_aupro(point_scores_list, point_labels_list, num_points=100):
    eval_points = np.linspace(0, 1, num_points)
    interp_pro_list = []
    for scores, labels in zip(point_scores_list, point_labels_list):
        labels = labels.astype(bool)
        n_anomaly = labels.sum()
        if n_anomaly == 0:
            continue
        sorted_idx = np.argsort(-scores)
        sorted_labels = labels[sorted_idx]
        pro_curve = np.cumsum(sorted_labels) / n_anomaly
        coverage_curve = np.arange(1, len(labels) + 1) / len(labels)
        interp_pro_list.append(np.interp(eval_points, coverage_curve, pro_curve))
    if not interp_pro_list:
        return float("nan")
    return auc(eval_points, np.mean(np.vstack(interp_pro_list), axis=0))


class DisjointSet:
    def __init__(self, size):
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a == root_b:
            return
        if self.rank[root_a] < self.rank[root_b]:
            root_a, root_b = root_b, root_a
        self.parent[root_b] = root_a
        if self.rank[root_a] == self.rank[root_b]:
            self.rank[root_a] += 1


def estimate_region_radius(points, positive_mask, k=6, scale=2.5):
    positive_points = points[positive_mask]
    if positive_points.shape[0] <= 1:
        return 0.0
    try:
        from scipy.spatial import cKDTree

        query_k = min(k + 1, positive_points.shape[0])
        distances, _ = cKDTree(positive_points).query(positive_points, k=query_k, workers=-1)
        nn_distances = distances[:, 1:].reshape(-1)
        nn_distances = nn_distances[np.isfinite(nn_distances) & (nn_distances > 0)]
        if nn_distances.size == 0:
            return 0.0
        return float(np.median(nn_distances) * scale)
    except Exception:
        diffs = positive_points[: min(positive_points.shape[0], 2048)]
        if diffs.shape[0] <= 1:
            return 0.0
        dist = torch.cdist(torch.from_numpy(diffs), torch.from_numpy(diffs))
        dist[dist == 0] = float("inf")
        nn_distances = dist.min(dim=1).values.numpy()
        nn_distances = nn_distances[np.isfinite(nn_distances)]
        return float(np.median(nn_distances) * scale) if nn_distances.size else 0.0


def connected_anomaly_regions(points, labels, radius=0.0, k=16, radius_scale=2.5, min_region_size=1):
    labels = np.asarray(labels).astype(bool).reshape(-1)
    positive_indices = np.flatnonzero(labels)
    if positive_indices.size == 0:
        return []
    if positive_indices.size == 1:
        return [positive_indices]

    points = np.asarray(points, dtype=np.float32)
    positive_points = points[positive_indices]
    if radius <= 0:
        radius = estimate_region_radius(points, labels, k=min(k, 6), scale=radius_scale)
    if radius <= 0:
        return [positive_indices]

    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(positive_points)
        neighbor_lists = tree.query_ball_point(positive_points, r=radius, workers=-1)
        dsu = DisjointSet(positive_indices.size)
        for i, neighbors in enumerate(neighbor_lists):
            for j in neighbors:
                if i != j:
                    dsu.union(i, j)
    except Exception:
        dsu = DisjointSet(positive_indices.size)
        pts = torch.from_numpy(positive_points)
        chunk_size = 4096
        for start in range(0, pts.shape[0], chunk_size):
            end = min(start + chunk_size, pts.shape[0])
            dist = torch.cdist(pts[start:end], pts)
            local_neighbors = dist <= radius
            rows, cols = torch.nonzero(local_neighbors, as_tuple=True)
            for row, col in zip(rows.tolist(), cols.tolist()):
                i = start + row
                if i != col:
                    dsu.union(i, col)

    groups = defaultdict(list)
    for local_idx, original_idx in enumerate(positive_indices):
        groups[dsu.find(local_idx)].append(original_idx)
    regions = [np.asarray(indices, dtype=np.int64) for indices in groups.values() if len(indices) >= min_region_size]
    if not regions and positive_indices.size:
        return [positive_indices]
    return regions


def compute_standard_fullres_aupro(
    score_list,
    label_list,
    point_list,
    fpr_limit=0.3,
    num_thresholds=200,
    region_radius=0.0,
    region_knn=16,
    region_radius_scale=2.5,
    min_region_size=1,
):
    all_scores = [np.asarray(scores, dtype=np.float32).reshape(-1) for scores in score_list]
    all_labels = [np.asarray(labels).astype(bool).reshape(-1) for labels in label_list]
    all_points = [np.asarray(points, dtype=np.float32).reshape(-1, 3) for points in point_list]

    regions_by_sample = []
    total_normal_points = 0
    for scores, labels, points in zip(all_scores, all_labels, all_points):
        if scores.shape[0] != labels.shape[0] or points.shape[0] != labels.shape[0]:
            raise ValueError("Full-res scores, labels, and points must have the same length for AU-PRO.")
        total_normal_points += int((~labels).sum())
        regions = connected_anomaly_regions(
            points,
            labels,
            radius=region_radius,
            k=region_knn,
            radius_scale=region_radius_scale,
            min_region_size=min_region_size,
        )
        regions_by_sample.append(regions)

    total_regions = sum(len(regions) for regions in regions_by_sample)
    if total_regions == 0 or total_normal_points == 0:
        return float("nan"), 0

    finite_scores = np.concatenate([scores[np.isfinite(scores)] for scores in all_scores])
    if finite_scores.size == 0:
        return float("nan"), total_regions
    min_score, max_score = float(finite_scores.min()), float(finite_scores.max())
    if min_score == max_score:
        return float("nan"), total_regions

    thresholds = np.linspace(max_score + 1e-12, min_score - 1e-12, num_thresholds)
    fprs, pros = [], []
    for threshold in thresholds:
        false_positive_count = 0
        region_overlaps = []
        for scores, labels, regions in zip(all_scores, all_labels, regions_by_sample):
            pred = scores >= threshold
            false_positive_count += int(np.logical_and(pred, ~labels).sum())
            for region in regions:
                region_overlaps.append(float(pred[region].mean()))

        fprs.append(false_positive_count / total_normal_points)
        pros.append(float(np.mean(region_overlaps)) if region_overlaps else 0.0)

    fprs = np.asarray(fprs, dtype=np.float64)
    pros = np.asarray(pros, dtype=np.float64)
    order = np.argsort(fprs, kind="mergesort")
    fprs, pros = fprs[order], pros[order]

    unique_fprs, inverse = np.unique(fprs, return_inverse=True)
    unique_pros = np.zeros_like(unique_fprs)
    for idx in range(unique_fprs.size):
        unique_pros[idx] = pros[inverse == idx].max()

    fprs = np.concatenate([[0.0], unique_fprs])
    pros = np.concatenate([[0.0], unique_pros])
    valid = fprs <= fpr_limit
    clipped_fprs = fprs[valid]
    clipped_pros = pros[valid]
    if clipped_fprs.size == 0 or clipped_fprs[-1] < fpr_limit:
        boundary_pro = np.interp(fpr_limit, fprs, pros)
        clipped_fprs = np.concatenate([clipped_fprs, [fpr_limit]])
        clipped_pros = np.concatenate([clipped_pros, [boundary_pro]])

    return float(auc(clipped_fprs, clipped_pros) / fpr_limit), total_regions


def loadtxt_auto_delimiter(path):
    try:
        return np.loadtxt(path, dtype=np.float32)
    except ValueError:
        return np.loadtxt(path, dtype=np.float32, delimiter=",")


def scalar_to_str(value):
    arr = np.asarray(value)
    return str(arr.item() if arr.shape == () else arr)


def nearest_sampled_scores_to_full(normalized_points, sample_indices, sampled_scores):
    sampled_points = normalized_points[sample_indices]
    try:
        from scipy.spatial import cKDTree

        nearest_idx = cKDTree(sampled_points).query(normalized_points, k=1, workers=-1)[1]
        return sampled_scores[nearest_idx].astype(np.float32)
    except Exception:
        full_scores = np.empty(normalized_points.shape[0], dtype=np.float32)
        query = torch.from_numpy(normalized_points.astype(np.float32))
        keys = torch.from_numpy(sampled_points.astype(np.float32))
        scores = torch.from_numpy(sampled_scores.astype(np.float32))
        chunk_size = 32768
        for start in range(0, query.shape[0], chunk_size):
            end = min(start + chunk_size, query.shape[0])
            dist = torch.cdist(query[start:end], keys)
            nearest_idx = dist.argmin(dim=1)
            full_scores[start:end] = scores[nearest_idx].numpy()
        return full_scores


def fullres_from_npz(sample, sampled_scores):
    path = sample["path"]
    if not path.endswith(".npz"):
        return None

    with np.load(path) as z:
        required = {"normalized_points", "sample_indices"}
        if not required.issubset(set(z.files)):
            return None

        normalized_points = z["normalized_points"].astype(np.float32)
        original_points = z["original_points"].astype(np.float32) if "original_points" in z.files else normalized_points
        sample_indices = z["sample_indices"].astype(np.int64)
        sampled_scores = np.asarray(sampled_scores, dtype=np.float32).reshape(-1)
        if sample_indices.shape[0] != sampled_scores.shape[0]:
            raise ValueError(
                f"sample_indices length {sample_indices.shape[0]} != sampled_scores length {sampled_scores.shape[0]} for {path}"
            )

        if "original_labels" in z.files:
            full_labels = z["original_labels"].astype(np.int64).reshape(-1)
        else:
            source = scalar_to_str(z["sampling_source"]) if "sampling_source" in z.files else ""
            if source == "gt":
                source_path = Path(scalar_to_str(z["source_path"])) if "source_path" in z.files else Path(path)
                gt_path = source_path.parent.parent / "gt" / f"{source_path.stem}.txt"
                if not gt_path.exists():
                    gt_path = source_path.parent.parent / "GT" / f"{source_path.stem}.txt"
                gt_data = loadtxt_auto_delimiter(gt_path)
                if gt_data.ndim == 1:
                    gt_data = gt_data.reshape(1, -1)
                full_labels = gt_data[:, 3].astype(np.int64)
            else:
                full_labels = np.zeros(normalized_points.shape[0], dtype=np.int64)

    if full_labels.shape[0] != normalized_points.shape[0]:
        raise ValueError(
            f"Full-res label/point count mismatch for {path}: labels={full_labels.shape[0]}, points={normalized_points.shape[0]}"
        )

    full_scores = nearest_sampled_scores_to_full(normalized_points, sample_indices, sampled_scores)
    full_scores[sample_indices] = sampled_scores
    return original_points, normalized_points, full_scores, full_labels, sample_indices


def build_models(args, device):
    encoder = ULIP2Encoder(
        args.model_path,
        device=device,
        num_points=args.num_points,
        return_layers=tuple(args.return_layers),
        return_clip=True,
    )
    encoder.model.eval()
    for param in encoder.model.parameters():
        param.requires_grad = False

    patch_feature_embedder = MultiLayerFeatureEncoder(fpfh_dim=args.geo_dim).to(device)
    patch_geo_encoder = GeometricFeatureCreationModule(out_dim=args.geo_dim).to(device)

    clip_model = encoder.open_clip_model
    prompt_learner = HybridPromptLearner(
        clip_model.to("cpu"),
        {
            "Prompt_length": args.n_ctx,
            "learnabel_text_embedding_depth": args.depth,
            "learnabel_text_embedding_length": args.t_n_ctx,
        },
    ).to(device)
    clip_model.to(device)

    for module in (patch_feature_embedder, patch_geo_encoder, prompt_learner):
        module.eval()

    return encoder, patch_feature_embedder, patch_geo_encoder, prompt_learner


def infer_train_class(args, checkpoint_path):
    if args.train_class:
        return args.train_class
    run_dir = os.path.dirname(checkpoint_path)
    return os.path.basename(os.path.dirname(run_dir))


def save_visual_npz(args, test_dataset, sample, points, point_probs, point_labels, category_idx):
    if not args.save_visual:
        return
    class_name = test_dataset.PRESETS[args.dataset_name][category_idx]
    out_dir = Path(args.save_dir) / class_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{Path(sample['path']).stem}.npz"
    np.savez_compressed(
        out_path,
        points=points.squeeze(0).detach().cpu().numpy(),
        score=point_probs.detach().cpu().numpy().reshape(-1),
        gt=point_labels.detach().cpu().numpy().reshape(-1),
    )


def save_fullres_npz(args, test_dataset, sample, fullres, sampled_points, sampled_scores, sampled_labels, category_idx):
    if not args.save_fullres or fullres is None:
        return

    class_name = test_dataset.PRESETS[args.dataset_name][category_idx]
    out_dir = Path(args.save_dir) / class_name / "fullres"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{Path(sample['path']).stem}.npz"

    original_points, normalized_points, full_scores, full_labels, sample_indices = fullres
    np.savez_compressed(
        out_path,
        points=original_points,
        normalized_points=normalized_points,
        score=full_scores,
        gt=full_labels,
        sample_indices=sample_indices,
        sampled_points=sampled_points,
        sampled_score=sampled_scores,
        sampled_gt=sampled_labels,
    )


@torch.no_grad()
def evaluate_checkpoint(args, device, models, checkpoint_path, test_dataset, result_file, per_class_csv):
    encoder, patch_feature_embedder, patch_geo_encoder, prompt_learner = models
    checkpoint = torch.load(checkpoint_path, map_location=device)
    patch_feature_embedder.load_state_dict(checkpoint["patch_feature_embedder"])
    patch_geo_encoder.load_state_dict(checkpoint["patch_geo_encoder"])
    prompt_learner.load_state_dict(checkpoint["prompt_learner"])

    for module in (patch_feature_embedder, patch_geo_encoder, prompt_learner):
        module.eval()

    _, tokenized_prompts = prompt_learner()
    text_normal_embed, text_anomaly_embed = encoder.encode_text_from_tokens(tokenized_prompts)

    class_names = test_dataset.PRESETS[args.dataset_name]
    category_scores = defaultdict(list)
    category_labels = defaultdict(list)
    category_point_scores = defaultdict(list)
    category_point_labels = defaultdict(list)
    category_point_points = defaultdict(list)
    category_full_point_scores = defaultdict(list)
    category_full_point_labels = defaultdict(list)
    category_full_point_points = defaultdict(list)

    for idx in tqdm(range(len(test_dataset)), desc=f"Eval {Path(checkpoint_path).parent.name}", dynamic_ncols=True):
        sample = test_dataset[idx]
        points = sample["points"].unsqueeze(0).to(device)
        point_labels = sample["labels"]
        global_label = float(point_labels.sum() > 0)
        category_idx = sample["category"]

        feat_dict = encoder.encode_pointcloud(points, return_intermediate=True)
        point_embeddings = feat_dict["concat"]
        global_embed = feat_dict["global"]
        layer_features = feat_dict["layer_feats"]
        patch_indices = feat_dict["patch_idx"]

        layer_features_list = [layer_features[layer][:, 1:, :].contiguous() for layer in args.return_layers]
        patch_geo_feats = patch_geo_encoder(points, patch_indices)
        patch_embeddings = patch_feature_embedder(layer_features_list, patch_geo_feats, global_embed)

        _, patch_probs = compute_patch_scores(patch_embeddings, text_normal_embed, text_anomaly_embed)
        point_probs = patch_scores_to_point_scores(patch_probs, patch_indices, args.num_points)
        _, global_probs = compute_global_anomaly_score(point_embeddings, text_normal_embed, text_anomaly_embed)

        k = max(1, int(patch_probs.shape[1] * args.topk_ratio))
        global_probs_from_patch = torch.topk(patch_probs, k=k, dim=1).values.mean(dim=1)
        global_probs_combined = args.global_alpha * global_probs + (1 - args.global_alpha) * global_probs_from_patch

        sampled_points = points.squeeze(0).detach().cpu().numpy()
        sampled_scores = point_probs.detach().cpu().numpy().reshape(-1)
        sampled_labels = point_labels.detach().cpu().numpy().reshape(-1)
        category_scores[category_idx].append(float(global_probs_combined.cpu().item()))
        category_labels[category_idx].append(global_label)
        category_point_scores[category_idx].append(sampled_scores)
        category_point_labels[category_idx].append(sampled_labels)
        category_point_points[category_idx].append(sampled_points)

        fullres = None
        if args.fullres_eval or args.save_fullres:
            fullres = fullres_from_npz(sample, sampled_scores)
            if fullres is not None and args.fullres_eval:
                original_points, _, full_scores, full_labels, _ = fullres
                category_full_point_scores[category_idx].append(full_scores)
                category_full_point_labels[category_idx].append(full_labels)
                category_full_point_points[category_idx].append(original_points)

        save_visual_npz(args, test_dataset, sample, points, point_probs, point_labels, category_idx)
        save_fullres_npz(args, test_dataset, sample, fullres, sampled_points, sampled_scores, sampled_labels, category_idx)

    train_class_name = infer_train_class(args, checkpoint_path)
    object_auroc_list, object_ap_list, object_f1max_list = [], [], []
    point_auroc_list, point_ap_list, point_aupro_list = [], [], []
    point_region_count_list = []
    full_point_auroc_list, full_point_ap_list, full_point_aupro_list = [], [], []
    full_region_count_list = []

    with open(result_file, "a") as f:
        header_lines = [
            f"Results for checkpoint: {checkpoint_path}",
        ]
        for line in header_lines:
            print(line)
            f.write(line + "\n")

        for category_idx, class_name in enumerate(class_names):
            obj_scores = np.asarray(category_scores[category_idx]).reshape(-1)
            obj_labels = np.asarray(category_labels[category_idx]).reshape(-1).astype(int)
            if obj_scores.size > 0 and len(np.unique(obj_labels)) >= 2:
                obj_auroc = roc_auc_score(obj_labels, obj_scores)
                obj_ap = average_precision_score(obj_labels, obj_scores)
                obj_f1max = f1_max_from_scores(obj_labels, obj_scores)
            else:
                obj_auroc = obj_ap = obj_f1max = float("nan")

            point_scores_list = category_point_scores[category_idx]
            point_labels_list = category_point_labels[category_idx]
            point_points_list = category_point_points[category_idx]
            point_aupro, point_region_count = compute_standard_fullres_aupro(
                point_scores_list,
                point_labels_list,
                point_points_list,
                fpr_limit=args.aupro_fpr_limit,
                num_thresholds=args.aupro_thresholds,
                region_radius=args.sampled_aupro_region_radius,
                region_knn=args.aupro_region_knn,
                region_radius_scale=args.aupro_region_radius_scale,
                min_region_size=args.aupro_min_region_size,
            )
            all_point_scores = np.concatenate(point_scores_list, axis=0) if point_scores_list else np.array([])
            all_point_labels = np.concatenate(point_labels_list, axis=0) if point_labels_list else np.array([])
            if all_point_scores.size > 0 and len(np.unique(all_point_labels)) >= 2:
                point_auroc = roc_auc_score(all_point_labels, all_point_scores)
                point_ap = average_precision_score(all_point_labels, all_point_scores)
            else:
                point_auroc = point_ap = float("nan")

            full_point_scores_list = category_full_point_scores[category_idx]
            full_point_labels_list = category_full_point_labels[category_idx]
            full_point_points_list = category_full_point_points[category_idx]
            full_point_aupro, full_region_count = compute_standard_fullres_aupro(
                full_point_scores_list,
                full_point_labels_list,
                full_point_points_list,
                fpr_limit=args.aupro_fpr_limit,
                num_thresholds=args.aupro_thresholds,
                region_radius=args.aupro_region_radius,
                region_knn=args.aupro_region_knn,
                region_radius_scale=args.aupro_region_radius_scale,
                min_region_size=args.aupro_min_region_size,
            )
            all_full_point_scores = np.concatenate(full_point_scores_list, axis=0) if full_point_scores_list else np.array([])
            all_full_point_labels = np.concatenate(full_point_labels_list, axis=0) if full_point_labels_list else np.array([])
            if all_full_point_scores.size > 0 and len(np.unique(all_full_point_labels)) >= 2:
                full_point_auroc = roc_auc_score(all_full_point_labels, all_full_point_scores)
                full_point_ap = average_precision_score(all_full_point_labels, all_full_point_scores)
            else:
                full_point_auroc = full_point_ap = float("nan")

            metric_lines = [
                f"{class_name} [Object] AUROC={obj_auroc:.4f}, AP={obj_ap:.4f}, F1max={obj_f1max:.4f}",
                f"{class_name} [Point sampled-2048] StdAU-PRO={point_aupro:.4f}, AUROC={point_auroc:.4f}, AP={point_ap:.4f}, regions={point_region_count}",
                f"{class_name} [Point full-res]     StdAU-PRO={full_point_aupro:.4f}, AUROC={full_point_auroc:.4f}, AP={full_point_ap:.4f}, regions={full_region_count}",
            ]
            for line in metric_lines:
                print(line)
                f.write(line + "\n")

            if class_name != train_class_name:
                if not np.isnan(obj_auroc):
                    object_auroc_list.append(obj_auroc)
                if not np.isnan(obj_ap):
                    object_ap_list.append(obj_ap)
                if not np.isnan(obj_f1max):
                    object_f1max_list.append(obj_f1max)
                if not np.isnan(point_auroc):
                    point_auroc_list.append(point_auroc)
                if not np.isnan(point_ap):
                    point_ap_list.append(point_ap)
                if not np.isnan(point_aupro):
                    point_aupro_list.append(point_aupro)
                    point_region_count_list.append(point_region_count)
                if not np.isnan(full_point_auroc):
                    full_point_auroc_list.append(full_point_auroc)
                if not np.isnan(full_point_ap):
                    full_point_ap_list.append(full_point_ap)
                if not np.isnan(full_point_aupro):
                    full_point_aupro_list.append(full_point_aupro)
                    full_region_count_list.append(full_region_count)

            need_header = not os.path.exists(per_class_csv)
            with open(per_class_csv, "a", newline="") as cf:
                writer = csv.writer(cf)
                if need_header:
                    writer.writerow([
                        "checkpoint", "train_class", "class",
                        "obj_AUROC", "obj_AP", "obj_F1max",
                        "point_std_AUPRO", "point_AUROC", "point_AP", "point_region_count",
                        "full_point_std_AUPRO", "full_point_AUROC", "full_point_AP", "full_point_region_count",
                    ])
                writer.writerow([
                    checkpoint_path, train_class_name, class_name,
                    obj_auroc, obj_ap, obj_f1max,
                    point_aupro, point_auroc, point_ap, point_region_count,
                    full_point_aupro, full_point_auroc, full_point_ap, full_region_count,
                ])

        if object_auroc_list:
            line = (
                "Average (excluding train class) - Object: "
                f"AUROC={np.mean(object_auroc_list):.4f}, AP={np.mean(object_ap_list):.4f}, F1max={np.mean(object_f1max_list):.4f}"
            )
            print(line)
            f.write(line + "\n")
        if point_auroc_list:
            line = (
                "Average (excluding train class) - Point sampled-2048: "
                f"StdAU-PRO={np.mean(point_aupro_list):.4f}, AUROC={np.mean(point_auroc_list):.4f}, AP={np.mean(point_ap_list):.4f}, "
                f"regions={np.sum(point_region_count_list)}"
            )
            print(line)
            f.write(line + "\n")
        if full_point_auroc_list:
            line = (
                "Average (excluding train class) - Point full-res: "
                f"StdAU-PRO={np.mean(full_point_aupro_list):.4f}, AUROC={np.mean(full_point_auroc_list):.4f}, AP={np.mean(full_point_ap_list):.4f}, "
                f"regions={np.sum(full_region_count_list)}"
            )
            print(line)
            f.write(line + "\n")
        f.write("\n")
        print("")


def find_checkpoints(ckpt_root):
    if os.path.isfile(ckpt_root):
        return [ckpt_root]
    ckpts = []
    for root, _, files in os.walk(ckpt_root):
        for fname in files:
            if fname.endswith("best.pth"):
                ckpts.append(os.path.join(root, fname))
    return sorted(ckpts)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate trained ULIP2 3D anomaly checkpoints with standard full-resolution AU-PRO."
    )
    parser.add_argument("--ckpt_root", type=str, required=True)
    parser.add_argument("--train_class", type=str, default=None, help="Class used for one-vs-rest training; excluded from macro averages.")
    parser.add_argument("--output_dir", type=str, default="./outputs/test")
    parser.add_argument("--save_dir", type=str, default="./outputs/test/visual")
    parser.add_argument("--save_visual", action="store_true")
    parser.add_argument(
        "--fullres_eval",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Map sampled 2048-point scores back to original point-cloud resolution and report full-res metrics.",
    )
    parser.add_argument(
        "--save_fullres",
        action="store_true",
        help="Save full-resolution score/GT npz files under <save_dir>/<class>/fullres.",
    )
    parser.add_argument("--data_root", type=str, default="./data/Real3D-AD-2048-npz")
    parser.add_argument("--model_path", type=str, default="./pretrained/ULIP-2-PointBERT-10k-xyzrgb-pc-vit_g-objaverse_shapenet-pretrained.pt")
    parser.add_argument("--dataset_name", type=str, default="Real3D", choices=list(PointCloudDataset.PRESETS))
    parser.add_argument("--classes", type=str, nargs="+", default=None)
    parser.add_argument("--num_points", type=int, default=2048)
    parser.add_argument("--return_layers", type=int, nargs="+", default=[2, 5, 8, 11])
    parser.add_argument("--depth", type=int, default=9)
    parser.add_argument("--n_ctx", type=int, default=8)
    parser.add_argument("--t_n_ctx", type=int, default=4)
    parser.add_argument("--geo_dim", type=int, default=33)
    parser.add_argument("--global_alpha", type=float, default=0.5)
    parser.add_argument("--topk_ratio", type=float, default=0.2)
    parser.add_argument("--aupro_fpr_limit", type=float, default=0.3)
    parser.add_argument("--aupro_thresholds", type=int, default=200)
    parser.add_argument(
        "--aupro_region_radius",
        type=float,
        default=0.0,
        help="Radius for connected full-res GT anomaly regions. <=0 estimates an adaptive radius per sample.",
    )
    parser.add_argument(
        "--sampled_aupro_region_radius",
        type=float,
        default=0.0,
        help="Radius for connected sampled-2048 GT anomaly regions. <=0 estimates an adaptive radius per sample.",
    )
    parser.add_argument("--aupro_region_knn", type=int, default=16)
    parser.add_argument("--aupro_region_radius_scale", type=float, default=2.5)
    parser.add_argument("--aupro_min_region_size", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    if args.save_visual or args.save_fullres:
        os.makedirs(args.save_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu"
    models = build_models(args, device)
    test_dataset = PointCloudDataset(
        root_dir=args.data_root,
        split="test",
        transform=None,
        classes=args.classes,
        dataset_name=args.dataset_name,
    )
    if len(test_dataset) == 0:
        raise FileNotFoundError(f"No test samples found under {args.data_root}")

    ckpts = find_checkpoints(args.ckpt_root)
    if not ckpts:
        raise FileNotFoundError(f"No .pth checkpoints found under {args.ckpt_root}")

    result_file = os.path.join(args.output_dir, "test_results.txt")
    per_class_csv = os.path.join(args.output_dir, "test_per_class.csv")
    with open(result_file, "a") as f:
        f.write("=== Eval start ===\n")

    for ckpt in ckpts:
        evaluate_checkpoint(args, device, models, ckpt, test_dataset, result_file, per_class_csv)

    print(f"Summary text: {result_file}")
    print(f"Per-class CSV: {per_class_csv}")


if __name__ == "__main__":
    main()
