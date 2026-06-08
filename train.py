import argparse
import csv
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from data.anomaly_datasets import PointCloudDataset
from loss import BinaryDiceLoss, BinaryFocalLoss, FPFHSupervisionLoss
from models.customized_network.GFCM import GeometricFeatureCreationModule
from models.customized_network.HybridPromptLearner import HybridPromptLearner
from models.customized_network.MultiLayerFeatureEmbedding import MultiLayerFeatureEncoder
from models.ulip2_encoder import ULIP2Encoder
from utils.utils import (
    compute_global_anomaly_score,
    compute_patch_FPFH,
    compute_patch_scores,
    patch_scores_to_point_scores,
)


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_trainable_modules(args, device):
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

    return encoder, patch_feature_embedder, patch_geo_encoder, prompt_learner


def build_optimizer(args, patch_feature_embedder, patch_geo_encoder, prompt_learner):
    decay, no_decay = [], []

    def add_param_groups(module):
        for name, param in module.named_parameters():
            if not param.requires_grad:
                continue
            lname = name.lower()
            if lname.endswith("bias") or "norm" in lname or "ln" in lname or "embed" in lname or "prompt" in lname:
                no_decay.append(param)
            else:
                decay.append(param)

    add_param_groups(patch_feature_embedder)
    add_param_groups(patch_geo_encoder)
    prompt_params = [p for p in prompt_learner.parameters() if p.requires_grad]

    return AdamW(
        [
            {"params": prompt_params, "lr": args.learning_rate, "weight_decay": 0.0},
            {"params": decay, "lr": args.learning_rate * 0.5, "weight_decay": args.weight_decay},
            {"params": no_decay, "lr": args.learning_rate * 0.5, "weight_decay": 0.0},
        ],
        betas=(0.9, 0.999),
    )


def save_checkpoint(path, patch_feature_embedder, patch_geo_encoder, prompt_learner, epoch, loss):
    torch.save(
        {
            "patch_feature_embedder": patch_feature_embedder.state_dict(),
            "patch_geo_encoder": patch_geo_encoder.state_dict(),
            "prompt_learner": prompt_learner.state_dict(),
            "epoch": epoch,
            "loss": loss,
        },
        path,
    )


def summarize_labels(dataset):
    total_points = 0
    positive_points = 0
    anomaly_samples = 0
    for sample in dataset:
        labels = sample["labels"]
        positives = int((labels > 0).sum().item())
        total_points += int(labels.numel())
        positive_points += positives
        anomaly_samples += int(positives > 0)
    return {
        "samples": len(dataset),
        "anomaly_samples": anomaly_samples,
        "positive_points": positive_points,
        "total_points": total_points,
    }


def train_one_run(args, class_name, run_idx, seed, run_dir):
    os.makedirs(run_dir, exist_ok=True)
    setup_seed(seed)
    device = "cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu"

    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump({"class": class_name, "run_idx": run_idx, "seed": seed, **vars(args)}, f, indent=2)

    train_dataset = PointCloudDataset(
        root_dir=args.data_root,
        split=args.train_split,
        transform=None,
        classes=[class_name],
        dataset_name=args.dataset_name,
    )
    if len(train_dataset) == 0:
        raise FileNotFoundError(f"No samples found for class={class_name}, split={args.train_split}, root={args.data_root}")

    label_summary = summarize_labels(train_dataset)
    label_msg = (
        f"Loaded {label_summary['samples']} samples for {class_name}/{args.train_split} | "
        f"anomaly_samples={label_summary['anomaly_samples']} | "
        f"positive_points={label_summary['positive_points']}/{label_summary['total_points']}"
    )
    print(label_msg)
    if label_summary["positive_points"] == 0:
        print("Warning: this split has no positive point labels; point/global anomaly losses may not learn anomalies.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        drop_last=args.drop_last,
        pin_memory=True,
        num_workers=args.num_workers,
    )

    encoder, patch_feature_embedder, patch_geo_encoder, prompt_learner = build_trainable_modules(args, device)
    optimizer = build_optimizer(args, patch_feature_embedder, patch_geo_encoder, prompt_learner)

    total_iters = args.epochs * max(1, len(train_loader))
    warmup_iters = max(1, int(args.warmup_ratio * total_iters))
    scheduler = SequentialLR(
        optimizer,
        schedulers=[
            LinearLR(optimizer, start_factor=0.1, total_iters=warmup_iters),
            CosineAnnealingLR(optimizer, T_max=max(1, total_iters - warmup_iters), eta_min=args.min_lr),
        ],
        milestones=[warmup_iters],
    )

    global_bce_criterion = nn.BCEWithLogitsLoss()
    point_focal_criterion = BinaryFocalLoss(gamma=args.focal_gamma, alpha=args.focal_alpha)
    point_dice_criterion = BinaryDiceLoss()
    geo_supervision_criterion = FPFHSupervisionLoss(mode=args.geo_loss)

    log_file = os.path.join(run_dir, "train_log.txt")
    best_loss = float("inf")
    best_epoch = -1
    start_time = time.time()

    for epoch in range(args.epochs):
        patch_feature_embedder.train()
        patch_geo_encoder.train()
        prompt_learner.train()

        totals = {"point_focal": 0.0, "point_dice": 0.0, "global": 0.0, "geo": 0.0, "total": 0.0}

        for batch in train_loader:
            points = batch["points"].to(device, non_blocking=True)
            point_labels = batch["labels"].to(device, non_blocking=True)
            global_labels = (point_labels.sum(dim=1) > 0).float()

            feat_dict = encoder.encode_pointcloud(points, return_intermediate=True)
            point_embeddings = feat_dict["concat"]
            global_embed = feat_dict["global"]
            layer_features = feat_dict["layer_feats"]
            patch_indices = feat_dict["patch_idx"]

            layer_features_list = [layer_features[layer][:, 1:, :].contiguous() for layer in args.return_layers]
            patch_fpfh_feats = compute_patch_FPFH(points, patch_indices, agg=args.fpfh_agg).to(device)
            patch_geo_feats = patch_geo_encoder(points, patch_indices)
            patch_embeddings = patch_feature_embedder(layer_features_list, patch_geo_feats, global_embed)

            _, tokenized_prompts = prompt_learner()
            text_normal_embed, text_anomaly_embed = encoder.encode_text_from_tokens(tokenized_prompts)

            patch_logits, patch_probs = compute_patch_scores(patch_embeddings, text_normal_embed, text_anomaly_embed)
            global_logits_from_patch, _ = torch.max(patch_logits, dim=1)
            point_logits = patch_scores_to_point_scores(patch_logits, patch_indices, args.num_points)
            global_logits, _ = compute_global_anomaly_score(point_embeddings, text_normal_embed, text_anomaly_embed)
            global_logits_combined = args.global_alpha * global_logits + (1 - args.global_alpha) * global_logits_from_patch

            loss_point_focal = point_focal_criterion(point_logits, point_labels)
            loss_point_dice = point_dice_criterion(point_logits, point_labels)
            loss_global = global_bce_criterion(global_logits_combined, global_labels)
            loss_geo = geo_supervision_criterion(patch_geo_feats, patch_fpfh_feats)
            loss_total = (
                loss_point_focal
                + loss_point_dice
                + args.global_loss_weight * loss_global
                + args.geo_loss_weight * loss_geo
            )

            optimizer.zero_grad(set_to_none=True)
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(
                list(prompt_learner.parameters())
                + list(patch_feature_embedder.parameters())
                + list(patch_geo_encoder.parameters()),
                max_norm=args.grad_clip,
            )
            optimizer.step()
            scheduler.step()

            totals["point_focal"] += loss_point_focal.item()
            totals["point_dice"] += loss_point_dice.item()
            totals["global"] += loss_global.item()
            totals["geo"] += loss_geo.item()
            totals["total"] += loss_total.item()

        num_batches = max(1, len(train_loader))
        averages = {key: value / num_batches for key, value in totals.items()}
        msg = (
            f"Epoch {epoch + 1}/{args.epochs} | LR {optimizer.param_groups[0]['lr']:.6g} | "
            f"PointFocal {averages['point_focal']:.4f} | PointDice {averages['point_dice']:.4f} | "
            f"Global {averages['global']:.4f} | Geo {averages['geo']:.4f} | Total {averages['total']:.4f}"
        )
        print(msg)
        with open(log_file, "a") as f:
            f.write(msg + "\n")

        if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
            save_checkpoint(
                os.path.join(run_dir, f"epoch{epoch + 1}.pth"),
                patch_feature_embedder, patch_geo_encoder, prompt_learner, epoch + 1, averages["total"],
            )

        if averages["total"] < best_loss:
            best_loss = averages["total"]
            best_epoch = epoch + 1
            save_checkpoint(
                os.path.join(run_dir, "best.pth"),
                patch_feature_embedder, patch_geo_encoder, prompt_learner, best_epoch, best_loss,
            )

    summary = {
        "class": class_name,
        "run_idx": run_idx,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_loss": float(best_loss),
        "elapsed_sec": time.time() - start_time,
        "run_dir": run_dir,
    }
    with open(os.path.join(run_dir, "run_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def write_summary_header(summary_csv):
    if not os.path.exists(summary_csv):
        with open(summary_csv, "w", newline="") as f:
            csv.writer(f).writerow(["timestamp", "dataset", "class", "run_idx", "seed", "best_epoch", "best_loss", "run_dir"])


def append_summary_row(summary_csv, args, summary):
    with open(summary_csv, "a", newline="") as f:
        csv.writer(f).writerow([
            time.strftime("%Y-%m-%d %H:%M:%S"),
            args.dataset_name,
            summary["class"],
            summary["run_idx"],
            summary["seed"],
            summary["best_epoch"],
            f"{summary['best_loss']:.6f}",
            summary["run_dir"],
        ])


def resolve_train_classes(args):
    valid_classes = PointCloudDataset.PRESETS[args.dataset_name]
    if args.train_class:
        classes = [args.train_class]
    elif args.classes:
        classes = args.classes
    else:
        classes = valid_classes

    invalid = [class_name for class_name in classes if class_name not in valid_classes]
    if invalid:
        raise ValueError(f"Unknown classes for {args.dataset_name}: {invalid}. Valid classes: {valid_classes}")
    return classes


def orchestrate_runs(args):
    os.makedirs(args.output_root, exist_ok=True)
    summary_csv = os.path.join(args.output_root, "summary.csv")
    write_summary_header(summary_csv)

    classes = resolve_train_classes(args)
    if args.train_class:
        run_specs = [(args.train_class, args.run_idx, args.seed)]
    else:
        run_specs = [
            (class_name, run_idx, args.seed_base + run_idx)
            for class_name in classes
            for run_idx in range(args.repeats)
        ]

    all_summaries = []
    for class_name, run_idx, seed in run_specs:
        run_dir = os.path.join(args.output_root, args.dataset_name, class_name, f"run_{run_idx:02d}_seed{seed}")
        print("=" * 80)
        print(f"One-vs-rest training class: {class_name} | split={args.train_split} | run={run_idx} | seed={seed}")
        print(f"Run dir: {run_dir}")
        print("=" * 80)

        summary = train_one_run(args, class_name, run_idx, seed, run_dir)
        all_summaries.append(summary)
        append_summary_row(summary_csv, args, summary)

    with open(os.path.join(args.output_root, "summary_all.json"), "w") as f:
        json.dump(all_summaries, f, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(description="Train one-vs-rest ULIP2 3D anomaly modules.")
    parser.add_argument("--output_root", type=str, default="./outputs/train")
    parser.add_argument("--data_root", type=str, default="./data/Real3D-AD-2048-npz")
    parser.add_argument("--model_path", type=str, default="./pretrained/ULIP-2-PointBERT-10k-xyzrgb-pc-vit_g-objaverse_shapenet-pretrained.pt")
    parser.add_argument("--dataset_name", type=str, default="Real3D", choices=list(PointCloudDataset.PRESETS))
    parser.add_argument("--train_class", type=str, default=None, help="Train a single class. If omitted, --classes/--repeats are used.")
    parser.add_argument("--classes", type=str, nargs="+", default=None, help="Classes for multi-run training. Defaults to all dataset classes.")
    parser.add_argument("--repeats", type=int, default=1, help="Number of repeats per class when --train_class is omitted.")
    parser.add_argument("--seed_base", type=int, default=111, help="Base seed for multi-run training; seed = seed_base + run_idx.")
    parser.add_argument("--train_split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--run_idx", type=int, default=0)
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--num_points", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--shuffle", action="store_true", help="Shuffle the one-class training samples. Ref script defaults to no shuffle.")
    parser.add_argument("--drop_last", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--return_layers", type=int, nargs="+", default=[2, 5, 8, 11])
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--depth", type=int, default=9)
    parser.add_argument("--n_ctx", type=int, default=8)
    parser.add_argument("--t_n_ctx", type=int, default=4)
    parser.add_argument("--geo_dim", type=int, default=33)
    parser.add_argument("--geo_loss", type=str, default="contrastive", choices=["mse", "cosine", "contrastive"])
    parser.add_argument("--fpfh_agg", type=str, default="max", choices=["mean", "max"])
    parser.add_argument("--global_alpha", type=float, default=0.5)
    parser.add_argument("--global_loss_weight", type=float, default=0.5)
    parser.add_argument("--geo_loss_weight", type=float, default=0.1)
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--focal_alpha", type=float, default=0.25)
    parser.add_argument("--save_every", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    return parser.parse_args()


def main():
    args = parse_args()
    orchestrate_runs(args)


if __name__ == "__main__":
    main()
