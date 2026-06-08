import argparse
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from models.ulip2_encoder import ULIP2Encoder


TEXT_NORMAL = "a 3D point cloud of a normal object"
TEXT_ANOMALY = "a 3D point cloud of a defective object"
SIM_LAYER = 11


class PointCloudDataset(Dataset):
    PRESETS = {
        "Real3D": [
            "airplane", "candybar", "car", "chicken", "diamond", "duck",
            "fish", "gemstone", "seahorse", "shell", "starfish", "toffees",
        ]
    }

    def __init__(self, root_dir, split="train", class_name="airplane", dataset_name="Real3D"):
        assert split in ["train", "test"]
        assert dataset_name in self.PRESETS

        self.root_dir = root_dir
        self.split = split
        self.class_name = class_name
        self.dataset_name = dataset_name

        cls_dir = os.path.join(root_dir, class_name, split)
        if not os.path.isdir(cls_dir):
            raise FileNotFoundError(f"Class dir not found: {cls_dir}")

        self.samples = []
        for fname in os.listdir(cls_dir):
            if fname.endswith(".npy"):
                self.samples.append(os.path.join(cls_dir, fname))

        if len(self.samples) == 0:
            raise FileNotFoundError(f"No .npy found under: {cls_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path = self.samples[idx]
        arr = np.load(path)
        if arr.ndim != 2 or arr.shape[1] < 4:
            raise ValueError(f"Invalid npy: {path}, shape={arr.shape} (expect (N,D>=4))")

        points = torch.from_numpy(arr[:, :3].astype(np.float32))
        labels = torch.from_numpy(arr[:, -1].astype(np.int64))
        return {"points": points, "labels": labels, "path": path, "class_name": self.class_name}


class ChannelAdapter(torch.nn.Module):
    def __init__(self, in_channels=384, out_channels=1280, drop_cls=True, use_ln=True, use_l2norm=True):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.drop_cls = bool(drop_cls)
        self.use_l2norm = bool(use_l2norm)
        self.proj = torch.nn.Linear(self.in_channels, self.out_channels, bias=True)
        self.ln = torch.nn.LayerNorm(self.out_channels) if use_ln else torch.nn.Identity()

    def forward(self, x):
        if x.dim() != 3:
            raise ValueError(f"Expected (B,T,C), got {tuple(x.shape)}")
        _, t, c = x.shape
        if c != self.in_channels:
            raise ValueError(f"in_channels mismatch: expect {self.in_channels}, got {c}")
        if self.drop_cls and t == 513:
            x = x[:, 1:, :]
        y = self.ln(self.proj(x))
        if self.use_l2norm:
            y = F.normalize(y, p=2, dim=-1)
        return y


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device(device="cuda"):
    if device == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


@torch.no_grad()
def encode_text_prompts(encoder, device):
    tn = encoder.encode_text(TEXT_NORMAL).to(device)
    ta = encoder.encode_text(TEXT_ANOMALY).to(device)
    tn = F.normalize(tn.squeeze(0), p=2, dim=-1)
    ta = F.normalize(ta.squeeze(0), p=2, dim=-1)
    return tn, ta


def build_loader(data_root, dataset_name, class_name, split, batch_size, num_workers, shuffle):
    ds = PointCloudDataset(root_dir=data_root, split=split, class_name=class_name, dataset_name=dataset_name)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        pin_memory=True,
        num_workers=num_workers,
    )
    return ds, dl


@torch.no_grad()
def run_single_batch_pipeline(args, encoder, adapter, loader, device, text_normal_vec, text_anomaly_vec):
    encoder.model.eval()
    adapter.eval()

    print(f"ULIP text encoding (normal): shape={tuple(text_normal_vec.shape)}")
    print(f"ULIP text encoding (anomaly): shape={tuple(text_anomaly_vec.shape)}")
    print("")

    for bidx, batch in enumerate(tqdm(loader, desc="Dump", dynamic_ncols=True)):
        points = batch["points"].to(device, non_blocking=True)
        feat = encoder.encode_pointcloud(points, return_intermediate=True)
        patch_idx = feat.get("patch_idx", None)
        layer_feats = feat.get("layer_feats", None)
        global_embedding = feat.get("global", None)

        print(f"[Batch {bidx}]")
        print(f"Point cloud input: shape={tuple(points.shape)}")
        if global_embedding is not None:
            print(f"ULIP global_embedding: shape={tuple(global_embedding.shape)}")
        else:
            print("ULIP global_embedding: None")
        if patch_idx is not None:
            print(f"ULIP intermediate patch_idx: shape={tuple(patch_idx.shape)}")
        else:
            print("ULIP intermediate patch_idx: None")

        if layer_feats is None:
            print("ULIP point encoding: layer_feats is None")
            break

        for lid in args.return_layers:
            if lid in layer_feats:
                print(f"ULIP intermediate layer_feats[{lid}]: shape={tuple(layer_feats[lid].shape)}")
            else:
                print(f"ULIP intermediate layer_feats[{lid}]: missing")

        if SIM_LAYER not in layer_feats:
            print(f"ULIP point encoding: layer_feats[{SIM_LAYER}] is missing")
            break

        layer_tokens = layer_feats[SIM_LAYER]
        if layer_tokens.size(1) == 513:
            cls_feature = layer_tokens[:, 0, :]
            patch_features = layer_tokens[:, 1:, :]
        else:
            cls_feature = None
            patch_features = layer_tokens

        print(f"ULIP cls_feature: shape={tuple(cls_feature.shape) if cls_feature is not None else None}")
        print(f"ULIP patch_features (without CLS): shape={tuple(patch_features.shape)}")

        adapted_patch_features = adapter(patch_features)
        print(f"Adapter patch_features: shape={tuple(adapted_patch_features.shape)}")
        break

    print("")


def main(args):
    set_seed(args.seed)
    device = pick_device(args.device)

    encoder = ULIP2Encoder(
        args.model_path,
        device=device,
        num_points=args.num_points,
        return_layers=tuple(args.return_layers),
    )
    encoder.model.eval()
    for p in encoder.model.parameters():
        p.requires_grad = False

    with torch.no_grad():
        text_normal_vec, text_anomaly_vec = encode_text_prompts(encoder, device)

    adapter = ChannelAdapter(
        in_channels=args.token_dim,
        out_channels=args.text_dim,
        drop_cls=True,
        use_ln=True,
        use_l2norm=True,
    ).to(device)
    adapter.eval()

    ds, dl = build_loader(
        data_root=args.data_root,
        dataset_name=args.dataset_name,
        class_name=args.class_name,
        split=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )
    print(f"[Data] class={args.class_name} split={args.split} samples={len(ds)} batch_size={args.batch_size}")

    run_single_batch_pipeline(args, encoder, adapter, dl, device, text_normal_vec, text_anomaly_vec)


def parse_args():
    p = argparse.ArgumentParser("Dump ULIP2 intermediate feature shapes (no training)")

    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--model_path", type=str, default="./pretrained/ULIP-2-PointBERT-10k-xyzrgb-pc-vit_g-objaverse_shapenet-pretrained.pt")
    p.add_argument("--dataset_name", type=str, default="Real3D")
    p.add_argument("--class_name", type=str, default="gemstone")
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--num_points", type=int, default=2048)
    p.add_argument("--return_layers", type=int, nargs="+", default=[2, 5, 8, 11])
    p.add_argument("--token_dim", type=int, default=384)
    p.add_argument("--text_dim", type=int, default=1280)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
