import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, targets):
        targets = targets.float()
        probs = torch.sigmoid(logits)
        ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_factor = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = alpha_factor * ((1 - p_t) ** self.gamma) * ce_loss

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class BinaryDiceLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits, targets):
        targets = targets.float()
        probs = torch.sigmoid(logits)
        numerator = 2 * (probs * targets).sum(dim=1)
        denominator = probs.sum(dim=1) + targets.sum(dim=1) + self.eps
        dice = numerator / denominator
        return 1 - dice.mean()

class FPFHSupervisionLoss(nn.Module):
    def __init__(self, mode: str = "mse", temperature: float = 0.1):
        super().__init__()
        self.mode = mode
        self.temperature = temperature

    def forward(self, pred_feat, fpfh_feat):
        if self.mode == "mse":
            loss = F.mse_loss(pred_feat, fpfh_feat)
        elif self.mode == "cosine":
            cos_sim = F.cosine_similarity(pred_feat, fpfh_feat, dim=-1)
            loss = (1 - cos_sim).mean()
        elif self.mode == "contrastive":
            B, G, D = pred_feat.shape
            pred_flat = F.normalize(pred_feat.view(B * G, D), dim=-1)
            fpfh_flat = F.normalize(fpfh_feat.view(B * G, D), dim=-1)
            logits = torch.matmul(pred_flat, fpfh_flat.T) / self.temperature
            labels = torch.arange(B * G, device=pred_feat.device)
            loss = F.cross_entropy(logits, labels)
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")
        return loss
