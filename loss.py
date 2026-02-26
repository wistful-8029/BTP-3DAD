import torch.nn.functional as F
import torch.nn as nn

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