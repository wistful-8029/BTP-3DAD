import torch.nn.functional as F
import torch
import torch.nn as nn


class MultiLayerFeatureEncoder(nn.Module):
    """
    Fuse multi-layer PointBERT features, patch FPFH features, and global embeddings.
    Each source is projected to the same dimension before concatenation.
    """

    def __init__(self, in_dim=384, fpfh_dim=33, out_dim=1280, hidden_dim=512, num_layers=4, global_dim=None):
        super().__init__()
        self.num_layers = num_layers
        self.out_dim = out_dim
        self.global_dim = in_dim if global_dim is None else global_dim

                                       
        self.mapper_pointbert = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

                                  
        self.mapper_fpfh = nn.Sequential(
            nn.Linear(fpfh_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

                                    
        self.mapper_global = nn.Sequential(
            nn.Linear(self.global_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

                                          
        self.fusion_proj = nn.Sequential(
            nn.Linear(out_dim * 3, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )

                                  
        self.layer_weights = nn.Parameter(torch.ones(num_layers))

    def forward(self, layer_feats_list, patch_fpfh_feats, global_embed):
        """
        Args:
            layer_feats_list: list of [B, N, in_dim], length = num_layers
            patch_fpfh_feats: [B, N, fpfh_dim]
            global_embed: [B, in_dim] global features
        Returns:
            patch_embeddings: [B, N, out_dim]
        """
        B, N, _ = layer_feats_list[0].shape
                                                          
        stacked_feats = torch.stack([self.mapper_pointbert(f) for f in layer_feats_list], dim=0)
        weights = F.softmax(self.layer_weights, dim=0).view(self.num_layers, 1, 1, 1)
        fused_pointbert_feat = (stacked_feats * weights).sum(dim=0)

                                                               
        features_to_concat = [
            fused_pointbert_feat,
            self.mapper_fpfh(patch_fpfh_feats),
            self.mapper_global(global_embed).unsqueeze(1).expand(-1, N, -1)
        ]
        patch_embeddings = self.fusion_proj(torch.cat(features_to_concat, dim=-1))
                                                                                      
        return patch_embeddings
