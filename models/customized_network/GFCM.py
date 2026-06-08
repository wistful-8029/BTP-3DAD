import torch.nn as nn
import torch
import torch.nn.functional as F


class GeometricFeatureCreationModule(nn.Module):
    def __init__(self, out_dim=33):
        super(GeometricFeatureCreationModule, self).__init__()
                                                          
        self.mlp1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.mlp2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.mlp3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, out_dim)                                         
        )

    def forward(self, points, patch_idx):
        """
        Args:
            points: [B, N, 3] original point cloud
            patch_idx: [B, G, M] point indices for each patch
        Returns:
            patch_feat: [B, G, out_dim] patch-level features
        """
        B, N, _ = points.shape
        G, M = patch_idx.shape[1], patch_idx.shape[2]

                                                                         
        idx_expand = patch_idx.unsqueeze(-1).expand(-1, -1, -1, 3)                
        pts_expand = points.unsqueeze(1).expand(-1, G, -1, -1)                
        neighborhood = torch.gather(pts_expand, 2, idx_expand)                

                                                  
                                  
        x = neighborhood.permute(0, 3, 1, 2).contiguous()

        feat = self.mlp1(x)
        feat = self.mlp2(feat)
        feat = self.mlp3(feat)                  

                                            
        patch_feat = torch.max(feat, dim=-1)[0]               
        patch_feat = patch_feat.permute(0, 2, 1).contiguous()               

                             
        patch_feat = self.fc(patch_feat)                   

        return patch_feat
