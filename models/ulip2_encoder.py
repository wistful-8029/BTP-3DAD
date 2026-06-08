import argparse
import numpy as np
import torch.nn.functional as F
from collections import OrderedDict
import torch
from utils.tokenizer import SimpleTokenizer
import models.ULIP_models as ulip_models


class ULIP2Encoder:
    """ULIP2-based encoder for 3D models and text."""

    def __init__(self, model_path: str, device: str = 'cuda', num_points: int = 10000,
                 return_layers: tuple = (2, 5, 8, 11), return_clip: bool = False):
        """
        Initialize the ULIP2 encoder.

        Args:
            return_clip (bool): return clip model
            return_layers (tuple or list of int): return middle layer of encoder.
            model_path: Path to the pretrained ULIP2 model
            device: Device to run the model on ('cuda' or 'cpu')
            num_points: Number of points to sample from 3D models
        """
        self.return_clip = return_clip
        self.return_layers = return_layers
        self.device = device
        self.num_points = num_points
        self.tokenizer = SimpleTokenizer()

        self.model = self._load_model(model_path)
        self.model.eval()

    def _load_model(self, model_path: str):
        """Load the ULIP2 model from checkpoint."""
        print(f"Loading ULIP2 model from {model_path}")

        args = argparse.Namespace()
        args.evaluate_3d_ulip2 = True
        args.npoints = self.num_points
        args.return_clip = self.return_clip

        ckpt = torch.load(model_path, map_location='cpu')
        state_dict = OrderedDict()
        for k, v in ckpt['state_dict'].items():
            state_dict[k.replace('module.', '')] = v

        loaded = getattr(ulip_models, 'ULIP2_PointBERT_Colored')(args=args)

        if self.return_clip:
            model, open_clip_model = loaded
            self.open_clip_model = open_clip_model
        else:
            model = loaded

        model.to(self.device)
        model.load_state_dict(state_dict, strict=False)

        print("ULIP2 model loaded successfully")
        return model

    def encode_pointcloud(self, points: torch.Tensor, return_intermediate=False) -> dict:
        """
        Encode a batch of point clouds (tensor input) to embedding vectors.

        Args:
            points: tensor of shape [B, N, 3] or [B, N, 6]
            return_intermediate: whether to return intermediate features (e.g. global_feat, local_feat, layer_feats...)
        Returns:
            dict with keys:
                'concat': [B, C]
                'global': [B, C] if return_intermediate=True
                'local': [B, num_patches, C_local] if return_intermediate=True
                'layer_feats': dict of layer features
                'neighborhood': group neighborhood,
                'center': group center
        """
        if points.ndim != 3:
            raise ValueError("Points must have shape [B, N, 3] or [B, N, 6]")

        B, N, C = points.shape
        if C < 3:
            raise ValueError("Points must have at least 3 coordinates (XYZ)")

                                                          
        if C >= 6:
            points = points[:, :, :6]
        else:
            dummy_rgb = torch.full((B, N, 3), 0.5, dtype=points.dtype, device=points.device)
            points = torch.cat([points[:, :, :3], dummy_rgb], dim=-1)             

                                        
        points = points.float().contiguous().to(self.device)

        with torch.no_grad():
            if return_intermediate:
                concat_f, global_feat, local_feat, features, neighborhood, center, patch_idx = self.model.encode_pc(
                    points, return_intermediate=True, return_layers=self.return_layers
                )
                concat_f = F.normalize(concat_f, dim=-1)
                global_feat = F.normalize(global_feat, dim=-1)
                local_feat = F.normalize(local_feat, dim=-1)
                for k in features:
                    features[k] = F.normalize(features[k], dim=-1)

                return {
                    'concat': concat_f,
                    'global': global_feat,
                    'local': local_feat,
                    'layer_feats': features,
                    'neighborhood': neighborhood,
                    'center': center,
                    'patch_idx': patch_idx
                }
            else:
                concat_f = self.model.encode_pc(points)
                concat_f = F.normalize(concat_f, dim=-1)
                return {'concat': concat_f}

    def encode_text(self, text: str) -> torch.Tensor:
        """
        Encode text to embedding vector.

        Args:
            text: Text description

        Returns:
            Embedding vector as numpy array
        """
                       
        text_tokens = self.tokenizer([text]).to(self.device)
                                                       

                                                                     
        if len(text_tokens.shape) < 2:
            text_tokens = text_tokens[None, ...]

        with torch.no_grad():
                         
            text_embed = self.model.encode_text(text_tokens)
            text_embed = F.normalize(text_embed, dim=-1)

        return text_embed

    def encode_text_templates(self, templates: list) -> torch.Tensor:
        """
        Encode a list of text templates into a single aggregated embedding vector.

        Args:
            templates (list): A list of text descriptions.

        Returns:
            torch.Tensor: Aggregated text embedding [1, D]
        """
        text_embeds = []

        with torch.no_grad():
            for text in templates:
                          
                text_tokens = self.tokenizer([text]).to(self.device)
                                                                             
                if len(text_tokens.shape) < 2:
                    text_tokens = text_tokens[None, ...]
                        
                text_embed = self.model.encode_text(text_tokens)
                text_embed = F.normalize(text_embed, dim=-1)          
                text_embeds.append(text_embed)

        text_embeds = torch.cat(text_embeds, dim=0)          
        return text_embeds.mean(dim=0, keepdim=True)          

    def encode_text_from_tokens(self, tokenized_prompts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode tokenized prompts to embedding vectors.

        Args:
            tokenized_prompts: Tensor of shape [2, seq_len] (0: normal, 1: anomaly)

        Returns:
            normal_embed: [1, embed_dim] embedding for normal text
            anomaly_embed: [1, embed_dim] embedding for anomaly text
        """
                                                   
        tokenized_prompts = tokenized_prompts.to(self.device)

                                
        if len(tokenized_prompts.shape) < 2:
            tokenized_prompts = tokenized_prompts[None, ...]

        with torch.no_grad():
                                
            text_embeds = self.model.encode_text(tokenized_prompts)                  
            text_embeds = F.normalize(text_embeds, dim=-1)

        normal_embed = text_embeds[0:1, :]                  
        anomaly_embed = text_embeds[1:2, :]                  

        return normal_embed, anomaly_embed

    def compute_similarity(self, embed1: np.ndarray, embed2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.

        Args:
            embed1: First embedding vector
            embed2: Second embedding vector

        Returns:
            Cosine similarity score
        """
        embed1_norm = embed1 / np.linalg.norm(embed1)
        embed2_norm = embed2 / np.linalg.norm(embed2)
        return float(np.dot(embed1_norm.flatten(), embed2_norm.flatten()))
