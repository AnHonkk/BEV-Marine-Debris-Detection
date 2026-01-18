# models/camera_bev/depth_net.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List

class DepthNet(nn.Module):
    """
    Depth Distribution 예측 네트워크
    출력: [B, D, H, W] (D개의 depth bin에 대한 확률 분포)
    """
    
    def __init__(
        self,
        in_channels: int = 512,
        depth_bins: int = 64,
        depth_range: Tuple[float, float] = (1.0, 50.0),
        use_uncertainty: bool = False
    ):
        super().__init__()
        
        self.depth_bins = depth_bins
        self.depth_min, self.depth_max = depth_range
        self.use_uncertainty = use_uncertainty
        
        # Depth bins (learnable or fixed)
        self.register_buffer(
            'depth_values',
            torch.linspace(depth_range[0], depth_range[1], depth_bins)
        )
        
        # Depth prediction head
        self.depth_head = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, depth_bins, 1)
        )
        
        # Uncertainty head (optional)
        if use_uncertainty:
            self.uncertainty_head = nn.Sequential(
                nn.Conv2d(in_channels, 64, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 1, 1),
                nn.Sigmoid()
            )
    
    def forward(self, img_features: torch.Tensor) -> dict:
        """
        Args:
            img_features: [B, C, H, W] 이미지 특징
        
        Returns:
            depth_logits: [B, D, H, W] depth bin logits
            depth_probs: [B, D, H, W] depth probabilities (softmax)
            depth_values: [D] depth bin values
            uncertainty: [B, 1, H, W] (optional)
        """
        B, C, H, W = img_features.shape
        
        # Depth logits
        depth_logits = self.depth_head(img_features)  # [B, D, H, W]
        
        # Softmax over depth dimension
        depth_probs = F.softmax(depth_logits, dim=1)
        
        # Expected depth (optional, for visualization)
        depth_expected = (depth_probs * self.depth_values.view(1, -1, 1, 1)).sum(dim=1, keepdim=True)
        
        outputs = {
            'depth_logits': depth_logits,
            'depth_probs': depth_probs,
            'depth_values': self.depth_values,
            'depth_expected': depth_expected
        }
        
        # Uncertainty estimation
        if self.use_uncertainty:
            uncertainty = self.uncertainty_head(img_features)
            outputs['uncertainty'] = uncertainty
        
        return outputs


class DepthNetWithContext(nn.Module):
    """
    Context-aware Depth Network
    ASPP(Atrous Spatial Pyramid Pooling) 사용
    """
    
    def __init__(
        self,
        in_channels: int = 512,
        depth_bins: int = 64,
        depth_range: Tuple[float, float] = (1.0, 50.0)
    ):
        super().__init__()
        
        self.depth_bins = depth_bins
        self.register_buffer(
            'depth_values',
            torch.linspace(depth_range[0], depth_range[1], depth_bins)
        )
        
        # ASPP module
        self.aspp = ASPP(in_channels, 256, [6, 12, 18])
        
        # Depth head
        self.depth_head = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, depth_bins, 1)
        )
    
    def forward(self, img_features: torch.Tensor) -> dict:
        # Context aggregation
        context_feat = self.aspp(img_features)
        
        # Depth prediction
        depth_logits = self.depth_head(context_feat)
        depth_probs = F.softmax(depth_logits, dim=1)
        
        return {
            'depth_logits': depth_logits,
            'depth_probs': depth_probs,
            'depth_values': self.depth_values
        }


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling"""
    
    def __init__(self, in_channels: int, out_channels: int, rates: List[int]):
        super().__init__()
        
        self.branches = nn.ModuleList()
        
        # 1x1 conv
        self.branches.append(
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        )
        
        # Atrous convs
        for rate in rates:
            self.branches.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 3, 
                             padding=rate, dilation=rate),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )
        
        # Global pooling
        self.branches.append(
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        )
        
        # Fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels * len(self.branches), out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        
        for branch in self.branches[:-1]:
            outputs.append(branch(x))
        
        # Global pooling branch
        global_feat = self.branches[-1](x)
        global_feat = F.interpolate(
            global_feat, size=x.shape[-2:], 
            mode='bilinear', align_corners=False
        )
        outputs.append(global_feat)
        
        # Concatenate and fuse
        concat = torch.cat(outputs, dim=1)
        return self.fusion(concat)