"""
BEV Fusion Module
Camera BEV Feature Map과 LiDAR BEV Feature Map을 융합

Reference: BEVFusion: Multi-Task Multi-Sensor Fusion with Unified Bird's-Eye View Representation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional


class ConvFuser(nn.Module):
    """
    단순 Concatenation + Convolution 기반 Fusion
    """
    def __init__(
        self,
        camera_channels: int = 256,
        lidar_channels: int = 256,
        out_channels: int = 256,
        num_conv_layers: int = 2,
    ):
        super().__init__()
        
        in_channels = camera_channels + lidar_channels
        
        layers = []
        for i in range(num_conv_layers):
            if i == 0:
                layers.append(nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False))
            else:
                layers.append(nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
        
        self.conv = nn.Sequential(*layers)
        self.out_channels = out_channels
        
    def forward(
        self,
        camera_bev: torch.Tensor,  # (B, C1, H, W)
        lidar_bev: torch.Tensor,   # (B, C2, H, W)
    ) -> torch.Tensor:
        """
        Args:
            camera_bev: Camera BEV features (B, C1, H, W)
            lidar_bev: LiDAR BEV features (B, C2, H, W)
            
        Returns:
            fused: Fused BEV features (B, C_out, H, W)
        """
        # Concatenate along channel dimension
        fused = torch.cat([camera_bev, lidar_bev], dim=1)
        fused = self.conv(fused)
        
        return fused


class ChannelAttentionFuser(nn.Module):
    """
    Channel Attention 기반 Fusion
    각 modality의 중요도를 학습하여 가중 합
    """
    def __init__(
        self,
        camera_channels: int = 256,
        lidar_channels: int = 256,
        out_channels: int = 256,
        reduction: int = 16,
    ):
        super().__init__()
        
        self.camera_channels = camera_channels
        self.lidar_channels = lidar_channels
        self.out_channels = out_channels
        
        # Channel projection (같은 차원으로 맞추기)
        self.camera_proj = nn.Conv2d(camera_channels, out_channels, 1, bias=False)
        self.lidar_proj = nn.Conv2d(lidar_channels, out_channels, 1, bias=False)
        
        # Channel attention for each modality
        self.camera_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // reduction, out_channels, 1, bias=False),
        )
        
        self.lidar_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // reduction, out_channels, 1, bias=False),
        )
        
        # Final fusion
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        
    def forward(
        self,
        camera_bev: torch.Tensor,
        lidar_bev: torch.Tensor,
    ) -> torch.Tensor:
        # Project to same dimension
        camera_feat = self.camera_proj(camera_bev)
        lidar_feat = self.lidar_proj(lidar_bev)
        
        # Compute attention weights
        camera_attn = self.camera_attention(camera_feat)
        lidar_attn = self.lidar_attention(lidar_feat)
        
        # Softmax over modalities
        attn = torch.stack([camera_attn, lidar_attn], dim=0)  # (2, B, C, 1, 1)
        attn = F.softmax(attn, dim=0)
        
        # Weighted sum
        camera_attn, lidar_attn = attn[0], attn[1]
        fused = camera_feat * camera_attn + lidar_feat * lidar_attn
        
        # Final fusion
        fused = self.fusion_conv(fused)
        
        return fused


class SpatialAttentionFuser(nn.Module):
    """
    Spatial Attention 기반 Fusion
    공간적으로 어느 modality가 더 신뢰할 수 있는지 학습
    """
    def __init__(
        self,
        camera_channels: int = 256,
        lidar_channels: int = 256,
        out_channels: int = 256,
    ):
        super().__init__()
        
        # Channel projection
        self.camera_proj = nn.Conv2d(camera_channels, out_channels, 1, bias=False)
        self.lidar_proj = nn.Conv2d(lidar_channels, out_channels, 1, bias=False)
        
        # Spatial attention
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, 2, 1),  # 2 modalities
        )
        
        # Final fusion
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        
        self.out_channels = out_channels
        
    def forward(
        self,
        camera_bev: torch.Tensor,
        lidar_bev: torch.Tensor,
    ) -> torch.Tensor:
        # Project
        camera_feat = self.camera_proj(camera_bev)
        lidar_feat = self.lidar_proj(lidar_bev)
        
        # Concatenate for attention computation
        concat = torch.cat([camera_feat, lidar_feat], dim=1)
        
        # Spatial attention weights (B, 2, H, W)
        attn = self.spatial_attention(concat)
        attn = F.softmax(attn, dim=1)
        
        # Weighted sum
        camera_attn = attn[:, 0:1, :, :]  # (B, 1, H, W)
        lidar_attn = attn[:, 1:2, :, :]   # (B, 1, H, W)
        
        fused = camera_feat * camera_attn + lidar_feat * lidar_attn
        
        # Final fusion
        fused = self.fusion_conv(fused)
        
        return fused


class CrossAttentionFuser(nn.Module):
    """
    Cross-Attention 기반 Fusion
    Transformer 스타일의 cross-attention으로 modality 간 상호작용
    """
    def __init__(
        self,
        camera_channels: int = 256,
        lidar_channels: int = 256,
        out_channels: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # Channel projection
        self.camera_proj = nn.Conv2d(camera_channels, out_channels, 1, bias=False)
        self.lidar_proj = nn.Conv2d(lidar_channels, out_channels, 1, bias=False)
        
        # Cross attention: camera attends to lidar
        self.camera_cross_attn = nn.MultiheadAttention(
            embed_dim=out_channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # Cross attention: lidar attends to camera
        self.lidar_cross_attn = nn.MultiheadAttention(
            embed_dim=out_channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # Layer norms
        self.camera_norm = nn.LayerNorm(out_channels)
        self.lidar_norm = nn.LayerNorm(out_channels)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(out_channels * 2, out_channels * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(out_channels * 4, out_channels),
            nn.Dropout(dropout),
        )
        
        self.final_norm = nn.LayerNorm(out_channels)
        
        # Reshape back to spatial
        self.final_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        
        self.out_channels = out_channels
        
    def forward(
        self,
        camera_bev: torch.Tensor,
        lidar_bev: torch.Tensor,
    ) -> torch.Tensor:
        B, _, H, W = camera_bev.shape
        
        # Project and flatten
        camera_feat = self.camera_proj(camera_bev)  # (B, C, H, W)
        lidar_feat = self.lidar_proj(lidar_bev)     # (B, C, H, W)
        
        # Flatten to sequence: (B, H*W, C)
        camera_seq = camera_feat.flatten(2).permute(0, 2, 1)
        lidar_seq = lidar_feat.flatten(2).permute(0, 2, 1)
        
        # Cross attention
        camera_attended, _ = self.camera_cross_attn(
            query=camera_seq, key=lidar_seq, value=lidar_seq
        )
        camera_attended = self.camera_norm(camera_seq + camera_attended)
        
        lidar_attended, _ = self.lidar_cross_attn(
            query=lidar_seq, key=camera_seq, value=camera_seq
        )
        lidar_attended = self.lidar_norm(lidar_seq + lidar_attended)
        
        # Concatenate and FFN
        combined = torch.cat([camera_attended, lidar_attended], dim=-1)  # (B, H*W, 2C)
        fused = self.ffn(combined)  # (B, H*W, C)
        fused = self.final_norm(fused)
        
        # Reshape back to spatial
        fused = fused.permute(0, 2, 1).reshape(B, -1, H, W)
        fused = self.final_conv(fused)
        
        return fused


class AdaptiveFuser(nn.Module):
    """
    Adaptive Fusion
    입력 조건에 따라 fusion 방식을 동적으로 선택
    """
    def __init__(
        self,
        camera_channels: int = 256,
        lidar_channels: int = 256,
        out_channels: int = 256,
        num_conv_layers: int = 3,
    ):
        super().__init__()
        
        in_channels = camera_channels + lidar_channels
        
        # Modality confidence predictor
        self.camera_confidence = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(camera_channels, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
        
        self.lidar_confidence = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(lidar_channels, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
        
        # Channel projection
        self.camera_proj = nn.Conv2d(camera_channels, out_channels, 1, bias=False)
        self.lidar_proj = nn.Conv2d(lidar_channels, out_channels, 1, bias=False)
        
        # Feature fusion layers
        layers = []
        for i in range(num_conv_layers):
            if i == 0:
                layers.append(nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False))
            else:
                layers.append(nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
        
        self.fusion_conv = nn.Sequential(*layers)
        
        # Residual connection
        self.residual_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        
        self.out_channels = out_channels
        
    def forward(
        self,
        camera_bev: torch.Tensor,
        lidar_bev: torch.Tensor,
    ) -> torch.Tensor:
        B = camera_bev.shape[0]
        
        # Compute modality confidences
        camera_conf = self.camera_confidence(camera_bev).view(B, 1, 1, 1)
        lidar_conf = self.lidar_confidence(lidar_bev).view(B, 1, 1, 1)
        
        # Normalize confidences
        total_conf = camera_conf + lidar_conf + 1e-6
        camera_weight = camera_conf / total_conf
        lidar_weight = lidar_conf / total_conf
        
        # Project and weighted sum (adaptive path)
        camera_feat = self.camera_proj(camera_bev)
        lidar_feat = self.lidar_proj(lidar_bev)
        adaptive_fused = camera_feat * camera_weight + lidar_feat * lidar_weight
        
        # Full concatenation fusion (complete path)
        concat = torch.cat([camera_bev, lidar_bev], dim=1)
        full_fused = self.fusion_conv(concat)
        
        # Combine both paths with residual
        output = full_fused + self.residual_conv(adaptive_fused)
        output = F.relu(output)
        
        return output


class BEVFusion(nn.Module):
    """
    Complete BEV Fusion Module
    Camera BEV와 LiDAR BEV를 융합하는 통합 모듈
    """
    def __init__(
        self,
        camera_channels: int = 256,
        lidar_channels: int = 256,
        out_channels: int = 256,
        fusion_method: str = 'adaptive',  # 'conv', 'channel_attn', 'spatial_attn', 'cross_attn', 'adaptive'
        # Spatial alignment
        align_corners: bool = True,
        # Resolution matching
        target_shape: Optional[Tuple[int, int]] = None,  # (H, W)
    ):
        super().__init__()
        
        self.camera_channels = camera_channels
        self.lidar_channels = lidar_channels
        self.out_channels = out_channels
        self.fusion_method = fusion_method
        self.align_corners = align_corners
        self.target_shape = target_shape
        
        # Fusion module 선택
        if fusion_method == 'conv':
            self.fuser = ConvFuser(camera_channels, lidar_channels, out_channels)
        elif fusion_method == 'channel_attn':
            self.fuser = ChannelAttentionFuser(camera_channels, lidar_channels, out_channels)
        elif fusion_method == 'spatial_attn':
            self.fuser = SpatialAttentionFuser(camera_channels, lidar_channels, out_channels)
        elif fusion_method == 'cross_attn':
            self.fuser = CrossAttentionFuser(camera_channels, lidar_channels, out_channels)
        elif fusion_method == 'adaptive':
            self.fuser = AdaptiveFuser(camera_channels, lidar_channels, out_channels)
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")
        
    def align_features(
        self,
        camera_bev: torch.Tensor,
        lidar_bev: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Camera BEV와 LiDAR BEV의 공간 해상도 맞추기
        """
        if self.target_shape is not None:
            target_H, target_W = self.target_shape
        else:
            # LiDAR BEV를 기준으로 맞춤
            target_H, target_W = lidar_bev.shape[2:]
        
        # Camera BEV resize
        if camera_bev.shape[2:] != (target_H, target_W):
            camera_bev = F.interpolate(
                camera_bev,
                size=(target_H, target_W),
                mode='bilinear',
                align_corners=self.align_corners,
            )
        
        # LiDAR BEV resize
        if lidar_bev.shape[2:] != (target_H, target_W):
            lidar_bev = F.interpolate(
                lidar_bev,
                size=(target_H, target_W),
                mode='bilinear',
                align_corners=self.align_corners,
            )
        
        return camera_bev, lidar_bev
    
    def forward(
        self,
        camera_bev: torch.Tensor,
        lidar_bev: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            camera_bev: Camera BEV features (B, C1, H1, W1)
            lidar_bev: LiDAR BEV features (B, C2, H2, W2)
            
        Returns:
            fused: Fused BEV features (B, C_out, H, W)
        """
        # Spatial alignment
        camera_bev, lidar_bev = self.align_features(camera_bev, lidar_bev)
        
        # Fusion
        fused = self.fuser(camera_bev, lidar_bev)
        
        return fused


class MultiScaleBEVFusion(nn.Module):
    """
    Multi-Scale BEV Fusion
    여러 스케일에서 fusion 후 통합
    """
    def __init__(
        self,
        camera_channels: List[int] = [64, 128, 256],
        lidar_channels: List[int] = [64, 128, 256],
        out_channels: int = 256,
        fusion_method: str = 'adaptive',
    ):
        super().__init__()
        
        assert len(camera_channels) == len(lidar_channels)
        self.num_scales = len(camera_channels)
        
        # Multi-scale fusers
        self.fusers = nn.ModuleList()
        fused_channels = []
        
        for cam_c, lid_c in zip(camera_channels, lidar_channels):
            fuser = BEVFusion(
                camera_channels=cam_c,
                lidar_channels=lid_c,
                out_channels=cam_c,  # 각 스케일에서 같은 채널 유지
                fusion_method=fusion_method,
            )
            self.fusers.append(fuser)
            fused_channels.append(cam_c)
        
        # FPN-style aggregation
        self.lateral_convs = nn.ModuleList()
        self.output_convs = nn.ModuleList()
        
        for i, c in enumerate(fused_channels):
            lateral_conv = nn.Conv2d(c, out_channels, 1, bias=False)
            output_conv = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
            self.lateral_convs.append(lateral_conv)
            self.output_convs.append(output_conv)
        
        # Final fusion
        self.final_conv = nn.Sequential(
            nn.Conv2d(out_channels * self.num_scales, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        
        self.out_channels = out_channels
        
    def forward(
        self,
        camera_bev_list: List[torch.Tensor],
        lidar_bev_list: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Args:
            camera_bev_list: List of camera BEV features at different scales
            lidar_bev_list: List of LiDAR BEV features at different scales
            
        Returns:
            fused: Fused BEV features (B, out_channels, H, W)
        """
        assert len(camera_bev_list) == len(lidar_bev_list) == self.num_scales
        
        # Fuse at each scale
        fused_features = []
        for i, (cam_bev, lid_bev) in enumerate(zip(camera_bev_list, lidar_bev_list)):
            fused = self.fusers[i](cam_bev, lid_bev)
            fused = self.lateral_convs[i](fused)
            fused = self.output_convs[i](fused)
            fused_features.append(fused)
        
        # Upsample all to the largest resolution
        target_size = fused_features[0].shape[2:]
        upsampled = []
        
        for feat in fused_features:
            if feat.shape[2:] != target_size:
                feat = F.interpolate(
                    feat, size=target_size, mode='bilinear', align_corners=True
                )
            upsampled.append(feat)
        
        # Concatenate and final fusion
        combined = torch.cat(upsampled, dim=1)
        output = self.final_conv(combined)
        
        return output
