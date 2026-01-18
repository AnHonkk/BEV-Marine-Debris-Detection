# models/camera_bev/depth_net_unet.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class UNetDepthNet(nn.Module):
    """
    U-Net 기반 Depth Distribution 예측
    
    Architecture:
        - Encoder: 4 levels (downsampling)
        - Bottleneck: 최하단
        - Decoder: 4 levels (upsampling + skip connections)
    """
    
    def __init__(
        self,
        in_channels: int = 256,
        depth_bins: int = 64,
        depth_range: tuple = (1.0, 50.0),
        base_channels: int = 64
    ):
        super().__init__()
        
        self.depth_bins = depth_bins
        self.depth_range = depth_range
        
        # ============================================
        # Encoder (Contracting Path)
        # ============================================
        self.enc1 = self.conv_block(in_channels, base_channels)
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = self.conv_block(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool2d(2)
        
        self.enc3 = self.conv_block(base_channels * 2, base_channels * 4)
        self.pool3 = nn.MaxPool2d(2)
        
        self.enc4 = self.conv_block(base_channels * 4, base_channels * 8)
        self.pool4 = nn.MaxPool2d(2)
        
        # ============================================
        # Bottleneck
        # ============================================
        self.bottleneck = self.conv_block(base_channels * 8, base_channels * 16)
        
        # ============================================
        # Decoder (Expanding Path)
        # ============================================
        self.up4 = nn.ConvTranspose2d(base_channels * 16, base_channels * 8, 2, stride=2)
        self.dec4 = self.conv_block(base_channels * 16, base_channels * 8)  # 16 = 8 + 8 (skip)
        
        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 2, stride=2)
        self.dec3 = self.conv_block(base_channels * 8, base_channels * 4)
        
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.dec2 = self.conv_block(base_channels * 4, base_channels * 2)
        
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
        self.dec1 = self.conv_block(base_channels * 2, base_channels)
        
        # ============================================
        # Final depth prediction
        # ============================================
        self.depth_head = nn.Conv2d(base_channels, depth_bins, 1)
        
        # Depth bin values
        depth_values = torch.linspace(
            depth_range[0], 
            depth_range[1], 
            depth_bins
        )
        self.register_buffer('depth_values', depth_values)
    
    def conv_block(self, in_ch, out_ch):
        """Double convolution block (U-Net 스타일)"""
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        """
        Args:
            x: [B, 256, H, W] - ConvNeXt features
        
        Returns:
            dict with depth predictions
        """
        # ============================================
        # Encoder
        # ============================================
        enc1 = self.enc1(x)        # [B, 64, H, W]
        x = self.pool1(enc1)       # [B, 64, H/2, W/2]
        
        enc2 = self.enc2(x)        # [B, 128, H/2, W/2]
        x = self.pool2(enc2)       # [B, 128, H/4, W/4]
        
        enc3 = self.enc3(x)        # [B, 256, H/4, W/4]
        x = self.pool3(enc3)       # [B, 256, H/8, W/8]
        
        enc4 = self.enc4(x)        # [B, 512, H/8, W/8]
        x = self.pool4(enc4)       # [B, 512, H/16, W/16]
        
        # ============================================
        # Bottleneck
        # ============================================
        x = self.bottleneck(x)     # [B, 1024, H/16, W/16]
        
        # ============================================
        # Decoder (with skip connections)
        # ============================================
        x = self.up4(x)            # [B, 512, H/8, W/8]
        x = torch.cat([x, enc4], dim=1)  # [B, 1024, H/8, W/8] (skip connection!)
        x = self.dec4(x)           # [B, 512, H/8, W/8]
        
        x = self.up3(x)            # [B, 256, H/4, W/4]
        x = torch.cat([x, enc3], dim=1)  # [B, 512, H/4, W/4]
        x = self.dec3(x)           # [B, 256, H/4, W/4]
        
        x = self.up2(x)            # [B, 128, H/2, W/2]
        x = torch.cat([x, enc2], dim=1)  # [B, 256, H/2, W/2]
        x = self.dec2(x)           # [B, 128, H/2, W/2]
        
        x = self.up1(x)            # [B, 64, H, W]
        x = torch.cat([x, enc1], dim=1)  # [B, 128, H, W]
        x = self.dec1(x)           # [B, 64, H, W]
        
        # ============================================
        # Depth prediction
        # ============================================
        depth_logits = self.depth_head(x)  # [B, D, H, W]
        
        # Softmax
        depth_probs = F.softmax(depth_logits, dim=1)
        
        # Expected depth
        depth_expected = (depth_probs * self.depth_values.view(1, -1, 1, 1)).sum(dim=1, keepdim=True)
        
        return {
            'depth_logits': depth_logits,
            'depth_probs': depth_probs,
            'depth_expected': depth_expected,
            'depth_values': self.depth_values
        }


# ============================================
# Lightweight U-Net
# ============================================

class LightUNetDepthNet(nn.Module):
    """
    Light U-Net (3 levels)

    Architecture:
        Input [B, 256, H, W]
        → Encoder (3 levels)
        → Bottleneck
        → Decoder (3 levels with skip)
        → Output [B, 64, H, W]
    """
    
    def __init__(
        self,
        in_channels: int = 256,
        depth_bins: int = 64,
        depth_range: tuple = (1.0, 50.0)
    ):
        super().__init__()
        
        self.depth_bins = depth_bins
        self.depth_range = depth_range
        
        # Encoder
        self.enc1 = self.conv_block(in_channels, 128)
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = self.conv_block(128, 256)
        self.pool2 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = self.conv_block(256, 512)
        
        # Decoder
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(512, 256)  # 256 + 256
        
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(256, 128)  # 128 + 128
        
        # Depth Head
        self.depth_head = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, depth_bins, 1)
        )
        
        # Depth values
        depth_values = torch.linspace(
            depth_range[0], 
            depth_range[1], 
            depth_bins,
            dtype=torch.float32
        )
        self.register_buffer('depth_values', depth_values)
        
        # Weight initialization
        self._init_weights()
    
    def conv_block(self, in_ch: int, out_ch: int) -> nn.Sequential:
        """
        Double convolution block (U-Net style)
        Conv → BN → ReLU → Conv → BN → ReLU
        """
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def _init_weights(self):
        """Weight initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    
    def forward(self, x: torch.Tensor) -> dict:
        """
        Forward pass
        
        Args:
            x: [B, 256, H, W] - Input features from ConvNeXt
        
        Returns:
            dict with:
                - depth_logits: [B, D, H, W] - Raw logits
                - depth_probs: [B, D, H, W] - Probability distribution
                - depth_expected: [B, 1, H, W] - Expected depth
                - depth_values: [D] - Depth bin values
        """
        # Encoder
        enc1 = self.enc1(x)           # [B, 128, H, W]
        x = self.pool1(enc1)          # [B, 128, H/2, W/2]
        
        enc2 = self.enc2(x)           # [B, 256, H/2, W/2]
        x = self.pool2(enc2)          # [B, 256, H/4, W/4]
        
        # Bottleneck
        x = self.bottleneck(x)        # [B, 512, H/4, W/4]
        
        # Decoder
        x = self.up2(x)               # [B, 256, H/2, W/2]
        x = torch.cat([x, enc2], dim=1)  # [B, 512, H/2, W/2]
        x = self.dec2(x)              # [B, 256, H/2, W/2]
        
        x = self.up1(x)               # [B, 128, H, W]
        x = torch.cat([x, enc1], dim=1)  # [B, 256, H, W]
        x = self.dec1(x)              # [B, 128, H, W]
        
        # Depth Prediction (BEVFusion style!)
        depth_logits = self.depth_head(x)  # [B, D, H, W]
        
        # Softmax → Probability distribution
        depth_probs = F.softmax(depth_logits, dim=1)
        
        # Expected depth (probability-weighted mean)
        depth_expected = (
            depth_probs * self.depth_values.view(1, -1, 1, 1)
        ).sum(dim=1, keepdim=True)
        
        return {
            'depth_logits': depth_logits,
            'depth_probs': depth_probs,
            'depth_expected': depth_expected,
            'depth_values': self.depth_values
        }