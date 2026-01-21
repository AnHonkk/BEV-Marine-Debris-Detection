"""
BEV Segmentation Head
Fused BEV Feature Map을 BEV Segmentation Map으로 변환

해양 부유쓰레기 탐지를 위한 클래스 (3개):
- 0: Background (배경/수면)
- 1: Land (육지/부두/구조물)
- 2: Debris (부유쓰레기)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional


class ConvBNReLU(nn.Module):
    """Conv + BatchNorm + ReLU block"""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        groups: int = 1,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding,
            dilation=dilation, groups=groups, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling
    다양한 스케일의 context 정보 추출
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        atrous_rates: List[int] = [6, 12, 18],
    ):
        super().__init__()
        
        modules = []
        
        # 1x1 convolution
        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        )
        
        # Atrous convolutions
        for rate in atrous_rates:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 3, padding=rate, dilation=rate, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                )
            )
        
        # Global average pooling
        modules.append(
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        )
        
        self.convs = nn.ModuleList(modules)
        
        # Project concatenated features
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * (len(atrous_rates) + 2), out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = []
        for conv in self.convs[:-1]:
            res.append(conv(x))
        
        # Global pooling branch
        global_feat = self.convs[-1](x)
        global_feat = F.interpolate(global_feat, size=x.shape[2:], mode='bilinear', align_corners=True)
        res.append(global_feat)
        
        # Concatenate and project
        res = torch.cat(res, dim=1)
        return self.project(res)


class DecoderBlock(nn.Module):
    """
    Decoder Block with skip connection support
    """
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        upsample_scale: int = 2,
    ):
        super().__init__()
        
        self.upsample_scale = upsample_scale
        
        # Skip connection projection
        if skip_channels > 0:
            self.skip_proj = nn.Sequential(
                nn.Conv2d(skip_channels, 48, 1, bias=False),
                nn.BatchNorm2d(48),
                nn.ReLU(inplace=True),
            )
            concat_channels = in_channels + 48
        else:
            self.skip_proj = None
            concat_channels = in_channels
        
        # Decoder convolutions
        self.conv = nn.Sequential(
            ConvBNReLU(concat_channels, out_channels, 3, padding=1),
            nn.Dropout(0.5),
            ConvBNReLU(out_channels, out_channels, 3, padding=1),
            nn.Dropout(0.1),
        )
        
    def forward(
        self,
        x: torch.Tensor,
        skip: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Upsample
        if self.upsample_scale > 1:
            x = F.interpolate(
                x, scale_factor=self.upsample_scale,
                mode='bilinear', align_corners=True
            )
        
        # Skip connection
        if skip is not None and self.skip_proj is not None:
            skip = self.skip_proj(skip)
            
            # Align sizes if needed
            if skip.shape[2:] != x.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
            
            x = torch.cat([x, skip], dim=1)
        
        return self.conv(x)


class BEVSegmentationHead(nn.Module):
    """
    기본 BEV Segmentation Head
    Fused BEV Feature → Segmentation Map
    """
    def __init__(
        self,
        in_channels: int = 256,
        num_classes: int = 3,  # Background, Land, Debris
        hidden_channels: int = 256,
        use_aspp: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.use_aspp = use_aspp
        
        if use_aspp:
            self.aspp = ASPP(in_channels, hidden_channels)
            decoder_in = hidden_channels
        else:
            self.aspp = None
            decoder_in = in_channels
        
        # Decoder
        self.decoder = nn.Sequential(
            ConvBNReLU(decoder_in, hidden_channels, 3, padding=1),
            nn.Dropout(dropout),
            ConvBNReLU(hidden_channels, hidden_channels // 2, 3, padding=1),
            nn.Dropout(dropout),
            nn.Conv2d(hidden_channels // 2, num_classes, 1),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Fused BEV features (B, C, H, W)
            
        Returns:
            seg_logits: Segmentation logits (B, num_classes, H, W)
        """
        if self.aspp is not None:
            x = self.aspp(x)
        
        seg_logits = self.decoder(x)
        
        return seg_logits


class UNetSegmentationHead(nn.Module):
    """
    UNet 스타일의 Segmentation Head
    Multi-scale features 활용
    """
    def __init__(
        self,
        in_channels: int = 256,
        skip_channels: List[int] = [128, 64],  # Skip connection channels
        num_classes: int = 3,
        hidden_channels: int = 256,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        
        # ASPP for bottleneck
        self.aspp = ASPP(in_channels, hidden_channels)
        
        # Decoder blocks
        self.decoders = nn.ModuleList()
        
        decoder_in = hidden_channels
        for i, skip_c in enumerate(skip_channels):
            decoder_out = hidden_channels // (2 ** (i + 1))
            decoder_out = max(decoder_out, 64)
            
            self.decoders.append(
                DecoderBlock(decoder_in, skip_c, decoder_out, upsample_scale=2)
            )
            decoder_in = decoder_out
        
        # Final classifier
        self.classifier = nn.Sequential(
            ConvBNReLU(decoder_in, 64, 3, padding=1),
            nn.Conv2d(64, num_classes, 1),
        )
        
    def forward(
        self,
        x: torch.Tensor,
        skip_features: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Fused BEV features (B, C, H, W)
            skip_features: List of skip connection features (optional)
            
        Returns:
            seg_logits: Segmentation logits (B, num_classes, H, W)
        """
        # Bottleneck
        x = self.aspp(x)
        
        # Decoder with skip connections
        if skip_features is None:
            skip_features = [None] * len(self.decoders)
        
        for i, decoder in enumerate(self.decoders):
            skip = skip_features[i] if i < len(skip_features) else None
            x = decoder(x, skip)
        
        # Final classification
        seg_logits = self.classifier(x)
        
        return seg_logits


class DeepLabV3PlusHead(nn.Module):
    """
    DeepLabV3+ 스타일의 Segmentation Head
    """
    def __init__(
        self,
        in_channels: int = 256,
        low_level_channels: int = 64,
        num_classes: int = 3,
        aspp_out_channels: int = 256,
        decoder_channels: int = 256,
    ):
        super().__init__()
        
        # ASPP
        self.aspp = ASPP(in_channels, aspp_out_channels)
        
        # Low-level feature projection
        self.low_level_proj = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            ConvBNReLU(aspp_out_channels + 48, decoder_channels, 3, padding=1),
            nn.Dropout(0.5),
            ConvBNReLU(decoder_channels, decoder_channels, 3, padding=1),
            nn.Dropout(0.1),
            nn.Conv2d(decoder_channels, num_classes, 1),
        )
        
        self.num_classes = num_classes
        
    def forward(
        self,
        x: torch.Tensor,
        low_level_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: High-level fused BEV features (B, C, H, W)
            low_level_features: Low-level features for skip connection (B, C_low, H_low, W_low)
            
        Returns:
            seg_logits: Segmentation logits (B, num_classes, H_out, W_out)
        """
        # ASPP
        aspp_out = self.aspp(x)
        
        if low_level_features is not None:
            # Project low-level features
            low_level_out = self.low_level_proj(low_level_features)
            
            # Upsample ASPP output
            aspp_out = F.interpolate(
                aspp_out, size=low_level_out.shape[2:],
                mode='bilinear', align_corners=True
            )
            
            # Concatenate
            x = torch.cat([aspp_out, low_level_out], dim=1)
        else:
            x = aspp_out
        
        # Decoder
        seg_logits = self.decoder(x)
        
        return seg_logits


class InstanceAwareBEVHead(nn.Module):
    """
    Instance-Aware BEV Segmentation Head
    Semantic segmentation + Instance center prediction
    (Panoptic-style segmentation)
    """
    def __init__(
        self,
        in_channels: int = 256,
        num_classes: int = 3,
        hidden_channels: int = 256,
        use_aspp: bool = True,
    ):
        super().__init__()
        
        # Shared encoder
        if use_aspp:
            self.encoder = ASPP(in_channels, hidden_channels)
        else:
            self.encoder = nn.Sequential(
                ConvBNReLU(in_channels, hidden_channels, 3, padding=1),
                ConvBNReLU(hidden_channels, hidden_channels, 3, padding=1),
            )
        
        # Semantic segmentation head
        self.semantic_head = nn.Sequential(
            ConvBNReLU(hidden_channels, hidden_channels // 2, 3, padding=1),
            nn.Conv2d(hidden_channels // 2, num_classes, 1),
        )
        
        # Instance center head (heatmap)
        self.center_head = nn.Sequential(
            ConvBNReLU(hidden_channels, hidden_channels // 2, 3, padding=1),
            nn.Conv2d(hidden_channels // 2, 1, 1),
            nn.Sigmoid(),
        )
        
        # Instance offset head (2D offset to center)
        self.offset_head = nn.Sequential(
            ConvBNReLU(hidden_channels, hidden_channels // 2, 3, padding=1),
            nn.Conv2d(hidden_channels // 2, 2, 1),  # (offset_x, offset_y)
        )
        
        self.num_classes = num_classes
        
    def forward(
        self,
        x: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: Fused BEV features (B, C, H, W)
            
        Returns:
            dict with:
                - semantic: Semantic segmentation logits (B, num_classes, H, W)
                - center: Instance center heatmap (B, 1, H, W)
                - offset: Instance offset (B, 2, H, W)
        """
        # Shared encoding
        features = self.encoder(x)
        
        # Multi-task outputs
        semantic = self.semantic_head(features)
        center = self.center_head(features)
        offset = self.offset_head(features)
        
        return {
            'semantic': semantic,
            'center': center,
            'offset': offset,
        }


class MarineDebrisSegHead(nn.Module):
    """
    해양 부유쓰레기 탐지를 위한 특화된 Segmentation Head
    
    클래스 (3개):
    0: Background (배경/수면)
    1: Land (육지/부두/구조물)
    2: Debris (부유쓰레기)
    """
    
    CLASSES = ['Background', 'Land', 'Debris']
    NUM_CLASSES = 3
    
    def __init__(
        self,
        in_channels: int = 256,
        num_classes: int = 3,
        hidden_channels: int = 256,
        output_stride: int = 1,  # Output resolution relative to input
        use_instance_head: bool = False,
        use_boundary_head: bool = True,  # 경계 검출 (작은 쓰레기 탐지 개선)
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.output_stride = output_stride
        self.use_instance_head = use_instance_head
        self.use_boundary_head = use_boundary_head
        
        # Main encoder
        self.encoder = ASPP(in_channels, hidden_channels)
        
        # Semantic segmentation head
        self.semantic_decoder = nn.Sequential(
            ConvBNReLU(hidden_channels, hidden_channels, 3, padding=1),
            nn.Dropout(0.5),
            ConvBNReLU(hidden_channels, hidden_channels // 2, 3, padding=1),
            nn.Dropout(0.1),
        )
        
        self.semantic_classifier = nn.Conv2d(hidden_channels // 2, num_classes, 1)
        
        # Boundary head (작은 물체 경계 검출)
        if use_boundary_head:
            self.boundary_head = nn.Sequential(
                ConvBNReLU(hidden_channels, hidden_channels // 2, 3, padding=1),
                nn.Conv2d(hidden_channels // 2, 1, 1),
                nn.Sigmoid(),
            )
        
        # Instance head (optional)
        if use_instance_head:
            self.instance_center = nn.Sequential(
                ConvBNReLU(hidden_channels, hidden_channels // 2, 3, padding=1),
                nn.Conv2d(hidden_channels // 2, 1, 1),
                nn.Sigmoid(),
            )
            self.instance_offset = nn.Sequential(
                ConvBNReLU(hidden_channels, hidden_channels // 2, 3, padding=1),
                nn.Conv2d(hidden_channels // 2, 2, 1),
            )
        
        # Size estimation head (쓰레기 크기 추정)
        self.size_head = nn.Sequential(
            ConvBNReLU(hidden_channels, hidden_channels // 4, 3, padding=1),
            nn.Conv2d(hidden_channels // 4, 1, 1),
            nn.ReLU(),  # Size is always positive
        )
        
    def forward(
        self,
        x: torch.Tensor,
        return_all: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: Fused BEV features (B, C, H, W)
            return_all: Return all intermediate outputs
            
        Returns:
            dict with:
                - semantic: Semantic segmentation logits (B, num_classes, H, W)
                - boundary: Boundary prediction (B, 1, H, W) [optional]
                - center: Instance center heatmap (B, 1, H, W) [optional]
                - offset: Instance offset (B, 2, H, W) [optional]
                - size: Size estimation (B, 1, H, W)
        """
        B, C, H, W = x.shape
        
        # Encode
        features = self.encoder(x)
        
        # Semantic segmentation
        semantic_feat = self.semantic_decoder(features)
        semantic = self.semantic_classifier(semantic_feat)
        
        outputs = {'semantic': semantic}
        
        # Boundary
        if self.use_boundary_head:
            boundary = self.boundary_head(features)
            outputs['boundary'] = boundary
        
        # Instance
        if self.use_instance_head:
            center = self.instance_center(features)
            offset = self.instance_offset(features)
            outputs['center'] = center
            outputs['offset'] = offset
        
        # Size estimation
        size = self.size_head(features)
        outputs['size'] = size
        
        # Upsample if needed
        if self.output_stride != 1:
            target_size = (int(H * self.output_stride), int(W * self.output_stride))
            for key in outputs:
                outputs[key] = F.interpolate(
                    outputs[key], size=target_size,
                    mode='bilinear', align_corners=True
                )
        
        return outputs
    
    def get_seg_map(self, outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Get final segmentation map from outputs
        """
        semantic = outputs['semantic']
        seg_map = semantic.argmax(dim=1)
        return seg_map
    
    @staticmethod
    def get_class_name(class_idx: int) -> str:
        return MarineDebrisSegHead.CLASSES[class_idx]
    
    @staticmethod
    def get_class_colors() -> Dict[int, Tuple[int, int, int]]:
        """
        Get RGB colors for each class (for visualization)
        """
        return {
            0: (0, 0, 255),      # Background - Blue (water)
            1: (0, 255, 0),      # Land - Green
            2: (255, 0, 0),      # Debris - Red
        }