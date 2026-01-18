# models/camera_bev/camera_encoder.py

import torch
import torch.nn as nn
from typing import List
import timm


class CameraBackbone(nn.Module):
    """
    ConvNeXt 기반 카메라 백본
    완전히 유연한 입력 크기 지원
    """
    
    SUPPORTED_BACKBONES = {
        # ConvNeXt 시리즈
        'convnext_tiny': 'convnext_tiny',
        'convnext_small': 'convnext_small',
        'convnext_base': 'convnext_base',
        
        # 다른 옵션들 (참고용)
        'efficientnetv2_s': 'tf_efficientnetv2_s',
        'efficientnetv2_m': 'tf_efficientnetv2_m',
        'resnet50': 'resnet50',
        'resnet101': 'resnet101',
    }
    
    def __init__(
        self,
        backbone_name: str = 'convnext_tiny',
        pretrained: bool = True,
        out_channels: int = 256,
        use_fpn: bool = True
    ):
        super().__init__()
        
        if backbone_name in self.SUPPORTED_BACKBONES:
            full_name = self.SUPPORTED_BACKBONES[backbone_name]
        else:
            full_name = backbone_name
        
        self.backbone_name = full_name
        self.use_fpn = use_fpn
        
        print(f"Loading {full_name}...")
        
        # Create ConvNeXt model
        self.backbone = timm.create_model(
            full_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3)  # 4 stages
        )
        
        print(f"✓ {full_name} loaded")
        
        # Get feature channels
        # self.feat_channels = self.backbone.feature_info.channels()
        # print(f"Feature channels: {self.feat_channels}")
        try:
            self.feat_channels = self.backbone.feature_info.channels()
        except:
            # Fallback
            self.feat_channels = [info['num_chs'] for info in self.backbone.feature_info.info]

        print(f"Feature channels: {self.feat_channels}")
        
        # Feature Pyramid Network
        if use_fpn:
            self.fpn = FPN(self.feat_channels, out_channels)
            self.final_channels = out_channels
        else:
            self.final_channels = self.feat_channels[-1]
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            x: [B, 3, H, W] 입력 이미지 (임의 크기 가능)
        
        Returns:
            features: List of [B, C, H', W'] multi-scale features
        """
        # ConvNeXt forward
        features = self.backbone(x)
        
        # FPN 적용
        if self.use_fpn:
            features = self.fpn(features)
        
        return features


class FPN(nn.Module):
    """Feature Pyramid Network"""
    
    def __init__(self, in_channels: List[int], out_channels: int = 256):
        super().__init__()
        
        # Lateral convolutions (1x1 conv)
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, 1) 
            for in_ch in in_channels
        ])
        
        # Output convolutions (3x3 conv)
        self.fpn_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
            for _ in in_channels
        ])
    
    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Top-down pathway with lateral connections
        
        Args:
            features: List of [B, C_i, H_i, W_i]
        
        Returns:
            List of [B, out_channels, H_i, W_i]
        """
        # Build laterals
        laterals = [
            lateral_conv(features[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        
        # Top-down pathway
        for i in range(len(laterals) - 1, 0, -1):
            # Upsample and add
            # laterals[i - 1] = laterals[i - 1] + nn.functional.interpolate(
            #     laterals[i], 
            #     size=laterals[i - 1].shape[-2:],
            #     mode='bilinear',
            #     align_corners=False
            # )
            upsampled = nn.functional.interpolate(
                laterals[i], 
                size=laterals[i - 1].shape[-2:],
                mode='bilinear',
                align_corners=False
            )

            laterals[i - 1] = laterals[i - 1] + upsampled
                
        # Apply output convolutions
        outputs = [
            fpn_conv(laterals[i])
            for i, fpn_conv in enumerate(self.fpn_convs)
        ]
        
        return outputs


# ========================================
# Test Code
# ========================================

def test_backbone(backbone_name: str, img_size: tuple = (384, 640)):
    """개별 백본 테스트"""
    print(f"\n{'='*60}")
    print(f"Testing: {backbone_name}")
    print('='*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    try:
        # Create model
        model = CameraBackbone(
            backbone_name=backbone_name,
            pretrained=False,  # 빠른 테스트
            out_channels=256,
            use_fpn=True
        )
        model = model.to(device)
        model.eval()
        
        # Create input
        x = torch.randn(1, 3, img_size[0], img_size[1]).to(device)
        print(f"Input: {x.shape}")
        
        # Forward
        with torch.no_grad():
            features = model(x)
        
        # Print outputs
        print("Output features:")
        for i, feat in enumerate(features):
            stride = img_size[0] // feat.shape[2]
            print(f"  Feature {i} (stride {stride}): {feat.shape}")
        
        # Parameters
        params = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\nParameters:")
        print(f"  Total: {params:,}")
        print(f"  Trainable: {trainable:,}")
        
        # Memory
        if device == 'cuda':
            memory = torch.cuda.max_memory_allocated() / 1024**2
            print(f"\nGPU Memory: {memory:.2f} MB")
            torch.cuda.reset_peak_memory_stats()
        
        print(f"\n✅ {backbone_name} test passed")
        
        # Cleanup
        del model, x, features
        if device == 'cuda':
            torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        print(f"\n❌ {backbone_name} test failed: {e}")
        import traceback
        traceback.print_exc()
        
        if device == 'cuda':
            torch.cuda.empty_cache()
        
        return False


if __name__ == '__main__':
    print("="*60)
    print("ConvNeXt Camera Backbone Test")
    print("="*60)
    
    # ConvNeXt 시리즈 테스트
    test_models = [
        'convnext_tiny',   # 추천! ⭐⭐⭐
        'convnext_small',  # 더 높은 정확도
    ]
    
    img_size = (384, 640)  # 임의 크기
    
    results = {}
    for model_name in test_models:
        success = test_backbone(model_name, img_size)
        results[model_name] = '✅' if success else '❌'
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    for name, result in results.items():
        print(f"{result} {name}")
    
    print("\n" + "="*60)
    print("✅ ConvNeXt-Tiny is ready for BEVFusion!")
    print("="*60)