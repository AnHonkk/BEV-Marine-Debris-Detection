# models/camera_bev/camera_bev_module.py

import torch
import torch.nn as nn
from typing import Dict
from .camera_encoder import CameraBackbone
from .depth_net_unet import LightUNetDepthNet  # ← Light U-Net! ⭐
from .calibration import LearnableCalibration
from .view_transform import LSSViewTransform


class CameraBEVModule(nn.Module):
    """
    Camera → BEV 변환 모듈
    
    Architecture:
        Input Image
          ↓
        ConvNeXt Backbone + FPN
          ↓
        Light U-Net Depth Head (improved!)
          ↓
        Learnable Calibration
          ↓
        LSS Transformation
          ↓
        BEV Features
    
    Components:
        1. CameraBackbone: ConvNeXt-Tiny + FPN
        2. LightUNetDepthNet: U-Net depth distribution
        3. LearnableCalibration: Learnable K, T
        4. LSSViewTransform: Lift-Splat-Shoot
    """
    
    def __init__(self, config: dict):
        super().__init__()
        
        # Validate config
        self._validate_config(config)
        self.config = config
        
        # 1. ConvNeXt Backbone
        self.backbone = CameraBackbone(
            backbone_name=config.get('backbone', 'convnext_tiny'),
            pretrained=config.get('pretrained', True),
            out_channels=config.get('feat_channels', 256),
            use_fpn=config.get('use_fpn', True)
        )
        
        # 2. Depth Network (Light U-Net!) ⭐
        self.depth_net = LightUNetDepthNet(
            in_channels=config.get('feat_channels', 256),
            depth_bins=config.get('depth_bins', 64),
            depth_range=config.get('depth_range', (1.0, 50.0))
        )
        
        # 3. Learnable Calibration
        self.calibration = LearnableCalibration(
            K_init=config['K_init'],
            T_init=config['T_init'],
            learn_intrinsic=config.get('learn_intrinsic', True),
            learn_extrinsic=config.get('learn_extrinsic', True)
        )
        
        # 4. LSS View Transformation
        # Feature stride (ConvNeXt first feature is stride 4)
        stride = config.get('feature_stride', 4)
        feat_h = config['img_size'][0] // stride
        feat_w = config['img_size'][1] // stride
        
        self.view_transform = LSSViewTransform(
            img_size=(feat_h, feat_w),
            bev_size=tuple(config['bev_size']),
            bev_range=tuple(config['bev_range']),
            feat_channels=config.get('feat_channels', 256),
            depth_bins=config.get('depth_bins', 64)
        )
    
    def _validate_config(self, config: dict):
        """Validate required configuration keys"""
        required_keys = [
            'K_init', 'T_init', 'img_size', 'bev_size', 'bev_range'
        ]
        
        missing_keys = [key for key in required_keys if key not in config]
        
        if missing_keys:
            raise ValueError(
                f"Missing required config keys: {missing_keys}\n"
                f"Required keys: {required_keys}"
            )
    
    def forward(
        self,
        images: torch.Tensor,
        refine_calibration: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass: Image → BEV Features
        
        Args:
            images: [B, 3, H, W] - Input RGB images
            refine_calibration: If True, calibration parameters are learnable
                               If False, they are frozen (detached from graph)
        
        Returns:
            Dictionary containing:
                - bev_features: [B, C, H_bev, W_bev] - BEV feature map
                - depth_probs: [B, D, H/4, W/4] - Depth distribution
                - depth_expected: [B, 1, H/4, W/4] - Expected depth
                - K: [3, 3] - Intrinsic matrix
                - T: [4, 4] - Extrinsic matrix
                - img_features: [B, C, H/4, W/4] - Image features (optional)
        """
        
        # 1. Extract multi-scale features
        multi_scale_features = self.backbone(images)
        
        # Use first feature (stride 4, highest resolution)
        img_features = multi_scale_features[0]  # [B, 256, H/4, W/4]
        
        # 2. Predict depth distribution (Light U-Net!)
        depth_outputs = self.depth_net(img_features)
        
        # 3. Get calibration parameters
        K, T = self.calibration()
        
        # Control gradient flow for calibration
        if not refine_calibration:
            K = K.detach()
            T = T.detach()
        
        # 4. LSS transformation: Image features → BEV
        bev_features = self.view_transform(
            img_features=img_features,
            depth_probs=depth_outputs['depth_probs'],
            depth_values=depth_outputs['depth_values'],
            K=K,
            T=T
        )
        
        return {
            'bev_features': bev_features,
            'depth_probs': depth_outputs['depth_probs'],
            'depth_expected': depth_outputs['depth_expected'],
            'depth_values': depth_outputs['depth_values'],
            'K': K,
            'T': T,
            'img_features': img_features  # For visualization or auxiliary tasks
        }