"""
BEV Fusion Network
- Camera Module: User's CameraBEVModule
- LiDAR Module: PointPillars
- Fusion: Adaptive Fusion
- Head: Segmentation Head
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import os
import numpy as np  # [수정] numpy 임포트 추가

from .lidar_bev import LiDARBEVEncoder, MarineLiDARBEVEncoder
from .fusion import BEVFusion
from .segmentation_head import MarineDebrisSegHead

# 사용자님의 모듈(CameraBEVModule) 임포트
try:
    from .camera_bev.camera_bev_module import CameraBEVModule
except ImportError:
    print("Warning: CameraBEVModule not found. Camera branch will fail.")
    pass

class BEVFusionNetwork(nn.Module):
    """LiDAR-only 모드"""
    def __init__(
        self,
        camera_bev_channels: int = 256,
        camera_bev_size: Tuple[int, int] = (200, 200),
        lidar_config: Optional[Dict] = None,
        lidar_bev_channels: int = 256,
        fusion_method: str = 'adaptive',
        fused_channels: int = 256,
        num_classes: int = 3,
        use_instance_head: bool = False,
        use_boundary_head: bool = True,
        output_stride: int = 1,
        bev_size: Tuple[int, int] = (200, 200),
        point_cloud_range: List[float] = [-20.0, -20.0, -2.0, 20.0, 20.0, 3.0],
        voxel_size: Tuple[float, float, float] = (0.2, 0.2, 5.0),
    ):
        super().__init__()
        self.bev_size = bev_size
        
        # LiDAR BEV Encoder
        if lidar_config is None:
            lidar_config = {
                'point_cloud_range': point_cloud_range,
                'voxel_size': voxel_size,
                'max_points_per_voxel': 32,
                'max_num_voxels': 25000,
                'backbone_out_channels': lidar_bev_channels,
            }
        
        self.lidar_encoder = MarineLiDARBEVEncoder(**lidar_config)
        
        # BEV Fusion
        self.fusion = BEVFusion(
            camera_channels=camera_bev_channels,
            lidar_channels=lidar_bev_channels,
            out_channels=fused_channels,
            fusion_method=fusion_method,
            target_shape=bev_size,
        )
        
        # Segmentation Head
        self.seg_head = MarineDebrisSegHead(
            in_channels=fused_channels,
            num_classes=num_classes,
            hidden_channels=fused_channels,
            use_instance_head=use_instance_head,
            use_boundary_head=use_boundary_head,
            output_stride=output_stride
        )
        
    def forward(self, camera_bev, points, batch_size):
        lidar_bev = self.lidar_encoder(points, batch_size)
        fused_bev = self.fusion(camera_bev, lidar_bev)
        outputs = self.seg_head(fused_bev)
        return outputs

class FullBEVFusionNetwork(nn.Module):
    """Camera Image + LiDAR Point Cloud -> Fusion (End-to-End)"""
    def __init__(
        self,
        camera_config: Dict,
        lidar_config: Dict,
        fusion_method: str = 'adaptive',
        fused_channels: int = 256,
        num_classes: int = 3,
        use_instance_head: bool = False,
        use_boundary_head: bool = True,
        output_stride: int = 1,
        bev_size: Tuple[int, int] = (200, 200),
    ):
        super().__init__()
        
        # Camera Module
        self.camera_encoder = CameraBEVModule(camera_config)
        
        # LiDAR Encoder
        self.lidar_encoder = MarineLiDARBEVEncoder(**lidar_config)
        
        # Fusion
        self.fusion = BEVFusion(
            camera_channels=camera_config.get('feat_channels', 256),
            lidar_channels=lidar_config['backbone_out_channels'],
            out_channels=fused_channels,
            fusion_method=fusion_method,
            target_shape=bev_size,
        )
        
        # Head
        self.seg_head = MarineDebrisSegHead(
            in_channels=fused_channels,
            num_classes=num_classes,
            hidden_channels=fused_channels,
            use_instance_head=use_instance_head,
            use_boundary_head=use_boundary_head,
            output_stride=output_stride
        )

    def forward(self, images, points, batch_size):
        # Camera Branch
        camera_outputs = self.camera_encoder(images)
        camera_bev = camera_outputs['bev_features']
        
        # LiDAR Branch
        lidar_bev = self.lidar_encoder(points, batch_size)
        
        # Fusion
        fused_bev = self.fusion(camera_bev, lidar_bev)
        
        # Segmentation
        outputs = self.seg_head(fused_bev)
        
        if self.training:
            outputs['depth_probs'] = camera_outputs['depth_probs']
            
        return outputs

def build_network(config: Dict) -> nn.Module:
    """Config -> Model 빌더"""
    
    if 'model' in config:
        model_cfg = config['model']
    else:
        model_cfg = config

    # 카메라 사용 여부 확인
    use_camera = config.get('camera', {}).get('use_camera', False) or \
                 model_cfg.get('camera', {}).get('use_camera', False) or \
                 'camera_bev_config' in config

    # 공통 설정
    output_stride = model_cfg.get('output_stride', 1)
    bev_size = tuple(model_cfg.get('bev_size', [200, 200]))
    point_cloud_range = model_cfg['point_cloud_range']
    voxel_size = tuple(model_cfg['voxel_size'])
    
    # LiDAR Config
    lidar_cfg = {
        'point_cloud_range': point_cloud_range,
        'voxel_size': voxel_size,
        'max_points_per_voxel': model_cfg.get('max_points_per_voxel', 32),
        'max_num_voxels': model_cfg.get('max_num_voxels', 25000),
        'backbone_out_channels': model_cfg.get('lidar_bev_channels', 256),
    }

    if use_camera:
        print("[Build Network] Building Full BEV Fusion Network (Camera + LiDAR)")
        
        # [수정] numpy array로 변환하여 설정 전달
        camera_cfg = {
            # Backbone & Structure
            'backbone': 'convnext_tiny',
            'pretrained': True,
            'feat_channels': model_cfg.get('camera_bev_channels', 256),
            'use_fpn': True,
            'feature_stride': 4,
            
            # Geometry
            'img_size': (368, 640),
            'bev_size': bev_size,
            'bev_range': point_cloud_range[:4],
            
            # Depth & Calibration
            'depth_bins': 112,
            'depth_range': (1.0, 50.0),
            # [중요] List -> np.array 변환 (dtype 명시)
            'K_init': np.array([[278.27, 0, 320.0], [0, 278.27, 184.0], [0, 0, 1]], dtype=np.float32),
            'T_init': np.eye(4, dtype=np.float32), 
            'learn_intrinsic': True,
            'learn_extrinsic': True,
            
            # Checkpoint
            'depth_net_checkpoint': '/home/anhong/BEVFusion/checkpoints/depth_net/best_depth_net.pth'
        }
        
        model = FullBEVFusionNetwork(
            camera_config=camera_cfg,
            lidar_config=lidar_cfg,
            fusion_method=model_cfg.get('fusion_method', 'adaptive'),
            fused_channels=model_cfg.get('fused_channels', 256),
            num_classes=model_cfg.get('num_classes', 3),
            use_instance_head=model_cfg.get('use_instance_head', False),
            use_boundary_head=model_cfg.get('use_boundary_head', True),
            output_stride=output_stride,
            bev_size=bev_size
        )
        
        ckpt_path = camera_cfg['depth_net_checkpoint']
        if os.path.exists(ckpt_path):
            print(f"Loading DepthNet weights from {ckpt_path}")
            # 모듈 내부 로직이 없다면 여기서 로드 (현재 사용자 모듈은 내부 로드 로직이 없어보여 추가하면 좋음)
            # 일단은 넘어가고, 필요하면 가중치 로드 코드 추가
            try:
                # CameraBEVModule 구조상 depth_net이 내부에 있음
                state_dict = torch.load(ckpt_path, map_location='cpu')
                model.camera_encoder.depth_net.load_state_dict(state_dict['model_state_dict'] if 'model_state_dict' in state_dict else state_dict, strict=False)
                print("✓ Pre-trained DepthNet weights loaded manually.")
            except:
                print("Warning: Failed to load DepthNet weights automatically.")

        return model

    else:
        print("[Build Network] Building Standard BEV Fusion Network (LiDAR input / Ext Camera Feature)")
        return BEVFusionNetwork(
            camera_bev_channels=model_cfg.get('camera_bev_channels', 256),
            lidar_bev_channels=model_cfg.get('lidar_bev_channels', 256),
            fusion_method=model_cfg.get('fusion_method', 'adaptive'),
            fused_channels=model_cfg.get('fused_channels', 256),
            num_classes=model_cfg.get('num_classes', 3),
            bev_size=bev_size,
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
            output_stride=output_stride,
        )