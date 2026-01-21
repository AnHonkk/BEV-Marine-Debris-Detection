"""
BEV Fusion Network for Marine Debris Detection (400x400 출력)
해양 부유쓰레기 탐지를 위한 통합 BEV Fusion 네트워크

Architecture:
1. Camera Branch: Image → Camera BEV Feature Map (400x400)
2. LiDAR Branch: Point Cloud → LiDAR BEV Feature Map (400x400)
3. Fusion: Camera BEV + LiDAR BEV → Fused BEV Feature Map (400x400)
4. Segmentation Head: Fused BEV → BEV Segmentation Map (400x400)

Classes (3):
- 0: Background (배경/수면)
- 1: Land (육지/부두/구조물)
- 2: Debris (부유쓰레기)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any

from .lidar_bev import LiDARBEVEncoder, MarineLiDARBEVEncoder
from .fusion import BEVFusion, MultiScaleBEVFusion
from .segmentation_head import MarineDebrisSegHead, BEVSegmentationHead


class BEVFusionNetwork(nn.Module):
    """
    완전한 BEV Fusion 네트워크 (400x400 출력)
    Camera와 LiDAR 센서 데이터를 BEV 공간에서 융합하여 Segmentation 수행
    """
    def __init__(
        self,
        # Camera BEV 설정
        camera_bev_channels: int = 256,
        camera_bev_size: Tuple[int, int] = (400, 400),
        # LiDAR BEV 설정
        lidar_config: Optional[Dict] = None,
        lidar_bev_channels: int = 256,
        # Fusion 설정
        fusion_method: str = 'adaptive',
        fused_channels: int = 256,
        # Segmentation 설정 (3 classes)
        num_classes: int = 3,
        use_instance_head: bool = False,
        use_boundary_head: bool = True,
        # BEV grid 설정 (400x400, 0.1m resolution)
        bev_size: Tuple[int, int] = (400, 400),
        point_cloud_range: List[float] = [-20.0, -20.0, -2.0, 20.0, 20.0, 3.0],
        voxel_size: Tuple[float, float, float] = (0.1, 0.1, 5.0),
    ):
        super().__init__()
        
        self.camera_bev_channels = camera_bev_channels
        self.lidar_bev_channels = lidar_bev_channels
        self.fused_channels = fused_channels
        self.num_classes = num_classes
        self.bev_size = bev_size
        self.point_cloud_range = point_cloud_range
        
        # LiDAR BEV Encoder (400x400, 0.1m resolution)
        if lidar_config is None:
            lidar_config = {
                'point_cloud_range': point_cloud_range,
                'voxel_size': voxel_size,
                'max_points_per_voxel': 32,
                'max_num_voxels': 60000,
                'backbone_out_channels': lidar_bev_channels,
            }
        
        self.lidar_encoder = MarineLiDARBEVEncoder(**lidar_config)
        
        # BEV Fusion (400x400)
        self.fusion = BEVFusion(
            camera_channels=camera_bev_channels,
            lidar_channels=lidar_bev_channels,
            out_channels=fused_channels,
            fusion_method=fusion_method,
            target_shape=bev_size,
        )
        
        # Segmentation Head (3 classes)
        self.seg_head = MarineDebrisSegHead(
            in_channels=fused_channels,
            num_classes=num_classes,
            hidden_channels=fused_channels,
            use_instance_head=use_instance_head,
            use_boundary_head=use_boundary_head,
        )
        
    def forward(
        self,
        camera_bev: torch.Tensor,
        points: torch.Tensor,
        batch_size: int,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            camera_bev: Camera BEV features (B, 256, 400, 400)
            points: LiDAR points (N, 5) - (batch_idx, x, y, z, intensity)
            batch_size: Batch size
            
        Returns:
            Dict containing:
                - semantic: Semantic segmentation (B, 3, 400, 400)
                - boundary: Boundary prediction (B, 1, 400, 400)
                - size: Size estimation (B, 1, 400, 400)
        """
        # LiDAR BEV encoding (400x400)
        lidar_bev = self.lidar_encoder(points, batch_size)
        
        # Fusion (400x400)
        fused_bev = self.fusion(camera_bev, lidar_bev)
        
        # Segmentation (400x400)
        outputs = self.seg_head(fused_bev)
        
        # Add intermediate features
        outputs['camera_bev'] = camera_bev
        outputs['lidar_bev'] = lidar_bev
        outputs['fused_bev'] = fused_bev
        
        return outputs
    
    def forward_with_voxels(
        self,
        camera_bev: torch.Tensor,
        voxels: torch.Tensor,
        num_points: torch.Tensor,
        coors: torch.Tensor,
        batch_size: int,
    ) -> Dict[str, torch.Tensor]:
        """미리 voxelize된 LiDAR 데이터로 forward"""
        lidar_bev = self.lidar_encoder.forward_with_voxels(
            voxels, num_points, coors, batch_size
        )
        
        fused_bev = self.fusion(camera_bev, lidar_bev)
        outputs = self.seg_head(fused_bev)
        
        outputs['camera_bev'] = camera_bev
        outputs['lidar_bev'] = lidar_bev
        outputs['fused_bev'] = fused_bev
        
        return outputs
    
    def get_seg_map(self, outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Get final segmentation map (400x400)"""
        return self.seg_head.get_seg_map(outputs)


class FullBEVFusionNetwork(nn.Module):
    """
    Camera Encoder를 포함한 완전한 End-to-End 네트워크 (400x400)
    Image + Point Cloud → BEV Segmentation Map
    """
    def __init__(
        self,
        camera_encoder: Optional[nn.Module] = None,
        camera_encoder_config: Optional[Dict] = None,
        lidar_config: Optional[Dict] = None,
        fusion_method: str = 'adaptive',
        fused_channels: int = 256,
        num_classes: int = 3,
        bev_size: Tuple[int, int] = (400, 400),
        point_cloud_range: List[float] = [-20.0, -20.0, -2.0, 20.0, 20.0, 3.0],
    ):
        super().__init__()
        
        self.bev_size = bev_size
        self.point_cloud_range = point_cloud_range
        self.num_classes = num_classes
        
        # Camera Encoder
        if camera_encoder is not None:
            self.camera_encoder = camera_encoder
            camera_bev_channels = getattr(camera_encoder, 'out_channels', 256)
        else:
            self.camera_encoder = None
            camera_bev_channels = camera_encoder_config.get('out_channels', 256) if camera_encoder_config else 256
        
        # LiDAR Encoder (400x400, 0.1m resolution)
        default_lidar_config = {
            'point_cloud_range': point_cloud_range,
            'voxel_size': (0.1, 0.1, 5.0),
            'max_points_per_voxel': 32,
            'max_num_voxels': 60000,
            'backbone_out_channels': 256,
        }
        if lidar_config:
            default_lidar_config.update(lidar_config)
        
        self.lidar_encoder = MarineLiDARBEVEncoder(**default_lidar_config)
        lidar_bev_channels = default_lidar_config['backbone_out_channels']
        
        # Fusion (400x400)
        self.fusion = BEVFusion(
            camera_channels=camera_bev_channels,
            lidar_channels=lidar_bev_channels,
            out_channels=fused_channels,
            fusion_method=fusion_method,
            target_shape=bev_size,
        )
        
        # Segmentation Head (3 classes)
        self.seg_head = MarineDebrisSegHead(
            in_channels=fused_channels,
            num_classes=num_classes,
            hidden_channels=fused_channels,
            use_instance_head=True,
            use_boundary_head=True,
        )
        
    def forward(
        self,
        images: Optional[torch.Tensor] = None,
        camera_intrinsics: Optional[torch.Tensor] = None,
        camera_extrinsics: Optional[torch.Tensor] = None,
        camera_bev: Optional[torch.Tensor] = None,
        points: torch.Tensor = None,
        batch_size: int = 1,
    ) -> Dict[str, torch.Tensor]:
        """End-to-end forward"""
        if camera_bev is None:
            if self.camera_encoder is not None and images is not None:
                camera_bev = self.camera_encoder(
                    images, camera_intrinsics, camera_extrinsics
                )
            else:
                raise ValueError("Either camera_bev or (camera_encoder + images) must be provided")
        
        lidar_bev = self.lidar_encoder(points, batch_size)
        fused_bev = self.fusion(camera_bev, lidar_bev)
        outputs = self.seg_head(fused_bev)
        
        return outputs


class LiDAROnlyNetwork(nn.Module):
    """LiDAR만 사용하는 네트워크 (400x400, 3 classes, 0.1m resolution)"""
    def __init__(
        self,
        lidar_config: Optional[Dict] = None,
        num_classes: int = 3,
        bev_size: Tuple[int, int] = (400, 400),
        point_cloud_range: List[float] = [-20.0, -20.0, -2.0, 20.0, 20.0, 3.0],
    ):
        super().__init__()
        
        default_lidar_config = {
            'point_cloud_range': point_cloud_range,
            'voxel_size': (0.1, 0.1, 5.0),
            'max_points_per_voxel': 32,
            'max_num_voxels': 60000,
            'backbone_out_channels': 256,
        }
        if lidar_config:
            default_lidar_config.update(lidar_config)
        
        self.lidar_encoder = MarineLiDARBEVEncoder(**default_lidar_config)
        
        self.seg_head = MarineDebrisSegHead(
            in_channels=default_lidar_config['backbone_out_channels'],
            num_classes=num_classes,
            hidden_channels=256,
        )
        
    def forward(self, points: torch.Tensor, batch_size: int) -> Dict[str, torch.Tensor]:
        lidar_bev = self.lidar_encoder(points, batch_size)
        outputs = self.seg_head(lidar_bev)
        outputs['lidar_bev'] = lidar_bev
        return outputs


class CameraOnlyNetwork(nn.Module):
    """Camera만 사용하는 네트워크 (400x400, 3 classes)"""
    def __init__(
        self,
        camera_bev_channels: int = 256,
        num_classes: int = 3,
    ):
        super().__init__()
        
        self.seg_head = MarineDebrisSegHead(
            in_channels=camera_bev_channels,
            num_classes=num_classes,
            hidden_channels=256,
        )
        
    def forward(self, camera_bev: torch.Tensor) -> Dict[str, torch.Tensor]:
        outputs = self.seg_head(camera_bev)
        outputs['camera_bev'] = camera_bev
        return outputs


def build_network(config: Dict) -> nn.Module:
    """Config로부터 네트워크 빌드 (400x400, 3 classes, 0.1m resolution)"""
    network_type = config.get('type', 'fusion')
    
    if network_type == 'fusion':
        return BEVFusionNetwork(
            camera_bev_channels=config.get('camera_bev_channels', 256),
            lidar_bev_channels=config.get('lidar_bev_channels', 256),
            fusion_method=config.get('fusion_method', 'adaptive'),
            fused_channels=config.get('fused_channels', 256),
            num_classes=config.get('num_classes', 3),
            bev_size=tuple(config.get('bev_size', [400, 400])),
            point_cloud_range=config.get('point_cloud_range', [-20.0, -20.0, -2.0, 20.0, 20.0, 3.0]),
            voxel_size=tuple(config.get('voxel_size', [0.1, 0.1, 5.0])),
        )
    elif network_type == 'lidar_only':
        return LiDAROnlyNetwork(
            num_classes=config.get('num_classes', 3),
            bev_size=tuple(config.get('bev_size', [400, 400])),
            point_cloud_range=config.get('point_cloud_range', [-20.0, -20.0, -2.0, 20.0, 20.0, 3.0]),
        )
    elif network_type == 'camera_only':
        return CameraOnlyNetwork(
            camera_bev_channels=config.get('camera_bev_channels', 256),
            num_classes=config.get('num_classes', 3),
        )
    else:
        raise ValueError(f"Unknown network type: {network_type}")