"""
BEV Fusion Network for Marine Debris Detection
해양 부유쓰레기 탐지를 위한 통합 BEV Fusion 네트워크

Architecture:
1. Camera Branch: Image → Camera BEV Feature Map
2. LiDAR Branch: Point Cloud → LiDAR BEV Feature Map
3. Fusion: Camera BEV + LiDAR BEV → Fused BEV Feature Map
4. Segmentation Head: Fused BEV → BEV Segmentation Map
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
    완전한 BEV Fusion 네트워크
    Camera와 LiDAR 센서 데이터를 BEV 공간에서 융합하여 Segmentation 수행
    """
    def __init__(
        self,
        # Camera BEV 설정
        camera_bev_channels: int = 256,
        camera_bev_size: Tuple[int, int] = (200, 200),  # (H, W)
        # LiDAR BEV 설정
        lidar_config: Optional[Dict] = None,
        lidar_bev_channels: int = 256,
        # Fusion 설정
        fusion_method: str = 'adaptive',  # 'conv', 'channel_attn', 'spatial_attn', 'cross_attn', 'adaptive'
        fused_channels: int = 256,
        # Segmentation 설정
        num_classes: int = 6,
        use_instance_head: bool = False,
        use_boundary_head: bool = True,
        # BEV grid 설정
        bev_size: Tuple[int, int] = (200, 200),  # 최종 BEV 크기
        point_cloud_range: List[float] = [-30.0, -30.0, -2.0, 30.0, 30.0, 3.0],
        voxel_size: Tuple[float, float, float] = (0.3, 0.3, 5.0),
    ):
        super().__init__()
        
        self.camera_bev_channels = camera_bev_channels
        self.lidar_bev_channels = lidar_bev_channels
        self.fused_channels = fused_channels
        self.num_classes = num_classes
        self.bev_size = bev_size
        self.point_cloud_range = point_cloud_range
        
        # LiDAR BEV Encoder
        if lidar_config is None:
            lidar_config = {
                'point_cloud_range': point_cloud_range,
                'voxel_size': voxel_size,
                'max_points_per_voxel': 32,
                'max_num_voxels': 40000,
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
            camera_bev: Camera BEV features (B, C, H, W) - from Camera BEV Encoder
            points: LiDAR points (N, 5) - (batch_idx, x, y, z, intensity)
            batch_size: Batch size
            
        Returns:
            Dict containing:
                - semantic: Semantic segmentation (B, num_classes, H, W)
                - boundary: Boundary prediction (B, 1, H, W)
                - size: Size estimation (B, 1, H, W)
                - (optional) center, offset for instance segmentation
        """
        # LiDAR BEV encoding
        lidar_bev = self.lidar_encoder(points, batch_size)
        
        # Fusion
        fused_bev = self.fusion(camera_bev, lidar_bev)
        
        # Segmentation
        outputs = self.seg_head(fused_bev)
        
        # Add intermediate features for debugging/visualization
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
        """
        미리 voxelize된 LiDAR 데이터로 forward
        (DataLoader에서 voxelize하는 경우)
        """
        # LiDAR BEV encoding
        lidar_bev = self.lidar_encoder.forward_with_voxels(
            voxels, num_points, coors, batch_size
        )
        
        # Fusion
        fused_bev = self.fusion(camera_bev, lidar_bev)
        
        # Segmentation
        outputs = self.seg_head(fused_bev)
        
        outputs['camera_bev'] = camera_bev
        outputs['lidar_bev'] = lidar_bev
        outputs['fused_bev'] = fused_bev
        
        return outputs
    
    def get_seg_map(self, outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Get final segmentation map"""
        return self.seg_head.get_seg_map(outputs)


class FullBEVFusionNetwork(nn.Module):
    """
    Camera Encoder를 포함한 완전한 End-to-End 네트워크
    Image + Point Cloud → BEV Segmentation Map
    """
    def __init__(
        self,
        # Camera encoder 설정 (외부에서 주입 또는 내부 생성)
        camera_encoder: Optional[nn.Module] = None,
        camera_encoder_config: Optional[Dict] = None,
        # LiDAR 설정
        lidar_config: Optional[Dict] = None,
        # Fusion 설정
        fusion_method: str = 'adaptive',
        fused_channels: int = 256,
        # Segmentation 설정
        num_classes: int = 6,
        # BEV 설정
        bev_size: Tuple[int, int] = (200, 200),
        point_cloud_range: List[float] = [-30.0, -30.0, -2.0, 30.0, 30.0, 3.0],
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
        
        # LiDAR Encoder
        default_lidar_config = {
            'point_cloud_range': point_cloud_range,
            'voxel_size': (0.3, 0.3, 5.0),
            'max_points_per_voxel': 32,
            'max_num_voxels': 40000,
            'backbone_out_channels': 256,
        }
        if lidar_config:
            default_lidar_config.update(lidar_config)
        
        self.lidar_encoder = MarineLiDARBEVEncoder(**default_lidar_config)
        lidar_bev_channels = default_lidar_config['backbone_out_channels']
        
        # Fusion
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
        # Camera BEV encoding
        if camera_bev is None:
            if self.camera_encoder is not None and images is not None:
                camera_bev = self.camera_encoder(
                    images, camera_intrinsics, camera_extrinsics
                )
            else:
                raise ValueError("Either camera_bev or (camera_encoder + images) must be provided")
        
        # LiDAR BEV encoding
        lidar_bev = self.lidar_encoder(points, batch_size)
        
        # Fusion
        fused_bev = self.fusion(camera_bev, lidar_bev)
        
        # Segmentation
        outputs = self.seg_head(fused_bev)
        
        return outputs


class LiDAROnlyNetwork(nn.Module):
    """LiDAR만 사용하는 네트워크 (Ablation study용)"""
    def __init__(
        self,
        lidar_config: Optional[Dict] = None,
        num_classes: int = 6,
        bev_size: Tuple[int, int] = (200, 200),
        point_cloud_range: List[float] = [-30.0, -30.0, -2.0, 30.0, 30.0, 3.0],
    ):
        super().__init__()
        
        default_lidar_config = {
            'point_cloud_range': point_cloud_range,
            'voxel_size': (0.3, 0.3, 5.0),
            'max_points_per_voxel': 32,
            'max_num_voxels': 40000,
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
    """Camera만 사용하는 네트워크 (Ablation study용)"""
    def __init__(
        self,
        camera_bev_channels: int = 256,
        num_classes: int = 6,
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
    """Config로부터 네트워크 빌드"""
    network_type = config.get('type', 'fusion')
    
    if network_type == 'fusion':
        return BEVFusionNetwork(
            camera_bev_channels=config.get('camera_bev_channels', 256),
            lidar_bev_channels=config.get('lidar_bev_channels', 256),
            fusion_method=config.get('fusion_method', 'adaptive'),
            fused_channels=config.get('fused_channels', 256),
            num_classes=config.get('num_classes', 6),
            bev_size=config.get('bev_size', (200, 200)),
            point_cloud_range=config.get('point_cloud_range', [-30.0, -30.0, -2.0, 30.0, 30.0, 3.0]),
        )
    elif network_type == 'lidar_only':
        return LiDAROnlyNetwork(
            num_classes=config.get('num_classes', 6),
            bev_size=config.get('bev_size', (200, 200)),
            point_cloud_range=config.get('point_cloud_range', [-30.0, -30.0, -2.0, 30.0, 30.0, 3.0]),
        )
    elif network_type == 'camera_only':
        return CameraOnlyNetwork(
            camera_bev_channels=config.get('camera_bev_channels', 256),
            num_classes=config.get('num_classes', 6),
        )
    else:
        raise ValueError(f"Unknown network type: {network_type}")
