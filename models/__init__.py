"""
BEV Fusion Models for Marine Debris Detection
해양 부유쓰레기 탐지를 위한 BEV Fusion 모델

Modules:
- lidar_bev: LiDAR 포인트 클라우드 → BEV Feature Map
- fusion: Camera BEV + LiDAR BEV Fusion
- segmentation_head: BEV Feature → Segmentation Map
- full_network: 전체 통합 네트워크
"""

from .lidar_bev import (
    LiDARBEVEncoder,
    MarineLiDARBEVEncoder,
    PillarFeatureNet,
    PointPillarsScatter,
    LiDARBackbone,
)

from .fusion import (
    BEVFusion,
    MultiScaleBEVFusion,
    ConvFuser,
    ChannelAttentionFuser,
    SpatialAttentionFuser,
    CrossAttentionFuser,
    AdaptiveFuser,
)

from .segmentation_head import (
    BEVSegmentationHead,
    UNetSegmentationHead,
    DeepLabV3PlusHead,
    InstanceAwareBEVHead,
    MarineDebrisSegHead,
    ASPP,
)

from .full_network import (
    BEVFusionNetwork,
    FullBEVFusionNetwork,
    LiDAROnlyNetwork,
    CameraOnlyNetwork,
    build_network,
)

__all__ = [
    # LiDAR BEV
    'LiDARBEVEncoder',
    'MarineLiDARBEVEncoder',
    'PillarFeatureNet',
    'PointPillarsScatter',
    'LiDARBackbone',
    # Fusion
    'BEVFusion',
    'MultiScaleBEVFusion',
    'ConvFuser',
    'ChannelAttentionFuser',
    'SpatialAttentionFuser',
    'CrossAttentionFuser',
    'AdaptiveFuser',
    # Segmentation Head
    'BEVSegmentationHead',
    'UNetSegmentationHead',
    'DeepLabV3PlusHead',
    'InstanceAwareBEVHead',
    'MarineDebrisSegHead',
    'ASPP',
    # Full Network
    'BEVFusionNetwork',
    'FullBEVFusionNetwork',
    'LiDAROnlyNetwork',
    'CameraOnlyNetwork',
    'build_network',
]
