"""
LiDAR BEV Module
LiDAR 포인트 클라우드를 BEV Feature Map으로 변환하는 모듈

PointPillars 기반으로 구현됨
- 사전학습 필요 없음 (End-to-End 학습)
- Segmentation Loss로 전체 네트워크와 함께 학습

주요 컴포넌트:
1. DynamicVoxelization: 포인트 → Pillar 변환
2. PillarFeatureNet: Pillar → Feature 인코딩 (PointNet 스타일)
3. PointPillarsScatter: Feature → BEV pseudo-image
4. LiDARBackbone: 2D CNN으로 최종 BEV feature 추출
"""

from .pointpillars import (
    # Main Encoders
    LiDARBEVEncoder,
    MarineLiDARBEVEncoder,
    # Components
    PillarFeatureNet,
    PFNLayer,
    PointPillarsScatter,
    LiDARBackbone,
    DynamicVoxelization,
    # Factory function
    create_lidar_bev_encoder,
)

__all__ = [
    # Main Encoders
    'LiDARBEVEncoder',
    'MarineLiDARBEVEncoder',
    # Components
    'PillarFeatureNet',
    'PFNLayer',
    'PointPillarsScatter',
    'LiDARBackbone',
    'DynamicVoxelization',
    # Factory
    'create_lidar_bev_encoder',
]