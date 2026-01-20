"""
PointPillars 기반 LiDAR BEV Encoder (개선된 버전)
해양 부유쓰레기 탐지를 위한 LiDAR 포인트 클라우드 → BEV Feature Map 변환

핵심 구조:
1. Voxelization: 포인트 클라우드를 pillar(세로 기둥)로 그룹화
2. PillarFeatureNet: 각 pillar 내 포인트들을 PointNet으로 인코딩
3. Scatter: Pillar features를 BEV pseudo-image로 scatter
4. Backbone: 2D CNN으로 BEV feature 추출

학습 방식:
- End-to-End로 전체 네트워크와 함께 학습 (별도 사전학습 불필요)
- Segmentation Loss로 역전파 → 모든 레이어 동시 학습

Reference: PointPillars: Fast Encoders for Object Detection from Point Clouds (CVPR 2019)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional


class PillarFeatureNet(nn.Module):
    """
    Pillar Feature Network (PFN)
    
    각 pillar 내의 포인트들을 feature로 인코딩
    PointNet 스타일의 shared MLP + max pooling 사용
    
    Input features per point:
    - x, y, z, intensity (4)
    - xc, yc, zc: offset to pillar mean (3)  
    - xp, yp: offset to pillar center (2)
    Total: 9 features (+ optional distance)
    """
    def __init__(
        self,
        in_channels: int = 4,  # x, y, z, intensity
        feat_channels: int = 64,
        with_distance: bool = False,
        voxel_size: Tuple[float, float, float] = (0.25, 0.25, 8.0),
        point_cloud_range: List[float] = [-50.0, -50.0, -5.0, 50.0, 50.0, 3.0],
        num_filters: List[int] = [64],  # Multi-layer PFN
    ):
        super().__init__()
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.with_distance = with_distance
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        
        # Input feature dimension
        # 기본: x, y, z, intensity (4) + xc, yc, zc (3) + xp, yp (2) = 9
        in_feat_channels = in_channels + 5
        if with_distance:
            in_feat_channels += 1
        
        # Build PFN layers
        pfn_layers = []
        in_ch = in_feat_channels
        
        for i, out_ch in enumerate(num_filters):
            pfn_layers.append(
                PFNLayer(in_ch, out_ch, last_layer=(i == len(num_filters) - 1))
            )
            in_ch = out_ch
        
        self.pfn_layers = nn.ModuleList(pfn_layers)
        self.out_channels = num_filters[-1] if num_filters else in_feat_channels
        
    def forward(
        self,
        voxels: torch.Tensor,      # (N, max_points, C)
        num_points: torch.Tensor,  # (N,)
        coors: torch.Tensor,       # (N, 3) - (batch_idx, x_idx, y_idx)
    ) -> torch.Tensor:
        """
        Args:
            voxels: 각 pillar의 포인트 features (N, max_points, C)
            num_points: 각 pillar의 실제 포인트 수 (N,)
            coors: pillar 좌표 (N, 3) - (batch_idx, x_idx, y_idx)
        
        Returns:
            pillar_features: (N, feat_channels)
        """
        # Augment point features
        features = self._augment_features(voxels, num_points, coors)
        
        # Apply PFN layers
        for pfn in self.pfn_layers:
            features = pfn(features, num_points)
        
        return features
    
    def _augment_features(
        self,
        voxels: torch.Tensor,
        num_points: torch.Tensor,
        coors: torch.Tensor,
    ) -> torch.Tensor:
        """포인트 features에 추가 정보 augment"""
        device = voxels.device
        N, max_points, C = voxels.shape
        
        # 1. Pillar 내 평균 좌표 계산
        # num_points로 나누어 평균 계산 (빈 pillar 처리)
        points_mean = voxels[:, :, :3].sum(dim=1, keepdim=True) / \
                      num_points.view(-1, 1, 1).clamp(min=1).float()
        
        # 2. xc, yc, zc: pillar 내 평균까지의 offset
        f_cluster = voxels[:, :, :3] - points_mean
        
        # 3. Pillar 중심 좌표 계산 (실제 월드 좌표)
        pillar_x = coors[:, 1].float() * self.voxel_size[0] + \
                   self.voxel_size[0] / 2 + self.point_cloud_range[0]
        pillar_y = coors[:, 2].float() * self.voxel_size[1] + \
                   self.voxel_size[1] / 2 + self.point_cloud_range[1]
        
        # 4. xp, yp: pillar 중심까지의 offset
        f_center = torch.zeros(N, max_points, 2, device=device, dtype=voxels.dtype)
        f_center[:, :, 0] = voxels[:, :, 0] - pillar_x.unsqueeze(1)
        f_center[:, :, 1] = voxels[:, :, 1] - pillar_y.unsqueeze(1)
        
        # Concatenate all features
        features = torch.cat([voxels, f_cluster, f_center], dim=-1)
        
        # 5. Optional: distance to origin
        if self.with_distance:
            distance = torch.norm(voxels[:, :, :3], p=2, dim=-1, keepdim=True)
            features = torch.cat([features, distance], dim=-1)
        
        # Apply mask for invalid points (padding을 0으로)
        point_mask = torch.arange(max_points, device=device).unsqueeze(0) < \
                     num_points.unsqueeze(1)
        features = features * point_mask.unsqueeze(-1).float()
        
        return features


class PFNLayer(nn.Module):
    """
    Single Pillar Feature Network Layer
    Linear → BatchNorm → ReLU → Max Pooling (if last layer)
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        last_layer: bool = False,
    ):
        super().__init__()
        self.last_layer = last_layer
        
        self.linear = nn.Linear(in_channels, out_channels, bias=False)
        self.bn = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
        
    def forward(
        self,
        features: torch.Tensor,    # (N, max_points, C)
        num_points: torch.Tensor,  # (N,)
    ) -> torch.Tensor:
        N, P, C = features.shape
        
        # Linear transformation
        x = self.linear(features)  # (N, P, out_channels)
        
        # BatchNorm (reshape for BN1d)
        x = x.permute(0, 2, 1).contiguous()  # (N, out_channels, P)
        x = self.bn(x)
        x = F.relu(x)
        x = x.permute(0, 2, 1).contiguous()  # (N, P, out_channels)
        
        if self.last_layer:
            # Max pooling over points
            # Mask out padding before max
            max_points = features.shape[1]
            device = features.device
            point_mask = torch.arange(max_points, device=device).unsqueeze(0) < \
                         num_points.unsqueeze(1)
            
            # Set padded positions to very negative for max pooling
            x = x.masked_fill(~point_mask.unsqueeze(-1), float('-inf'))
            x = x.max(dim=1)[0]  # (N, out_channels)
            
            # Handle empty pillars
            x = torch.where(
                num_points.unsqueeze(-1) > 0,
                x,
                torch.zeros_like(x)
            )
        
        return x


class PointPillarsScatter(nn.Module):
    """
    Pillar features를 BEV pseudo-image로 scatter
    
    효율적인 index_put_ 사용
    """
    def __init__(
        self,
        in_channels: int = 64,
        output_shape: Tuple[int, int] = (400, 400),  # (H, W)
    ):
        super().__init__()
        self.in_channels = in_channels
        self.output_shape = output_shape
        
    def forward(
        self,
        pillar_features: torch.Tensor,  # (N, C)
        coors: torch.Tensor,            # (N, 3) - (batch_idx, x_idx, y_idx)
        batch_size: int,
    ) -> torch.Tensor:
        """
        Args:
            pillar_features: (N, C) pillar features
            coors: (N, 3) pillar 좌표 (batch_idx, x_idx, y_idx)
            batch_size: 배치 크기
            
        Returns:
            bev_features: (B, C, H, W) BEV pseudo-image
        """
        device = pillar_features.device
        dtype = pillar_features.dtype
        H, W = self.output_shape
        C = self.in_channels
        
        # BEV canvas 초기화
        bev_features = torch.zeros(
            batch_size, C, H, W,
            dtype=dtype, device=device
        )
        
        # Coordinate 추출
        batch_idx = coors[:, 0].long()
        x_idx = coors[:, 1].long()
        y_idx = coors[:, 2].long()
        
        # 범위 체크 및 필터링
        valid_mask = (
            (x_idx >= 0) & (x_idx < H) &
            (y_idx >= 0) & (y_idx < W) &
            (batch_idx >= 0) & (batch_idx < batch_size)
        )
        
        if valid_mask.sum() == 0:
            return bev_features
        
        batch_idx = batch_idx[valid_mask]
        x_idx = x_idx[valid_mask]
        y_idx = y_idx[valid_mask]
        features = pillar_features[valid_mask]  # (M, C)
        
        # Efficient scatter using advanced indexing
        # bev_features[batch_idx, :, x_idx, y_idx] = features
        bev_features[batch_idx, :, x_idx, y_idx] = features
        
        return bev_features


class LiDARBackbone(nn.Module):
    """
    BEV Pseudo-image를 처리하는 2D CNN Backbone
    
    구조:
    - Multi-scale feature extraction (3 blocks)
    - Feature Pyramid Network (FPN) style upsampling
    - Final feature fusion
    """
    def __init__(
        self,
        in_channels: int = 64,
        out_channels: int = 256,
        layer_nums: List[int] = [3, 5, 5],
        layer_strides: List[int] = [2, 2, 2],
        num_filters: List[int] = [64, 128, 256],
        upsample_strides: List[int] = [1, 2, 4],
        num_upsample_filters: List[int] = [128, 128, 128],
    ):
        super().__init__()
        
        assert len(layer_nums) == len(layer_strides) == len(num_filters)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Downsampling blocks
        in_filters = [in_channels] + list(num_filters[:-1])
        self.blocks = nn.ModuleList()
        
        for i, (in_f, out_f, num_layers, stride) in enumerate(
            zip(in_filters, num_filters, layer_nums, layer_strides)
        ):
            block = self._make_block(in_f, out_f, num_layers, stride)
            self.blocks.append(block)
        
        # Upsampling blocks (FPN style)
        self.deblocks = nn.ModuleList()
        for in_f, out_f, stride in zip(num_filters, num_upsample_filters, upsample_strides):
            if stride > 1:
                deblock = nn.Sequential(
                    nn.ConvTranspose2d(in_f, out_f, stride, stride=stride, bias=False),
                    nn.BatchNorm2d(out_f, eps=1e-3, momentum=0.01),
                    nn.ReLU(inplace=True),
                )
            else:
                deblock = nn.Sequential(
                    nn.Conv2d(in_f, out_f, 3, padding=1, bias=False),
                    nn.BatchNorm2d(out_f, eps=1e-3, momentum=0.01),
                    nn.ReLU(inplace=True),
                )
            self.deblocks.append(deblock)
        
        # Final conv to output channels
        total_upsample_channels = sum(num_upsample_filters)
        self.final_conv = nn.Sequential(
            nn.Conv2d(total_upsample_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.01),
            nn.ReLU(inplace=True),
        )
        
    def _make_block(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int,
        stride: int,
    ) -> nn.Sequential:
        """Residual-like block"""
        layers = [
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.01),
            nn.ReLU(inplace=True),
        ]
        
        for _ in range(num_layers - 1):
            layers.extend([
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.01),
                nn.ReLU(inplace=True),
            ])
        
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) BEV pseudo-image
            
        Returns:
            out: (B, out_channels, H', W') BEV features
        """
        # Multi-scale feature extraction
        multi_scale_features = []
        
        for i, block in enumerate(self.blocks):
            x = block(x)
            multi_scale_features.append(x)
        
        # Upsample and concatenate
        ups = []
        for i, (feat, deblock) in enumerate(zip(multi_scale_features, self.deblocks)):
            up = deblock(feat)
            ups.append(up)
        
        # Align spatial sizes and concatenate
        target_size = ups[0].shape[2:]
        aligned_ups = []
        for up in ups:
            if up.shape[2:] != target_size:
                up = F.interpolate(up, size=target_size, mode='bilinear', align_corners=True)
            aligned_ups.append(up)
        
        out = torch.cat(aligned_ups, dim=1)
        out = self.final_conv(out)
        
        return out


class DynamicVoxelization(nn.Module):
    """
    Dynamic Voxelization
    
    포인트를 pillar에 할당하여 voxel 생성
    """
    def __init__(
        self,
        voxel_size: Tuple[float, float, float],
        point_cloud_range: List[float],
        max_points_per_voxel: int = 32,
        max_num_voxels: int = 40000,
    ):
        super().__init__()
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.max_points_per_voxel = max_points_per_voxel
        self.max_num_voxels = max_num_voxels
        
        # Grid size 계산
        self.grid_size = (
            int((point_cloud_range[3] - point_cloud_range[0]) / voxel_size[0]),
            int((point_cloud_range[4] - point_cloud_range[1]) / voxel_size[1]),
        )
        
    def forward(
        self,
        points: torch.Tensor,  # (N, 5) - (batch_idx, x, y, z, intensity)
        batch_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            voxels: (M, max_points, 4) voxel features
            num_points: (M,) number of points per voxel
            coors: (M, 3) voxel coordinates (batch_idx, x_idx, y_idx)
        """
        device = points.device
        dtype = points.dtype
        
        # Filter points in range
        x_min, y_min, z_min = self.point_cloud_range[:3]
        x_max, y_max, z_max = self.point_cloud_range[3:]
        
        in_range_mask = (
            (points[:, 1] >= x_min) & (points[:, 1] < x_max) &
            (points[:, 2] >= y_min) & (points[:, 2] < y_max) &
            (points[:, 3] >= z_min) & (points[:, 3] < z_max)
        )
        points = points[in_range_mask]
        
        if len(points) == 0:
            return (
                torch.zeros(1, self.max_points_per_voxel, 4, device=device, dtype=dtype),
                torch.zeros(1, device=device, dtype=torch.long),
                torch.zeros(1, 3, device=device, dtype=torch.long),
            )
        
        # Compute voxel indices
        batch_idx = points[:, 0].long()
        x_idx = ((points[:, 1] - x_min) / self.voxel_size[0]).long()
        y_idx = ((points[:, 2] - y_min) / self.voxel_size[1]).long()
        
        # Clamp to valid range
        x_idx = x_idx.clamp(0, self.grid_size[0] - 1)
        y_idx = y_idx.clamp(0, self.grid_size[1] - 1)
        
        # Create voxel coordinates
        voxel_coors = torch.stack([batch_idx, x_idx, y_idx], dim=1)
        
        # Find unique voxels
        unique_coors, inverse_indices = torch.unique(
            voxel_coors, dim=0, return_inverse=True
        )
        num_voxels = len(unique_coors)
        
        # Limit number of voxels
        if num_voxels > self.max_num_voxels:
            # Sample voxels
            perm = torch.randperm(num_voxels, device=device)[:self.max_num_voxels]
            unique_coors = unique_coors[perm]
            
            # Update inverse indices
            valid_voxel_mask = torch.zeros(num_voxels, dtype=torch.bool, device=device)
            valid_voxel_mask[perm] = True
            valid_point_mask = valid_voxel_mask[inverse_indices]
            
            points = points[valid_point_mask]
            inverse_indices = inverse_indices[valid_point_mask]
            
            # Remap indices
            new_indices = torch.zeros(num_voxels, dtype=torch.long, device=device)
            new_indices[perm] = torch.arange(len(perm), device=device)
            inverse_indices = new_indices[inverse_indices]
            
            num_voxels = len(perm)
        
        # Initialize voxel tensors
        voxels = torch.zeros(
            num_voxels, self.max_points_per_voxel, 4,
            dtype=dtype, device=device
        )
        num_points_per_voxel = torch.zeros(num_voxels, dtype=torch.long, device=device)
        
        # Assign points to voxels
        point_features = points[:, 1:5]  # x, y, z, intensity
        
        # Count points per voxel
        ones = torch.ones(len(inverse_indices), dtype=torch.long, device=device)
        num_points_per_voxel.scatter_add_(0, inverse_indices, ones)
        num_points_per_voxel = num_points_per_voxel.clamp(max=self.max_points_per_voxel)
        
        # Assign points (need loop for variable-length assignment)
        point_counts = torch.zeros(num_voxels, dtype=torch.long, device=device)
        
        for i in range(len(points)):
            voxel_idx = inverse_indices[i].item()
            point_idx = point_counts[voxel_idx].item()
            
            if point_idx < self.max_points_per_voxel:
                voxels[voxel_idx, point_idx] = point_features[i]
                point_counts[voxel_idx] += 1
        
        return voxels, num_points_per_voxel, unique_coors


class LiDARBEVEncoder(nn.Module):
    """
    완전한 LiDAR BEV Encoder
    PointPillars 기반으로 LiDAR 포인트 클라우드를 BEV Feature Map으로 변환
    
    Pipeline:
    1. Voxelization: Points → Pillars
    2. PillarFeatureNet: Pillars → Pillar Features  
    3. Scatter: Pillar Features → BEV Pseudo-image
    4. Backbone: BEV Pseudo-image → BEV Feature Map
    """
    def __init__(
        self,
        # Point cloud config
        point_cloud_range: List[float] = [-50.0, -50.0, -5.0, 50.0, 50.0, 3.0],
        voxel_size: Tuple[float, float, float] = (0.25, 0.25, 8.0),
        max_points_per_voxel: int = 32,
        max_num_voxels: int = 30000,
        # PillarFeatureNet config
        in_channels: int = 4,
        feat_channels: int = 64,
        with_distance: bool = False,
        # Backbone config
        backbone_out_channels: int = 256,
        layer_nums: List[int] = [3, 5, 5],
        layer_strides: List[int] = [2, 2, 2],
        num_filters: List[int] = [64, 128, 256],
        upsample_strides: List[int] = [1, 2, 4],
        num_upsample_filters: List[int] = [128, 128, 128],
    ):
        super().__init__()
        
        self.point_cloud_range = point_cloud_range
        self.voxel_size = voxel_size
        self.max_points_per_voxel = max_points_per_voxel
        self.max_num_voxels = max_num_voxels
        
        # BEV grid size 계산
        self.grid_size = (
            int((point_cloud_range[3] - point_cloud_range[0]) / voxel_size[0]),
            int((point_cloud_range[4] - point_cloud_range[1]) / voxel_size[1]),
        )
        
        # Voxelization
        self.voxelizer = DynamicVoxelization(
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            max_points_per_voxel=max_points_per_voxel,
            max_num_voxels=max_num_voxels,
        )
        
        # Pillar Feature Network
        self.pillar_feature_net = PillarFeatureNet(
            in_channels=in_channels,
            feat_channels=feat_channels,
            with_distance=with_distance,
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            num_filters=[feat_channels],
        )
        
        # Scatter to BEV
        self.scatter = PointPillarsScatter(
            in_channels=feat_channels,
            output_shape=self.grid_size,
        )
        
        # 2D CNN Backbone
        self.backbone = LiDARBackbone(
            in_channels=feat_channels,
            out_channels=backbone_out_channels,
            layer_nums=layer_nums,
            layer_strides=layer_strides,
            num_filters=num_filters,
            upsample_strides=upsample_strides,
            num_upsample_filters=num_upsample_filters,
        )
        
        self.out_channels = backbone_out_channels
        
    def forward(
        self,
        points: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """
        Args:
            points: (N, 5) - (batch_idx, x, y, z, intensity)
            batch_size: batch size
            
        Returns:
            bev_features: (B, C, H, W) BEV feature map
        """
        # 1. Voxelization
        voxels, num_points, coors = self.voxelizer(points, batch_size)
        
        # 2. Pillar Feature Extraction
        pillar_features = self.pillar_feature_net(voxels, num_points, coors)
        
        # 3. Scatter to BEV pseudo-image
        bev_pseudo_image = self.scatter(pillar_features, coors, batch_size)
        
        # 4. Backbone CNN
        bev_features = self.backbone(bev_pseudo_image)
        
        return bev_features
    
    def forward_with_voxels(
        self,
        voxels: torch.Tensor,
        num_points: torch.Tensor,
        coors: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """
        사전에 voxelize된 데이터로 forward
        (DataLoader에서 미리 voxelize하는 경우)
        """
        pillar_features = self.pillar_feature_net(voxels, num_points, coors)
        bev_pseudo_image = self.scatter(pillar_features, coors, batch_size)
        bev_features = self.backbone(bev_pseudo_image)
        return bev_features


class MarineLiDARBEVEncoder(LiDARBEVEncoder):
    """
    해양 부유쓰레기 탐지에 최적화된 LiDAR BEV Encoder
    
    해양 환경 특화 기능:
    - 근거리 고해상도 (부유쓰레기는 주로 가까운 거리에서 탐지)
    - 수면 반사 노이즈 필터링
    - 강도(intensity) 기반 수면/물체 구분
    """
    def __init__(
        self,
        # 해양 환경 맞춤 설정 (더 좁은 범위, 더 작은 voxel)
        point_cloud_range: List[float] = [-30.0, -30.0, -2.0, 30.0, 30.0, 3.0],
        voxel_size: Tuple[float, float, float] = (0.15, 0.15, 5.0),
        max_points_per_voxel: int = 32,
        max_num_voxels: int = 40000,
        # Feature config
        in_channels: int = 4,
        feat_channels: int = 64,
        backbone_out_channels: int = 256,
        # Marine-specific config
        filter_water_reflection: bool = True,
        water_level_threshold: float = -0.5,
        min_intensity_threshold: float = 0.0,  # 낮은 intensity 필터링
    ):
        super().__init__(
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
            max_points_per_voxel=max_points_per_voxel,
            max_num_voxels=max_num_voxels,
            in_channels=in_channels,
            feat_channels=feat_channels,
            backbone_out_channels=backbone_out_channels,
        )
        
        self.filter_water_reflection = filter_water_reflection
        self.water_level_threshold = water_level_threshold
        self.min_intensity_threshold = min_intensity_threshold
        
    def preprocess_points(self, points: torch.Tensor) -> torch.Tensor:
        """
        해양 환경 전처리
        
        1. 수면 아래 포인트 필터링 (수면 반사 노이즈)
        2. 너무 낮은 intensity 필터링 (노이즈)
        """
        if not self.filter_water_reflection:
            return points
        
        # 1. 수면 아래 필터링
        z_values = points[:, 3]  # z 좌표
        above_water = z_values > self.water_level_threshold
        
        # 2. Intensity 필터링 (optional)
        if self.min_intensity_threshold > 0:
            intensity = points[:, 4]
            valid_intensity = intensity > self.min_intensity_threshold
            valid_mask = above_water & valid_intensity
        else:
            valid_mask = above_water
        
        return points[valid_mask]
    
    def forward(
        self,
        points: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """Forward with marine preprocessing"""
        # 전처리
        points = self.preprocess_points(points)
        
        # 포인트가 없는 경우 처리
        if len(points) == 0:
            device = points.device if len(points) > 0 else 'cpu'
            return torch.zeros(
                batch_size, self.out_channels, 
                self.grid_size[0] // 4, self.grid_size[1] // 4,  # Backbone output size
                device=device
            )
        
        return super().forward(points, batch_size)


# Utility function
def create_lidar_bev_encoder(config: Dict) -> LiDARBEVEncoder:
    """Config로부터 LiDAR BEV Encoder 생성"""
    encoder_type = config.get('type', 'marine')
    
    if encoder_type == 'marine':
        return MarineLiDARBEVEncoder(
            point_cloud_range=config.get('point_cloud_range', [-30.0, -30.0, -2.0, 30.0, 30.0, 3.0]),
            voxel_size=config.get('voxel_size', (0.15, 0.15, 5.0)),
            max_points_per_voxel=config.get('max_points_per_voxel', 32),
            max_num_voxels=config.get('max_num_voxels', 40000),
            feat_channels=config.get('feat_channels', 64),
            backbone_out_channels=config.get('backbone_out_channels', 256),
            filter_water_reflection=config.get('filter_water_reflection', True),
            water_level_threshold=config.get('water_level_threshold', -0.5),
        )
    else:
        return LiDARBEVEncoder(
            point_cloud_range=config.get('point_cloud_range', [-50.0, -50.0, -5.0, 50.0, 50.0, 3.0]),
            voxel_size=config.get('voxel_size', (0.25, 0.25, 8.0)),
            max_points_per_voxel=config.get('max_points_per_voxel', 32),
            max_num_voxels=config.get('max_num_voxels', 30000),
            feat_channels=config.get('feat_channels', 64),
            backbone_out_channels=config.get('backbone_out_channels', 256),
        )