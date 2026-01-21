"""
PointPillars 기반 LiDAR BEV Encoder (400x400 출력)
해양 부유쓰레기 탐지를 위한 LiDAR 포인트 클라우드 → BEV Feature Map 변환

핵심 구조:
1. Voxelization: 포인트 클라우드를 pillar(세로 기둥)로 그룹화
2. PillarFeatureNet: 각 pillar 내 포인트들을 PointNet으로 인코딩
3. Scatter: Pillar features를 BEV pseudo-image로 scatter
4. Backbone: 2D CNN으로 BEV feature 추출

BEV 출력 크기: 400x400
- Point cloud range: [-20m, 20m] (40m)
- Voxel size: 0.1m
- Grid size: 40m / 0.1m = 400

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
    """
    def __init__(
        self,
        in_channels: int = 4,  # x, y, z, intensity
        feat_channels: int = 64,
        with_distance: bool = False,
        voxel_size: Tuple[float, float, float] = (0.1, 0.1, 5.0),
        point_cloud_range: List[float] = [-20.0, -20.0, -2.0, 20.0, 20.0, 3.0],
        num_filters: List[int] = [64],
    ):
        super().__init__()
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.with_distance = with_distance
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        
        # Input feature dimension
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
        voxels: torch.Tensor,
        num_points: torch.Tensor,
        coors: torch.Tensor,
    ) -> torch.Tensor:
        features = self._augment_features(voxels, num_points, coors)
        
        for pfn in self.pfn_layers:
            features = pfn(features, num_points)
        
        return features
    
    def _augment_features(
        self,
        voxels: torch.Tensor,
        num_points: torch.Tensor,
        coors: torch.Tensor,
    ) -> torch.Tensor:
        device = voxels.device
        N, max_points, C = voxels.shape
        
        points_mean = voxels[:, :, :3].sum(dim=1, keepdim=True) / \
                      num_points.view(-1, 1, 1).clamp(min=1).float()
        
        f_cluster = voxels[:, :, :3] - points_mean
        
        pillar_x = coors[:, 1].float() * self.voxel_size[0] + \
                   self.voxel_size[0] / 2 + self.point_cloud_range[0]
        pillar_y = coors[:, 2].float() * self.voxel_size[1] + \
                   self.voxel_size[1] / 2 + self.point_cloud_range[1]
        
        f_center = torch.zeros(N, max_points, 2, device=device, dtype=voxels.dtype)
        f_center[:, :, 0] = voxels[:, :, 0] - pillar_x.unsqueeze(1)
        f_center[:, :, 1] = voxels[:, :, 1] - pillar_y.unsqueeze(1)
        
        features = torch.cat([voxels, f_cluster, f_center], dim=-1)
        
        if self.with_distance:
            distance = torch.norm(voxels[:, :, :3], p=2, dim=-1, keepdim=True)
            features = torch.cat([features, distance], dim=-1)
        
        point_mask = torch.arange(max_points, device=device).unsqueeze(0) < \
                     num_points.unsqueeze(1)
        features = features * point_mask.unsqueeze(-1).float()
        
        return features


class PFNLayer(nn.Module):
    """Single Pillar Feature Network Layer"""
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
        features: torch.Tensor,
        num_points: torch.Tensor,
    ) -> torch.Tensor:
        N, P, C = features.shape
        
        x = self.linear(features)
        x = x.permute(0, 2, 1).contiguous()
        x = self.bn(x)
        x = F.relu(x)
        x = x.permute(0, 2, 1).contiguous()
        
        if self.last_layer:
            max_points = features.shape[1]
            device = features.device
            point_mask = torch.arange(max_points, device=device).unsqueeze(0) < \
                         num_points.unsqueeze(1)
            
            x = x.masked_fill(~point_mask.unsqueeze(-1), float('-inf'))
            x = x.max(dim=1)[0]
            
            x = torch.where(
                num_points.unsqueeze(-1) > 0,
                x,
                torch.zeros_like(x)
            )
        
        return x


class PointPillarsScatter(nn.Module):
    """Pillar features를 BEV pseudo-image로 scatter (400x400)"""
    def __init__(
        self,
        in_channels: int = 64,
        output_shape: Tuple[int, int] = (400, 400),
    ):
        super().__init__()
        self.in_channels = in_channels
        self.output_shape = output_shape
        
    def forward(
        self,
        pillar_features: torch.Tensor,
        coors: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        device = pillar_features.device
        dtype = pillar_features.dtype
        H, W = self.output_shape
        C = self.in_channels
        
        bev_features = torch.zeros(
            batch_size, C, H, W,
            dtype=dtype, device=device
        )
        
        batch_idx = coors[:, 0].long()
        x_idx = coors[:, 1].long()
        y_idx = coors[:, 2].long()
        
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
        features = pillar_features[valid_mask]
        
        bev_features[batch_idx, :, x_idx, y_idx] = features
        
        return bev_features


class LiDARBackbone(nn.Module):
    """
    BEV Pseudo-image를 처리하는 2D CNN Backbone
    입력: (B, 64, 400, 400)
    출력: (B, 256, 400, 400)
    """
    def __init__(
        self,
        in_channels: int = 64,
        out_channels: int = 256,
        layer_nums: List[int] = [3, 5, 5],
        layer_strides: List[int] = [1, 2, 2],  # stride 조정하여 해상도 유지
        num_filters: List[int] = [64, 128, 256],
        upsample_strides: List[int] = [1, 2, 4],
        num_upsample_filters: List[int] = [128, 128, 128],
    ):
        super().__init__()
        
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
        
        # Upsampling blocks
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
        
        # Final conv
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
        
        # Align to target size (400, 400)
        target_size = (400, 400)
        aligned_ups = []
        for up in ups:
            if up.shape[2:] != target_size:
                up = F.interpolate(up, size=target_size, mode='bilinear', align_corners=True)
            aligned_ups.append(up)
        
        out = torch.cat(aligned_ups, dim=1)
        out = self.final_conv(out)
        
        return out


class DynamicVoxelization(nn.Module):
    """Dynamic Voxelization for 400x400 BEV (0.1m resolution)"""
    def __init__(
        self,
        voxel_size: Tuple[float, float, float] = (0.1, 0.1, 5.0),
        point_cloud_range: List[float] = [-20.0, -20.0, -2.0, 20.0, 20.0, 3.0],
        max_points_per_voxel: int = 32,
        max_num_voxels: int = 60000,
    ):
        super().__init__()
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.max_points_per_voxel = max_points_per_voxel
        self.max_num_voxels = max_num_voxels
        
        # Grid size: 60m / 0.15m = 400
        self.grid_size = (
            int((point_cloud_range[3] - point_cloud_range[0]) / voxel_size[0]),
            int((point_cloud_range[4] - point_cloud_range[1]) / voxel_size[1]),
        )
        
        print(f"[DynamicVoxelization] Grid size: {self.grid_size}")  # Should be (400, 400)
        
    def forward(
        self,
        points: torch.Tensor,
        batch_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = points.device
        dtype = points.dtype
        
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
        
        batch_idx = points[:, 0].long()
        x_idx = ((points[:, 1] - x_min) / self.voxel_size[0]).long()
        y_idx = ((points[:, 2] - y_min) / self.voxel_size[1]).long()
        
        x_idx = x_idx.clamp(0, self.grid_size[0] - 1)
        y_idx = y_idx.clamp(0, self.grid_size[1] - 1)
        
        voxel_coors = torch.stack([batch_idx, x_idx, y_idx], dim=1)
        
        unique_coors, inverse_indices = torch.unique(
            voxel_coors, dim=0, return_inverse=True
        )
        num_voxels = len(unique_coors)
        
        if num_voxels > self.max_num_voxels:
            perm = torch.randperm(num_voxels, device=device)[:self.max_num_voxels]
            unique_coors = unique_coors[perm]
            
            valid_voxel_mask = torch.zeros(num_voxels, dtype=torch.bool, device=device)
            valid_voxel_mask[perm] = True
            valid_point_mask = valid_voxel_mask[inverse_indices]
            
            points = points[valid_point_mask]
            inverse_indices = inverse_indices[valid_point_mask]
            
            new_indices = torch.zeros(num_voxels, dtype=torch.long, device=device)
            new_indices[perm] = torch.arange(len(perm), device=device)
            inverse_indices = new_indices[inverse_indices]
            
            num_voxels = len(perm)
        
        voxels = torch.zeros(
            num_voxels, self.max_points_per_voxel, 4,
            dtype=dtype, device=device
        )
        num_points_per_voxel = torch.zeros(num_voxels, dtype=torch.long, device=device)
        
        point_features = points[:, 1:5]
        
        ones = torch.ones(len(inverse_indices), dtype=torch.long, device=device)
        num_points_per_voxel.scatter_add_(0, inverse_indices, ones)
        num_points_per_voxel = num_points_per_voxel.clamp(max=self.max_points_per_voxel)
        
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
    LiDAR BEV Encoder (400x400 출력, 0.1m resolution)
    
    Pipeline:
    1. Voxelization: Points → Pillars (400x400 grid)
    2. PillarFeatureNet: Pillars → Pillar Features  
    3. Scatter: Pillar Features → BEV Pseudo-image (400x400)
    4. Backbone: BEV Pseudo-image → BEV Feature Map (400x400)
    
    Coverage: 40m x 40m (±20m)
    Resolution: 0.1m per pixel
    """
    def __init__(
        self,
        point_cloud_range: List[float] = [-20.0, -20.0, -2.0, 20.0, 20.0, 3.0],
        voxel_size: Tuple[float, float, float] = (0.1, 0.1, 5.0),
        max_points_per_voxel: int = 32,
        max_num_voxels: int = 60000,
        in_channels: int = 4,
        feat_channels: int = 64,
        with_distance: bool = False,
        backbone_out_channels: int = 256,
        layer_nums: List[int] = [3, 5, 5],
        layer_strides: List[int] = [1, 2, 2],
        num_filters: List[int] = [64, 128, 256],
        upsample_strides: List[int] = [1, 2, 4],
        num_upsample_filters: List[int] = [128, 128, 128],
    ):
        super().__init__()
        
        self.point_cloud_range = point_cloud_range
        self.voxel_size = voxel_size
        
        # BEV grid size: 400x400
        self.grid_size = (
            int((point_cloud_range[3] - point_cloud_range[0]) / voxel_size[0]),
            int((point_cloud_range[4] - point_cloud_range[1]) / voxel_size[1]),
        )
        
        print(f"[LiDARBEVEncoder] Output BEV size: {self.grid_size}")
        
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
        
        # Scatter to BEV (400x400)
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
            bev_features: (B, 256, 400, 400) BEV feature map
        """
        voxels, num_points, coors = self.voxelizer(points, batch_size)
        pillar_features = self.pillar_feature_net(voxels, num_points, coors)
        bev_pseudo_image = self.scatter(pillar_features, coors, batch_size)
        bev_features = self.backbone(bev_pseudo_image)
        
        return bev_features
    
    def forward_with_voxels(
        self,
        voxels: torch.Tensor,
        num_points: torch.Tensor,
        coors: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        pillar_features = self.pillar_feature_net(voxels, num_points, coors)
        bev_pseudo_image = self.scatter(pillar_features, coors, batch_size)
        bev_features = self.backbone(bev_pseudo_image)
        return bev_features


class MarineLiDARBEVEncoder(LiDARBEVEncoder):
    """
    해양 부유쓰레기 탐지에 최적화된 LiDAR BEV Encoder
    
    Output: 400x400 BEV feature map
    Coverage: 40m x 40m (±20m)
    Resolution: 0.1m per pixel
    """
    def __init__(
        self,
        point_cloud_range: List[float] = [-20.0, -20.0, -2.0, 20.0, 20.0, 3.0],
        voxel_size: Tuple[float, float, float] = (0.1, 0.1, 5.0),
        max_points_per_voxel: int = 32,
        max_num_voxels: int = 60000,
        in_channels: int = 4,
        feat_channels: int = 64,
        backbone_out_channels: int = 256,
        filter_water_reflection: bool = True,
        water_level_threshold: float = -0.5,
        min_intensity_threshold: float = 0.0,
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
        if not self.filter_water_reflection:
            return points
        
        z_values = points[:, 3]
        above_water = z_values > self.water_level_threshold
        
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
        """
        Returns:
            bev_features: (B, 256, 400, 400) BEV feature map
        """
        points = self.preprocess_points(points)
        
        if len(points) == 0:
            device = points.device if len(points) > 0 else 'cpu'
            return torch.zeros(
                batch_size, self.out_channels, 400, 400,
                device=device
            )
        
        return super().forward(points, batch_size)


def create_lidar_bev_encoder(config: Dict) -> LiDARBEVEncoder:
    """Config로부터 LiDAR BEV Encoder 생성 (400x400, 0.1m resolution)"""
    encoder_type = config.get('type', 'marine')
    
    if encoder_type == 'marine':
        return MarineLiDARBEVEncoder(
            point_cloud_range=config.get('point_cloud_range', [-20.0, -20.0, -2.0, 20.0, 20.0, 3.0]),
            voxel_size=config.get('voxel_size', (0.1, 0.1, 5.0)),
            max_points_per_voxel=config.get('max_points_per_voxel', 32),
            max_num_voxels=config.get('max_num_voxels', 60000),
            feat_channels=config.get('feat_channels', 64),
            backbone_out_channels=config.get('backbone_out_channels', 256),
            filter_water_reflection=config.get('filter_water_reflection', True),
            water_level_threshold=config.get('water_level_threshold', -0.5),
        )
    else:
        return LiDARBEVEncoder(
            point_cloud_range=config.get('point_cloud_range', [-20.0, -20.0, -2.0, 20.0, 20.0, 3.0]),
            voxel_size=config.get('voxel_size', (0.1, 0.1, 5.0)),
            max_points_per_voxel=config.get('max_points_per_voxel', 32),
            max_num_voxels=config.get('max_num_voxels', 60000),
            feat_channels=config.get('feat_channels', 64),
            backbone_out_channels=config.get('backbone_out_channels', 256),
        )