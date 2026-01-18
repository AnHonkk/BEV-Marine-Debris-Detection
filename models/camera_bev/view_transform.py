# models/camera_bev/view_transform.py

import torch
import torch.nn as nn
from typing import Tuple


class LSSViewTransform(nn.Module):
    """
    Lift-Splat-Shoot Transformation (Optimized)
    
    2D image features → 3D → BEV feature map
    
    Process:
        1. Lift: 2D pixels → 3D rays (camera frame)
        2. Transform: Camera → LiDAR frame
        3. Shoot: 3D points → BEV grid (probabilistic scatter)
    """
    
    def __init__(
        self,
        img_size: Tuple[int, int],  # (H, W)
        bev_size: Tuple[int, int],  # (H_bev, W_bev)
        bev_range: Tuple[float, float, float, float],  # (x_min, x_max, y_min, y_max)
        feat_channels: int = 256,
        depth_bins: int = 64
    ):
        super().__init__()
        
        self.img_h, self.img_w = img_size
        self.bev_h, self.bev_w = bev_size
        self.x_min, self.x_max, self.y_min, self.y_max = bev_range
        
        self.bev_res_x = (self.x_max - self.x_min) / self.bev_w
        self.bev_res_y = (self.y_max - self.y_min) / self.bev_h
        
        self.feat_channels = feat_channels
        self.depth_bins = depth_bins
        
        # Create image grid
        self._create_image_grid()
    
    def _create_image_grid(self):
        """Create image pixel grid in homogeneous coordinates"""
        xs = torch.linspace(0, self.img_w - 1, self.img_w)
        ys = torch.linspace(0, self.img_h - 1, self.img_h)
        
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
        img_grid = torch.stack([grid_x, grid_y], dim=-1)
        
        # Homogeneous coordinates [H, W, 3]
        ones = torch.ones_like(grid_x).unsqueeze(-1)
        img_grid_homo = torch.cat([img_grid, ones], dim=-1)
        
        self.register_buffer('img_grid_homo', img_grid_homo)
    
    def unproject_to_3d(
        self,
        depth_values: torch.Tensor,  # [D]
        K: torch.Tensor  # [3, 3]
    ) -> torch.Tensor:
        """
        Unproject 2D pixels to 3D camera frame
        
        Returns:
            points_3d: [D, H, W, 3]
        """
        # Inverse intrinsic
        K_inv = torch.inverse(K)
        
        # Rays in camera coordinates [H, W, 3]
        rays = (K_inv @ self.img_grid_homo.view(-1, 3).T).T
        rays = rays.view(self.img_h, self.img_w, 3)
        
        # Scale by depth [D, H, W, 3]
        points_3d = depth_values.view(-1, 1, 1, 1) * rays.unsqueeze(0)
        
        return points_3d
    
    def transform_to_lidar(
        self,
        points_cam: torch.Tensor,  # [D, H, W, 3]
        T: torch.Tensor  # [4, 4]
    ) -> torch.Tensor:
        """
        Transform from camera to LiDAR frame
        
        Returns:
            points_lidar: [D, H, W, 3]
        """
        D, H, W, _ = points_cam.shape
        
        # Flatten
        points_flat = points_cam.reshape(-1, 3)
        
        # Apply transformation
        R = T[:3, :3]
        t = T[:3, 3]
        points_lidar_flat = (R @ points_flat.T).T + t
        
        # Reshape
        points_lidar = points_lidar_flat.reshape(D, H, W, 3)
        
        return points_lidar
    
    def project_to_bev(
        self,
        points_3d: torch.Tensor,   # [D, H, W, 3]
        features: torch.Tensor,    # [C, H, W]
        depth_probs: torch.Tensor  # [D, H, W]
    ) -> torch.Tensor:
        """
        Project 3D points to BEV grid (vectorized)
        
        Returns:
            bev_features: [C, H_bev, W_bev]
        """
        D, H, W, _ = points_3d.shape
        C = features.shape[0]
        
        # Extract coordinates
        x = points_3d[..., 0]  # [D, H, W]
        y = points_3d[..., 1]
        z = points_3d[..., 2]
        
        # BEV grid indices
        bev_x = ((x - self.x_min) / self.bev_res_x).long()
        bev_y = ((y - self.y_min) / self.bev_res_y).long()
        
        # Valid mask
        valid = (
            (bev_x >= 0) & (bev_x < self.bev_w) &
            (bev_y >= 0) & (bev_y < self.bev_h) &
            (z > 0)
        )
        
        # Weighted features [C, D, H, W]
        weighted_feat = features.unsqueeze(1) * depth_probs.unsqueeze(0)
        
        # Flatten
        bev_x = bev_x.flatten()
        bev_y = bev_y.flatten()
        valid = valid.flatten()
        weighted_feat = weighted_feat.permute(1, 2, 3, 0).reshape(-1, C)  # [D*H*W, C]
        
        # Filter valid
        bev_x = bev_x[valid]
        bev_y = bev_y[valid]
        weighted_feat = weighted_feat[valid]
        
        # Linear indices
        indices = bev_y * self.bev_w + bev_x
        
        # Scatter (vectorized!)
        bev_flat = torch.zeros(
            self.bev_h * self.bev_w, C,
            device=features.device,
            dtype=features.dtype
        )
        weighted_feat = weighted_feat.to(bev_flat.dtype)
        bev_flat.scatter_add_(0, indices.unsqueeze(1).expand(-1, C), weighted_feat)
        
        # Reshape [H*W, C] → [C, H, W]
        bev_features = bev_flat.T.view(C, self.bev_h, self.bev_w)
        
        return bev_features
    
    def forward(
        self,
        img_features: torch.Tensor,  # [B, C, H, W]
        depth_probs: torch.Tensor,   # [B, D, H, W]
        depth_values: torch.Tensor,  # [D]
        K: torch.Tensor,             # [3, 3] or [B, 3, 3]
        T: torch.Tensor              # [4, 4] or [B, 4, 4]
    ) -> torch.Tensor:
        """
        Complete LSS transformation
        
        Returns:
            bev_features: [B, C, H_bev, W_bev]
        """
        B, C, H, W = img_features.shape
        
        bev_features_list = []
        
        for b in range(B):
            # Get batch-specific calibration
            K_b = K if K.dim() == 2 else K[b]
            T_b = T if T.dim() == 2 else T[b]
            
            # 1. Lift: 2D → 3D (camera frame)
            points_cam = self.unproject_to_3d(depth_values, K_b)
            
            # 2. Transform: Camera → LiDAR
            points_lidar = self.transform_to_lidar(points_cam, T_b)
            
            # 3. Shoot: 3D → BEV (vectorized!)
            bev_feat = self.project_to_bev(
                points_lidar,
                img_features[b],
                depth_probs[b]
            )
            
            bev_features_list.append(bev_feat)
        
        bev_features = torch.stack(bev_features_list, dim=0)
        
        return bev_features