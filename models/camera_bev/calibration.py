# models/camera_bev/calibration.py

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple


class LearnableCalibration(nn.Module):
    """
    학습 가능한 캘리브레이션 파라미터
    
    사전 캘리브레이션을 초기값으로 사용하고
    학습 중 bounded residual을 통해 미세 조정
    
    Intrinsic (K):
        K_refined = K_base * (I + bounded_delta)
    
    Extrinsic (T):
        T_refined = T_base ⊕ SE(3)_exp(bounded_delta)
        SE(3) manifold을 유지
    """
    
    def __init__(
        self,
        K_init: np.ndarray,  # [3, 3] intrinsic matrix
        T_init: np.ndarray,  # [4, 4] extrinsic matrix
        learn_intrinsic: bool = True,
        learn_extrinsic: bool = True,
        max_delta_focal: float = 0.05,  # 5% 변화 허용
        max_delta_pose: float = 0.05    # ±0.05m translation, ±0.05rad rotation
    ):
        super().__init__()
        
        self.learn_intrinsic = learn_intrinsic
        self.learn_extrinsic = learn_extrinsic
        self.max_delta_focal = max_delta_focal
        self.max_delta_pose = max_delta_pose
        
        # Base calibration (fixed, non-trainable)
        self.register_buffer('K_base', torch.from_numpy(K_init).float())
        self.register_buffer('T_base', torch.from_numpy(T_init).float())
        
        # Learnable residuals
        if learn_intrinsic:
            # Delta for [fx, fy, cx, cy]
            self.delta_K = nn.Parameter(torch.zeros(4))
        else:
            self.register_buffer('delta_K', torch.zeros(4))
        
        if learn_extrinsic:
            # Delta in se(3): [tx, ty, tz, rx, ry, rz]
            self.delta_pose = nn.Parameter(torch.zeros(6))
        else:
            self.register_buffer('delta_pose', torch.zeros(6))
    
    def get_intrinsic(self) -> torch.Tensor:
        """
        Get refined intrinsic matrix
        
        Returns:
            K: [3, 3] refined intrinsic matrix
        """
        K = self.K_base.clone()
        
        if self.learn_intrinsic:
            # Apply bounded residuals via tanh
            delta = torch.tanh(self.delta_K) * self.max_delta_focal
            
            # Multiplicative update for focal lengths
            K[0, 0] = self.K_base[0, 0] * (1 + delta[0])  # fx
            K[1, 1] = self.K_base[1, 1] * (1 + delta[1])  # fy
            
            # Additive update for principal point (relative to image size)
            # Assume image center is approximately at (cx, cy)
            img_w = self.K_base[0, 2] * 2
            img_h = self.K_base[1, 2] * 2
            K[0, 2] = self.K_base[0, 2] + delta[2] * img_w * self.max_delta_focal
            K[1, 2] = self.K_base[1, 2] + delta[3] * img_h * self.max_delta_focal
        
        return K
    
    def get_extrinsic(self) -> torch.Tensor:
        """
        Get refined extrinsic matrix
        
        Returns:
            T: [4, 4] refined extrinsic matrix (SE(3))
        """
        T = self.T_base.clone()
        
        if self.learn_extrinsic:
            # Bounded delta via tanh
            delta = torch.tanh(self.delta_pose) * self.max_delta_pose
            
            # SE(3) exponential map
            T_delta = self.se3_exp(delta)
            
            # Compose transformations (right multiplication)
            T = T @ T_delta
        
        return T
    
    def se3_exp(self, xi: torch.Tensor) -> torch.Tensor:
        """
        SE(3) exponential map (numerically stable version)
        
        Args:
            xi: [6] Lie algebra element [tx, ty, tz, rx, ry, rz]
        
        Returns:
            T: [4, 4] SE(3) matrix
        """
        # Translation part
        t = xi[:3]
        
        # Rotation part (so(3))
        omega = xi[3:]
        theta = torch.norm(omega)
        
        # Numerical stability: avoid division by zero
        eps = 1e-8
        
        # Compute coefficients (with Taylor series for small angles)
        if theta > 1e-6:
            # Normal case: use Rodrigues' formula
            a = torch.sin(theta) / theta
            b = (1 - torch.cos(theta)) / (theta ** 2)
        else:
            # Small angle: use Taylor expansion
            theta_sq = theta ** 2
            a = 1.0 - theta_sq / 6.0
            b = 0.5 - theta_sq / 24.0
        
        # Skew-symmetric matrix
        K = self.skew_symmetric(omega)
        
        # Rotation matrix: R = I + a*K + b*K^2
        R = (torch.eye(3, device=xi.device, dtype=xi.dtype) + 
             a * K + 
             b * (K @ K))
        
        # Combine to SE(3)
        T = torch.eye(4, device=xi.device, dtype=xi.dtype)
        T[:3, :3] = R
        T[:3, 3] = t
        
        return T
    
    @staticmethod
    def skew_symmetric(v: torch.Tensor) -> torch.Tensor:
        """
        Skew-symmetric matrix from 3D vector
        
        Args:
            v: [3] vector [v1, v2, v3]
        
        Returns:
            K: [3, 3] skew-symmetric matrix
                [[  0, -v3,  v2],
                 [ v3,   0, -v1],
                 [-v2,  v1,   0]]
        """
        K = torch.zeros(3, 3, device=v.device, dtype=v.dtype)
        K[0, 1] = -v[2]
        K[0, 2] = v[1]
        K[1, 0] = v[2]
        K[1, 2] = -v[0]
        K[2, 0] = -v[1]
        K[2, 1] = v[0]
        return K
    
    def forward(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get both refined intrinsic and extrinsic
        
        Returns:
            K: [3, 3] intrinsic matrix
            T: [4, 4] extrinsic matrix
        """
        return self.get_intrinsic(), self.get_extrinsic()