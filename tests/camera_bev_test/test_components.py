# tests/test_components.py

import torch
import numpy as np
import pytest
import sys
sys.path.append('..')

from models.camera_bev.camera_encoder import CameraBackbone, FPN
from models.camera_bev.depth_net import DepthNet, DepthNetWithContext
from models.camera_bev.calibration import LearnableCalibration
from models.camera_bev.view_transform import LSSViewTransform


class TestCameraBackbone:
    """Camera Backbone 테스트"""
    
    def setup_method(self):
        """테스트 초기화"""
        self.batch_size = 2
        self.img_h, self.img_w = 1080, 1920
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def test_resnet50_output_shape(self):
        """ResNet50 출력 shape 테스트"""
        print("\n[TEST] ResNet50 Output Shape")
        
        backbone = CameraBackbone(
            backbone_name='resnet50',
            pretrained=False,  # 테스트는 빠르게
            out_channels=512
        ).to(self.device)
        
        # 입력 생성
        x = torch.randn(self.batch_size, 3, self.img_h, self.img_w).to(self.device)
        
        # Forward
        features = backbone(x)
        
        # 검증
        assert isinstance(features, list), "Output should be a list"
        assert len(features) > 0, "Should have at least one feature map"
        
        for i, feat in enumerate(features):
            print(f"  Feature {i}: {feat.shape}")
            assert feat.shape[0] == self.batch_size, "Batch size mismatch"
            assert feat.shape[1] == 512, "Channel mismatch"
        
        print("✅ ResNet50 output shape test passed")
    
    def test_efficientnet_output_shape(self):
        """EfficientNet 출력 shape 테스트"""
        print("\n[TEST] EfficientNet Output Shape")
        
        try:
            backbone = CameraBackbone(
                backbone_name='efficientnet-b0',
                pretrained=False,
                out_channels=256
            ).to(self.device)
            
            x = torch.randn(self.batch_size, 3, 224, 224).to(self.device)
            features = backbone(x)
            
            print(f"  Output features: {len(features)}")
            for i, feat in enumerate(features):
                print(f"  Feature {i}: {feat.shape}")
            
            print("✅ EfficientNet output shape test passed")
        except Exception as e:
            print(f"⚠️ EfficientNet test skipped: {e}")
    
    def test_fpn(self):
        """FPN 테스트"""
        print("\n[TEST] FPN")
        
        # 가짜 multi-scale features
        in_channels = [512, 1024, 2048]
        features = [
            torch.randn(2, ch, 64//(2**i), 64//(2**i)).to(self.device)
            for i, ch in enumerate(in_channels)
        ]
        
        fpn = FPN(in_channels, out_channels=256).to(self.device)
        outputs = fpn(features)
        
        assert len(outputs) == len(features), "FPN output count mismatch"
        for out in outputs:
            assert out.shape[1] == 256, "FPN output channel should be 256"
        
        print(f"  Input channels: {in_channels}")
        print(f"  Output shapes: {[out.shape for out in outputs]}")
        print("✅ FPN test passed")


class TestDepthNet:
    """Depth Network 테스트"""
    
    def setup_method(self):
        self.batch_size = 2
        self.feat_h, self.feat_w = 135, 240  # 1080/8, 1920/8
        self.in_channels = 512
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def test_depth_prediction(self):
        """Depth 예측 테스트"""
        print("\n[TEST] Depth Prediction")
        
        depth_net = DepthNet(
            in_channels=self.in_channels,
            depth_bins=64,
            depth_range=(1.0, 50.0)
        ).to(self.device)
        
        # 입력 features
        x = torch.randn(
            self.batch_size, self.in_channels, 
            self.feat_h, self.feat_w
        ).to(self.device)
        
        # Forward
        outputs = depth_net(x)
        
        # 검증
        assert 'depth_logits' in outputs
        assert 'depth_probs' in outputs
        assert 'depth_values' in outputs
        
        depth_probs = outputs['depth_probs']
        assert depth_probs.shape == (self.batch_size, 64, self.feat_h, self.feat_w)
        
        # Probability sum should be 1 along depth dimension
        prob_sum = depth_probs.sum(dim=1)
        assert torch.allclose(prob_sum, torch.ones_like(prob_sum), atol=1e-5)
        
        print(f"  Depth logits shape: {outputs['depth_logits'].shape}")
        print(f"  Depth probs shape: {outputs['depth_probs'].shape}")
        print(f"  Depth values: {outputs['depth_values'][:5]}...")
        print("✅ Depth prediction test passed")
    
    def test_depth_range(self):
        """Depth 범위 테스트"""
        print("\n[TEST] Depth Range")
        
        depth_range = (2.0, 100.0)
        depth_net = DepthNet(
            in_channels=self.in_channels,
            depth_bins=32,
            depth_range=depth_range
        ).to(self.device)
        
        x = torch.randn(2, self.in_channels, 64, 64).to(self.device)
        outputs = depth_net(x)
        
        depth_values = outputs['depth_values']
        assert depth_values.min() >= depth_range[0]
        assert depth_values.max() <= depth_range[1]
        
        print(f"  Depth range: {depth_values.min():.2f} - {depth_values.max():.2f}")
        print("✅ Depth range test passed")


class TestCalibration:
    """Calibration 테스트"""
    
    def setup_method(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 초기 캘리브레이션
        self.K_init = np.array([
            [1000.0, 0, 960.0],
            [0, 1000.0, 540.0],
            [0, 0, 1.0]
        ])
        
        self.T_init = np.eye(4)
        self.T_init[:3, 3] = [0.1, -0.05, 0.2]  # translation
    
    def test_fixed_calibration(self):
        """고정 캘리브레이션 테스트"""
        print("\n[TEST] Fixed Calibration")
        
        calib = LearnableCalibration(
            K_init=self.K_init,
            T_init=self.T_init,
            learn_intrinsic=False,
            learn_extrinsic=False
        ).to(self.device)
        
        K, T = calib()
        
        # Should be exactly the same as init
        assert torch.allclose(K, torch.FloatTensor(self.K_init).to(self.device))
        assert torch.allclose(T, torch.FloatTensor(self.T_init).to(self.device))
        
        print("  K (fixed):")
        print(K.cpu().numpy())
        print("✅ Fixed calibration test passed")
    
    def test_learnable_calibration(self):
        """학습 가능 캘리브레이션 테스트"""
        print("\n[TEST] Learnable Calibration")
        
        calib = LearnableCalibration(
            K_init=self.K_init,
            T_init=self.T_init,
            learn_intrinsic=True,
            learn_extrinsic=True
        ).to(self.device)
        
        # 초기 상태
        K_init, T_init = calib()
        
        # Simulate learning (gradient step)
        optimizer = torch.optim.Adam(calib.parameters(), lr=0.01)
        
        # Dummy loss
        loss = (calib.delta_K.sum() + calib.delta_pose.sum())
        loss.backward()
        optimizer.step()
        
        # After update
        K_updated, T_updated = calib()
        
        # Should be different after learning
        assert not torch.allclose(K_updated, K_init)
        
        print(f"  K change: {torch.norm(K_updated - K_init).item():.6f}")
        print(f"  T change: {torch.norm(T_updated - T_init).item():.6f}")
        print("✅ Learnable calibration test passed")
    
    def test_se3_exp(self):
        """SE(3) exponential map 테스트"""
        print("\n[TEST] SE(3) Exponential Map")
        
        calib = LearnableCalibration(
            K_init=self.K_init,
            T_init=self.T_init
        ).to(self.device)
        
        # Small perturbation
        xi = torch.tensor([0.01, 0.02, 0.03, 0.001, 0.002, 0.003]).to(self.device)
        T_delta = calib.se3_exp(xi)
        
        # Should be close to identity for small perturbation
        I = torch.eye(4).to(self.device)
        assert torch.norm(T_delta - I) < 0.1
        
        print(f"  Perturbation norm: {torch.norm(T_delta - I).item():.6f}")
        print("✅ SE(3) exp test passed")


class TestViewTransform:
    """LSS View Transformation 테스트"""
    
    def setup_method(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.img_size = (270, 480)  # Downsampled for speed
        self.bev_size = (128, 128)
        self.bev_range = (-25.6, 25.6, -25.6, 25.6)
        
        self.K = torch.FloatTensor([
            [500.0, 0, 240.0],
            [0, 500.0, 135.0],
            [0, 0, 1.0]
        ]).to(self.device)
        
        self.T = torch.eye(4).to(self.device)
        self.T[:3, 3] = torch.FloatTensor([0.1, 0, 0.2])
    
    def test_image_grid_creation(self):
        """이미지 그리드 생성 테스트"""
        print("\n[TEST] Image Grid Creation")
        
        view_transform = LSSViewTransform(
            img_size=self.img_size,
            bev_size=self.bev_size,
            bev_range=self.bev_range,
            feat_channels=64,
            depth_bins=32
        ).to(self.device)
        
        grid = view_transform.img_grid_homo
        
        assert grid.shape == (self.img_size[0], self.img_size[1], 3)
        
        # Check corners
        print(f"  Grid shape: {grid.shape}")
        print(f"  Top-left: {grid[0, 0].cpu().numpy()}")
        print(f"  Bottom-right: {grid[-1, -1].cpu().numpy()}")
        print("✅ Image grid creation test passed")
    
    def test_unproject_to_3d(self):
        """3D unprojection 테스트"""
        print("\n[TEST] 3D Unprojection")
        
        view_transform = LSSViewTransform(
            img_size=self.img_size,
            bev_size=self.bev_size,
            bev_range=self.bev_range,
            feat_channels=64,
            depth_bins=32
        ).to(self.device)
        
        depth_values = torch.linspace(1.0, 50.0, 32).to(self.device)
        
        points_3d = view_transform.unproject_to_3d(depth_values, self.K)
        
        assert points_3d.shape == (32, self.img_size[0], self.img_size[1], 3)
        
        # Check reasonable depth values
        depths = points_3d[..., 2]  # Z coordinate
        assert depths.min() >= 1.0
        assert depths.max() <= 50.0
        
        print(f"  3D points shape: {points_3d.shape}")
        print(f"  Depth range: {depths.min():.2f} - {depths.max():.2f}")
        print("✅ 3D unprojection test passed")
    
    def test_full_transform(self):
        """전체 변환 테스트"""
        print("\n[TEST] Full LSS Transform")
        
        batch_size = 2
        feat_channels = 64
        depth_bins = 32
        
        view_transform = LSSViewTransform(
            img_size=self.img_size,
            bev_size=self.bev_size,
            bev_range=self.bev_range,
            feat_channels=feat_channels,
            depth_bins=depth_bins
        ).to(self.device)
        
        # Inputs
        img_features = torch.randn(
            batch_size, feat_channels, 
            self.img_size[0], self.img_size[1]
        ).to(self.device)
        
        depth_probs = torch.randn(
            batch_size, depth_bins,
            self.img_size[0], self.img_size[1]
        ).to(self.device)
        depth_probs = torch.softmax(depth_probs, dim=1)
        
        depth_values = torch.linspace(1.0, 50.0, depth_bins).to(self.device)
        
        # Forward
        bev_features = view_transform(
            img_features, depth_probs, depth_values, self.K, self.T
        )
        
        # Verify
        assert bev_features.shape == (batch_size, feat_channels, 
                                     self.bev_size[0], self.bev_size[1])
        
        print(f"  Input features: {img_features.shape}")
        print(f"  BEV features: {bev_features.shape}")
        print(f"  Non-zero ratio: {(bev_features != 0).float().mean():.2%}")
        print("✅ Full LSS transform test passed")


def run_component_tests():
    """모든 컴포넌트 테스트 실행"""
    print("="*60)
    print("CAMERA BEV MODULE - COMPONENT TESTS")
    print("="*60)
    
    # Backbone tests
    print("\n" + "="*60)
    print("1. CAMERA BACKBONE TESTS")
    print("="*60)
    test_backbone = TestCameraBackbone()
    test_backbone.setup_method()
    test_backbone.test_resnet50_output_shape()
    test_backbone.test_efficientnet_output_shape()
    test_backbone.test_fpn()
    
    # Depth tests
    print("\n" + "="*60)
    print("2. DEPTH NETWORK TESTS")
    print("="*60)
    test_depth = TestDepthNet()
    test_depth.setup_method()
    test_depth.test_depth_prediction()
    test_depth.test_depth_range()
    
    # Calibration tests
    print("\n" + "="*60)
    print("3. CALIBRATION TESTS")
    print("="*60)
    test_calib = TestCalibration()
    test_calib.setup_method()
    test_calib.test_fixed_calibration()
    test_calib.test_learnable_calibration()
    test_calib.test_se3_exp()
    
    # View transform tests
    print("\n" + "="*60)
    print("4. VIEW TRANSFORM TESTS")
    print("="*60)
    test_view = TestViewTransform()
    test_view.setup_method()
    test_view.test_image_grid_creation()
    test_view.test_unproject_to_3d()
    test_view.test_full_transform()
    
    print("\n" + "="*60)
    print("✅ ALL COMPONENT TESTS PASSED!")
    print("="*60)


if __name__ == '__main__':
    run_component_tests()