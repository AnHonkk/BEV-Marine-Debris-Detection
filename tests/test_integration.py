# tests/test_integration.py

import torch
import numpy as np
import time
import sys
sys.path.append('..')

from models.camera_bev.camera_bev_module import CameraBEVModule


class TestCameraBEVIntegration:
    """전체 모듈 통합 테스트"""
    
    def setup_method(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Configuration
        K_init = np.array([
            [1000.0, 0, 960.0],
            [0, 1000.0, 540.0],
            [0, 0, 1.0]
        ])
        
        T_init = np.eye(4)
        T_init[:3, 3] = [0.1, -0.05, 0.2]
        
        self.config = {
            'backbone': 'resnet50',
            'pretrained': False,  # Faster for testing
            'feat_channels': 512,
            'depth_bins': 64,
            'depth_range': (1.0, 50.0),
            'img_size': (1080, 1920),
            'bev_size': (256, 256),
            'bev_range': (-25.6, 25.6, -25.6, 25.6),
            'K_init': K_init,
            'T_init': T_init,
            'learn_intrinsic': True,
            'learn_extrinsic': True
        }
    
    def test_module_creation(self):
        """모듈 생성 테스트"""
        print("\n[TEST] Module Creation")
        
        model = CameraBEVModule(self.config).to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print("✅ Module creation test passed")
        
        return model
    
    def test_forward_pass(self):
        """Forward pass 테스트"""
        print("\n[TEST] Forward Pass")
        
        model = CameraBEVModule(self.config).to(self.device)
        model.eval()
        
        batch_size = 2
        images = torch.randn(batch_size, 3, 1080, 1920).to(self.device)
        
        with torch.no_grad():
            outputs = model(images, refine_calibration=False)
        
        # Verify outputs
        assert 'bev_features' in outputs
        assert 'depth_probs' in outputs
        assert 'K' in outputs
        assert 'T' in outputs
        
        bev_features = outputs['bev_features']
        assert bev_features.shape == (batch_size, 512, 256, 256)
        
        print(f"  Input shape: {images.shape}")
        print(f"  BEV features shape: {bev_features.shape}")
        print(f"  Depth probs shape: {outputs['depth_probs'].shape}")
        print("✅ Forward pass test passed")
        
        return outputs
    
    def test_backward_pass(self):
        """Backward pass 테스트"""
        print("\n[TEST] Backward Pass")
        
        model = CameraBEVModule(self.config).to(self.device)
        model.train()
        
        images = torch.randn(2, 3, 1080, 1920).to(self.device)
        
        # Forward
        outputs = model(images, refine_calibration=True)
        
        # Dummy loss
        loss = outputs['bev_features'].sum()
        
        # Backward
        loss.backward()
        
        # Check gradients
        has_grad = False
        for name, param in model.named_parameters():
            if param.grad is not None:
                has_grad = True
                print(f"  {name}: grad norm = {param.grad.norm().item():.6f}")
        
        assert has_grad, "No gradients computed"
        print("✅ Backward pass test passed")
    
    def test_calibration_refinement(self):
        """캘리브레이션 refinement 테스트"""
        print("\n[TEST] Calibration Refinement")
        
        model = CameraBEVModule(self.config).to(self.device)
        
        # Initial calibration
        K_init, T_init = model.calibration()
        
        # Train for a few steps
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        
        images = torch.randn(2, 3, 1080, 1920).to(self.device)
        
        for step in range(5):
            optimizer.zero_grad()
            
            outputs = model(images, refine_calibration=True)
            loss = outputs['bev_features'].sum()
            
            loss.backward()
            optimizer.step()
        
        # After training
        K_refined, T_refined = model.calibration()
        
        K_change = torch.norm(K_refined - K_init).item()
        T_change = torch.norm(T_refined - T_init).item()
        
        print(f"  K change: {K_change:.6f}")
        print(f"  T change: {T_change:.6f}")
        
        assert K_change > 0, "K should change during training"
        assert T_change > 0, "T should change during training"
        
        print("✅ Calibration refinement test passed")
    
    def test_inference_speed(self):
        """추론 속도 테스트"""
        print("\n[TEST] Inference Speed")
        
        model = CameraBEVModule(self.config).to(self.device)
        model.eval()
        
        images = torch.randn(1, 3, 1080, 1920).to(self.device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(images, refine_calibration=False)
        
        # Measure
        if self.device == 'cuda':
            torch.cuda.synchronize()
        
        num_runs = 50
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(images, refine_calibration=False)
        
        if self.device == 'cuda':
            torch.cuda.synchronize()
        
        elapsed_time = time.time() - start_time
        fps = num_runs / elapsed_time
        latency = elapsed_time / num_runs * 1000
        
        print(f"  Device: {self.device}")
        print(f"  Average FPS: {fps:.2f}")
        print(f"  Average latency: {latency:.2f} ms")
        print("✅ Inference speed test passed")
    
    def test_memory_usage(self):
        """메모리 사용량 테스트"""
        print("\n[TEST] Memory Usage")
        
        if self.device != 'cuda':
            print("  ⚠️ Skipped (CPU mode)")
            return
        
        torch.cuda.reset_peak_memory_stats()
        
        model = CameraBEVModule(self.config).to(self.device)
        
        images = torch.randn(4, 3, 1080, 1920).to(self.device)
        
        # Forward
        outputs = model(images, refine_calibration=False)
        
        # Backward
        loss = outputs['bev_features'].sum()
        loss.backward()
        
        peak_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
        
        print(f"  Peak memory: {peak_memory:.2f} GB")
        print("✅ Memory usage test passed")


def run_integration_tests():
    """통합 테스트 실행"""
    print("="*60)
    print("CAMERA BEV MODULE - INTEGRATION TESTS")
    print("="*60)
    
    tester = TestCameraBEVIntegration()
    tester.setup_method()
    
    tester.test_module_creation()
    tester.test_forward_pass()
    tester.test_backward_pass()
    tester.test_calibration_refinement()
    tester.test_inference_speed()
    tester.test_memory_usage()
    
    print("\n" + "="*60)
    print("✅ ALL INTEGRATION TESTS PASSED!")
    print("="*60)


if __name__ == '__main__':
    run_integration_tests()