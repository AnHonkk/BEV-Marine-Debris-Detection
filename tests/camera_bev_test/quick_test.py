# tests/quick_test.py - 최종 수정 버전

import torch
import sys
import os
import gc

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from models.camera_bev.camera_bev_module import CameraBEVModule
from utils.config import get_convnext_tiny_config


# Global constants
DEFAULT_IMG_SIZE = (384, 640)  # ← 384로 통일!


def quick_test():
    """Camera BEV Module - 전체 시스템 테스트"""
    print("="*60)
    print("Camera BEV Module - Full System Test")
    print("ConvNeXt-Tiny + Light U-Net Depth Head")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    
    if device == 'cuda':
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"Total GPU Memory: {total_memory:.2f} GB")
    
    # Configuration
    test_img_size = DEFAULT_IMG_SIZE
    
    config = get_convnext_tiny_config(img_size=test_img_size)
    config['pretrained'] = False  # 빠른 테스트
    
    print(f"\nConfiguration:")
    print(f"  Image size: {test_img_size}")
    print(f"  BEV size: {config['bev_size']}")
    print(f"  Depth bins: {config['depth_bins']}")
    print(f"  Pretrained: {config['pretrained']}")
    
    print(f"\n{'='*60}")
    print(f"Testing: ConvNeXt-Tiny + Light U-Net")
    print('='*60)
    
    try:
        if device == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        
        # ==========================================
        # 1. Model Creation
        # ==========================================
        print("\n1. Creating model...")
        model = CameraBEVModule(config).to(device)
        print("   ✓ Model created")
        
        # Model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"\n   Model Statistics:")
        print(f"     Total parameters: {total_params:,}")
        print(f"     Trainable: {trainable_params:,}")
        
        # Module-wise breakdown
        backbone_params = sum(p.numel() for p in model.backbone.parameters())
        depth_params = sum(p.numel() for p in model.depth_net.parameters())
        calib_params = sum(p.numel() for p in model.calibration.parameters())
        
        print(f"\n   Module Breakdown:")
        print(f"     Backbone: {backbone_params:,}")
        print(f"     Depth Net: {depth_params:,}")
        print(f"     Calibration: {calib_params:,}")
        
        if device == 'cuda':
            model_memory = torch.cuda.memory_allocated() / 1024**3
            print(f"\n   GPU Memory (model): {model_memory:.2f} GB")
        
        # ==========================================
        # 2. Forward Pass (Batch=2)
        # ==========================================
        print("\n" + "-"*60)
        print("2. Forward Pass Test (Batch=2)")
        print("-"*60)
        
        images = torch.randn(2, 3, *test_img_size).to(device)
        print(f"   Input: {images.shape}")
        
        model.eval()
        with torch.no_grad():
            outputs = model(images, refine_calibration=False)
        
        print("   ✓ Forward pass successful")
        
        # Check outputs
        print(f"\n   Outputs:")
        print(f"     bev_features: {outputs['bev_features'].shape}")
        print(f"     depth_probs: {outputs['depth_probs'].shape}")
        print(f"     depth_expected: {outputs['depth_expected'].shape}")
        print(f"     K: {outputs['K'].shape}")
        print(f"     T: {outputs['T'].shape}")
        
        # Validate
        assert not torch.isnan(outputs['bev_features']).any(), "NaN in BEV features!"
        assert not torch.isinf(outputs['bev_features']).any(), "Inf in BEV features!"
        print("   ✓ Output validity passed")
        
        # Depth probability check
        prob_sum = outputs['depth_probs'].sum(dim=1)
        assert torch.allclose(prob_sum, torch.ones_like(prob_sum), atol=1e-5), "Depth probs don't sum to 1!"
        print("   ✓ Depth distribution valid")
        
        if device == 'cuda':
            forward_memory = torch.cuda.max_memory_allocated() / 1024**3
            print(f"\n   GPU Memory (forward): {forward_memory:.2f} GB")
        
        # ==========================================
        # 3. Backward Pass (Batch=1, Memory Safe)
        # ==========================================
        print("\n" + "-"*60)
        print("3. Backward Pass Test (Batch=1)")
        print("-"*60)
        
        # Cleanup
        del outputs, images
        gc.collect()
        if device == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        
        # Single image for backward
        small_images = torch.randn(1, 3, *test_img_size).to(device)
        print(f"   Input: {small_images.shape}")
        
        model.train()
        
        # Forward (no mixed precision for stability)
        outputs = model(small_images, refine_calibration=True)
        loss = outputs['bev_features'].mean()
        
        print(f"   Loss: {loss.item():.6f}")
        
        # Backward
        loss.backward()
        print("   ✓ Backward pass successful")
        
        # Gradient check
        has_grad = any(p.grad is not None for p in model.parameters() if p.requires_grad)
        assert has_grad, "No gradients computed!"
        print("   ✓ Gradients computed")
        
        # Check specific modules
        depth_grads = sum(
            1 for p in model.depth_net.parameters() 
            if p.grad is not None and p.requires_grad
        )
        print(f"   Depth Net layers with gradients: {depth_grads}")
        
        if device == 'cuda':
            peak_memory = torch.cuda.max_memory_allocated() / 1024**3
            print(f"\n   GPU Memory (peak): {peak_memory:.2f} GB")
        
        # ==========================================
        # Summary
        # ==========================================
        print(f"\n{'='*60}")
        print(f"✅ ALL TESTS PASSED!")
        print('='*60)
        
        print("\nTest Summary:")
        print(f"  ✓ Model creation: OK")
        print(f"  ✓ Forward pass (batch=2): OK")
        print(f"  ✓ Backward pass (batch=1): OK")
        print(f"  ✓ Gradient computation: OK")
        
        # Cleanup
        del small_images, outputs, loss
        gc.collect()
        if device == 'cuda':
            torch.cuda.empty_cache()
        
        return model
        
    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"\n❌ OUT OF MEMORY")
            if device == 'cuda':
                allocated = torch.cuda.memory_allocated() / 1024**3
                peak = torch.cuda.max_memory_allocated() / 1024**3
                print(f"   Allocated: {allocated:.2f} GB")
                print(f"   Peak: {peak:.2f} GB")
            print("\n💡 Solution:")
            print("   - Reduce batch size")
            print("   - Reduce image size")
            print("   - Use gradient checkpointing")
        else:
            print(f"\n❌ Runtime Error: {e}")
            import traceback
            traceback.print_exc()
        
        gc.collect()
        if device == 'cuda':
            torch.cuda.empty_cache()
        
        return None
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        
        gc.collect()
        if device == 'cuda':
            torch.cuda.empty_cache()
        
        return None


def benchmark_speed():
    """추론 속도 벤치마크"""
    import time
    
    print("\n" + "="*60)
    print("Inference Speed Benchmark")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if device == 'cpu':
        print("⚠️ Running on CPU - benchmark will be slow")
    
    if device == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    # Create model
    print("\nCreating model...")
    config = get_convnext_tiny_config(img_size=DEFAULT_IMG_SIZE)
    config['pretrained'] = False
    
    try:
        model = CameraBEVModule(config).to(device)
        model.eval()
        print("✓ Model created")
        
        images = torch.randn(1, 3, *DEFAULT_IMG_SIZE).to(device)
        print(f"Input: {images.shape}")
        
        # Warmup
        print("\nWarming up (10 iterations)...")
        with torch.no_grad():
            for _ in range(10):
                _ = model(images, refine_calibration=False)
        
        if device == 'cuda':
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        
        # Benchmark
        num_runs = 50
        print(f"Benchmarking ({num_runs} iterations)...")
        
        if device == 'cuda':
            torch.cuda.synchronize()
        
        start = time.time()
        
        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(images, refine_calibration=False)
        
        if device == 'cuda':
            torch.cuda.synchronize()
        
        elapsed = time.time() - start
        fps = num_runs / elapsed
        latency = elapsed / num_runs * 1000
        
        print(f"\nResults (ConvNeXt-Tiny + Light U-Net):")
        print(f"  Image size: {DEFAULT_IMG_SIZE}")
        print(f"  Batch size: 1")
        print(f"  FPS: {fps:.2f}")
        print(f"  Latency: {latency:.2f} ms")
        
        if device == 'cuda':
            memory = torch.cuda.max_memory_allocated() / 1024**3
            print(f"  GPU Memory: {memory:.2f} GB")
        
        # Cleanup
        del model, images
        gc.collect()
        if device == 'cuda':
            torch.cuda.empty_cache()
    
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    print("="*60)
    print("BEVFusion Camera Module - Test Suite")
    print("="*60)
    print("\nComponents:")
    print("  - ConvNeXt-Tiny Backbone")
    print("  - Light U-Net Depth Head")
    print("  - Learnable Calibration")
    print("  - LSS Transformation")
    
    # Main test
    model = quick_test()
    
    if model is not None:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Benchmark
        benchmark_speed()
    else:
        print("\n⚠️ Skipping benchmark due to test failure")
    
    print("\n" + "="*60)
    print("Testing completed!")
    print("="*60)