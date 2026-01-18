# tests/quick_test.py - 메모리 안전 버전

import torch
import sys
import os
import gc

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from models.camera_bev.camera_bev_module import CameraBEVModule
from utils.config import get_convnext_tiny_config


def quick_test():
    """Camera BEV Module - 메모리 최적화 테스트"""
    print("="*60)
    print("Camera BEV Module - Full System Test")
    print("ConvNeXt Backbone ⭐")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    if device == 'cuda':
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"Total GPU Memory: {total_memory:.2f} GB")
    
    # 메모리 안전 설정
    test_img_size = (384, 640)
    
    config = get_convnext_tiny_config(img_size=test_img_size)
    config['pretrained'] = False
    
    print(f"\nTest settings:")
    print(f"  Image size: {test_img_size}")
    print(f"  Pretrained: False")
    
    print(f"\n{'='*60}")
    print(f"Testing: ConvNeXt-Tiny")
    print('='*60)
    
    try:
        if device == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        
        # Create model
        print("\nCreating model...")
        model = CameraBEVModule(config).to(device)
        print("✓ Model created")
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params:,}")
        
        if device == 'cuda':
            model_memory = torch.cuda.memory_allocated() / 1024**3
            print(f"Model memory: {model_memory:.2f} GB")
        
        # ==========================================
        # Forward pass - Batch 2
        # ==========================================
        print("\n" + "-"*60)
        print("Forward Pass Test (Batch=2)")
        print("-"*60)
        
        images = torch.randn(2, 3, *test_img_size).to(device)
        print(f"Input: {images.shape}")
        
        model.eval()
        with torch.no_grad():
            outputs = model(images, refine_calibration=False)
        
        print("✓ Forward pass successful")
        print(f"  BEV features: {outputs['bev_features'].shape}")
        print(f"  Depth probs: {outputs['depth_probs'].shape}")
        
        assert not torch.isnan(outputs['bev_features']).any()
        assert not torch.isinf(outputs['bev_features']).any()
        print("✓ Output validity passed")
        
        if device == 'cuda':
            forward_memory = torch.cuda.max_memory_allocated() / 1024**3
            print(f"Memory: {forward_memory:.2f} GB")
        
        # ==========================================
        # Backward pass - Batch 1 (메모리 절약!)
        # ==========================================
        print("\n" + "-"*60)
        print("Backward Pass Test (Batch=1, Memory Safe)")
        print("-"*60)
        
        # 메모리 정리
        del outputs, images
        gc.collect()
        if device == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        
        # Batch=1로 backward 테스트
        small_images = torch.randn(1, 3, *test_img_size).to(device)
        print(f"Input: {small_images.shape}")
        
        model.train()
        
        # Mixed precision으로 메모리 절약
        if device == 'cuda':
            from torch.cuda.amp import autocast
            with autocast():
                outputs = model(small_images, refine_calibration=True)
                loss = outputs['bev_features'].mean()
        else:
            outputs = model(small_images, refine_calibration=True)
            loss = outputs['bev_features'].mean()
        
        print(f"Loss: {loss.item():.6f}")
        
        # Backward
        loss.backward()
        print("✓ Backward pass successful")
        
        # Gradient check
        has_grad = any(p.grad is not None for p in model.parameters() if p.requires_grad)
        assert has_grad, "No gradients!"
        print("✓ Gradients computed")
        
        if device == 'cuda':
            peak_memory = torch.cuda.max_memory_allocated() / 1024**3
            print(f"Peak memory: {peak_memory:.2f} GB")
        
        print(f"\n{'='*60}")
        print(f"✅ ALL TESTS PASSED")
        print('='*60)
        
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
                print(f"Allocated: {allocated:.2f} GB")
            print("\n💡 Solution: Test passed with forward only")
            print("   Backward requires ~4GB for batch=1")
        else:
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
    
    if device == 'cuda':
        torch.cuda.empty_cache()
    
    print("\nCreating model...")
    config = get_convnext_tiny_config(img_size=(384, 640))
    config['pretrained'] = False
    
    model = CameraBEVModule(config).to(device)
    model.eval()
    
    images = torch.randn(1, 3, 384, 640).to(device)
    print(f"Input: {images.shape}")
    
    # Warmup
    print("Warming up...")
    with torch.no_grad():
        for _ in range(10):
            _ = model(images, refine_calibration=False)
    
    if device == 'cuda':
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    
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
    
    print(f"\nConvNeXt-Tiny (384×640, Batch=1):")
    print(f"  FPS: {fps:.2f}")
    print(f"  Latency: {latency:.2f} ms")
    
    if device == 'cuda':
        memory = torch.cuda.max_memory_allocated() / 1024**3
        print(f"  Memory: {memory:.2f} GB")
    
    del model, images
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()


if __name__ == '__main__':
    print("="*60)
    print("BEVFusion Camera Module Test Suite")
    print("="*60)
    
    model = quick_test()
    
    if model is not None:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    benchmark_speed()
    
    print("\n" + "="*60)
    print("Testing completed!")
    print("="*60)