# tests/visualize_results.py

import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# 프로젝트 루트 경로 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from models.camera_bev.camera_bev_module import CameraBEVModule
from utils.config import get_convnext_tiny_config


def visualize_depth_prediction(model, image):
    """Depth 예측 시각화"""
    print("  - Visualizing Depth Prediction...")
    model.eval()
    
    with torch.no_grad():
        outputs = model(image.unsqueeze(0), refine_calibration=False)
    
    depth_expected = outputs['depth_expected'][0, 0].cpu().numpy()
    depth_probs = outputs['depth_probs'][0].cpu().numpy()
    depth_values = outputs['depth_values'].cpu().numpy()
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # 1. Original image
    img_np = image.permute(1, 2, 0).cpu().numpy()
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
    axes[0, 0].imshow(img_np)
    axes[0, 0].set_title(f'Input Image {image.shape[1:]}')
    axes[0, 0].axis('off')
    
    # 2. Expected depth map
    im = axes[0, 1].imshow(depth_expected, cmap='plasma', vmin=depth_values.min(), vmax=depth_values.max())
    axes[0, 1].set_title('Expected Depth Map')
    axes[0, 1].axis('off')
    plt.colorbar(im, ax=axes[0, 1], label='Depth (m)')
    
    # 3. Depth Uncertainty (Entropy)
    entropy = -(depth_probs * np.log(depth_probs + 1e-10)).sum(axis=0)
    im = axes[0, 2].imshow(entropy, cmap='inferno')
    axes[0, 2].set_title('Uncertainty (Entropy)')
    axes[0, 2].axis('off')
    plt.colorbar(im, ax=axes[0, 2])
    
    # 4. Center Pixel Distribution
    h, w = depth_expected.shape
    center_y, center_x = h // 2, w // 2
    center_probs = depth_probs[:, center_y, center_x]
    
    axes[1, 0].bar(depth_values, center_probs, width=(depth_values[1]-depth_values[0])*0.8)
    axes[1, 0].set_title(f'Depth Distribution @ Center ({center_x}, {center_y})')
    axes[1, 0].set_xlabel('Depth (m)')
    axes[1, 0].set_ylabel('Probability')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Global Depth Histogram
    axes[1, 1].hist(depth_expected.flatten(), bins=50, color='skyblue', edgecolor='black')
    axes[1, 1].set_title('Global Depth Histogram')
    axes[1, 1].set_xlabel('Depth (m)')
    axes[1, 1].set_ylabel('Pixel Count')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Statistics
    stats_text = (
        f"Statistics:\n"
        f"─────────────────\n"
        f"Min:  {depth_expected.min():.2f} m\n"
        f"Max:  {depth_expected.max():.2f} m\n"
        f"Mean: {depth_expected.mean():.2f} m\n"
        f"Std:  {depth_expected.std():.2f} m\n\n"
        f"Center: {depth_expected[center_y, center_x]:.2f} m\n"
        f"Range: {depth_values.min():.1f}~{depth_values.max():.1f}m"
    )
    axes[1, 2].text(0.1, 0.5, stats_text, fontsize=12, family='monospace', va='center')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('results/depth_prediction.png', dpi=150, bbox_inches='tight')
    print("  ✅ Saved: results/depth_prediction.png")
    plt.close()


def visualize_bev_features(model, image):
    """BEV 특징맵 시각화"""
    print("  - Visualizing BEV Features...")
    model.eval()
    
    with torch.no_grad():
        outputs = model(image.unsqueeze(0), refine_calibration=False)
    
    bev_features = outputs['bev_features'][0].cpu().numpy()  # [C, H, W]
    
    # Channel aggregation
    bev_mean = bev_features.mean(axis=0)
    bev_max = bev_features.max(axis=0)
    bev_norm = np.linalg.norm(bev_features, axis=0)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 1. Mean Activation
    im = axes[0].imshow(bev_mean, cmap='viridis', origin='lower')
    axes[0].set_title('BEV Features (Mean)')
    plt.colorbar(im, ax=axes[0])
    
    # 2. Max Activation
    im = axes[1].imshow(bev_max, cmap='hot', origin='lower')
    axes[1].set_title('BEV Features (Max)')
    plt.colorbar(im, ax=axes[1])
    
    # 3. Feature Magnitude
    im = axes[2].imshow(bev_norm, cmap='plasma', origin='lower')
    axes[2].set_title('Feature Magnitude (L2 Norm)')
    plt.colorbar(im, ax=axes[2])
    
    for ax in axes:
        ax.set_xlabel('X (Right, m)')
        ax.set_ylabel('Y (Forward, m)')
        ax.grid(False)
    
    plt.tight_layout()
    plt.savefig('results/bev_features.png', dpi=150, bbox_inches='tight')
    print("  ✅ Saved: results/bev_features.png")
    plt.close()


def visualize_calibration_refinement(model, image, num_steps=50):
    """
    캘리브레이션 refinement 시각화
    
    Note: 실제로는 BEV segmentation loss로 학습
          여기서는 데모 목적으로 간단한 loss 사용
    """
    print(f"  - Visualizing Calibration Refinement ({num_steps} steps)...")
    print("    (This is a demo - real training uses segmentation loss)")
    
    model.train()
    
    # Freeze everything except calibration
    for param in model.parameters():
        param.requires_grad = False
    
    # Enable calibration
    if hasattr(model.calibration, 'delta_K'):
        model.calibration.delta_K.requires_grad = True
    if hasattr(model.calibration, 'delta_pose'):
        model.calibration.delta_pose.requires_grad = True
    
    # Optimizer
    params = [p for p in model.calibration.parameters() if p.requires_grad]
    
    if not params:
        print("  ⚠️ No learnable calibration parameters!")
        return
    
    optimizer = torch.optim.Adam(params, lr=1e-3)
    
    # History
    K_init, T_init = model.calibration()
    K_history = [K_init.detach().cpu().numpy().copy()]
    loss_history = []
    
    for step in range(num_steps):
        optimizer.zero_grad()
        
        outputs = model(image.unsqueeze(0), refine_calibration=True)
        
        # Demo loss (실제로는 segmentation loss 사용)
        bev_feat = outputs['bev_features']
        loss = -bev_feat.abs().mean()  # Maximize feature magnitude
        
        # Regularization
        if hasattr(model.calibration, 'delta_K'):
            loss += 0.01 * model.calibration.delta_K.pow(2).sum()
        if hasattr(model.calibration, 'delta_pose'):
            loss += 0.01 * model.calibration.delta_pose.pow(2).sum()
        
        loss.backward()
        optimizer.step()
        
        # Record
        K_curr, _ = model.calibration()
        K_history.append(K_curr.detach().cpu().numpy().copy())
        loss_history.append(loss.item())
    
    K_history = np.array(K_history)
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. Focal Length
    axes[0].plot(K_history[:, 0, 0], label='fx', linewidth=2)
    axes[0].plot(K_history[:, 1, 1], label='fy', linewidth=2)
    axes[0].axhline(K_init.cpu().numpy()[0, 0], color='gray', linestyle='--', alpha=0.5, label='Initial')
    axes[0].set_title('Focal Length Refinement')
    axes[0].set_xlabel('Step')
    axes[0].set_ylabel('Focal Length (pixels)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 2. Principal Point
    axes[1].plot(K_history[:, 0, 2], label='cx', linewidth=2)
    axes[1].plot(K_history[:, 1, 2], label='cy', linewidth=2)
    axes[1].axhline(K_init.cpu().numpy()[0, 2], color='gray', linestyle='--', alpha=0.5, label='Initial')
    axes[1].set_title('Principal Point Refinement')
    axes[1].set_xlabel('Step')
    axes[1].set_ylabel('Principal Point (pixels)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # 3. Loss
    axes[2].plot(loss_history, color='red', linewidth=2)
    axes[2].set_title('Demo Loss (not real training)')
    axes[2].set_xlabel('Step')
    axes[2].set_ylabel('Loss')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/calibration_refinement.png', dpi=150, bbox_inches='tight')
    print("  ✅ Saved: results/calibration_refinement.png")
    plt.close()


def run_visualization_tests():
    """메인 테스트 함수"""
    os.makedirs('results', exist_ok=True)
    
    print("="*60)
    print("CAMERA BEV MODULE - VISUALIZATION TESTS")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    
    # Configuration (수정된 버전!)
    img_size = (384, 640)  # ← 4와 16의 배수!
    
    K_init = np.array([
        [800.0,   0.0, 320.0],  # fx, 0, cx (640/2)
        [  0.0, 800.0, 192.0],  # 0, fy, cy (384/2)
        [  0.0,   0.0,   1.0]
    ], dtype=np.float32)
    
    T_init = np.eye(4, dtype=np.float32)
    T_init[:3, 3] = [0.0, 0.0, 1.5]  # Camera height: 1.5m
    
    config = get_convnext_tiny_config(img_size=img_size)
    config.update({
        'K_init': K_init,
        'T_init': T_init,
        'bev_size': (256, 256),                   # 10cm resolution
        'bev_range': (-12.8, 12.8, -12.8, 12.8),  # ±12.8m
        'depth_bins': 64,
        'depth_range': (1.0, 50.0),               # 1~50m (marine)
        'pretrained': True
    })
    
    print("\nConfiguration:")
    print(f"  Image Size : {img_size}")
    print(f"  BEV Size   : {config['bev_size']}")
    print(f"  BEV Range  : {config['bev_range']}")
    print(f"  BEV Resolution: {(config['bev_range'][1]-config['bev_range'][0])/config['bev_size'][0]:.2f}m/pixel")
    print(f"  Depth Range: {config['depth_range']}")
    print(f"  Depth Bins : {config['depth_bins']}")
    
    # Create model
    print("\nCreating Model...")
    try:
        model = CameraBEVModule(config).to(device)
        print("✓ Model created successfully")
        
        # Model info
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Total parameters: {total_params:,}")
        
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Create dummy image
    print("\nGenerating dummy image...")
    dummy_image = torch.randn(1, 3, *img_size).to(device)
    dummy_image = (dummy_image - dummy_image.min()) / (dummy_image.max() - dummy_image.min())
    print(f"  Image shape: {dummy_image.shape}")
    
    # Run visualizations
    print("\nRunning Visualizations:")
    print("-"*60)
    
    try:
        # 1. Depth
        visualize_depth_prediction(model, dummy_image.squeeze(0))
        
        # 2. BEV
        visualize_bev_features(model, dummy_image.squeeze(0))
        
        # 3. Calibration
        visualize_calibration_refinement(model, dummy_image.squeeze(0), num_steps=30)
        
    except Exception as e:
        print(f"\n❌ Visualization failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "="*60)
    print("✅ ALL VISUALIZATIONS COMPLETED!")
    print("="*60)
    print("\nGenerated files in 'results/' directory:")
    print("  - depth_prediction.png")
    print("  - bev_features.png")
    print("  - calibration_refinement.png")


if __name__ == '__main__':
    run_visualization_tests()