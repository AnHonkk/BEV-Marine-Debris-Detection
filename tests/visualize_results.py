# tests/visualize_results.py

import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys
sys.path.append('..')

from models.camera_bev.camera_bev_module import CameraBEVModule


def visualize_depth_prediction(model, image):
    """Depth 예측 시각화"""
    model.eval()
    
    with torch.no_grad():
        outputs = model(image.unsqueeze(0), refine_calibration=False)
    
    depth_expected = outputs['depth_expected'][0, 0].cpu().numpy()
    depth_probs = outputs['depth_probs'][0].cpu().numpy()  # [D, H, W]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original image
    img_np = image.permute(1, 2, 0).cpu().numpy()
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
    axes[0, 0].imshow(img_np)
    axes[0, 0].set_title('Input Image')
    axes[0, 0].axis('off')
    
    # Expected depth
    im = axes[0, 1].imshow(depth_expected, cmap='plasma')
    axes[0, 1].set_title('Expected Depth')
    axes[0, 1].axis('off')
    plt.colorbar(im, ax=axes[0, 1])
    
    # Depth uncertainty (entropy)
    entropy = -(depth_probs * np.log(depth_probs + 1e-10)).sum(axis=0)
    im = axes[0, 2].imshow(entropy, cmap='hot')
    axes[0, 2].set_title('Depth Uncertainty')
    axes[0, 2].axis('off')
    plt.colorbar(im, ax=axes[0, 2])
    
    # Depth distribution at center
    h, w = depth_expected.shape
    center_probs = depth_probs[:, h//2, w//2]
    depth_values = outputs['depth_values'].cpu().numpy()
    
    axes[1, 0].bar(range(len(center_probs)), center_probs)
    axes[1, 0].set_title('Depth Distribution (Center)')
    axes[1, 0].set_xlabel('Depth Bin')
    axes[1, 0].set_ylabel('Probability')
    
    # Depth histogram
    axes[1, 1].hist(depth_expected.flatten(), bins=50)
    axes[1, 1].set_title('Depth Histogram')
    axes[1, 1].set_xlabel('Depth (m)')
    axes[1, 1].set_ylabel('Count')
    
    # Statistics
    stats_text = f"""
    Min depth: {depth_expected.min():.2f} m
    Max depth: {depth_expected.max():.2f} m
    Mean depth: {depth_expected.mean():.2f} m
    Std depth: {depth_expected.std():.2f} m
    """
    axes[1, 2].text(0.1, 0.5, stats_text, fontsize=12, family='monospace')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('results/depth_prediction.png', dpi=150, bbox_inches='tight')
    print("✅ Depth visualization saved to results/depth_prediction.png")
    
    return fig


def visualize_bev_features(model, image):
    """BEV 특징맵 시각화"""
    model.eval()
    
    with torch.no_grad():
        outputs = model(image.unsqueeze(0), refine_calibration=False)
    
    bev_features = outputs['bev_features'][0].cpu().numpy()  # [C, H, W]
    
    # Aggregate channels
    bev_aggregated = bev_features.mean(axis=0)  # [H, W]
    bev_max = bev_features.max(axis=0)
    bev_std = bev_features.std(axis=0)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # Mean activation
    im = axes[0, 0].imshow(bev_aggregated, cmap='viridis')
    axes[0, 0].set_title('BEV Features (Mean)')
    plt.colorbar(im, ax=axes[0, 0])
    
    # Max activation
    im = axes[0, 1].imshow(bev_max, cmap='hot')
    axes[0, 1].set_title('BEV Features (Max)')
    plt.colorbar(im, ax=axes[0, 1])
    
    # Std deviation
    im = axes[1, 0].imshow(bev_std, cmap='plasma')
    axes[1, 0].set_title('BEV Features (Std)')
    plt.colorbar(im, ax=axes[1, 0])
    
    # Non-zero mask
    non_zero_mask = (bev_aggregated > bev_aggregated.mean()).astype(float)
    axes[1, 1].imshow(non_zero_mask, cmap='gray')
    axes[1, 1].set_title(f'Active Regions ({non_zero_mask.sum()/(256*256)*100:.1f}%)')
    
    for ax in axes.flat:
        ax.set_xlabel('X (BEV)')
        ax.set_ylabel('Y (BEV)')
    
    plt.tight_layout()
    plt.savefig('results/bev_features.png', dpi=150, bbox_inches='tight')
    print("✅ BEV visualization saved to results/bev_features.png")
    
    return fig


def visualize_calibration_refinement(model, image, num_steps=100):
    """캘리브레이션 refinement 시각화"""
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    K_init, T_init = model.calibration()
    K_history = [K_init.detach().cpu().numpy()]
    T_history = [T_init.detach().cpu().numpy()]
    
    losses = []
    
    for step in range(num_steps):
        optimizer.zero_grad()
        
        outputs = model(image.unsqueeze(0), refine_calibration=True)
        loss = outputs['bev_features'].sum()
        
        loss.backward()
        optimizer.step()
        
        K_current, T_current = model.calibration()
        K_history.append(K_current.detach().cpu().numpy())
        T_history.append(T_current.detach().cpu().numpy())
        losses.append(loss.item())
    
    K_history = np.array(K_history)
    T_history = np.array(T_history)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Focal length changes
    axes[0, 0].plot(K_history[:, 0, 0], label='fx')
    axes[0, 0].plot(K_history[:, 1, 1], label='fy')
    axes[0, 0].set_title('Focal Length Refinement')
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('Focal Length')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Principal point changes
    axes[0, 1].plot(K_history[:, 0, 2], label='cx')
    axes[0, 1].plot(K_history[:, 1, 2], label='cy')
    axes[0, 1].set_title('Principal Point Refinement')
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('Principal Point')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Translation changes
    axes[1, 0].plot(T_history[:, 0, 3], label='tx')
    axes[1, 0].plot(T_history[:, 1, 3], label='ty')
    axes[1, 0].plot(T_history[:, 2, 3], label='tz')
    axes[1, 0].set_title('Translation Refinement')
    axes[1, 0].set_xlabel('Iteration')
    axes[1, 0].set_ylabel('Translation (m)')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Loss curve
    axes[1, 1].plot(losses)
    axes[1, 1].set_title('Training Loss')
    axes[1, 1].set_xlabel('Iteration')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].grid(True)
    axes[1, 1].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('results/calibration_refinement.png', dpi=150, bbox_inches='tight')
    print("✅ Calibration refinement visualization saved")
    
    return fig


def run_visualization_tests():
    """시각화 테스트 실행"""
    import os
    os.makedirs('results', exist_ok=True)
    
    print("="*60)
    print("CAMERA BEV MODULE - VISUALIZATION TESTS")
    print("="*60)
    
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    K_init = np.array([
        [1000.0, 0, 960.0],
        [0, 1000.0, 540.0],
        [0, 0, 1.0]
    ])
    
    T_init = np.eye(4)
    T_init[:3, 3] = [0.1, -0.05, 0.2]
    
    config = {
        'backbone': 'resnet50',
        'pretrained': False,
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
    
    model = CameraBEVModule(config).to(device)
    
    # Dummy image
    image = torch.randn(3, 1080, 1920).to(device)
    
    print("\n[VIZ] Depth Prediction")
    visualize_depth_prediction(model, image)
    
    print("\n[VIZ] BEV Features")
    visualize_bev_features(model, image)
    
    print("\n[VIZ] Calibration Refinement")
    visualize_calibration_refinement(model, image, num_steps=50)
    
    print("\n" + "="*60)
    print("✅ ALL VISUALIZATIONS COMPLETED!")
    print("Check results/ directory for outputs")
    print("="*60)


if __name__ == '__main__':
    run_visualization_tests()