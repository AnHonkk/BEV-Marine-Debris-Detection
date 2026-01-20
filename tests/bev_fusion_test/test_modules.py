"""
Test Script for BEV Fusion Network
모든 모듈의 동작을 검증하는 테스트 코드
"""

import os
import sys
import torch
import torch.nn as nn

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_lidar_bev_encoder():
    """Test LiDAR BEV Encoder"""
    print("=" * 50)
    print("Testing LiDAR BEV Encoder...")
    print("=" * 50)
    
    from models.lidar_bev import LiDARBEVEncoder, MarineLiDARBEVEncoder
    
    # Configuration
    point_cloud_range = [-30.0, -30.0, -2.0, 30.0, 30.0, 3.0]
    voxel_size = (0.3, 0.3, 5.0)
    
    # Create encoder
    encoder = MarineLiDARBEVEncoder(
        point_cloud_range=point_cloud_range,
        voxel_size=voxel_size,
        backbone_out_channels=256,
    )
    
    # Create dummy point cloud
    # (N, 5) - (batch_idx, x, y, z, intensity)
    batch_size = 2
    num_points = 10000
    
    points = torch.randn(num_points * batch_size, 5)
    points[:, 0] = torch.randint(0, batch_size, (num_points * batch_size,)).float()  # batch idx
    points[:, 1] = torch.rand(num_points * batch_size) * 60 - 30  # x: [-30, 30]
    points[:, 2] = torch.rand(num_points * batch_size) * 60 - 30  # y: [-30, 30]
    points[:, 3] = torch.rand(num_points * batch_size) * 5 - 2    # z: [-2, 3]
    points[:, 4] = torch.rand(num_points * batch_size)             # intensity: [0, 1]
    
    # Forward pass
    bev_features = encoder(points, batch_size)
    
    print(f"Input points shape: {points.shape}")
    print(f"Output BEV features shape: {bev_features.shape}")
    print(f"Expected shape: (batch_size, 256, H, W)")
    print(f"Grid size: {encoder.grid_size}")
    
    # Check output
    assert bev_features.shape[0] == batch_size
    assert bev_features.shape[1] == 256
    print("✓ LiDAR BEV Encoder test passed!")
    print()
    
    return encoder, bev_features


def test_fusion_module():
    """Test BEV Fusion Module"""
    print("=" * 50)
    print("Testing BEV Fusion Module...")
    print("=" * 50)
    
    from models.fusion import BEVFusion, MultiScaleBEVFusion
    
    batch_size = 2
    camera_channels = 256
    lidar_channels = 256
    H, W = 200, 200
    
    # Create dummy BEV features
    camera_bev = torch.randn(batch_size, camera_channels, H, W)
    lidar_bev = torch.randn(batch_size, lidar_channels, H, W)
    
    # Test different fusion methods
    fusion_methods = ['conv', 'channel_attn', 'spatial_attn', 'adaptive']
    
    for method in fusion_methods:
        print(f"\nTesting {method} fusion...")
        
        fuser = BEVFusion(
            camera_channels=camera_channels,
            lidar_channels=lidar_channels,
            out_channels=256,
            fusion_method=method,
        )
        
        fused = fuser(camera_bev, lidar_bev)
        
        print(f"  Camera BEV shape: {camera_bev.shape}")
        print(f"  LiDAR BEV shape: {lidar_bev.shape}")
        print(f"  Fused BEV shape: {fused.shape}")
        
        assert fused.shape == (batch_size, 256, H, W)
        print(f"  ✓ {method} fusion test passed!")
    
    # Test cross-attention (separate due to higher memory)
    print(f"\nTesting cross_attn fusion...")
    fuser_cross = BEVFusion(
        camera_channels=camera_channels,
        lidar_channels=lidar_channels,
        out_channels=256,
        fusion_method='cross_attn',
    )
    
    # Use smaller size for cross-attention
    small_camera = torch.randn(batch_size, camera_channels, 50, 50)
    small_lidar = torch.randn(batch_size, lidar_channels, 50, 50)
    fused_cross = fuser_cross(small_camera, small_lidar)
    print(f"  ✓ cross_attn fusion test passed!")
    
    print("\n✓ All Fusion Module tests passed!")
    print()


def test_segmentation_head():
    """Test Segmentation Head"""
    print("=" * 50)
    print("Testing Segmentation Head...")
    print("=" * 50)
    
    from models.segmentation_head import (
        BEVSegmentationHead,
        MarineDebrisSegHead,
        UNetSegmentationHead,
    )
    
    batch_size = 2
    in_channels = 256
    H, W = 200, 200
    num_classes = 6
    
    # Create dummy fused BEV features
    fused_bev = torch.randn(batch_size, in_channels, H, W)
    
    # Test basic segmentation head
    print("\nTesting BEVSegmentationHead...")
    basic_head = BEVSegmentationHead(
        in_channels=in_channels,
        num_classes=num_classes,
    )
    
    seg_logits = basic_head(fused_bev)
    print(f"  Input shape: {fused_bev.shape}")
    print(f"  Output shape: {seg_logits.shape}")
    assert seg_logits.shape == (batch_size, num_classes, H, W)
    print("  ✓ BEVSegmentationHead test passed!")
    
    # Test marine debris head
    print("\nTesting MarineDebrisSegHead...")
    marine_head = MarineDebrisSegHead(
        in_channels=in_channels,
        num_classes=num_classes,
        use_instance_head=True,
        use_boundary_head=True,
    )
    
    outputs = marine_head(fused_bev)
    print(f"  Output keys: {list(outputs.keys())}")
    print(f"  Semantic shape: {outputs['semantic'].shape}")
    print(f"  Boundary shape: {outputs['boundary'].shape}")
    print(f"  Size shape: {outputs['size'].shape}")
    print(f"  Center shape: {outputs['center'].shape}")
    print(f"  Offset shape: {outputs['offset'].shape}")
    
    # Get segmentation map
    seg_map = marine_head.get_seg_map(outputs)
    print(f"  Segmentation map shape: {seg_map.shape}")
    
    print("  ✓ MarineDebrisSegHead test passed!")
    print("\n✓ All Segmentation Head tests passed!")
    print()


def test_full_network():
    """Test Full BEV Fusion Network"""
    print("=" * 50)
    print("Testing Full BEV Fusion Network...")
    print("=" * 50)
    
    from models import BEVFusionNetwork, build_network
    
    # Configuration
    config = {
        'type': 'fusion',
        'camera_bev_channels': 256,
        'lidar_bev_channels': 256,
        'fusion_method': 'adaptive',
        'fused_channels': 256,
        'num_classes': 6,
        'bev_size': (100, 100),  # Smaller for testing
        'point_cloud_range': [-30.0, -30.0, -2.0, 30.0, 30.0, 3.0],
    }
    
    # Build network
    network = build_network(config)
    
    print(f"Network type: {type(network).__name__}")
    
    # Create dummy inputs
    batch_size = 2
    camera_bev = torch.randn(batch_size, 256, 100, 100)
    
    # Create dummy point cloud
    num_points = 5000
    points = torch.zeros(num_points * batch_size, 5)
    points[:, 0] = torch.randint(0, batch_size, (num_points * batch_size,)).float()
    points[:, 1] = torch.rand(num_points * batch_size) * 60 - 30
    points[:, 2] = torch.rand(num_points * batch_size) * 60 - 30
    points[:, 3] = torch.rand(num_points * batch_size) * 5 - 2
    points[:, 4] = torch.rand(num_points * batch_size)
    
    # Forward pass
    outputs = network(camera_bev, points, batch_size)
    
    print(f"Input camera BEV shape: {camera_bev.shape}")
    print(f"Input points shape: {points.shape}")
    print(f"Output keys: {list(outputs.keys())}")
    print(f"Semantic output shape: {outputs['semantic'].shape}")
    
    # Get segmentation map
    seg_map = network.get_seg_map(outputs)
    print(f"Segmentation map shape: {seg_map.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in network.parameters())
    trainable_params = sum(p.numel() for p in network.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    print("\n✓ Full Network test passed!")
    print()


def test_losses():
    """Test Loss Functions"""
    print("=" * 50)
    print("Testing Loss Functions...")
    print("=" * 50)
    
    from training.losses import MultiTaskLoss, FocalLoss, DiceLoss
    
    batch_size = 2
    num_classes = 6
    H, W = 100, 100
    
    # Create dummy outputs
    outputs = {
        'semantic': torch.randn(batch_size, num_classes, H, W),
        'boundary': torch.sigmoid(torch.randn(batch_size, 1, H, W)),
        'size': torch.relu(torch.randn(batch_size, 1, H, W)),
    }
    
    # Create dummy targets
    targets = {
        'semantic': torch.randint(0, num_classes, (batch_size, H, W)),
        'boundary': torch.randint(0, 2, (batch_size, H, W)).float(),
        'size': torch.rand(batch_size, 1, H, W),
    }
    
    # Test Focal Loss
    print("\nTesting FocalLoss...")
    focal_loss = FocalLoss(gamma=2.0)
    loss_focal = focal_loss(outputs['semantic'], targets['semantic'])
    print(f"  Focal Loss: {loss_focal.item():.4f}")
    print("  ✓ FocalLoss test passed!")
    
    # Test Dice Loss
    print("\nTesting DiceLoss...")
    dice_loss = DiceLoss()
    loss_dice = dice_loss(outputs['semantic'], targets['semantic'])
    print(f"  Dice Loss: {loss_dice.item():.4f}")
    print("  ✓ DiceLoss test passed!")
    
    # Test Multi-Task Loss
    print("\nTesting MultiTaskLoss...")
    multi_loss = MultiTaskLoss(
        num_classes=num_classes,
        use_instance_loss=False,
    )
    losses = multi_loss(outputs, targets)
    print(f"  Total Loss: {losses['total'].item():.4f}")
    print(f"  Focal Loss: {losses['focal'].item():.4f}")
    print(f"  Dice Loss: {losses['dice'].item():.4f}")
    print(f"  Boundary Loss: {losses['boundary'].item():.4f}")
    print("  ✓ MultiTaskLoss test passed!")
    
    print("\n✓ All Loss Function tests passed!")
    print()


def test_training_step():
    """Test one training step"""
    print("=" * 50)
    print("Testing Training Step...")
    print("=" * 50)
    
    from models import BEVFusionNetwork
    from training.losses import MultiTaskLoss
    
    # Configuration
    batch_size = 2
    num_classes = 6
    H, W = 100, 100
    
    # Build network
    network = BEVFusionNetwork(
        camera_bev_channels=256,
        lidar_bev_channels=256,
        fusion_method='conv',  # Use simpler fusion for speed
        num_classes=num_classes,
        bev_size=(H, W),
    )
    
    # Build loss
    criterion = MultiTaskLoss(num_classes=num_classes)
    
    # Build optimizer
    optimizer = torch.optim.AdamW(network.parameters(), lr=1e-4)
    
    # Create dummy data
    camera_bev = torch.randn(batch_size, 256, H, W)
    
    num_points = 5000
    points = torch.zeros(num_points * batch_size, 5)
    points[:, 0] = torch.randint(0, batch_size, (num_points * batch_size,)).float()
    points[:, 1] = torch.rand(num_points * batch_size) * 60 - 30
    points[:, 2] = torch.rand(num_points * batch_size) * 60 - 30
    points[:, 3] = torch.rand(num_points * batch_size) * 5 - 2
    points[:, 4] = torch.rand(num_points * batch_size)
    
    targets = {
        'semantic': torch.randint(0, num_classes, (batch_size, H, W)),
        'boundary': torch.randint(0, 2, (batch_size, H, W)).float(),
        'size': torch.rand(batch_size, 1, H, W),
    }
    
    # Training step
    network.train()
    optimizer.zero_grad()
    
    # Forward
    outputs = network(camera_bev, points, batch_size)
    
    # Loss
    losses = criterion(outputs, targets)
    
    # Backward
    losses['total'].backward()
    
    # Check gradients
    has_grad = False
    for name, param in network.named_parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_grad = True
            break
    
    assert has_grad, "No gradients computed!"
    
    # Optimizer step
    optimizer.step()
    
    print(f"Loss: {losses['total'].item():.4f}")
    print("✓ Training step completed successfully!")
    print()


def run_all_tests():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("BEV Fusion Network Test Suite")
    print("=" * 60 + "\n")
    
    try:
        test_lidar_bev_encoder()
        test_fusion_module()
        test_segmentation_head()
        test_full_network()
        test_losses()
        test_training_step()
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED! ✓")
        print("=" * 60 + "\n")
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == '__main__':
    run_all_tests()
