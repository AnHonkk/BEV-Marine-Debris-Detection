"""
LiDAR BEV Encoder 독립 테스트 스크립트
다른 모듈(fusion, segmentation_head)에 의존하지 않고 LiDAR BEV만 테스트
"""

import os
import sys
import torch
import torch.nn as nn

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_pillar_feature_net():
    """Test PillarFeatureNet"""
    print("=" * 50)
    print("Testing PillarFeatureNet...")
    print("=" * 50)
    
    from models.lidar_bev.pointpillars import PillarFeatureNet
    
    # Config
    voxel_size = (0.25, 0.25, 8.0)
    point_cloud_range = [-30.0, -30.0, -2.0, 30.0, 30.0, 3.0]
    
    pfn = PillarFeatureNet(
        in_channels=4,
        feat_channels=64,
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
    )
    
    # Dummy input
    num_voxels = 100
    max_points = 32
    
    voxels = torch.randn(num_voxels, max_points, 4)
    num_points = torch.randint(1, max_points + 1, (num_voxels,))
    coors = torch.zeros(num_voxels, 3, dtype=torch.long)
    coors[:, 0] = torch.randint(0, 2, (num_voxels,))  # batch idx
    coors[:, 1] = torch.randint(0, 200, (num_voxels,))  # x idx
    coors[:, 2] = torch.randint(0, 200, (num_voxels,))  # y idx
    
    # Forward
    pillar_features = pfn(voxels, num_points, coors)
    
    print(f"  Input voxels shape: {voxels.shape}")
    print(f"  Output pillar_features shape: {pillar_features.shape}")
    
    assert pillar_features.shape == (num_voxels, 64)
    print("  ✓ PillarFeatureNet test passed!")
    print()


def test_scatter():
    """Test PointPillarsScatter"""
    print("=" * 50)
    print("Testing PointPillarsScatter...")
    print("=" * 50)
    
    from models.lidar_bev.pointpillars import PointPillarsScatter
    
    scatter = PointPillarsScatter(
        in_channels=64,
        output_shape=(200, 200),
    )
    
    # Dummy input
    num_pillars = 500
    batch_size = 2
    
    pillar_features = torch.randn(num_pillars, 64)
    coors = torch.zeros(num_pillars, 3, dtype=torch.long)
    coors[:, 0] = torch.randint(0, batch_size, (num_pillars,))
    coors[:, 1] = torch.randint(0, 200, (num_pillars,))
    coors[:, 2] = torch.randint(0, 200, (num_pillars,))
    
    # Forward
    bev_features = scatter(pillar_features, coors, batch_size)
    
    print(f"  Input pillar_features shape: {pillar_features.shape}")
    print(f"  Output BEV shape: {bev_features.shape}")
    
    assert bev_features.shape == (batch_size, 64, 200, 200)
    print("  ✓ PointPillarsScatter test passed!")
    print()


def test_backbone():
    """Test LiDARBackbone"""
    print("=" * 50)
    print("Testing LiDARBackbone...")
    print("=" * 50)
    
    from models.lidar_bev.pointpillars import LiDARBackbone
    
    backbone = LiDARBackbone(
        in_channels=64,
        out_channels=256,
    )
    
    # Dummy input
    batch_size = 2
    bev_input = torch.randn(batch_size, 64, 200, 200)
    
    # Forward
    bev_output = backbone(bev_input)
    
    print(f"  Input shape: {bev_input.shape}")
    print(f"  Output shape: {bev_output.shape}")
    
    assert bev_output.shape[0] == batch_size
    assert bev_output.shape[1] == 256
    print("  ✓ LiDARBackbone test passed!")
    print()


def test_voxelization():
    """Test DynamicVoxelization"""
    print("=" * 50)
    print("Testing DynamicVoxelization...")
    print("=" * 50)
    
    from models.lidar_bev.pointpillars import DynamicVoxelization
    
    voxelizer = DynamicVoxelization(
        voxel_size=(0.25, 0.25, 8.0),
        point_cloud_range=[-30.0, -30.0, -2.0, 30.0, 30.0, 3.0],
        max_points_per_voxel=32,
        max_num_voxels=30000,
    )
    
    # Create dummy points
    batch_size = 2
    num_points_per_batch = 5000
    total_points = batch_size * num_points_per_batch
    
    points = torch.zeros(total_points, 5)
    for b in range(batch_size):
        start = b * num_points_per_batch
        end = (b + 1) * num_points_per_batch
        points[start:end, 0] = b  # batch idx
        points[start:end, 1] = torch.rand(num_points_per_batch) * 60 - 30  # x
        points[start:end, 2] = torch.rand(num_points_per_batch) * 60 - 30  # y
        points[start:end, 3] = torch.rand(num_points_per_batch) * 5 - 2   # z
        points[start:end, 4] = torch.rand(num_points_per_batch)            # intensity
    
    # Forward
    voxels, num_points, coors = voxelizer(points, batch_size)
    
    print(f"  Input points shape: {points.shape}")
    print(f"  Output voxels shape: {voxels.shape}")
    print(f"  Output num_points shape: {num_points.shape}")
    print(f"  Output coors shape: {coors.shape}")
    print(f"  Number of voxels: {len(voxels)}")
    
    assert voxels.shape[1] == 32  # max_points_per_voxel
    assert voxels.shape[2] == 4   # x, y, z, intensity
    print("  ✓ DynamicVoxelization test passed!")
    print()


def test_lidar_bev_encoder():
    """Test full LiDARBEVEncoder"""
    print("=" * 50)
    print("Testing LiDARBEVEncoder (Full Pipeline)...")
    print("=" * 50)
    
    from models.lidar_bev.pointpillars import LiDARBEVEncoder
    
    encoder = LiDARBEVEncoder(
        point_cloud_range=[-30.0, -30.0, -2.0, 30.0, 30.0, 3.0],
        voxel_size=(0.3, 0.3, 5.0),
        max_points_per_voxel=32,
        max_num_voxels=20000,
        feat_channels=64,
        backbone_out_channels=256,
    )
    
    # Create dummy points
    batch_size = 2
    num_points_per_batch = 5000
    total_points = batch_size * num_points_per_batch
    
    points = torch.zeros(total_points, 5)
    for b in range(batch_size):
        start = b * num_points_per_batch
        end = (b + 1) * num_points_per_batch
        points[start:end, 0] = b
        points[start:end, 1] = torch.rand(num_points_per_batch) * 60 - 30
        points[start:end, 2] = torch.rand(num_points_per_batch) * 60 - 30
        points[start:end, 3] = torch.rand(num_points_per_batch) * 5 - 2
        points[start:end, 4] = torch.rand(num_points_per_batch)
    
    # Forward
    bev_features = encoder(points, batch_size)
    
    print(f"  Input points shape: {points.shape}")
    print(f"  Output BEV features shape: {bev_features.shape}")
    print(f"  Grid size: {encoder.grid_size}")
    print(f"  Output channels: {encoder.out_channels}")
    
    assert bev_features.shape[0] == batch_size
    assert bev_features.shape[1] == 256
    print("  ✓ LiDARBEVEncoder test passed!")
    print()
    
    return encoder


def test_marine_encoder():
    """Test MarineLiDARBEVEncoder"""
    print("=" * 50)
    print("Testing MarineLiDARBEVEncoder...")
    print("=" * 50)
    
    from models.lidar_bev.pointpillars import MarineLiDARBEVEncoder
    
    encoder = MarineLiDARBEVEncoder(
        point_cloud_range=[-30.0, -30.0, -2.0, 30.0, 30.0, 3.0],
        voxel_size=(0.15, 0.15, 5.0),
        filter_water_reflection=True,
        water_level_threshold=-0.5,
    )
    
    # Create dummy points with some below water level
    batch_size = 2
    num_points_per_batch = 5000
    total_points = batch_size * num_points_per_batch
    
    points = torch.zeros(total_points, 5)
    for b in range(batch_size):
        start = b * num_points_per_batch
        end = (b + 1) * num_points_per_batch
        points[start:end, 0] = b
        points[start:end, 1] = torch.rand(num_points_per_batch) * 60 - 30
        points[start:end, 2] = torch.rand(num_points_per_batch) * 60 - 30
        # z: some below water level (-0.5)
        points[start:end, 3] = torch.rand(num_points_per_batch) * 5 - 2
        points[start:end, 4] = torch.rand(num_points_per_batch)
    
    # Count points that should be filtered
    below_water = (points[:, 3] <= -0.5).sum().item()
    print(f"  Points below water level: {below_water}/{total_points}")
    
    # Forward
    bev_features = encoder(points, batch_size)
    
    print(f"  Output BEV features shape: {bev_features.shape}")
    print("  ✓ MarineLiDARBEVEncoder test passed!")
    print()


def test_gradient_flow():
    """Test gradient flow through the encoder"""
    print("=" * 50)
    print("Testing Gradient Flow...")
    print("=" * 50)
    
    from models.lidar_bev.pointpillars import LiDARBEVEncoder
    
    encoder = LiDARBEVEncoder(
        point_cloud_range=[-30.0, -30.0, -2.0, 30.0, 30.0, 3.0],
        voxel_size=(0.5, 0.5, 5.0),  # Larger voxels for faster test
        max_num_voxels=10000,
        backbone_out_channels=64,
    )
    
    # Create dummy points
    batch_size = 2
    num_points = 3000
    
    points = torch.zeros(num_points, 5)
    points[:, 0] = torch.randint(0, batch_size, (num_points,)).float()
    points[:, 1] = torch.rand(num_points) * 60 - 30
    points[:, 2] = torch.rand(num_points) * 60 - 30
    points[:, 3] = torch.rand(num_points) * 5 - 2
    points[:, 4] = torch.rand(num_points)
    
    # Forward
    bev_features = encoder(points, batch_size)
    
    # Create dummy loss
    loss = bev_features.mean()
    
    # Backward
    loss.backward()
    
    # Check gradients
    has_grad = False
    grad_info = []
    for name, param in encoder.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            if grad_norm > 0:
                has_grad = True
                grad_info.append(f"    {name}: grad_norm = {grad_norm:.6f}")
    
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Gradients computed: {has_grad}")
    if grad_info:
        print("  Sample gradients:")
        for info in grad_info[:5]:  # Show first 5
            print(info)
    
    assert has_grad, "No gradients computed!"
    print("  ✓ Gradient flow test passed!")
    print()


def test_training_step():
    """Test a complete training step"""
    print("=" * 50)
    print("Testing Training Step...")
    print("=" * 50)
    
    from models.lidar_bev.pointpillars import LiDARBEVEncoder
    
    # Simple segmentation head for testing
    class SimpleSegHead(nn.Module):
        def __init__(self, in_channels, num_classes):
            super().__init__()
            self.conv = nn.Conv2d(in_channels, num_classes, 1)
        
        def forward(self, x):
            return self.conv(x)
    
    # Build mini network
    encoder = LiDARBEVEncoder(
        point_cloud_range=[-30.0, -30.0, -2.0, 30.0, 30.0, 3.0],
        voxel_size=(0.5, 0.5, 5.0),
        max_num_voxels=10000,
        backbone_out_channels=64,
    )
    seg_head = SimpleSegHead(64, 6)
    
    # Optimizer
    params = list(encoder.parameters()) + list(seg_head.parameters())
    optimizer = torch.optim.Adam(params, lr=1e-4)
    
    # Create dummy data
    batch_size = 2
    num_points = 3000
    
    points = torch.zeros(num_points, 5)
    points[:, 0] = torch.randint(0, batch_size, (num_points,)).float()
    points[:, 1] = torch.rand(num_points) * 60 - 30
    points[:, 2] = torch.rand(num_points) * 60 - 30
    points[:, 3] = torch.rand(num_points) * 5 - 2
    points[:, 4] = torch.rand(num_points)
    
    # Forward
    bev_features = encoder(points, batch_size)
    seg_output = seg_head(bev_features)
    
    # Create dummy target
    H, W = seg_output.shape[2:]
    target = torch.randint(0, 6, (batch_size, H, W))
    
    # Loss
    loss = nn.functional.cross_entropy(seg_output, target)
    
    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"  BEV features shape: {bev_features.shape}")
    print(f"  Segmentation output shape: {seg_output.shape}")
    print(f"  Loss: {loss.item():.4f}")
    print("  ✓ Training step completed!")
    print()


def count_parameters():
    """Count model parameters"""
    print("=" * 50)
    print("Model Parameters Count...")
    print("=" * 50)
    
    from models.lidar_bev.pointpillars import LiDARBEVEncoder, PillarFeatureNet, LiDARBackbone
    
    # PillarFeatureNet
    pfn = PillarFeatureNet(in_channels=4, feat_channels=64)
    pfn_params = sum(p.numel() for p in pfn.parameters())
    
    # Backbone
    backbone = LiDARBackbone(in_channels=64, out_channels=256)
    backbone_params = sum(p.numel() for p in backbone.parameters())
    
    # Full encoder
    encoder = LiDARBEVEncoder(backbone_out_channels=256)
    total_params = sum(p.numel() for p in encoder.parameters())
    trainable_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    
    print(f"  PillarFeatureNet: {pfn_params:,} params")
    print(f"  LiDARBackbone: {backbone_params:,} params")
    print(f"  Total: {total_params:,} params")
    print(f"  Trainable: {trainable_params:,} params")
    print()


def run_all_tests():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("LiDAR BEV Encoder Test Suite")
    print("=" * 60 + "\n")
    
    try:
        test_pillar_feature_net()
        test_scatter()
        test_backbone()
        test_voxelization()
        test_lidar_bev_encoder()
        test_marine_encoder()
        test_gradient_flow()
        test_training_step()
        count_parameters()
        
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
