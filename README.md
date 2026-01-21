# BEV Fusion - Marine Debris Detection

해양 부유쓰레기 탐지를 위한 BEV Fusion 네트워크

## 프로젝트 구조

```
BEV-Marine-Debris-Detection/
├── models/
│   ├── __init__.py
│   ├── full_network.py          # 통합 네트워크
│   ├── lidar_bev/               # LiDAR BEV 모듈
│   │   ├── __init__.py
│   │   └── pointpillars.py      # PointPillars 기반 LiDAR 인코더
│   ├── fusion/                  # Fusion 모듈
│   │   ├── __init__.py
│   │   └── bev_fusion.py        # 다양한 Fusion 방법
│   └── segmentation_head/       # Segmentation Head
│       ├── __init__.py
│       └── seg_head.py          # BEV Segmentation Head
├── training/
│   ├── __init__.py
│   ├── losses.py                # Loss 함수들
│   └── train.py                 # 학습 스크립트
├── configs/
│   └── train_config.yaml        # 학습 설정
├── tests/
│   └── test_modules.py          # 모듈 테스트
└── README.md
```

## 아키텍처

```
┌─────────────────┐     ┌─────────────────┐
│   Camera Input  │     │   LiDAR Input   │
│   (Images)      │     │  (Point Cloud)  │
└────────┬────────┘     └────────┬────────┘
         │                       │
         ▼                       ▼
┌─────────────────┐     ┌─────────────────┐
│  Camera BEV     │     │  LiDAR BEV      │
│    Encoder      │     │  Encoder        │
│    [ LSS ]      │     │  (PointPillars) │
└────────┬────────┘     └────────┬────────┘
         │                       │
         ▼                       ▼
┌─────────────────┐     ┌─────────────────┐
│  Camera BEV     │     │  LiDAR BEV      │
│  Feature Map    │     │  Feature Map    │
│  (B, 256, H, W) │     │  (B, 256, H, W) │
└────────┬────────┘     └────────┬────────┘
         │                       │
         └──────────┬────────────┘
                    │
                    ▼
         ┌─────────────────────┐
         │    BEV Fusion       │
         │  (Adaptive/Attn)    │
         └──────────┬──────────┘
                    │
                    ▼
         ┌─────────────────────┐
         │  Fused BEV Feature  │
         │  (B, 256, H, W)     │
         └──────────┬──────────┘
                    │
                    ▼
         ┌─────────────────────┐
         │  Segmentation Head  │
         │  (ASPP + Decoder)   │
         └──────────┬──────────┘
                    │
                    ▼
         ┌─────────────────────┐
         │  BEV Segmentation   │
         │  Map (B, 6, H, W)   │
         └─────────────────────┘
```

## 주요 모듈

### 1. LiDAR BEV Encoder (`models/lidar_bev/`)
- **PointPillars 기반**: 포인트 클라우드를 pillar로 변환 후 2D CNN 처리
- **MarineLiDARBEVEncoder**: 해양 환경에 최적화 (수면 반사 필터링)

```python
from models.lidar_bev import MarineLiDARBEVEncoder

encoder = MarineLiDARBEVEncoder(
    point_cloud_range=[-30.0, -30.0, -2.0, 30.0, 30.0, 3.0],
    voxel_size=(0.3, 0.3, 5.0),
    backbone_out_channels=256,
)
# Input: points (N, 5) - (batch_idx, x, y, z, intensity)
# Output: bev_features (B, 256, H, W)
```

### 2. BEV Fusion (`models/fusion/`)
- **ConvFuser**: 단순 Concatenation + Conv
- **ChannelAttentionFuser**: 채널 어텐션 기반
- **SpatialAttentionFuser**: 공간 어텐션 기반
- **CrossAttentionFuser**: Transformer 스타일
- **AdaptiveFuser**: 입력 조건에 따른 적응형 융합

```python
from models.fusion import BEVFusion

fusion = BEVFusion(
    camera_channels=256,
    lidar_channels=256,
    out_channels=256,
    fusion_method='adaptive',  # 'conv', 'channel_attn', 'spatial_attn', 'cross_attn', 'adaptive'
)
# Output: fused_bev (B, 256, H, W)
```

### 3. Segmentation Head (`models/segmentation_head/`)
- **MarineDebrisSegHead**: 해양 쓰레기 탐지 전용
  - Semantic Segmentation (6 classes)
  - Boundary Detection
  - Instance Center/Offset (optional)
  - Size Estimation

```python
from models.segmentation_head import MarineDebrisSegHead

head = MarineDebrisSegHead(
    in_channels=256,
    num_classes=6,  # Background, Plastic, Wood, Styrofoam, Rope/Net, Other
    use_instance_head=True,
    use_boundary_head=True,
)
# Output: dict with semantic, boundary, center, offset, size
```

### 4. 통합 네트워크 (`models/full_network.py`)

```python
from models import BEVFusionNetwork

network = BEVFusionNetwork(
    camera_bev_channels=256,
    lidar_bev_channels=256,
    fusion_method='adaptive',
    num_classes=6,
    bev_size=(200, 200),
)

# Forward
outputs = network(camera_bev, points, batch_size)
seg_map = network.get_seg_map(outputs)
```

## 클래스 정의

| Index | Class Name | Description |
|-------|------------|-------------|
| 0 | Background | 배경/수면 |
| 1 | Plastic | 플라스틱 쓰레기 |
| 2 | Wood | 나무/목재 |
| 3 | Styrofoam | 스티로폼 |
| 4 | Rope/Net | 로프/그물 |
| 5 | Other | 기타 쓰레기 |

## 설치

```bash
pip install -r requirements.txt
```

## 학습

```bash
python training/train.py --config configs/train_config.yaml
```

## 테스트

```bash
python tests/test_modules.py
```

## Loss Functions

- **Focal Loss**: 클래스 불균형 처리
- **Dice Loss**: IoU 최적화
- **Boundary Loss**: 경계 검출 향상
- **Instance Loss**: 인스턴스 분할 (optional)

## 설정

`configs/train_config.yaml`에서 설정 변경 가능:
- Model architecture
- Fusion method
- Loss weights
- Optimizer settings
- Learning rate scheduler

## Fusion Methods 비교

| Method | Description | Pros | Cons |
|--------|-------------|------|------|
| conv | Concat + Conv | 단순, 빠름 | 제한된 융합 |
| channel_attn | Channel attention | 적응형 가중치 | 공간 정보 무시 |
| spatial_attn | Spatial attention | 위치별 가중치 | 계산량 증가 |
| cross_attn | Cross attention | 풍부한 상호작용 | 메모리 사용량 높음 |
| adaptive | 적응형 융합 | 균형잡힌 성능 | 복잡한 구조 |

## TODO

- [ ] 실제 데이터셋 연동
- [ ] Data augmentation 구현
- [ ] Inference 스크립트
- [ ] Visualization 도구
- [ ] Model export (ONNX, TensorRT)

## References

- [PointPillars](https://arxiv.org/abs/1812.05784)
- [BEVFusion](https://arxiv.org/abs/2205.13542)
- [LSS (Lift, Splat, Shoot)](https://arxiv.org/abs/2008.05711)
