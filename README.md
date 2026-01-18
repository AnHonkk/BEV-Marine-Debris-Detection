# BEV Fusion - Camera BEV Module

해양 부유쓰레기 탐지를 위한 BEV Fusion 카메라 모듈

## 프로젝트 구조
```
BEVFusion/
├── models/camera_bev/     # 카메라 BEV 변환 모듈
├── tests/                 # 테스트 코드
├── configs/               # 설정 파일
└── utils/                 # 유틸리티 함수
```

## 설치
```bash
pip install -r requirements.txt
```

## 빠른 테스트
```bash
python tests/quick_test.py
```

## 전체 테스트
```bash
bash tests/run_all_tests.sh
```

## 주요 기능

- ✅ LSS 기반 Camera → BEV 변환
- ✅ 학습 가능한 Calibration Refinement
- ✅ Depth Distribution 예측
- ✅ Multi-scale Feature 추출
# BEV-Marine-Debris-Detection
