import torch
import cv2
import numpy as np
import os
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from models.camera_bev.depth_net_unet import LightUNetDepthNet

# ==========================================
# 1. 설정
# ==========================================
class Config:
    INPUT_SIZE = (368, 640)
    DEPTH_RANGE = (1.0, 20.0) # 학습 시 설정한 범위
    DEPTH_BINS = 64
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 학습된 가중치 경로
    CHECKPOINT_PATH = "checkpoints/depth_net/best_depth_net.pth"
    
    # 테스트 이미지 폴더
    TEST_IMG_DIR = "dataset/depth_gt/images"
    
    # 결과 저장 폴더
    RESULT_DIR = "results_comparison"

# ==========================================
# 2. 모델 로드 함수
# ==========================================
def get_model(load_weights=False):
    # 모델 구조 초기화 (Random Weights)
    model = LightUNetDepthNet(
        in_channels=3,
        depth_bins=Config.DEPTH_BINS,
        depth_range=Config.DEPTH_RANGE
    ).to(Config.DEVICE)
    
    if load_weights:
        if os.path.exists(Config.CHECKPOINT_PATH):
            checkpoint = torch.load(Config.CHECKPOINT_PATH, map_location=Config.DEVICE)
            model.load_state_dict(checkpoint)
            print(">>> [Trained Model] Weights loaded successfully.")
        else:
            print(f">>> [Error] Checkpoint not found: {Config.CHECKPOINT_PATH}")
            return None
    else:
        print(">>> [Untrained Model] Initialized with random weights.")
        
    model.eval()
    return model

# ==========================================
# 3. 전처리 및 시각화
# ==========================================
def preprocess_image(img_path):
    original_img = cv2.imread(img_path)
    if original_img is None: return None, None
    
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(original_img, (Config.INPUT_SIZE[1], Config.INPUT_SIZE[0]))
    
    # 정규화
    img = img.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    
    img_tensor = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0)
    return img_tensor.to(Config.DEVICE), original_img

def save_comparison(original_img, depth_untrained, depth_trained, save_path):
    # Depth Map을 numpy로 변환 (H, W)
    d_untrained = depth_untrained.squeeze().cpu().numpy()
    d_trained = depth_trained.squeeze().cpu().numpy()
    
    plt.figure(figsize=(18, 5))
    
    # 1. 원본 이미지
    plt.subplot(1, 3, 1)
    plt.title("Input Image")
    plt.imshow(original_img)
    plt.axis('off')
    
    # 2. 학습 안 한 결과 (Untrained)
    plt.subplot(1, 3, 2)
    plt.title("Untrained")
    plt.imshow(d_untrained, cmap='plasma', vmin=Config.DEPTH_RANGE[0], vmax=Config.DEPTH_RANGE[1])
    plt.colorbar(label='Depth (m)')
    plt.axis('off')
    
    # 3. 학습된 결과 (Trained)
    plt.subplot(1, 3, 3)
    plt.title("Trained")
    plt.imshow(d_trained, cmap='plasma', vmin=Config.DEPTH_RANGE[0], vmax=Config.DEPTH_RANGE[1])
    plt.colorbar(label='Depth (m)')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")

# ==========================================
# 4. 메인 실행
# ==========================================
def main():
    os.makedirs(Config.RESULT_DIR, exist_ok=True)
    
    # 두 모델 준비
    model_untrained = get_model(load_weights=False) # 학습 X
    model_trained = get_model(load_weights=True)    # 학습 O
    
    if model_trained is None: return

    # 이미지 목록
    image_files = sorted(glob.glob(os.path.join(Config.TEST_IMG_DIR, "*.jpg")))
    if not image_files:
        print("No images found.")
        return

    print(f"Processing {len(image_files)} images (saving first 5)...")
    
    with torch.no_grad():
        for i, img_path in enumerate(image_files):
            img_tensor, original_img = preprocess_image(img_path)
            if img_tensor is None: continue
            
            # 추론 (Untrained)
            out_u = model_untrained(img_tensor)
            depth_u = out_u['depth_expected']
            
            # 추론 (Trained)
            out_t = model_trained(img_tensor)
            depth_t = out_t['depth_expected']
            
            # 비교 저장
            save_name = os.path.join(Config.RESULT_DIR, f"compare_{os.path.basename(img_path)}")
            save_comparison(original_img, depth_u, depth_t, save_name)

if __name__ == "__main__":
    main()