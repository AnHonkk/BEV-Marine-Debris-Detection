import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import os
import glob
from tqdm import tqdm

# 모델 클래스 Import
from models.camera_bev.depth_net_unet import LightUNetDepthNet 

# ==========================================
# 1. Dataset 정의
# ==========================================
class DepthDataset(Dataset):
    def __init__(self, data_root, input_size=(368, 640), mode='train'):
        self.data_root = data_root
        self.input_size = input_size # (H, W)
        
        # 이미지와 Depth 파일 경로 매칭
        self.image_files = sorted(glob.glob(os.path.join(data_root, "images", "*.jpg")))
        self.depth_files = sorted(glob.glob(os.path.join(data_root, "depths", "*.npy")))
        
        # 데이터 매칭 확인
        if len(self.image_files) != len(self.depth_files):
            min_len = min(len(self.image_files), len(self.depth_files))
            self.image_files = self.image_files[:min_len]
            self.depth_files = self.depth_files[:min_len]
            print(f"[WARNING] File count mismatch. Truncated to {min_len} samples.")

        # Train/Val Split (8:2)
        split_idx = int(len(self.image_files) * 0.8)
        if mode == 'train':
            self.image_files = self.image_files[:split_idx]
            self.depth_files = self.depth_files[:split_idx]
        else:
            self.image_files = self.image_files[split_idx:]
            self.depth_files = self.depth_files[split_idx:]
            
        print(f"[{mode.upper()}] Dataset loaded: {len(self.image_files)} samples")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # 1. Load Image
        img_path = self.image_files[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize
        img = cv2.resize(img, (self.input_size[1], self.input_size[0]))
        
        # Normalize (0~1)
        # 여기서 numpy 연산 중 float64로 변할 수 있음
        img = img.astype(np.float32) / 255.0
        
        # Mean/Std Normalization
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        
        # [수정] Tensor 변환 시 .float() 강제 적용하여 Float32 보장
        img_tensor = torch.from_numpy(img).float().permute(2, 0, 1)

        # 2. Load Sparse Depth GT
        depth_path = self.depth_files[idx]
        depth_gt = np.load(depth_path).astype(np.float32)
        
        # Resize
        if depth_gt.shape[:2] != self.input_size:
            depth_gt = cv2.resize(depth_gt, (self.input_size[1], self.input_size[0]), interpolation=cv2.INTER_NEAREST)
        
        # Depth도 float32 보장
        depth_tensor = torch.from_numpy(depth_gt).float().unsqueeze(0) # (1, H, W)

        return img_tensor, depth_tensor

# ==========================================
# 2. Masked Loss Function
# ==========================================
class MaskedL1Loss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, pred, target):
        mask = (target > 0)
        
        if mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
            
        loss = torch.abs(pred[mask] - target[mask])
        return loss.mean()

# ==========================================
# 3. Training Loop
# ==========================================
def train():
    # 설정
    DATA_ROOT = "/home/anhong/BEVFusion/dataset/depth_gt" 
    BATCH_SIZE = 2
    EPOCHS = 50
    LR = 1e-4
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    SAVE_DIR = "checkpoints/depth_net"
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 데이터셋 로드
    train_dataset = DepthDataset(DATA_ROOT, mode='train')
    val_dataset = DepthDataset(DATA_ROOT, mode='val')
    
    if len(train_dataset) == 0:
        print("No training data found! Check DATA_ROOT.")
        return

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # 모델 초기화
    model = LightUNetDepthNet(
        in_channels=3, 
        depth_bins=64, 
        depth_range=(0.5, 20.0)
    ).to(DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = MaskedL1Loss()

    print(f"Start training on {DEVICE}...")

    best_loss = float('inf')

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for imgs, depths_gt in loop:
            imgs = imgs.to(DEVICE)
            depths_gt = depths_gt.to(DEVICE)

            # Forward
            outputs = model(imgs) 
            depth_pred = outputs['depth_expected'] # [B, 1, H, W]
            
            # Loss Calculation
            loss = criterion(depth_pred, depths_gt)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, depths_gt in val_loader:
                imgs = imgs.to(DEVICE)
                depths_gt = depths_gt.to(DEVICE)
                
                outputs = model(imgs)
                depth_pred = outputs['depth_expected']
                
                loss = criterion(depth_pred, depths_gt)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0.0
        
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # Save Best Model
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_depth_net.pth"))
            print(">>> Best Model Saved!")
            
        # Save Last Model
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, "last_depth_net.pth"))

if __name__ == "__main__":
    train()