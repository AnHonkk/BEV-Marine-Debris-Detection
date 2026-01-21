import torch
import numpy as np
import cv2
import pickle
from pathlib import Path
from torch.utils.data import Dataset

class MarineDebrisDataset(Dataset):
    def __init__(self, info_path, bev_size=(400, 400), point_cloud_range=[-20, -20, -5, 20, 20, 3], training=True):
        with open(info_path, 'rb') as f:
            self.infos = pickle.load(f)
            
        self.bev_size = bev_size
        self.pc_range = point_cloud_range
        self.training = training
        
    def __len__(self):
        return len(self.infos)
    
    def __getitem__(self, idx):
        info = self.infos[idx]
        
        # 1. Point Cloud 로드
        pts_path = info['pts_path']
        points = np.fromfile(pts_path, dtype=np.float32).reshape(-1, 4)
        
        # 2. Camera Image 로드 [수정됨]
        cam_path = info['cam_path']
        image = cv2.imread(cam_path)
        if image is None:
            # 이미지가 없는 경우 (예외처리): 0으로 채움
            image = np.zeros((368, 640, 3), dtype=np.uint8)
        else:
            # 리사이즈 (모델 입력 크기에 맞춤, 예: 640x368)
            # config의 img_size와 일치시켜야 함
            image = cv2.resize(image, (640, 368))
            
        # 정규화 및 텐서 변환 (H, W, C) -> (C, H, W)
        image = image.astype(np.float32) / 255.0
        # ImageNet Mean/Std (일반적인 값)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image = (image - mean) / std
        
        image_tensor = torch.from_numpy(image).permute(2, 0, 1) # (3, H, W)
        
        # 3. Ground Truth 로드
        seg_path = info['seg_path']
        seg_map = np.load(seg_path).astype(np.int64)
        
        points_tensor = torch.from_numpy(points)
        target_tensor = torch.from_numpy(seg_map)
        
        return {
            'points': points_tensor,
            'images': image_tensor,     # Camera BEV 대신 Raw Image 전달
            'targets': {'semantic': target_tensor}
        }

def custom_collate_fn(batch):
    """Batching"""
    images = []
    points_list = []
    targets_list = []
    
    for i, item in enumerate(batch):
        images.append(item['images'])
        targets_list.append(item['targets']['semantic'])
        
        # Points Batch Indexing
        p = item['points']
        num_points = p.shape[0]
        p_with_batch_idx = torch.zeros((num_points, 5), dtype=torch.float32)
        p_with_batch_idx[:, 0] = i
        p_with_batch_idx[:, 1:] = p
        points_list.append(p_with_batch_idx)
    
    points_tensor = torch.cat(points_list, dim=0) 
    
    return {
        'images': torch.stack(images), # (B, 3, H, W)
        'points': points_tensor,
        'targets': {'semantic': torch.stack(targets_list)}
    }