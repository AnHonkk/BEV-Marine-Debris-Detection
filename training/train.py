import sys
import os

# 경로 추가 (가장 먼저 실행되어야 함)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml
import argparse
from tqdm import tqdm
from typing import Dict, Optional
import time
import logging

# 모듈 임포트
from models import build_network
from training.dataset import MarineDebrisDataset, custom_collate_fn
from training.losses import MultiTaskLoss
from utils.logger import setup_logging

class Trainer:
    def __init__(
        self,
        config: Dict,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        device: str = 'cuda',
    ):
        self.config = config
        self.device = device
        
        # Model
        self.model = model.to(device)
        
        # Data loaders
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Loss function
        loss_config = config.get('loss', {})
        self.criterion = MultiTaskLoss(
            num_classes=config.get('model', {}).get('num_classes', 3),
            class_weights=loss_config.get('class_weights'),
            semantic_weight=loss_config.get('semantic_weight', 1.0),
            dice_weight=loss_config.get('dice_weight', 1.0),
            boundary_weight=loss_config.get('boundary_weight', 0.5),
            use_instance_loss=loss_config.get('use_instance_loss', False),
        )
        
        # [수정] 손실 함수(와 그 내부의 가중치)를 GPU로 이동
        self.criterion = self.criterion.to(self.device)

        # Training settings
        self.num_epochs = config.get('num_epochs', 100)
        self.save_interval = config.get('save_interval', 10)
        self.val_interval = config.get('val_interval', 1)
        self.log_interval = config.get('log_interval', 10)
        
        # Optimizer
        opt_config = config.get('optimizer', {})
        self.optimizer = self._build_optimizer(opt_config)
        
        # Scheduler
        sched_config = config.get('scheduler', {})
        self.scheduler = self._build_scheduler(sched_config)
        
        # Directories
        self.output_dir = config.get('output_dir', './outputs')
        self.checkpoint_dir = os.path.join(self.output_dir, 'checkpoints')
        self.log_dir = os.path.join(self.output_dir, 'logs')
        
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Logger
        self.logger = setup_logging(self.log_dir)
        self.writer = SummaryWriter(self.log_dir)
        
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_miou = 0.0
        
    def _build_optimizer(self, config):
        lr = config.get('lr', 0.0001)
        weight_decay = config.get('weight_decay', 0.01)
        return optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
    
    def _build_scheduler(self, config):
        T_max = config.get('T_max', self.num_epochs)
        eta_min = float(config.get('eta_min', 1e-6))
        return optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=T_max, eta_min=eta_min
        )

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch+1}/{self.num_epochs}")
        
        for batch_idx, batch in enumerate(pbar):
            self.optimizer.zero_grad()
            
            # Targets 이동
            targets_dict = {k: v.to(self.device) for k, v in batch['targets'].items()}
            
            points = batch['points'].to(self.device)
            batch_size = len(targets_dict['semantic'])
            
            # 입력 데이터 처리
            if 'images' in batch:
                images = batch['images'].to(self.device)
                outputs = self.model(images, points, batch_size)
            elif 'camera_bev' in batch:
                camera_bev = batch['camera_bev'].to(self.device)
                outputs = self.model(camera_bev, points, batch_size)
            else:
                bev_h, bev_w = self.config['model'].get('camera_bev_size', (200, 200))
                dummy_bev = torch.zeros((batch_size, 256, bev_h, bev_w)).to(self.device)
                outputs = self.model(dummy_bev, points, batch_size)

            # Loss 계산
            loss, loss_dict = self.criterion(outputs, targets_dict)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            self.global_step += 1
            
            # Logging
            if batch_idx % self.log_interval == 0:
                self.writer.add_scalar('Train/Loss', loss.item(), self.global_step)
                for k, v in loss_dict.items():
                    if isinstance(v, torch.Tensor):
                        self.writer.add_scalar(f'Train/{k}', v.item(), self.global_step)
                    else:
                        self.writer.add_scalar(f'Train/{k}', v, self.global_step)
                
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        if self.scheduler:
            self.scheduler.step()
            
        return total_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        total_loss = 0
        total_intersection = 0
        total_union = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                targets_dict = {k: v.to(self.device) for k, v in batch['targets'].items()}
                points = batch['points'].to(self.device)
                batch_size = len(targets_dict['semantic'])
                
                if 'images' in batch:
                    images = batch['images'].to(self.device)
                    outputs = self.model(images, points, batch_size)
                elif 'camera_bev' in batch:
                    camera_bev = batch['camera_bev'].to(self.device)
                    outputs = self.model(camera_bev, points, batch_size)
                else:
                    bev_h, bev_w = self.config['model'].get('camera_bev_size', (200, 200))
                    dummy_bev = torch.zeros((batch_size, 256, bev_h, bev_w)).to(self.device)
                    outputs = self.model(dummy_bev, points, batch_size)
                
                loss, _ = self.criterion(outputs, targets_dict)
                total_loss += loss.item()
                
                # mIoU calculation
                pred = outputs['semantic'].argmax(dim=1)
                targets = targets_dict['semantic']
                
                num_classes = self.criterion.num_classes
                for cls in range(num_classes):
                    pred_mask = (pred == cls)
                    target_mask = (targets == cls)
                    intersection = (pred_mask & target_mask).sum().item()
                    union = (pred_mask | target_mask).sum().item()
                    
                    if union > 0:
                        total_intersection += intersection
                        total_union += union
                        
        miou = total_intersection / (total_union + 1e-6)
        return total_loss / len(self.val_loader), miou

    def train(self):
        self.logger.info("Starting training...")
        self.logger.info(f"Config: {self.config}")
        
        for epoch in range(self.num_epochs):
            self.current_epoch = epoch
            
            train_loss = self.train_epoch()
            val_loss, val_miou = self.validate()
            
            self.logger.info(f"Epoch {epoch+1}/{self.num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val mIoU: {val_miou:.4f}")
            self.writer.add_scalar('Val/Loss', val_loss, epoch)
            self.writer.add_scalar('Val/mIoU', val_miou, epoch)
            
            if (epoch + 1) % self.save_interval == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pth')
                
            if val_miou > self.best_val_miou:
                self.best_val_miou = val_miou
                self.save_checkpoint('best_model.pth')
                self.logger.info(f"New best model saved with mIoU: {val_miou:.4f}")

    def save_checkpoint(self, filename):
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save({
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_miou': self.best_val_miou,
            'config': self.config,
        }, path)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_miou = checkpoint['best_val_miou']
        self.logger.info(f"Loaded checkpoint from {path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    model = build_network(config)
    
    print("Loading datasets...")
    data_cfg = config.get('data', {})
    model_cfg = config.get('model', {})
    
    train_dataset = MarineDebrisDataset(
        info_path=data_cfg['train_path'],
        bev_size=tuple(model_cfg.get('bev_size', [200, 200])),
        point_cloud_range=model_cfg['point_cloud_range'],
        training=True
    )
    
    val_dataset = MarineDebrisDataset(
        info_path=data_cfg['val_path'],
        bev_size=tuple(model_cfg.get('bev_size', [200, 200])),
        point_cloud_range=model_cfg['point_cloud_range'],
        training=False
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training'].get('batch_size', 4),
        shuffle=True,
        num_workers=4,
        collate_fn=custom_collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training'].get('batch_size', 4),
        shuffle=False,
        num_workers=4,
        collate_fn=custom_collate_fn
    )
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    trainer = Trainer(
        config=config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=args.device,
    )
    
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    trainer.train()

if __name__ == '__main__':
    main()