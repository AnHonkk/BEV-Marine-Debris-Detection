"""
Training Script for BEV Fusion Network
해양 부유쓰레기 탐지 네트워크 학습
"""

import os
import sys
import argparse
import yaml
import logging
from datetime import datetime
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import BEVFusionNetwork, build_network
from training.losses import MultiTaskLoss


def setup_logging(log_dir: str) -> logging.Logger:
    """Setup logging"""
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger('BEVFusion')
    logger.setLevel(logging.INFO)
    
    # File handler
    fh = logging.FileHandler(os.path.join(log_dir, 'train.log'))
    fh.setLevel(logging.INFO)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger


class Trainer:
    """Training class for BEV Fusion Network"""
    
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
            num_classes=config.get('num_classes', 6),
            class_weights=loss_config.get('class_weights'),
            semantic_weight=loss_config.get('semantic_weight', 1.0),
            dice_weight=loss_config.get('dice_weight', 1.0),
            boundary_weight=loss_config.get('boundary_weight', 0.5),
            use_instance_loss=loss_config.get('use_instance_loss', False),
        )
        
        # Optimizer
        opt_config = config.get('optimizer', {})
        self.optimizer = self._build_optimizer(opt_config)
        
        # Scheduler
        sched_config = config.get('scheduler', {})
        self.scheduler = self._build_scheduler(sched_config)
        
        # Training settings
        self.num_epochs = config.get('num_epochs', 100)
        self.save_interval = config.get('save_interval', 10)
        self.val_interval = config.get('val_interval', 1)
        self.log_interval = config.get('log_interval', 10)
        
        # Directories
        self.output_dir = config.get('output_dir', './outputs')
        self.checkpoint_dir = os.path.join(self.output_dir, 'checkpoints')
        self.log_dir = os.path.join(self.output_dir, 'logs')
        
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Logger
        self.logger = setup_logging(self.log_dir)
        
        # TensorBoard
        self.writer = SummaryWriter(self.log_dir)
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_miou = 0.0
        
    def _build_optimizer(self, config: Dict) -> optim.Optimizer:
        """Build optimizer"""
        opt_type = config.get('type', 'AdamW')
        lr = config.get('lr', 1e-4)
        weight_decay = config.get('weight_decay', 0.01)
        
        if opt_type == 'AdamW':
            return optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif opt_type == 'SGD':
            momentum = config.get('momentum', 0.9)
            return optim.SGD(self.model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        elif opt_type == 'Adam':
            return optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {opt_type}")
    
    def _build_scheduler(self, config: Dict) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Build learning rate scheduler"""
        sched_type = config.get('type', 'CosineAnnealing')
        
        if sched_type == 'CosineAnnealing':
            T_max = config.get('T_max', self.num_epochs)
            eta_min = config.get('eta_min', 1e-6)
            return optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=T_max, eta_min=eta_min)
        elif sched_type == 'StepLR':
            step_size = config.get('step_size', 30)
            gamma = config.get('gamma', 0.1)
            return optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)
        elif sched_type == 'OneCycleLR':
            max_lr = config.get('max_lr', self.config.get('optimizer', {}).get('lr', 1e-4))
            total_steps = len(self.train_loader) * self.num_epochs
            return optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=max_lr, total_steps=total_steps)
        elif sched_type == 'none':
            return None
        else:
            raise ValueError(f"Unknown scheduler: {sched_type}")
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        epoch_losses = {}
        num_batches = len(self.train_loader)
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move data to device
            camera_bev = batch['camera_bev'].to(self.device)
            points = batch['points'].to(self.device)
            targets = {k: v.to(self.device) for k, v in batch['targets'].items()}
            batch_size = camera_bev.shape[0]
            
            # Forward pass
            outputs = self.model(camera_bev, points, batch_size)
            
            # Compute loss
            losses = self.criterion(outputs, targets)
            
            # Backward pass
            self.optimizer.zero_grad()
            losses['total'].backward()
            
            # Gradient clipping
            max_grad_norm = self.config.get('max_grad_norm', 1.0)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
            
            self.optimizer.step()
            
            # Update scheduler if OneCycleLR
            if isinstance(self.scheduler, optim.lr_scheduler.OneCycleLR):
                self.scheduler.step()
            
            # Accumulate losses
            for k, v in losses.items():
                if k not in epoch_losses:
                    epoch_losses[k] = 0.0
                epoch_losses[k] += v.item()
            
            # Logging
            if batch_idx % self.log_interval == 0:
                self.logger.info(
                    f"Epoch [{self.current_epoch}/{self.num_epochs}] "
                    f"Batch [{batch_idx}/{num_batches}] "
                    f"Loss: {losses['total'].item():.4f}"
                )
            
            # TensorBoard logging
            self.writer.add_scalar('train/loss', losses['total'].item(), self.global_step)
            for k, v in losses.items():
                if k != 'total':
                    self.writer.add_scalar(f'train/loss_{k}', v.item(), self.global_step)
            
            self.global_step += 1
        
        # Average losses
        for k in epoch_losses:
            epoch_losses[k] /= num_batches
        
        return epoch_losses
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate model"""
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        val_losses = {}
        num_batches = len(self.val_loader)
        
        # Metrics
        total_correct = 0
        total_pixels = 0
        intersection = torch.zeros(self.config.get('num_classes', 6), device=self.device)
        union = torch.zeros(self.config.get('num_classes', 6), device=self.device)
        
        for batch_idx, batch in enumerate(self.val_loader):
            camera_bev = batch['camera_bev'].to(self.device)
            points = batch['points'].to(self.device)
            targets = {k: v.to(self.device) for k, v in batch['targets'].items()}
            batch_size = camera_bev.shape[0]
            
            # Forward pass
            outputs = self.model(camera_bev, points, batch_size)
            
            # Compute loss
            losses = self.criterion(outputs, targets)
            
            # Accumulate losses
            for k, v in losses.items():
                if k not in val_losses:
                    val_losses[k] = 0.0
                val_losses[k] += v.item()
            
            # Compute metrics
            pred = outputs['semantic'].argmax(dim=1)
            target = targets['semantic']
            
            # Accuracy
            total_correct += (pred == target).sum().item()
            total_pixels += target.numel()
            
            # IoU
            for c in range(self.config.get('num_classes', 6)):
                pred_c = (pred == c)
                target_c = (target == c)
                intersection[c] += (pred_c & target_c).sum()
                union[c] += (pred_c | target_c).sum()
        
        # Average losses
        for k in val_losses:
            val_losses[k] /= num_batches
        
        # Compute metrics
        accuracy = total_correct / total_pixels
        iou = intersection / (union + 1e-6)
        miou = iou.mean().item()
        
        val_losses['accuracy'] = accuracy
        val_losses['miou'] = miou
        
        # TensorBoard logging
        self.writer.add_scalar('val/loss', val_losses['total'], self.current_epoch)
        self.writer.add_scalar('val/accuracy', accuracy, self.current_epoch)
        self.writer.add_scalar('val/miou', miou, self.current_epoch)
        
        return val_losses
    
    def save_checkpoint(self, filename: str, is_best: bool = False):
        """Save checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_miou': self.best_val_miou,
            'config': self.config,
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save(checkpoint, path)
        self.logger.info(f"Checkpoint saved to {path}")
        
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best.pth')
            torch.save(checkpoint, best_path)
            self.logger.info(f"Best model saved to {best_path}")
    
    def load_checkpoint(self, path: str):
        """Load checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_miou = checkpoint.get('best_val_miou', 0.0)
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.logger.info(f"Checkpoint loaded from {path}")
    
    def train(self):
        """Full training loop"""
        self.logger.info("Starting training...")
        self.logger.info(f"Config: {self.config}")
        
        for epoch in range(self.current_epoch, self.num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_losses = self.train_epoch()
            self.logger.info(
                f"Epoch [{epoch}/{self.num_epochs}] "
                f"Train Loss: {train_losses['total']:.4f}"
            )
            
            # Update scheduler (except OneCycleLR)
            if self.scheduler is not None and not isinstance(self.scheduler, optim.lr_scheduler.OneCycleLR):
                self.scheduler.step()
            
            # Validate
            if self.val_loader is not None and epoch % self.val_interval == 0:
                val_losses = self.validate()
                self.logger.info(
                    f"Epoch [{epoch}/{self.num_epochs}] "
                    f"Val Loss: {val_losses['total']:.4f} "
                    f"mIoU: {val_losses['miou']:.4f}"
                )
                
                # Save best model
                if val_losses['miou'] > self.best_val_miou:
                    self.best_val_miou = val_losses['miou']
                    self.save_checkpoint('best.pth', is_best=True)
            
            # Save checkpoint
            if epoch % self.save_interval == 0:
                self.save_checkpoint(f'epoch_{epoch}.pth')
        
        # Save final model
        self.save_checkpoint('final.pth')
        self.logger.info("Training completed!")
        
        self.writer.close()


def main():
    parser = argparse.ArgumentParser(description='Train BEV Fusion Network')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Build model
    model = build_network(config.get('model', {}))
    
    # Create dummy data loaders for demonstration
    # In practice, replace with actual data loaders
    from torch.utils.data import TensorDataset
    
    # Dummy data
    B, C, H, W = 2, 256, 200, 200
    num_points = 10000
    num_classes = config.get('model', {}).get('num_classes', 6)
    
    # Create dummy dataset
    camera_bev = torch.randn(10, C, H, W)
    points = torch.randn(10, num_points, 5)  # (batch_idx, x, y, z, intensity)
    semantic_targets = torch.randint(0, num_classes, (10, H, W))
    boundary_targets = torch.randint(0, 2, (10, H, W)).float()
    
    # Note: This is a simplified example. Real implementation needs proper data loading
    print("Note: Using dummy data for demonstration. Replace with actual data loaders.")
    
    # Create trainer
    # trainer = Trainer(
    #     config=config,
    #     model=model,
    #     train_loader=train_loader,
    #     val_loader=val_loader,
    #     device=args.device,
    # )
    
    # Resume from checkpoint if provided
    # if args.resume:
    #     trainer.load_checkpoint(args.resume)
    
    # Start training
    # trainer.train()
    
    print("Training script ready. Provide actual data loaders to start training.")


if __name__ == '__main__':
    main()
