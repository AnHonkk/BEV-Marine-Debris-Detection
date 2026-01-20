import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean', ignore_index=-100):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.alpha = alpha
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        """
        Args:
            inputs: (B, C, H, W) logits
            targets: (B, H, W) labels
        """
        # Cross Entropy Loss 계산 (reduction='none'으로 픽셀별 loss 구함)
        ce_loss = F.cross_entropy(
            inputs, 
            targets, 
            weight=self.alpha, 
            reduction='none', 
            ignore_index=self.ignore_index
        )
        
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation
    """
    def __init__(self, smooth=1.0, ignore_index=-100):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        # inputs: (B, C, H, W) logits
        num_classes = inputs.shape[1]
        inputs = F.softmax(inputs, dim=1)
        
        # targets: (B, H, W) labels -> One-hot encoding
        # ignore_index 처리: 유효한 영역만 마스킹
        valid_mask = (targets != self.ignore_index)
        targets = targets * valid_mask.long() # ignore 인덱스를 0으로 임시 변환
        
        targets_one_hot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()
        
        # ignore_index였던 픽셀은 마스크를 이용해 계산에서 제외하거나 0으로 처리
        # 간단한 구현을 위해 여기서는 전체 계산 후 평균
        
        intersection = (inputs * targets_one_hot).sum(dim=(2, 3))
        union = inputs.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        
        # 전체 배치 및 클래스에 대해 평균 (1 - Dice)
        return 1 - dice.mean()


class BoundaryLoss(nn.Module):
    """
    Binary Cross Entropy Loss for boundary detection
    """
    def __init__(self, pos_weight=None):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, pred, target):
        return self.bce(pred, target)


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss for Marine Debris Detection
    Combines Semantic (Focal + Dice), Boundary, and Instance losses
    """
    def __init__(self, num_classes=6, class_weights=None, 
                 semantic_weight=1.0, dice_weight=1.0, boundary_weight=0.5,
                 center_weight=1.0, offset_weight=0.2, size_weight=0.1,
                 use_instance_loss=False, focal_gamma=2.0, ignore_index=-100):
        super().__init__()
        
        if class_weights is not None:
            if not isinstance(class_weights, torch.Tensor):
                class_weights = torch.tensor(class_weights)
            # device는 forward 시 처리되도록 init에서는 to(device) 하지 않음 (또는 모델 device 따름)
            
        self.focal_loss = FocalLoss(alpha=class_weights, gamma=focal_gamma, ignore_index=ignore_index)
        self.dice_loss = DiceLoss(ignore_index=ignore_index)
        self.boundary_loss = BoundaryLoss()
        self.reg_loss = nn.L1Loss() # For regression (center, offset, size)
        
        self.semantic_weight = semantic_weight
        self.dice_weight = dice_weight
        self.boundary_weight = boundary_weight
        self.center_weight = center_weight
        self.offset_weight = offset_weight
        self.size_weight = size_weight
        self.use_instance_loss = use_instance_loss

    def forward(self, outputs, targets):
        """
        Args:
            outputs: Dictionary of model predictions
            targets: Dictionary of ground truth targets
        """
        losses = {}
        total_loss = 0.0
        
        # 1. Semantic Loss (Focal + Dice)
        if 'semantic' in outputs and 'semantic' in targets:
            # device 맞춰주기 (class_weights가 있다면)
            if self.focal_loss.alpha is not None:
                self.focal_loss.alpha = self.focal_loss.alpha.to(outputs['semantic'].device)

            focal = self.focal_loss(outputs['semantic'], targets['semantic'])
            dice = self.dice_loss(outputs['semantic'], targets['semantic'])
            
            losses['focal'] = focal
            losses['dice'] = dice
            
            sem_loss = (self.semantic_weight * focal) + (self.dice_weight * dice)
            losses['semantic'] = sem_loss
            total_loss += sem_loss
        
        # 2. Boundary Loss
        if 'boundary' in outputs and 'boundary' in targets:
            pred_boundary = outputs['boundary']
            target_boundary = targets['boundary']
            
            # 차원 맞추기: target이 (B, H, W)면 (B, 1, H, W)로
            if target_boundary.dim() == 3:
                target_boundary = target_boundary.unsqueeze(1)
                
            b_loss = self.boundary_loss(pred_boundary, target_boundary.float())
            losses['boundary'] = b_loss
            total_loss += self.boundary_weight * b_loss
            
        # 3. Instance Losses (Optional)
        if self.use_instance_loss:
            # Center (Heatmap) Loss
            if 'center' in outputs and 'center' in targets:
                c_loss = self.reg_loss(outputs['center'], targets['center'])
                losses['center'] = c_loss
                total_loss += self.center_weight * c_loss
            
            # Offset Loss
            if 'offset' in outputs and 'offset' in targets:
                o_loss = self.reg_loss(outputs['offset'], targets['offset'])
                losses['offset'] = o_loss
                total_loss += self.offset_weight * o_loss
                
            # Size Loss
            if 'size' in outputs and 'size' in targets:
                s_loss = self.reg_loss(outputs['size'], targets['size'])
                losses['size'] = s_loss
                total_loss += self.size_weight * s_loss

        losses['total'] = total_loss
        
        return losses