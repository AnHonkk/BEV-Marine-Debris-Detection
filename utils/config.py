# utils/config.py

import yaml
from pathlib import Path
import numpy as np


def load_config(config_path: str = 'configs/camera_bev_config.yaml'):
    """Load configuration from YAML file"""
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Add calibration matrices
    img_h, img_w = config['data']['img_size']
    
    config['model']['K_init'] = np.array([
        [1000.0, 0, img_w / 2],
        [0, 1000.0, img_h / 2],
        [0, 0, 1.0]
    ], dtype=np.float32)
    
    config['model']['T_init'] = np.eye(4, dtype=np.float32)
    config['model']['T_init'][:3, 3] = [0.1, -0.05, 0.2]
    
    return config


def get_model_config(backbone='convnext_tiny'):
    """Get model configuration"""
    try:
        config = load_config()
        config['model']['backbone'] = backbone
        return config['model']
    except FileNotFoundError:
        # Fallback to default config
        print("⚠️ Config file not found, using default config")
        return get_convnext_tiny_config()


# Preset configs
def get_convnext_tiny_config(img_size=(384, 640)):
    """ConvNeXt-Tiny preset config"""
    return {
        'backbone': 'convnext_tiny',
        'pretrained': True,
        'feat_channels': 256,
        'depth_bins': 64,
        'depth_range': (1.0, 50.0),
        'img_size': list(img_size),
        'bev_size': [256, 256],
        'bev_range': [-25.6, 25.6, -25.6, 25.6],
        'K_init': np.array([
            [1000.0, 0, img_size[1]/2],
            [0, 1000.0, img_size[0]/2],
            [0, 0, 1.0]
        ], dtype=np.float32),
        'T_init': np.eye(4, dtype=np.float32),
        'learn_intrinsic': True,
        'learn_extrinsic': True,
        'use_fpn': True
    }


def get_convnext_small_config(img_size=(384, 640)):
    """ConvNeXt-Small preset config"""
    config = get_convnext_tiny_config(img_size)
    config['backbone'] = 'convnext_small'
    return config


# Backward compatibility (Swin 제거)
def get_swin_config(*args, **kwargs):
    """Deprecated: Swin not supported, use ConvNeXt instead"""
    print("⚠️ Swin is not supported. Using ConvNeXt-Tiny instead.")
    return get_convnext_tiny_config(*args, **kwargs)