"""
Configuration settings for the medical imaging pipeline.
"""
import os
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from enum import Enum


class ModelType(Enum):
    RESNET50 = "resnet50"
    EFFICIENTNET_B4 = "efficientnet_b4"
    MULTIMODAL = "multimodal"


@dataclass
class DataConfig:
    """Data preprocessing configuration."""
    # Dataset paths
    data_root: str = "./data/chest_xray"
    train_dir: str = "train"
    val_dir: str = "val"
    test_dir: str = "test"
    
    # Image settings
    image_size: Tuple[int, int] = (224, 224)
    num_channels: int = 3
    
    # Normalization (ImageNet statistics)
    normalize_mean: Tuple[float, ...] = (0.485, 0.456, 0.406)
    normalize_std: Tuple[float, ...] = (0.229, 0.224, 0.225)
    
    # Data split ratios (if creating custom split)
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    
    # Class names
    class_names: List[str] = field(default_factory=lambda: ["NORMAL", "PNEUMONIA"])
    num_classes: int = 2
    
    # Data loading
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    
    # Augmentation settings
    augmentation_prob: float = 0.5
    rotation_degrees: int = 15
    horizontal_flip: bool = True
    vertical_flip: bool = False
    color_jitter: bool = True
    random_erasing: bool = True


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    model_type: ModelType = ModelType.RESNET50
    pretrained: bool = True
    
    # Feature dimensions
    vision_feature_dim: int = 2048  # ResNet-50 output
    text_feature_dim: int = 768     # BERT output
    fusion_dim: int = 512           # Multimodal fusion dimension
    
    # Dropout
    dropout_rate: float = 0.5
    
    # Transfer learning
    freeze_backbone: bool = True
    unfreeze_layers: int = 2  # Number of layers to unfreeze during fine-tuning


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Optimization
    learning_rate: float = 1e-4
    backbone_lr: float = 1e-5  # Lower LR for pretrained layers
    weight_decay: float = 1e-4
    
    # Scheduler
    scheduler_type: str = "cosine"  # "step", "cosine", "plateau"
    step_size: int = 10
    gamma: float = 0.1
    
    # Training duration
    num_epochs: int = 50
    early_stopping_patience: int = 10
    
    # Checkpointing
    checkpoint_dir: str = "./checkpoints"
    save_every_n_epochs: int = 5
    
    # Mixed precision
    use_amp: bool = True
    
    # Gradient clipping
    max_grad_norm: float = 1.0
    
    # Class weights for imbalanced data
    use_class_weights: bool = True


@dataclass
class OptimizationConfig:
    """Model optimization configuration."""
    # Quantization
    quantize: bool = True
    quantization_backend: str = "fbgemm"  # "fbgemm" for x86, "qnnpack" for ARM
    
    # Pruning
    prune: bool = True
    pruning_amount: float = 0.3  # Percentage of weights to prune
    
    # Export
    export_onnx: bool = True
    onnx_opset_version: int = 14


@dataclass
class DistributedConfig:
    """Distributed training configuration."""
    # DDP settings
    backend: str = "nccl"  # "nccl" for GPU, "gloo" for CPU
    world_size: int = 2
    
    # Gradient accumulation
    gradient_accumulation_steps: int = 1
    
    # Sync batch norm
    sync_batchnorm: bool = True


@dataclass
class InferenceConfig:
    """Inference and deployment configuration."""
    # API settings
    host: str = "0.0.0.0"
    port: int = 8000
    
    # Model loading
    model_path: str = "./checkpoints/best_model.pth"
    use_quantized: bool = False
    
    # Inference settings
    confidence_threshold: float = 0.5


@dataclass
class Config:
    """Master configuration."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    
    # Experiment tracking
    experiment_name: str = "medical_imaging_v1"
    seed: int = 42
    device: str = "cuda"
    
    def __post_init__(self):
        """Create necessary directories."""
        os.makedirs(self.training.checkpoint_dir, exist_ok=True)


def get_config() -> Config:
    """Get default configuration."""
    return Config()


def load_config_from_yaml(yaml_path: str) -> Config:
    """Load configuration from YAML file."""
    import yaml
    
    with open(yaml_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    config = Config()
    
    # Update nested dataclasses
    for key, value in config_dict.items():
        if hasattr(config, key) and isinstance(value, dict):
            nested_config = getattr(config, key)
            for k, v in value.items():
                if hasattr(nested_config, k):
                    setattr(nested_config, k, v)
    
    return config
