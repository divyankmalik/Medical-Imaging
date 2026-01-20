"""
Model architectures for medical image classification.
Includes ResNet-50, EfficientNet-B4, and custom classification heads.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List
import timm
from torchvision import models


class AttentionPool(nn.Module):
    """Attention pooling layer for feature aggregation."""
    
    def __init__(self, in_features: int, hidden_dim: int = 256):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, seq_len, features)
        Returns:
            Pooled tensor of shape (batch, features)
        """
        weights = F.softmax(self.attention(x), dim=1)
        return torch.sum(weights * x, dim=1)


class ClassificationHead(nn.Module):
    """Classification head with dropout and optional intermediate layers."""
    
    def __init__(
        self,
        in_features: int,
        num_classes: int,
        hidden_dim: Optional[int] = None,
        dropout_rate: float = 0.5
    ):
        super().__init__()
        
        layers = []
        
        if hidden_dim:
            layers.extend([
                nn.Linear(in_features, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dim, num_classes)
            ])
        else:
            layers.extend([
                nn.Dropout(dropout_rate),
                nn.Linear(in_features, num_classes)
            ])
        
        self.classifier = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


class ResNet50Classifier(nn.Module):
    """ResNet-50 based classifier with transfer learning support."""
    
    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        freeze_backbone: bool = True,
        dropout_rate: float = 0.5
    ):
        super().__init__()
        
        # Load pretrained ResNet-50
        self.backbone = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        )
        
        # Get feature dimension
        self.feature_dim = self.backbone.fc.in_features  # 2048
        
        # Remove original classifier
        self.backbone.fc = nn.Identity()
        
        # Freeze backbone if specified
        if freeze_backbone:
            self._freeze_backbone()
        
        # Add custom classification head
        self.classifier = ClassificationHead(
            in_features=self.feature_dim,
            num_classes=num_classes,
            hidden_dim=512,
            dropout_rate=dropout_rate
        )
    
    def _freeze_backbone(self):
        """Freeze all backbone layers."""
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def unfreeze_layers(self, num_layers: int = 2):
        """
        Gradually unfreeze layers for fine-tuning.
        Unfreezes from the last layer backwards.
        """
        # ResNet-50 layers: conv1, bn1, layer1, layer2, layer3, layer4
        layers = [
            self.backbone.layer4,
            self.backbone.layer3,
            self.backbone.layer2,
            self.backbone.layer1,
        ]
        
        for i, layer in enumerate(layers[:num_layers]):
            for param in layer.parameters():
                param.requires_grad = True
            print(f"Unfroze layer{4-i}")
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features without classification."""
        return self.backbone(x)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits


class EfficientNetClassifier(nn.Module):
    """EfficientNet-B4 based classifier using timm library."""
    
    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        freeze_backbone: bool = True,
        dropout_rate: float = 0.5,
        model_name: str = "efficientnet_b4"
    ):
        super().__init__()
        
        # Load pretrained EfficientNet
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0  # Remove classifier
        )
        
        # Get feature dimension
        self.feature_dim = self.backbone.num_features  # 1792 for B4
        
        # Freeze backbone if specified
        if freeze_backbone:
            self._freeze_backbone()
        
        # Add custom classification head
        self.classifier = ClassificationHead(
            in_features=self.feature_dim,
            num_classes=num_classes,
            hidden_dim=512,
            dropout_rate=dropout_rate
        )
    
    def _freeze_backbone(self):
        """Freeze all backbone layers."""
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def unfreeze_layers(self, num_blocks: int = 2):
        """
        Unfreeze last N blocks of EfficientNet.
        EfficientNet-B4 has 7 blocks.
        """
        # Get all blocks
        blocks = list(self.backbone.blocks.children())
        
        # Unfreeze last N blocks
        for block in blocks[-num_blocks:]:
            for param in block.parameters():
                param.requires_grad = True
        
        print(f"Unfroze last {num_blocks} blocks")
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features without classification."""
        return self.backbone(x)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits


class DenseNetClassifier(nn.Module):
    """DenseNet-121 based classifier as an alternative architecture."""
    
    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        freeze_backbone: bool = True,
        dropout_rate: float = 0.5
    ):
        super().__init__()
        
        # Load pretrained DenseNet-121
        self.backbone = models.densenet121(
            weights=models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        )
        
        # Get feature dimension
        self.feature_dim = self.backbone.classifier.in_features  # 1024
        
        # Remove original classifier
        self.backbone.classifier = nn.Identity()
        
        # Freeze backbone if specified
        if freeze_backbone:
            self._freeze_backbone()
        
        # Add custom classification head
        self.classifier = ClassificationHead(
            in_features=self.feature_dim,
            num_classes=num_classes,
            hidden_dim=512,
            dropout_rate=dropout_rate
        )
    
    def _freeze_backbone(self):
        """Freeze all backbone layers."""
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features without classification."""
        features = self.backbone.features(x)
        features = F.relu(features, inplace=True)
        features = F.adaptive_avg_pool2d(features, (1, 1))
        features = torch.flatten(features, 1)
        return features
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.get_features(x)
        logits = self.classifier(features)
        return logits


class EnsembleModel(nn.Module):
    """Ensemble of multiple models for improved predictions."""
    
    def __init__(
        self,
        models: List[nn.Module],
        weights: Optional[List[float]] = None
    ):
        super().__init__()
        self.models = nn.ModuleList(models)
        
        if weights is None:
            weights = [1.0 / len(models)] * len(models)
        self.weights = weights
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Average predictions from all models."""
        outputs = []
        for model, weight in zip(self.models, self.weights):
            outputs.append(weight * model(x))
        return torch.stack(outputs).sum(dim=0)


def create_model(
    model_type: str = "resnet50",
    num_classes: int = 2,
    pretrained: bool = True,
    freeze_backbone: bool = True,
    dropout_rate: float = 0.5
) -> nn.Module:
    """
    Factory function to create models.
    
    Args:
        model_type: One of 'resnet50', 'efficientnet_b4', 'densenet121'
        num_classes: Number of output classes
        pretrained: Whether to use ImageNet pretrained weights
        freeze_backbone: Whether to freeze backbone layers
        dropout_rate: Dropout rate for classification head
    
    Returns:
        Initialized model
    """
    model_type = model_type.lower()
    
    if model_type == "resnet50":
        return ResNet50Classifier(
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            dropout_rate=dropout_rate
        )
    elif model_type == "efficientnet_b4":
        return EfficientNetClassifier(
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            dropout_rate=dropout_rate
        )
    elif model_type == "densenet121":
        return DenseNetClassifier(
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            dropout_rate=dropout_rate
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def get_parameter_groups(
    model: nn.Module,
    backbone_lr: float = 1e-5,
    head_lr: float = 1e-4,
    weight_decay: float = 1e-4
) -> List[Dict]:
    """
    Create parameter groups with differential learning rates.
    Lower LR for pretrained backbone, higher for new classification head.
    """
    backbone_params = []
    head_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        if 'classifier' in name:
            head_params.append(param)
        else:
            backbone_params.append(param)
    
    return [
        {'params': backbone_params, 'lr': backbone_lr, 'weight_decay': weight_decay},
        {'params': head_params, 'lr': head_lr, 'weight_decay': weight_decay}
    ]


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


if __name__ == "__main__":
    # Test model creation
    print("Testing ResNet-50:")
    model = create_model("resnet50", num_classes=2, freeze_backbone=True)
    total, trainable = count_parameters(model)
    print(f"  Total params: {total:,}")
    print(f"  Trainable params: {trainable:,}")
    
    # Test forward pass
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    print(f"  Output shape: {out.shape}")
    
    # Test unfreezing
    model.unfreeze_layers(2)
    total, trainable = count_parameters(model)
    print(f"  After unfreezing - Trainable params: {trainable:,}")
    
    print("\nTesting EfficientNet-B4:")
    model = create_model("efficientnet_b4", num_classes=2)
    total, trainable = count_parameters(model)
    print(f"  Total params: {total:,}")
    print(f"  Trainable params: {trainable:,}")
    
    out = model(x)
    print(f"  Output shape: {out.shape}")
