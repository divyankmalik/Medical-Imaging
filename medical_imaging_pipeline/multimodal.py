"""
Multimodal fusion module combining vision (CNN) and text (BERT) features.
Implements various fusion strategies: concatenation, attention, cross-attention.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from transformers import AutoModel, AutoTokenizer

from models import ResNet50Classifier, EfficientNetClassifier, ClassificationHead


class TextEncoder(nn.Module):
    """BERT-based text encoder for clinical notes."""
    
    def __init__(
        self,
        model_name: str = "dmis-lab/biobert-v1.1",  # BioBERT for medical text
        freeze: bool = True,
        pooling: str = "cls"  # "cls", "mean", "max"
    ):
        super().__init__()
        
        self.bert = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.pooling = pooling
        self.feature_dim = self.bert.config.hidden_size  # 768
        
        if freeze:
            self._freeze_bert()
    
    def _freeze_bert(self):
        """Freeze BERT layers."""
        for param in self.bert.parameters():
            param.requires_grad = False
    
    def unfreeze_layers(self, num_layers: int = 2):
        """Unfreeze last N transformer layers."""
        # BERT has 12 encoder layers
        for layer in self.bert.encoder.layer[-num_layers:]:
            for param in layer.parameters():
                param.requires_grad = True
        print(f"Unfroze last {num_layers} BERT layers")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            input_ids: Tokenized input (batch, seq_len)
            attention_mask: Attention mask (batch, seq_len)
        
        Returns:
            Text features (batch, feature_dim)
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        if self.pooling == "cls":
            # Use [CLS] token representation
            return outputs.last_hidden_state[:, 0, :]
        elif self.pooling == "mean":
            # Mean pooling over all tokens
            hidden = outputs.last_hidden_state
            mask = attention_mask.unsqueeze(-1).expand(hidden.size()).float()
            return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        elif self.pooling == "max":
            # Max pooling
            hidden = outputs.last_hidden_state
            hidden[attention_mask == 0] = float('-inf')
            return hidden.max(dim=1).values
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")
    
    def tokenize(self, texts: list, max_length: int = 128) -> Dict[str, torch.Tensor]:
        """Tokenize a list of texts."""
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )


class CrossAttention(nn.Module):
    """Cross-attention module for fusing vision and text features."""
    
    def __init__(
        self,
        query_dim: int,
        key_dim: int,
        hidden_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Project to common dimension
        self.query_proj = nn.Linear(query_dim, hidden_dim)
        self.key_proj = nn.Linear(key_dim, hidden_dim)
        self.value_proj = nn.Linear(key_dim, hidden_dim)
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            query: Query features (batch, query_dim)
            key: Key features (batch, key_dim)
            value: Value features, defaults to key
        
        Returns:
            Attended features (batch, hidden_dim)
        """
        if value is None:
            value = key
        
        # Add sequence dimension
        query = self.query_proj(query).unsqueeze(1)  # (batch, 1, hidden)
        key = self.key_proj(key).unsqueeze(1)        # (batch, 1, hidden)
        value = self.value_proj(value).unsqueeze(1)  # (batch, 1, hidden)
        
        # Cross-attention
        attended, _ = self.attention(query, key, value)
        
        # Output projection with residual
        output = self.output_proj(attended.squeeze(1))
        output = self.layer_norm(output + query.squeeze(1))
        output = self.dropout(output)
        
        return output


class GatedFusion(nn.Module):
    """Gated fusion mechanism for combining modalities."""
    
    def __init__(
        self,
        vision_dim: int,
        text_dim: int,
        hidden_dim: int = 512
    ):
        super().__init__()
        
        # Project to common dimension
        self.vision_proj = nn.Linear(vision_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        
        # Gating mechanism
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        
        self.output_dim = hidden_dim
    
    def forward(
        self,
        vision_features: torch.Tensor,
        text_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            vision_features: (batch, vision_dim)
            text_features: (batch, text_dim)
        
        Returns:
            Fused features (batch, hidden_dim)
        """
        v = self.vision_proj(vision_features)
        t = self.text_proj(text_features)
        
        # Compute gate
        concat = torch.cat([v, t], dim=-1)
        gate = self.gate(concat)
        
        # Gated fusion
        fused = gate * v + (1 - gate) * t
        
        return fused


class MultimodalClassifier(nn.Module):
    """
    Multimodal classifier combining vision (CNN) and text (BERT) branches.
    """
    
    def __init__(
        self,
        num_classes: int = 2,
        vision_model: str = "resnet50",
        text_model: str = "dmis-lab/biobert-v1.1",
        fusion_type: str = "gated",  # "concat", "attention", "gated"
        fusion_dim: int = 512,
        dropout_rate: float = 0.5,
        freeze_vision: bool = True,
        freeze_text: bool = True
    ):
        super().__init__()
        
        self.fusion_type = fusion_type
        
        # Vision encoder
        if vision_model == "resnet50":
            self.vision_encoder = ResNet50Classifier(
                num_classes=num_classes,
                freeze_backbone=freeze_vision
            )
            self.vision_dim = self.vision_encoder.feature_dim
        elif vision_model == "efficientnet_b4":
            self.vision_encoder = EfficientNetClassifier(
                num_classes=num_classes,
                freeze_backbone=freeze_vision
            )
            self.vision_dim = self.vision_encoder.feature_dim
        
        # Text encoder
        self.text_encoder = TextEncoder(
            model_name=text_model,
            freeze=freeze_text
        )
        self.text_dim = self.text_encoder.feature_dim
        
        # Fusion module
        if fusion_type == "concat":
            self.fusion = nn.Sequential(
                nn.Linear(self.vision_dim + self.text_dim, fusion_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            )
            self.fusion_output_dim = fusion_dim
        
        elif fusion_type == "attention":
            self.fusion = CrossAttention(
                query_dim=self.vision_dim,
                key_dim=self.text_dim,
                hidden_dim=fusion_dim,
                dropout=dropout_rate
            )
            self.fusion_output_dim = fusion_dim
        
        elif fusion_type == "gated":
            self.fusion = GatedFusion(
                vision_dim=self.vision_dim,
                text_dim=self.text_dim,
                hidden_dim=fusion_dim
            )
            self.fusion_output_dim = fusion_dim
        
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")
        
        # Final classifier
        self.classifier = ClassificationHead(
            in_features=self.fusion_output_dim,
            num_classes=num_classes,
            hidden_dim=256,
            dropout_rate=dropout_rate
        )
    
    def get_vision_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extract vision features."""
        return self.vision_encoder.get_features(images)
    
    def get_text_features(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Extract text features."""
        return self.text_encoder(input_ids, attention_mask)
    
    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with both vision and text inputs.
        
        Args:
            images: Input images (batch, 3, H, W)
            input_ids: Tokenized text (batch, seq_len)
            attention_mask: Text attention mask (batch, seq_len)
        
        Returns:
            Dictionary with logits and intermediate features
        """
        # Extract features
        vision_features = self.get_vision_features(images)
        text_features = self.get_text_features(input_ids, attention_mask)
        
        # Fuse modalities
        if self.fusion_type == "concat":
            combined = torch.cat([vision_features, text_features], dim=-1)
            fused = self.fusion(combined)
        elif self.fusion_type == "attention":
            fused = self.fusion(vision_features, text_features)
        elif self.fusion_type == "gated":
            fused = self.fusion(vision_features, text_features)
        
        # Classify
        logits = self.classifier(fused)
        
        return {
            'logits': logits,
            'vision_features': vision_features,
            'text_features': text_features,
            'fused_features': fused
        }
    
    def forward_vision_only(self, images: torch.Tensor) -> torch.Tensor:
        """Forward pass with only vision input (for inference without text)."""
        vision_features = self.get_vision_features(images)
        
        # Use zeros for text features
        batch_size = images.size(0)
        text_features = torch.zeros(
            batch_size,
            self.text_dim,
            device=images.device
        )
        
        # Fuse with zero text
        if self.fusion_type == "concat":
            combined = torch.cat([vision_features, text_features], dim=-1)
            fused = self.fusion(combined)
        elif self.fusion_type == "attention":
            fused = self.fusion(vision_features, text_features)
        elif self.fusion_type == "gated":
            fused = self.fusion(vision_features, text_features)
        
        return self.classifier(fused)


class LateMultimodalFusion(nn.Module):
    """
    Late fusion: Train separate vision and text classifiers,
    combine predictions at inference time.
    """
    
    def __init__(
        self,
        num_classes: int = 2,
        vision_model: str = "resnet50",
        text_model: str = "dmis-lab/biobert-v1.1",
        vision_weight: float = 0.7,
        text_weight: float = 0.3
    ):
        super().__init__()
        
        self.vision_weight = vision_weight
        self.text_weight = text_weight
        
        # Vision classifier
        if vision_model == "resnet50":
            self.vision_classifier = ResNet50Classifier(num_classes=num_classes)
        else:
            self.vision_classifier = EfficientNetClassifier(num_classes=num_classes)
        
        # Text classifier
        self.text_encoder = TextEncoder(model_name=text_model)
        self.text_classifier = ClassificationHead(
            in_features=self.text_encoder.feature_dim,
            num_classes=num_classes
        )
    
    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Combined forward pass."""
        # Vision prediction
        vision_logits = self.vision_classifier(images)
        
        # Text prediction
        text_features = self.text_encoder(input_ids, attention_mask)
        text_logits = self.text_classifier(text_features)
        
        # Weighted combination
        combined_logits = (
            self.vision_weight * vision_logits +
            self.text_weight * text_logits
        )
        
        return {
            'logits': combined_logits,
            'vision_logits': vision_logits,
            'text_logits': text_logits
        }


def create_multimodal_model(
    num_classes: int = 2,
    vision_model: str = "resnet50",
    text_model: str = "dmis-lab/biobert-v1.1",
    fusion_type: str = "gated",
    fusion_dim: int = 512,
    dropout_rate: float = 0.5
) -> MultimodalClassifier:
    """Factory function to create multimodal models."""
    return MultimodalClassifier(
        num_classes=num_classes,
        vision_model=vision_model,
        text_model=text_model,
        fusion_type=fusion_type,
        fusion_dim=fusion_dim,
        dropout_rate=dropout_rate
    )


if __name__ == "__main__":
    # Test multimodal model
    print("Testing Multimodal Classifier...")
    
    model = create_multimodal_model(
        num_classes=2,
        fusion_type="gated"
    )
    
    # Dummy inputs
    batch_size = 2
    images = torch.randn(batch_size, 3, 224, 224)
    
    # Tokenize sample text
    texts = [
        "Normal chest X-ray with clear lung fields.",
        "Bilateral infiltrates consistent with pneumonia."
    ]
    tokens = model.text_encoder.tokenize(texts)
    
    # Forward pass
    outputs = model(
        images,
        tokens['input_ids'],
        tokens['attention_mask']
    )
    
    print(f"Logits shape: {outputs['logits'].shape}")
    print(f"Vision features shape: {outputs['vision_features'].shape}")
    print(f"Text features shape: {outputs['text_features'].shape}")
    print(f"Fused features shape: {outputs['fused_features'].shape}")
    
    # Count parameters
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")
