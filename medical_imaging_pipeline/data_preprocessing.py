"""
Data preprocessing, augmentation, and loading utilities.
Handles class imbalance, data augmentation, and creates data loaders.
"""
import os
import random
from pathlib import Path
from typing import Tuple, Dict, List, Optional, Callable
from collections import Counter

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

from config import DataConfig, get_config


class MedicalImageDataset(Dataset):
    """Custom dataset for medical images with optional text data."""
    
    def __init__(
        self,
        root_dir: str,
        transform: Optional[Callable] = None,
        include_text: bool = False,
        class_names: List[str] = None
    ):
        """
        Args:
            root_dir: Root directory containing class subdirectories
            transform: Albumentations transform pipeline
            include_text: Whether to include synthetic clinical text
            class_names: List of class names (subdirectory names)
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.include_text = include_text
        self.class_names = class_names or ["NORMAL", "PNEUMONIA"]
        
        # Build image paths and labels
        self.samples = []
        self.labels = []
        
        for class_idx, class_name in enumerate(self.class_names):
            class_dir = self.root_dir / class_name
            if class_dir.exists():
                for img_path in class_dir.glob("*"):
                    if img_path.suffix.lower() in ['.jpeg', '.jpg', '.png', '.bmp']:
                        self.samples.append(str(img_path))
                        self.labels.append(class_idx)
        
        print(f"Loaded {len(self.samples)} images from {root_dir}")
        print(f"Class distribution: {Counter(self.labels)}")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sample."""
        img_path = self.samples[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)
        
        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        else:
            # Default: convert to tensor
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        sample = {
            'image': image,
            'label': torch.tensor(label, dtype=torch.long),
            'path': img_path
        }
        
        # Add synthetic clinical text for multimodal training
        if self.include_text:
            sample['text'] = self._generate_clinical_text(label, img_path)
        
        return sample
    
    def _generate_clinical_text(self, label: int, img_path: str) -> str:
        """Generate synthetic clinical notes for multimodal training."""
        # In real scenarios, this would come from actual clinical records
        normal_templates = [
            "Chest X-ray shows clear lung fields. No infiltrates or consolidation. "
            "Heart size is normal. No pleural effusion.",
            "PA and lateral chest radiograph demonstrates no acute cardiopulmonary "
            "abnormality. Lungs are clear bilaterally.",
            "Normal chest X-ray. No evidence of pneumonia, mass, or effusion. "
            "Cardiac silhouette within normal limits.",
        ]
        
        pneumonia_templates = [
            "Chest X-ray reveals consolidation in the lower lobe consistent with "
            "pneumonia. Recommend clinical correlation and follow-up.",
            "Bilateral infiltrates noted on chest radiograph. Findings suggestive "
            "of bacterial pneumonia. Consider antibiotic therapy.",
            "Opacity in right middle lobe with air bronchograms. Clinical picture "
            "consistent with community-acquired pneumonia.",
        ]
        
        templates = normal_templates if label == 0 else pneumonia_templates
        return random.choice(templates)
    
    def get_class_weights(self) -> torch.Tensor:
        """Calculate class weights for handling imbalanced data."""
        class_counts = Counter(self.labels)
        total = len(self.labels)
        weights = []
        
        for i in range(len(self.class_names)):
            count = class_counts.get(i, 1)
            weight = total / (len(self.class_names) * count)
            weights.append(weight)
        
        return torch.tensor(weights, dtype=torch.float32)
    
    def get_sample_weights(self) -> List[float]:
        """Get per-sample weights for WeightedRandomSampler."""
        class_weights = self.get_class_weights()
        return [class_weights[label].item() for label in self.labels]


def get_train_transforms(config: DataConfig) -> A.Compose:
    """Get training augmentation pipeline using Albumentations."""
    return A.Compose([
        # Resize
        A.Resize(config.image_size[0], config.image_size[1]),
        
        # Geometric augmentations
        A.Rotate(limit=config.rotation_degrees, p=config.augmentation_prob),
        A.HorizontalFlip(p=0.5 if config.horizontal_flip else 0),
        A.VerticalFlip(p=0.5 if config.vertical_flip else 0),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.1,
            rotate_limit=15,
            p=config.augmentation_prob
        ),
        
        # Color augmentations
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
            A.RandomGamma(gamma_limit=(80, 120)),
            A.CLAHE(clip_limit=4.0),
        ], p=config.augmentation_prob if config.color_jitter else 0),
        
        # Noise and blur
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0)),
            A.GaussianBlur(blur_limit=(3, 5)),
            A.MedianBlur(blur_limit=3),
        ], p=0.3),
        
        # Dropout augmentations
        A.OneOf([
            A.CoarseDropout(
                max_holes=8,
                max_height=config.image_size[0] // 8,
                max_width=config.image_size[1] // 8,
                fill_value=0
            ),
            A.GridDropout(ratio=0.3),
        ], p=0.3 if config.random_erasing else 0),
        
        # Normalize and convert to tensor
        A.Normalize(mean=config.normalize_mean, std=config.normalize_std),
        ToTensorV2(),
    ])


def get_val_transforms(config: DataConfig) -> A.Compose:
    """Get validation/test transform pipeline (no augmentation)."""
    return A.Compose([
        A.Resize(config.image_size[0], config.image_size[1]),
        A.Normalize(mean=config.normalize_mean, std=config.normalize_std),
        ToTensorV2(),
    ])


def create_data_loaders(
    config: DataConfig,
    include_text: bool = False
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test data loaders."""
    
    # Create datasets
    train_dataset = MedicalImageDataset(
        root_dir=os.path.join(config.data_root, config.train_dir),
        transform=get_train_transforms(config),
        include_text=include_text,
        class_names=config.class_names
    )
    
    val_dataset = MedicalImageDataset(
        root_dir=os.path.join(config.data_root, config.val_dir),
        transform=get_val_transforms(config),
        include_text=include_text,
        class_names=config.class_names
    )
    
    test_dataset = MedicalImageDataset(
        root_dir=os.path.join(config.data_root, config.test_dir),
        transform=get_val_transforms(config),
        include_text=include_text,
        class_names=config.class_names
    )
    
    # Create weighted sampler for imbalanced training data
    sample_weights = train_dataset.get_sample_weights()
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )
    
    return train_loader, val_loader, test_loader


def create_synthetic_dataset(
    output_dir: str = "./data/chest_xray",
    num_samples_per_class: int = 100
) -> None:
    """
    Create a synthetic dataset for testing the pipeline.
    In production, use real medical imaging data.
    """
    import cv2
    
    os.makedirs(output_dir, exist_ok=True)
    
    for split in ['train', 'val', 'test']:
        for class_name in ['NORMAL', 'PNEUMONIA']:
            class_dir = os.path.join(output_dir, split, class_name)
            os.makedirs(class_dir, exist_ok=True)
            
            # Adjust sample count by split
            if split == 'train':
                n_samples = num_samples_per_class
            elif split == 'val':
                n_samples = num_samples_per_class // 4
            else:
                n_samples = num_samples_per_class // 4
            
            for i in range(n_samples):
                # Create synthetic X-ray-like image
                img = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)
                
                # Add class-specific patterns
                if class_name == 'PNEUMONIA':
                    # Add some "opacity" patterns
                    cv2.circle(
                        img,
                        (random.randint(50, 174), random.randint(50, 174)),
                        random.randint(20, 50),
                        (random.randint(100, 150),) * 3,
                        -1
                    )
                
                # Add Gaussian blur for realism
                img = cv2.GaussianBlur(img, (5, 5), 0)
                
                # Save image
                img_path = os.path.join(class_dir, f"{class_name.lower()}_{split}_{i:04d}.jpeg")
                cv2.imwrite(img_path, img)
    
    print(f"Created synthetic dataset at {output_dir}")


class TextTokenizer:
    """Simple tokenizer wrapper for clinical text."""
    
    def __init__(self, model_name: str = "bert-base-uncased", max_length: int = 128):
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length
    
    def __call__(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """Tokenize a batch of texts."""
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )


def collate_multimodal(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Custom collate function for multimodal data."""
    images = torch.stack([item['image'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])
    texts = [item.get('text', '') for item in batch]
    paths = [item['path'] for item in batch]
    
    return {
        'image': images,
        'label': labels,
        'text': texts,
        'path': paths
    }


if __name__ == "__main__":
    # Test data loading
    config = get_config()
    
    # Create synthetic data for testing
    create_synthetic_dataset()
    
    # Test data loaders
    train_loader, val_loader, test_loader = create_data_loaders(config.data)
    
    # Print sample batch
    batch = next(iter(train_loader))
    print(f"Batch image shape: {batch['image'].shape}")
    print(f"Batch label shape: {batch['label'].shape}")
    print(f"Labels: {batch['label']}")
