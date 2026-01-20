"""
Training module with training loops, schedulers, and utilities.
Supports single-GPU training with mixed precision.
"""
import os
import time
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

from config import Config, TrainingConfig, get_config
from models import create_model, get_parameter_groups, count_parameters
from data_preprocessing import create_data_loaders, TextTokenizer
from evaluate import MetricsCalculator, compute_metrics


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.001,
        mode: str = 'min'
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = None
        self.should_stop = False
    
    def __call__(self, value: float) -> bool:
        """Check if training should stop."""
        if self.best_value is None:
            self.best_value = value
            return False
        
        if self.mode == 'min':
            improved = value < self.best_value - self.min_delta
        else:
            improved = value > self.best_value + self.min_delta
        
        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        
        return self.should_stop


class CheckpointManager:
    """Manage model checkpoints."""
    
    def __init__(
        self,
        checkpoint_dir: str,
        max_checkpoints: int = 5
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.checkpoints = []
    
    def save(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: Optional[object],
        epoch: int,
        metrics: Dict[str, float],
        is_best: bool = False
    ) -> str:
        """Save checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics
        }
        
        if scheduler:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        # Save regular checkpoint
        filename = f"checkpoint_epoch_{epoch:03d}.pth"
        filepath = self.checkpoint_dir / filename
        torch.save(checkpoint, filepath)
        
        self.checkpoints.append(filepath)
        
        # Remove old checkpoints
        while len(self.checkpoints) > self.max_checkpoints:
            old_checkpoint = self.checkpoints.pop(0)
            if old_checkpoint.exists():
                old_checkpoint.unlink()
        
        # Save best model separately
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            print(f"Saved best model with metrics: {metrics}")
        
        return str(filepath)
    
    def load(
        self,
        model: nn.Module,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[object] = None,
        checkpoint_path: Optional[str] = None
    ) -> Tuple[nn.Module, int, Dict]:
        """Load checkpoint."""
        if checkpoint_path is None:
            checkpoint_path = self.checkpoint_dir / "best_model.pth"
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if scheduler and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        return model, checkpoint['epoch'], checkpoint.get('metrics', {})


def create_optimizer(
    model: nn.Module,
    config: TrainingConfig
) -> optim.Optimizer:
    """Create optimizer with differential learning rates."""
    param_groups = get_parameter_groups(
        model,
        backbone_lr=config.backbone_lr,
        head_lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    return optim.AdamW(param_groups)


def create_scheduler(
    optimizer: optim.Optimizer,
    config: TrainingConfig,
    num_training_steps: int
) -> object:
    """Create learning rate scheduler."""
    if config.scheduler_type == "step":
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.step_size,
            gamma=config.gamma
        )
    elif config.scheduler_type == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_training_steps,
            eta_min=1e-7
        )
    elif config.scheduler_type == "plateau":
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=config.gamma,
            patience=5
        )
    else:
        return None


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    scaler: Optional[GradScaler] = None,
    scheduler: Optional[object] = None,
    max_grad_norm: float = 1.0
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(train_loader, desc="Training")
    
    for batch_idx, batch in enumerate(pbar):
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision forward pass
        if scaler:
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
        
        # Update scheduler (for step-based schedulers)
        if scheduler and not isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step()
        
        # Track metrics
        running_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'lr': f"{optimizer.param_groups[0]['lr']:.2e}"
        })
    
    # Compute epoch metrics
    epoch_loss = running_loss / len(train_loader)
    metrics = compute_metrics(
        np.array(all_labels),
        np.array(all_preds),
        None  # No probabilities in training
    )
    metrics['loss'] = epoch_loss
    
    return metrics


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Dict[str, float]:
    """Validate the model."""
    model.eval()
    
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    for batch in tqdm(val_loader, desc="Validation"):
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        running_loss += loss.item()
        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
    
    # Compute metrics
    epoch_loss = running_loss / len(val_loader)
    metrics = compute_metrics(
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probs)
    )
    metrics['loss'] = epoch_loss
    
    return metrics


def train(
    config: Config,
    model: Optional[nn.Module] = None,
    resume_from: Optional[str] = None
) -> nn.Module:
    """Main training function."""
    
    # Set device
    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set random seed
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(config.data)
    
    # Create model
    if model is None:
        model = create_model(
            model_type=config.model.model_type.value,
            num_classes=config.data.num_classes,
            pretrained=config.model.pretrained,
            freeze_backbone=config.model.freeze_backbone,
            dropout_rate=config.model.dropout_rate
        )
    
    model = model.to(device)
    
    # Print model info
    total, trainable = count_parameters(model)
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    
    # Create optimizer and scheduler
    optimizer = create_optimizer(model, config.training)
    num_training_steps = len(train_loader) * config.training.num_epochs
    scheduler = create_scheduler(optimizer, config.training, num_training_steps)
    
    # Create loss function with class weights
    if config.training.use_class_weights:
        class_weights = train_loader.dataset.get_class_weights().to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Mixed precision scaler
    scaler = GradScaler() if config.training.use_amp else None
    
    # Checkpoint manager
    checkpoint_manager = CheckpointManager(config.training.checkpoint_dir)
    
    # Resume if specified
    start_epoch = 0
    if resume_from:
        model, start_epoch, _ = checkpoint_manager.load(
            model, optimizer, scheduler, resume_from
        )
        start_epoch += 1
        print(f"Resumed from epoch {start_epoch}")
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=config.training.early_stopping_patience,
        mode='max'  # Maximize validation accuracy
    )
    
    # TensorBoard writer
    writer = SummaryWriter(log_dir=os.path.join(config.training.checkpoint_dir, 'logs'))
    
    # Training loop
    best_val_acc = 0.0
    
    for epoch in range(start_epoch, config.training.num_epochs):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch + 1}/{config.training.num_epochs}")
        print(f"{'='*50}")
        
        # Train
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            scaler, scheduler if not isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau) else None,
            config.training.max_grad_norm
        )
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device)
        
        # Update plateau scheduler
        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_metrics['loss'])
        
        # Print metrics
        print(f"\nTrain - Loss: {train_metrics['loss']:.4f}, "
              f"Acc: {train_metrics['accuracy']:.4f}, "
              f"F1: {train_metrics['f1']:.4f}")
        print(f"Val   - Loss: {val_metrics['loss']:.4f}, "
              f"Acc: {val_metrics['accuracy']:.4f}, "
              f"F1: {val_metrics['f1']:.4f}, "
              f"AUC: {val_metrics.get('auc_roc', 0):.4f}")
        
        # Log to TensorBoard
        writer.add_scalars('Loss', {
            'train': train_metrics['loss'],
            'val': val_metrics['loss']
        }, epoch)
        writer.add_scalars('Accuracy', {
            'train': train_metrics['accuracy'],
            'val': val_metrics['accuracy']
        }, epoch)
        writer.add_scalar('Val/F1', val_metrics['f1'], epoch)
        writer.add_scalar('Val/AUC', val_metrics.get('auc_roc', 0), epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        
        # Save checkpoint
        is_best = val_metrics['accuracy'] > best_val_acc
        if is_best:
            best_val_acc = val_metrics['accuracy']
        
        if (epoch + 1) % config.training.save_every_n_epochs == 0 or is_best:
            checkpoint_manager.save(
                model, optimizer, scheduler, epoch, val_metrics, is_best
            )
        
        # Check early stopping
        if early_stopping(val_metrics['accuracy']):
            print(f"\nEarly stopping triggered at epoch {epoch + 1}")
            break
        
        # Gradual unfreezing
        if epoch == config.training.num_epochs // 3:
            print("\nUnfreezing backbone layers...")
            if hasattr(model, 'unfreeze_layers'):
                model.unfreeze_layers(config.model.unfreeze_layers)
                # Reset optimizer with new parameters
                optimizer = create_optimizer(model, config.training)
    
    writer.close()
    
    # Load best model and evaluate on test set
    model, _, _ = checkpoint_manager.load(model)
    test_metrics = validate(model, test_loader, criterion, device)
    
    print(f"\n{'='*50}")
    print("Final Test Results:")
    print(f"{'='*50}")
    print(f"Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall: {test_metrics['recall']:.4f}")
    print(f"F1-Score: {test_metrics['f1']:.4f}")
    print(f"AUC-ROC: {test_metrics.get('auc_roc', 0):.4f}")
    
    return model


def train_multimodal(
    config: Config,
    model: Optional[nn.Module] = None
) -> nn.Module:
    """Train multimodal model with both vision and text."""
    from multimodal import create_multimodal_model
    from data_preprocessing import collate_multimodal
    
    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    
    # Create data loaders with text
    train_loader, val_loader, test_loader = create_data_loaders(
        config.data, include_text=True
    )
    
    # Update collate function
    train_loader.collate_fn = collate_multimodal
    val_loader.collate_fn = collate_multimodal
    test_loader.collate_fn = collate_multimodal
    
    # Create multimodal model
    if model is None:
        model = create_multimodal_model(
            num_classes=config.data.num_classes,
            vision_model=config.model.model_type.value,
            fusion_type="gated",
            fusion_dim=config.model.fusion_dim,
            dropout_rate=config.model.dropout_rate
        )
    
    model = model.to(device)
    
    # Tokenizer for text
    tokenizer = model.text_encoder.tokenize
    
    # Training setup
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay
    )
    
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler() if config.training.use_amp else None
    
    checkpoint_manager = CheckpointManager(config.training.checkpoint_dir)
    best_val_acc = 0.0
    
    for epoch in range(config.training.num_epochs):
        print(f"\nEpoch {epoch + 1}/{config.training.num_epochs}")
        
        # Train
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch in tqdm(train_loader, desc="Training"):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            texts = batch['text']
            
            # Tokenize text
            tokens = tokenizer(texts)
            input_ids = tokens['input_ids'].to(device)
            attention_mask = tokens['attention_mask'].to(device)
            
            optimizer.zero_grad()
            
            if scaler:
                with autocast():
                    outputs = model(images, input_ids, attention_mask)
                    loss = criterion(outputs['logits'], labels)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images, input_ids, attention_mask)
                loss = criterion(outputs['logits'], labels)
                loss.backward()
                optimizer.step()
            
            running_loss += loss.item()
            preds = outputs['logits'].argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        
        train_acc = correct / total
        train_loss = running_loss / len(train_loader)
        
        # Validate
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                labels = batch['label'].to(device)
                texts = batch['text']
                
                tokens = tokenizer(texts)
                input_ids = tokens['input_ids'].to(device)
                attention_mask = tokens['attention_mask'].to(device)
                
                outputs = model(images, input_ids, attention_mask)
                preds = outputs['logits'].argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        
        val_acc = val_correct / val_total
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_manager.save(
                model, optimizer, None, epoch,
                {'accuracy': val_acc, 'loss': train_loss},
                is_best=True
            )
    
    return model


if __name__ == "__main__":
    config = get_config()
    
    # Create synthetic data for testing
    from data_preprocessing import create_synthetic_dataset
    create_synthetic_dataset()
    
    # Train model
    model = train(config)
