"""
Distributed training module with PyTorch DDP and Mixed Precision.
Supports multi-GPU training for faster convergence.
"""
import os
import argparse
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np

from config import Config, get_config
from models import create_model, get_parameter_groups
from data_preprocessing import MedicalImageDataset, get_train_transforms, get_val_transforms
from evaluate import compute_metrics


def setup_distributed(rank: int, world_size: int, backend: str = 'nccl'):
    """Initialize distributed training environment."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_distributed():
    """Clean up distributed training."""
    dist.destroy_process_group()


def create_distributed_dataloaders(
    config: Config,
    rank: int,
    world_size: int
) -> Tuple[DataLoader, DataLoader]:
    """Create data loaders with distributed sampler."""
    
    # Create datasets
    train_dataset = MedicalImageDataset(
        root_dir=os.path.join(config.data.data_root, config.data.train_dir),
        transform=get_train_transforms(config.data),
        class_names=config.data.class_names
    )
    
    val_dataset = MedicalImageDataset(
        root_dir=os.path.join(config.data.data_root, config.data.val_dir),
        transform=get_val_transforms(config.data),
        class_names=config.data.class_names
    )
    
    # Create distributed samplers
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    
    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        sampler=train_sampler,
        num_workers=config.data.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.data.batch_size,
        sampler=val_sampler,
        num_workers=config.data.num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


def train_one_epoch_distributed(
    model: DDP,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scaler: Optional[GradScaler],
    rank: int,
    config: Config
) -> Dict[str, float]:
    """Train for one epoch with DDP."""
    model.train()
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Set epoch for distributed sampler
    train_loader.sampler.set_epoch(rank)
    
    pbar = tqdm(train_loader, desc=f"Training [GPU {rank}]", disable=(rank != 0))
    
    for batch_idx, batch in enumerate(pbar):
        images = batch['image'].cuda(rank, non_blocking=True)
        labels = batch['label'].cuda(rank, non_blocking=True)
        
        optimizer.zero_grad()
        
        # Mixed precision forward pass
        if scaler and config.training.use_amp:
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)
            optimizer.step()
        
        # Track metrics
        running_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        if rank == 0:
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{correct/total:.4f}"
            })
    
    # Gather metrics from all processes
    metrics = torch.tensor([running_loss, correct, total], dtype=torch.float64).cuda(rank)
    dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
    
    total_loss = metrics[0].item() / (len(train_loader) * dist.get_world_size())
    total_acc = metrics[1].item() / metrics[2].item()
    
    return {
        'loss': total_loss,
        'accuracy': total_acc
    }


@torch.no_grad()
def validate_distributed(
    model: DDP,
    val_loader: DataLoader,
    criterion: nn.Module,
    rank: int
) -> Dict[str, float]:
    """Validate with DDP."""
    model.eval()
    
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    for batch in val_loader:
        images = batch['image'].cuda(rank, non_blocking=True)
        labels = batch['label'].cuda(rank, non_blocking=True)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        running_loss += loss.item()
        preds = outputs.argmax(dim=1)
        
        all_preds.append(preds)
        all_labels.append(labels)
    
    # Gather predictions from all processes
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    
    # Gather across all GPUs
    gathered_preds = [torch.zeros_like(all_preds) for _ in range(dist.get_world_size())]
    gathered_labels = [torch.zeros_like(all_labels) for _ in range(dist.get_world_size())]
    
    dist.all_gather(gathered_preds, all_preds)
    dist.all_gather(gathered_labels, all_labels)
    
    if rank == 0:
        all_preds = torch.cat(gathered_preds).cpu().numpy()
        all_labels = torch.cat(gathered_labels).cpu().numpy()
        
        metrics = compute_metrics(all_labels, all_preds)
        metrics['loss'] = running_loss / len(val_loader)
        return metrics
    
    return {}


def train_distributed(
    rank: int,
    world_size: int,
    config: Config
):
    """Main distributed training function."""
    
    # Setup distributed
    setup_distributed(rank, world_size, config.distributed.backend)
    
    # Create data loaders
    train_loader, val_loader = create_distributed_dataloaders(config, rank, world_size)
    
    # Create model
    model = create_model(
        model_type=config.model.model_type.value,
        num_classes=config.data.num_classes,
        pretrained=config.model.pretrained,
        freeze_backbone=config.model.freeze_backbone
    ).cuda(rank)
    
    # Sync batch norm
    if config.distributed.sync_batchnorm:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    
    # Wrap with DDP
    model = DDP(model, device_ids=[rank])
    
    # Optimizer
    param_groups = get_parameter_groups(
        model.module,
        backbone_lr=config.training.backbone_lr,
        head_lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay
    )
    optimizer = optim.AdamW(param_groups)
    
    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.training.num_epochs
    )
    
    # Loss function with class weights
    class_weights = train_loader.dataset.get_class_weights().cuda(rank)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Mixed precision scaler
    scaler = GradScaler() if config.training.use_amp else None
    
    # Training loop
    best_val_acc = 0.0
    
    for epoch in range(config.training.num_epochs):
        if rank == 0:
            print(f"\n{'='*60}")
            print(f"Epoch {epoch + 1}/{config.training.num_epochs}")
            print(f"{'='*60}")
        
        # Train
        train_metrics = train_one_epoch_distributed(
            model, train_loader, criterion, optimizer, scaler, rank, config
        )
        
        # Validate
        val_metrics = validate_distributed(model, val_loader, criterion, rank)
        
        # Update scheduler
        scheduler.step()
        
        # Log metrics (only on rank 0)
        if rank == 0:
            print(f"\nTrain - Loss: {train_metrics['loss']:.4f}, "
                  f"Acc: {train_metrics['accuracy']:.4f}")
            
            if val_metrics:
                print(f"Val   - Loss: {val_metrics['loss']:.4f}, "
                      f"Acc: {val_metrics['accuracy']:.4f}, "
                      f"F1: {val_metrics['f1']:.4f}")
                
                # Save best model
                if val_metrics['accuracy'] > best_val_acc:
                    best_val_acc = val_metrics['accuracy']
                    
                    checkpoint = {
                        'epoch': epoch,
                        'model_state_dict': model.module.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_metrics': val_metrics
                    }
                    
                    torch.save(
                        checkpoint,
                        os.path.join(config.training.checkpoint_dir, 'best_model_ddp.pth')
                    )
                    print(f"Saved best model with accuracy: {best_val_acc:.4f}")
    
    # Cleanup
    cleanup_distributed()


def run_distributed_training(config: Config):
    """Launch distributed training across multiple GPUs."""
    world_size = torch.cuda.device_count()
    
    if world_size < 2:
        print("Distributed training requires at least 2 GPUs.")
        print("Falling back to single GPU training...")
        from train import train
        return train(config)
    
    print(f"Starting distributed training on {world_size} GPUs")
    
    # Use torch.multiprocessing.spawn to launch processes
    import torch.multiprocessing as mp
    
    mp.spawn(
        train_distributed,
        args=(world_size, config),
        nprocs=world_size,
        join=True
    )


class GradientAccumulator:
    """Helper class for gradient accumulation in distributed training."""
    
    def __init__(self, accumulation_steps: int = 1):
        self.accumulation_steps = accumulation_steps
        self.step_count = 0
    
    def should_step(self) -> bool:
        """Check if optimizer should step."""
        self.step_count += 1
        return self.step_count % self.accumulation_steps == 0
    
    def reset(self):
        """Reset step counter."""
        self.step_count = 0


def calculate_effective_batch_size(
    batch_size: int,
    world_size: int,
    accumulation_steps: int = 1
) -> int:
    """Calculate effective batch size for distributed training."""
    return batch_size * world_size * accumulation_steps


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Distributed Training')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    
    config = get_config()
    
    # Check GPU availability
    if not torch.cuda.is_available():
        print("CUDA is not available. Distributed training requires GPUs.")
        exit(1)
    
    num_gpus = torch.cuda.device_count()
    print(f"Found {num_gpus} GPUs")
    
    # Calculate effective batch size
    effective_bs = calculate_effective_batch_size(
        config.data.batch_size,
        num_gpus,
        config.distributed.gradient_accumulation_steps
    )
    print(f"Effective batch size: {effective_bs}")
    
    # Launch training
    # Use: torchrun --nproc_per_node=NUM_GPUS distributed_train.py
    
    if 'RANK' in os.environ:
        # Launched with torchrun
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(rank)
        
        train_distributed(rank, world_size, config)
    else:
        # Launch with multiprocessing
        run_distributed_training(config)
