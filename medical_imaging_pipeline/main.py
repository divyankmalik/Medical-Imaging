#!/usr/bin/env python3
"""
Main entry point for the Medical Image Classification Pipeline.
Provides CLI interface for training, evaluation, optimization, and deployment.
"""
import argparse
import os
import sys
from pathlib import Path

import torch
import numpy as np

from config import Config, get_config, ModelType


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_model(args):
    """Train a vision model."""
    from train import train
    from models import create_model
    
    config = get_config()
    config.model.model_type = ModelType(args.model)
    config.training.num_epochs = args.epochs
    config.data.batch_size = args.batch_size
    config.training.learning_rate = args.lr
    
    if args.data_dir:
        config.data.data_root = args.data_dir
    
    set_seed(config.seed)
    
    print("=" * 60)
    print("Medical Image Classification - Training")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Device: {config.device}")
    print("=" * 60)
    
    model = train(config, resume_from=args.resume)
    
    print("\nTraining complete!")
    return model


def train_multimodal_model(args):
    """Train a multimodal (vision + text) model."""
    from train import train_multimodal
    
    config = get_config()
    config.training.num_epochs = args.epochs
    config.data.batch_size = args.batch_size
    
    if args.data_dir:
        config.data.data_root = args.data_dir
    
    set_seed(config.seed)
    
    print("=" * 60)
    print("Medical Image Classification - Multimodal Training")
    print("=" * 60)
    print(f"Vision model: {args.model}")
    print(f"Text model: BioBERT")
    print(f"Fusion type: gated")
    print("=" * 60)
    
    model = train_multimodal(config)
    
    print("\nMultimodal training complete!")
    return model


def evaluate_model(args):
    """Evaluate a trained model."""
    from evaluate import evaluate_model
    from models import create_model
    from data_preprocessing import create_data_loaders
    
    config = get_config()
    
    if args.data_dir:
        config.data.data_root = args.data_dir
    
    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = create_model(
        model_type=args.model,
        num_classes=config.data.num_classes,
        pretrained=False,
        freeze_backbone=False
    )
    
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    
    model = model.to(device)
    
    # Create data loader
    _, _, test_loader = create_data_loaders(config.data)
    
    print("=" * 60)
    print("Medical Image Classification - Evaluation")
    print("=" * 60)
    
    metrics = evaluate_model(
        model,
        test_loader,
        device,
        config.data.class_names,
        output_dir=args.output_dir
    )
    
    print("\nEvaluation complete!")
    print(f"Results saved to {args.output_dir}")
    
    return metrics


def optimize_model(args):
    """Optimize a trained model for deployment."""
    from optimization import optimize_model
    from models import create_model
    
    config = get_config()
    device = torch.device('cpu')  # Optimization on CPU
    
    # Load model
    model = create_model(
        model_type=args.model,
        num_classes=config.data.num_classes,
        pretrained=False,
        freeze_backbone=False
    )
    
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    
    print("=" * 60)
    print("Medical Image Classification - Model Optimization")
    print("=" * 60)
    
    config.optimization.prune = args.prune
    config.optimization.quantize = args.quantize
    config.optimization.pruning_amount = args.prune_amount
    
    optimized = optimize_model(
        model,
        config.optimization,
        output_dir=args.output_dir
    )
    
    print("\nOptimization complete!")
    print(f"Optimized models saved to {args.output_dir}")
    
    return optimized


def run_distributed(args):
    """Run distributed training."""
    from distributed_train import run_distributed_training
    
    config = get_config()
    config.model.model_type = ModelType(args.model)
    config.training.num_epochs = args.epochs
    config.data.batch_size = args.batch_size
    
    if args.data_dir:
        config.data.data_root = args.data_dir
    
    print("=" * 60)
    print("Medical Image Classification - Distributed Training")
    print("=" * 60)
    print(f"Available GPUs: {torch.cuda.device_count()}")
    print("=" * 60)
    
    run_distributed_training(config)


def run_api(args):
    """Run the inference API server."""
    from inference_api import run_server
    
    print("=" * 60)
    print("Medical Image Classification - API Server")
    print("=" * 60)
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print("=" * 60)
    
    run_server(
        host=args.host,
        port=args.port,
        model_path=args.checkpoint,
        reload=args.reload
    )


def create_demo_data(args):
    """Create synthetic demo dataset."""
    from data_preprocessing import create_synthetic_dataset
    
    print("=" * 60)
    print("Creating Synthetic Demo Dataset")
    print("=" * 60)
    
    create_synthetic_dataset(
        output_dir=args.output_dir,
        num_samples_per_class=args.num_samples
    )
    
    print(f"\nDataset created at {args.output_dir}")


def run_quick_demo(args):
    """Run a quick demonstration of the pipeline."""
    from data_preprocessing import create_synthetic_dataset, create_data_loaders
    from models import create_model, count_parameters
    from train import train_one_epoch, validate
    
    config = get_config()
    config.training.num_epochs = 3
    config.data.batch_size = 16
    
    print("=" * 60)
    print("Medical Image Classification - Quick Demo")
    print("=" * 60)
    
    # Create synthetic data
    print("\n[1/4] Creating synthetic dataset...")
    create_synthetic_dataset(num_samples_per_class=50)
    
    # Create data loaders
    print("\n[2/4] Loading data...")
    train_loader, val_loader, test_loader = create_data_loaders(config.data)
    
    # Create model
    print("\n[3/4] Creating model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_model(
        model_type='resnet50',
        num_classes=2,
        pretrained=True,
        freeze_backbone=True
    ).to(device)
    
    total, trainable = count_parameters(model)
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    
    # Quick training
    print("\n[4/4] Running quick training (3 epochs)...")
    
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4
    )
    criterion = torch.nn.CrossEntropyLoss()
    
    for epoch in range(3):
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_metrics = validate(model, val_loader, criterion, device)
        
        print(f"Epoch {epoch+1}: "
              f"Train Loss={train_metrics['loss']:.4f}, "
              f"Train Acc={train_metrics['accuracy']:.4f}, "
              f"Val Acc={val_metrics['accuracy']:.4f}")
    
    print("\n" + "=" * 60)
    print("Demo complete! Pipeline is working correctly.")
    print("=" * 60)
    
    print("\nNext steps:")
    print("1. Download real chest X-ray dataset from Kaggle")
    print("2. Run full training: python main.py train --epochs 50")
    print("3. Evaluate: python main.py evaluate --checkpoint checkpoints/best_model.pth")
    print("4. Deploy API: python main.py api --checkpoint checkpoints/best_model.pth")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Medical Image Classification Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick demo
  python main.py demo
  
  # Train ResNet-50
  python main.py train --model resnet50 --epochs 50
  
  # Train EfficientNet-B4
  python main.py train --model efficientnet_b4 --epochs 50
  
  # Train multimodal model
  python main.py train_multimodal --epochs 30
  
  # Evaluate model
  python main.py evaluate --checkpoint checkpoints/best_model.pth
  
  # Optimize model
  python main.py optimize --checkpoint checkpoints/best_model.pth --quantize --prune
  
  # Distributed training (multi-GPU)
  python main.py distributed --model resnet50 --epochs 50
  
  # Run API server
  python main.py api --checkpoint checkpoints/best_model.pth --port 8000
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Demo command
    demo_parser = subparsers.add_parser('demo', help='Run quick demonstration')
    
    # Create data command
    data_parser = subparsers.add_parser('create_data', help='Create synthetic dataset')
    data_parser.add_argument('--output-dir', type=str, default='./data/chest_xray',
                            help='Output directory')
    data_parser.add_argument('--num-samples', type=int, default=100,
                            help='Samples per class')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train vision model')
    train_parser.add_argument('--model', type=str, default='resnet50',
                             choices=['resnet50', 'efficientnet_b4', 'densenet121'],
                             help='Model architecture')
    train_parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    train_parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    train_parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    train_parser.add_argument('--data-dir', type=str, default=None, help='Data directory')
    train_parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    
    # Train multimodal command
    mm_parser = subparsers.add_parser('train_multimodal', help='Train multimodal model')
    mm_parser.add_argument('--model', type=str, default='resnet50', help='Vision model')
    mm_parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    mm_parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    mm_parser.add_argument('--data-dir', type=str, default=None, help='Data directory')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate model')
    eval_parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint')
    eval_parser.add_argument('--model', type=str, default='resnet50', help='Model architecture')
    eval_parser.add_argument('--data-dir', type=str, default=None, help='Data directory')
    eval_parser.add_argument('--output-dir', type=str, default='./evaluation',
                            help='Output directory for results')
    
    # Optimize command
    opt_parser = subparsers.add_parser('optimize', help='Optimize model for deployment')
    opt_parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint')
    opt_parser.add_argument('--model', type=str, default='resnet50', help='Model architecture')
    opt_parser.add_argument('--quantize', action='store_true', help='Apply INT8 quantization')
    opt_parser.add_argument('--prune', action='store_true', help='Apply pruning')
    opt_parser.add_argument('--prune-amount', type=float, default=0.3, help='Pruning amount')
    opt_parser.add_argument('--output-dir', type=str, default='./optimized_models',
                            help='Output directory')
    
    # Distributed training command
    dist_parser = subparsers.add_parser('distributed', help='Distributed training')
    dist_parser.add_argument('--model', type=str, default='resnet50', help='Model architecture')
    dist_parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    dist_parser.add_argument('--batch-size', type=int, default=32, help='Batch size per GPU')
    dist_parser.add_argument('--data-dir', type=str, default=None, help='Data directory')
    
    # API command
    api_parser = subparsers.add_parser('api', help='Run inference API server')
    api_parser.add_argument('--checkpoint', type=str, default=None, help='Model checkpoint')
    api_parser.add_argument('--host', type=str, default='0.0.0.0', help='Host')
    api_parser.add_argument('--port', type=int, default=8000, help='Port')
    api_parser.add_argument('--reload', action='store_true', help='Auto-reload')
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    # Execute command
    if args.command == 'demo':
        run_quick_demo(args)
    elif args.command == 'create_data':
        create_demo_data(args)
    elif args.command == 'train':
        train_model(args)
    elif args.command == 'train_multimodal':
        train_multimodal_model(args)
    elif args.command == 'evaluate':
        evaluate_model(args)
    elif args.command == 'optimize':
        optimize_model(args)
    elif args.command == 'distributed':
        run_distributed(args)
    elif args.command == 'api':
        run_api(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
