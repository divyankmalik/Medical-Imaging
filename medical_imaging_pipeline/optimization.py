"""
Model optimization module with quantization, pruning, and ONNX export.
Reduces model size and inference latency for deployment.
"""
import os
import copy
import time
from typing import Dict, Optional, Tuple
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torch.quantization import quantize_dynamic, get_default_qconfig
from torch.utils.data import DataLoader
import numpy as np

from config import OptimizationConfig, get_config


class ModelOptimizer:
    """Optimize models for deployment using quantization and pruning."""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
    
    def quantize_dynamic(
        self,
        model: nn.Module,
        dtype: torch.dtype = torch.qint8
    ) -> nn.Module:
        """
        Apply dynamic quantization (INT8).
        Quantizes weights statically, activations dynamically at runtime.
        
        Args:
            model: Model to quantize
            dtype: Quantization data type (qint8 or float16)
        
        Returns:
            Quantized model
        """
        # Dynamic quantization for Linear and LSTM layers
        quantized_model = quantize_dynamic(
            model,
            {nn.Linear, nn.Conv2d},
            dtype=dtype
        )
        
        return quantized_model
    
    def quantize_static(
        self,
        model: nn.Module,
        calibration_loader: DataLoader,
        device: torch.device,
        num_calibration_batches: int = 100
    ) -> nn.Module:
        """
        Apply static quantization with calibration.
        Quantizes both weights and activations using calibration data.
        
        Args:
            model: Model to quantize
            calibration_loader: DataLoader for calibration
            device: Device for calibration
            num_calibration_batches: Number of batches for calibration
        
        Returns:
            Quantized model
        """
        model.eval()
        model_fp32 = copy.deepcopy(model).cpu()
        
        # Fuse modules
        model_fp32 = self._fuse_modules(model_fp32)
        
        # Set quantization configuration
        model_fp32.qconfig = get_default_qconfig(self.config.quantization_backend)
        
        # Prepare for quantization
        torch.quantization.prepare(model_fp32, inplace=True)
        
        # Calibrate with representative data
        with torch.no_grad():
            for i, batch in enumerate(calibration_loader):
                if i >= num_calibration_batches:
                    break
                images = batch['image']
                model_fp32(images)
        
        # Convert to quantized model
        torch.quantization.convert(model_fp32, inplace=True)
        
        return model_fp32
    
    def _fuse_modules(self, model: nn.Module) -> nn.Module:
        """Fuse Conv-BN-ReLU patterns for better quantization."""
        # This is model-specific and should be customized
        # Example for ResNet-like models:
        modules_to_fuse = []
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Sequential):
                # Look for Conv-BN-ReLU patterns
                children = list(module.children())
                for i in range(len(children) - 2):
                    if (isinstance(children[i], nn.Conv2d) and
                        isinstance(children[i+1], nn.BatchNorm2d) and
                        isinstance(children[i+2], nn.ReLU)):
                        modules_to_fuse.append([f'{i}', f'{i+1}', f'{i+2}'])
        
        if modules_to_fuse:
            try:
                model = torch.quantization.fuse_modules(model, modules_to_fuse)
            except Exception as e:
                print(f"Module fusion failed: {e}")
        
        return model
    
    def prune_model(
        self,
        model: nn.Module,
        amount: float = 0.3,
        pruning_method: str = 'l1_unstructured'
    ) -> nn.Module:
        """
        Apply weight pruning to reduce model size.
        
        Args:
            model: Model to prune
            amount: Fraction of weights to prune (0-1)
            pruning_method: 'l1_unstructured', 'l1_structured', or 'random'
        
        Returns:
            Pruned model
        """
        model = copy.deepcopy(model)
        
        # Get all prunable layers
        layers_to_prune = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                layers_to_prune.append((module, 'weight'))
        
        # Apply global pruning
        if pruning_method == 'l1_unstructured':
            prune.global_unstructured(
                layers_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=amount
            )
        elif pruning_method == 'random':
            for module, name in layers_to_prune:
                prune.random_unstructured(module, name=name, amount=amount)
        elif pruning_method == 'l1_structured':
            for module, name in layers_to_prune:
                if isinstance(module, nn.Conv2d):
                    prune.ln_structured(
                        module, name=name, amount=amount, n=1, dim=0
                    )
        
        # Make pruning permanent
        for module, name in layers_to_prune:
            try:
                prune.remove(module, name)
            except Exception:
                pass
        
        return model
    
    def compute_sparsity(self, model: nn.Module) -> Dict[str, float]:
        """Compute sparsity statistics for pruned model."""
        total_zeros = 0
        total_elements = 0
        layer_sparsity = {}
        
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                weight = module.weight.data
                zeros = (weight == 0).sum().item()
                elements = weight.numel()
                
                total_zeros += zeros
                total_elements += elements
                layer_sparsity[name] = zeros / elements
        
        return {
            'global_sparsity': total_zeros / total_elements,
            'layer_sparsity': layer_sparsity,
            'total_zeros': total_zeros,
            'total_elements': total_elements
        }
    
    def export_onnx(
        self,
        model: nn.Module,
        output_path: str,
        input_shape: Tuple[int, ...] = (1, 3, 224, 224),
        opset_version: int = 14,
        dynamic_axes: Optional[Dict] = None
    ) -> str:
        """
        Export model to ONNX format.
        
        Args:
            model: Model to export
            output_path: Path to save ONNX model
            input_shape: Input tensor shape
            opset_version: ONNX opset version
            dynamic_axes: Dynamic axes for variable batch size
        
        Returns:
            Path to exported model
        """
        model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(*input_shape)
        
        # Default dynamic axes for variable batch size
        if dynamic_axes is None:
            dynamic_axes = {
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        
        # Export
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes=dynamic_axes
        )
        
        print(f"Model exported to {output_path}")
        
        # Verify ONNX model
        try:
            import onnx
            onnx_model = onnx.load(output_path)
            onnx.checker.check_model(onnx_model)
            print("ONNX model verified successfully")
        except ImportError:
            print("Install onnx package to verify exported model")
        
        return output_path
    
    def export_torchscript(
        self,
        model: nn.Module,
        output_path: str,
        input_shape: Tuple[int, ...] = (1, 3, 224, 224),
        method: str = 'trace'
    ) -> str:
        """
        Export model to TorchScript format.
        
        Args:
            model: Model to export
            output_path: Path to save TorchScript model
            input_shape: Input tensor shape for tracing
            method: 'trace' or 'script'
        
        Returns:
            Path to exported model
        """
        model.eval()
        
        if method == 'trace':
            dummy_input = torch.randn(*input_shape)
            traced_model = torch.jit.trace(model, dummy_input)
        else:
            traced_model = torch.jit.script(model)
        
        traced_model.save(output_path)
        print(f"TorchScript model saved to {output_path}")
        
        return output_path


def benchmark_model(
    model: nn.Module,
    input_shape: Tuple[int, ...] = (1, 3, 224, 224),
    num_runs: int = 100,
    warmup_runs: int = 10,
    device: str = 'cpu'
) -> Dict[str, float]:
    """
    Benchmark model inference latency.
    
    Args:
        model: Model to benchmark
        input_shape: Input tensor shape
        num_runs: Number of inference runs
        warmup_runs: Number of warmup runs
        device: Device to run on
    
    Returns:
        Benchmark statistics
    """
    model = model.to(device)
    model.eval()
    
    dummy_input = torch.randn(*input_shape).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(dummy_input)
    
    # Benchmark
    if device == 'cuda':
        torch.cuda.synchronize()
    
    latencies = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = model(dummy_input)
            
            if device == 'cuda':
                torch.cuda.synchronize()
            
            latencies.append((time.perf_counter() - start) * 1000)  # ms
    
    return {
        'mean_latency_ms': np.mean(latencies),
        'std_latency_ms': np.std(latencies),
        'min_latency_ms': np.min(latencies),
        'max_latency_ms': np.max(latencies),
        'p50_latency_ms': np.percentile(latencies, 50),
        'p95_latency_ms': np.percentile(latencies, 95),
        'p99_latency_ms': np.percentile(latencies, 99),
        'throughput_fps': 1000 / np.mean(latencies)
    }


def get_model_size(model: nn.Module) -> Dict[str, float]:
    """Get model size in MB."""
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    
    return {
        'param_size_mb': param_size / (1024 ** 2),
        'buffer_size_mb': buffer_size / (1024 ** 2),
        'total_size_mb': (param_size + buffer_size) / (1024 ** 2),
        'num_parameters': sum(p.numel() for p in model.parameters())
    }


def optimize_model(
    model: nn.Module,
    config: OptimizationConfig,
    calibration_loader: Optional[DataLoader] = None,
    output_dir: str = './optimized_models'
) -> Dict[str, nn.Module]:
    """
    Full optimization pipeline: pruning, quantization, export.
    
    Args:
        model: Model to optimize
        config: Optimization configuration
        calibration_loader: DataLoader for static quantization calibration
        output_dir: Directory to save optimized models
    
    Returns:
        Dictionary of optimized models
    """
    os.makedirs(output_dir, exist_ok=True)
    
    optimizer = ModelOptimizer(config)
    optimized_models = {'original': model}
    
    # Get original model stats
    original_size = get_model_size(model)
    original_benchmark = benchmark_model(model)
    
    print("=" * 60)
    print("Original Model Statistics")
    print("=" * 60)
    print(f"Size: {original_size['total_size_mb']:.2f} MB")
    print(f"Parameters: {original_size['num_parameters']:,}")
    print(f"Latency: {original_benchmark['mean_latency_ms']:.2f} ms")
    print(f"Throughput: {original_benchmark['throughput_fps']:.2f} FPS")
    
    # Pruning
    if config.prune:
        print("\n" + "=" * 60)
        print(f"Applying Pruning (amount={config.pruning_amount})")
        print("=" * 60)
        
        pruned_model = optimizer.prune_model(model, amount=config.pruning_amount)
        optimized_models['pruned'] = pruned_model
        
        sparsity = optimizer.compute_sparsity(pruned_model)
        pruned_size = get_model_size(pruned_model)
        pruned_benchmark = benchmark_model(pruned_model)
        
        print(f"Global Sparsity: {sparsity['global_sparsity']*100:.1f}%")
        print(f"Size: {pruned_size['total_size_mb']:.2f} MB")
        print(f"Latency: {pruned_benchmark['mean_latency_ms']:.2f} ms")
        print(f"Speedup: {original_benchmark['mean_latency_ms']/pruned_benchmark['mean_latency_ms']:.2f}x")
    
    # Quantization
    if config.quantize:
        print("\n" + "=" * 60)
        print("Applying Dynamic Quantization (INT8)")
        print("=" * 60)
        
        quantized_model = optimizer.quantize_dynamic(model)
        optimized_models['quantized'] = quantized_model
        
        quantized_size = get_model_size(quantized_model)
        quantized_benchmark = benchmark_model(quantized_model)
        
        print(f"Size: {quantized_size['total_size_mb']:.2f} MB")
        print(f"Size Reduction: {(1 - quantized_size['total_size_mb']/original_size['total_size_mb'])*100:.1f}%")
        print(f"Latency: {quantized_benchmark['mean_latency_ms']:.2f} ms")
        print(f"Speedup: {original_benchmark['mean_latency_ms']/quantized_benchmark['mean_latency_ms']:.2f}x")
    
    # Export to ONNX
    if config.export_onnx:
        print("\n" + "=" * 60)
        print("Exporting to ONNX")
        print("=" * 60)
        
        onnx_path = os.path.join(output_dir, 'model.onnx')
        optimizer.export_onnx(
            model,
            onnx_path,
            opset_version=config.onnx_opset_version
        )
        
        # Also export optimized model
        if 'pruned' in optimized_models:
            optimizer.export_onnx(
                optimized_models['pruned'],
                os.path.join(output_dir, 'model_pruned.onnx'),
                opset_version=config.onnx_opset_version
            )
    
    # Export TorchScript
    print("\n" + "=" * 60)
    print("Exporting to TorchScript")
    print("=" * 60)
    
    torchscript_path = os.path.join(output_dir, 'model.pt')
    optimizer.export_torchscript(model, torchscript_path)
    
    # Save summary
    summary_path = os.path.join(output_dir, 'optimization_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("Model Optimization Summary\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("Original Model:\n")
        f.write(f"  Size: {original_size['total_size_mb']:.2f} MB\n")
        f.write(f"  Parameters: {original_size['num_parameters']:,}\n")
        f.write(f"  Latency: {original_benchmark['mean_latency_ms']:.2f} ms\n\n")
        
        if 'pruned' in optimized_models:
            f.write(f"Pruned Model (amount={config.pruning_amount}):\n")
            f.write(f"  Sparsity: {sparsity['global_sparsity']*100:.1f}%\n")
            f.write(f"  Size: {pruned_size['total_size_mb']:.2f} MB\n")
            f.write(f"  Latency: {pruned_benchmark['mean_latency_ms']:.2f} ms\n\n")
        
        if 'quantized' in optimized_models:
            f.write("Quantized Model (INT8):\n")
            f.write(f"  Size: {quantized_size['total_size_mb']:.2f} MB\n")
            f.write(f"  Size Reduction: {(1 - quantized_size['total_size_mb']/original_size['total_size_mb'])*100:.1f}%\n")
            f.write(f"  Latency: {quantized_benchmark['mean_latency_ms']:.2f} ms\n")
    
    print(f"\nOptimization summary saved to {summary_path}")
    
    return optimized_models


if __name__ == "__main__":
    from models import create_model
    
    # Create a model
    model = create_model("resnet50", num_classes=2, freeze_backbone=False)
    
    # Optimize
    config = get_config().optimization
    optimized = optimize_model(model, config)
