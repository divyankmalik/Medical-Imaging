"""
Evaluation module with comprehensive metrics and visualization.
Includes confusion matrix, ROC curves, and clinical validation metrics.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    classification_report,
    cohen_kappa_score
)
from tqdm import tqdm


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    average: str = 'weighted'
) -> Dict[str, float]:
    """
    Compute comprehensive classification metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_prob: Prediction probabilities (optional)
        average: Averaging method for multi-class metrics
    
    Returns:
        Dictionary of metric names and values
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
        'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
        'f1': f1_score(y_true, y_pred, average=average, zero_division=0),
        'specificity': compute_specificity(y_true, y_pred),
        'cohen_kappa': cohen_kappa_score(y_true, y_pred)
    }
    
    # Add AUC-ROC if probabilities are available
    if y_prob is not None:
        try:
            if y_prob.ndim == 2 and y_prob.shape[1] == 2:
                metrics['auc_roc'] = roc_auc_score(y_true, y_prob[:, 1])
            else:
                metrics['auc_roc'] = roc_auc_score(
                    y_true, y_prob, multi_class='ovr', average='weighted'
                )
        except ValueError:
            metrics['auc_roc'] = 0.0
    
    return metrics


def compute_specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute specificity (true negative rate)."""
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        return tn / (tn + fp) if (tn + fp) > 0 else 0.0
    else:
        specificities = []
        for i in range(cm.shape[0]):
            tn = np.sum(cm) - np.sum(cm[i, :]) - np.sum(cm[:, i]) + cm[i, i]
            fp = np.sum(cm[:, i]) - cm[i, i]
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            specificities.append(spec)
        return np.mean(specificities)


class MetricsCalculator:
    """Accumulate predictions and compute metrics over an entire dataset."""
    
    def __init__(self, num_classes: int = 2):
        self.num_classes = num_classes
        self.reset()
    
    def reset(self):
        """Reset accumulated predictions."""
        self.all_labels = []
        self.all_preds = []
        self.all_probs = []
    
    def update(
        self,
        labels: torch.Tensor,
        preds: torch.Tensor,
        probs: Optional[torch.Tensor] = None
    ):
        """Add batch predictions."""
        self.all_labels.extend(labels.cpu().numpy())
        self.all_preds.extend(preds.cpu().numpy())
        if probs is not None:
            self.all_probs.extend(probs.cpu().numpy())
    
    def compute(self) -> Dict[str, float]:
        """Compute all metrics."""
        y_true = np.array(self.all_labels)
        y_pred = np.array(self.all_preds)
        y_prob = np.array(self.all_probs) if self.all_probs else None
        return compute_metrics(y_true, y_pred, y_prob)
    
    def get_arrays(self) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Get accumulated arrays."""
        y_true = np.array(self.all_labels)
        y_pred = np.array(self.all_preds)
        y_prob = np.array(self.all_probs) if self.all_probs else None
        return y_true, y_pred, y_prob


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    save_path: Optional[str] = None,
    normalize: bool = True
) -> plt.Figure:
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(
        cm, annot=True, fmt='.2%' if normalize else 'd',
        cmap='Blues', xticklabels=class_names,
        yticklabels=class_names, ax=ax
    )
    
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    ax.set_title('Confusion Matrix')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: List[str],
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot ROC curve."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    if y_prob.ndim == 2 and y_prob.shape[1] == 2:
        fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
        auc = roc_auc_score(y_true, y_prob[:, 1])
        ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {auc:.3f})')
    else:
        for i, class_name in enumerate(class_names):
            y_true_binary = (y_true == i).astype(int)
            fpr, tpr, _ = roc_curve(y_true_binary, y_prob[:, i])
            auc = roc_auc_score(y_true_binary, y_prob[:, i])
            ax.plot(fpr, tpr, linewidth=2, label=f'{class_name} (AUC = {auc:.3f})')
    
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: List[str],
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot precision-recall curve."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    if y_prob.ndim == 2 and y_prob.shape[1] == 2:
        precision, recall, _ = precision_recall_curve(y_true, y_prob[:, 1])
        ax.plot(recall, precision, 'b-', linewidth=2, label='Precision-Recall')
    else:
        for i, class_name in enumerate(class_names):
            y_true_binary = (y_true == i).astype(int)
            precision, recall, _ = precision_recall_curve(y_true_binary, y_prob[:, i])
            ax.plot(recall, precision, linewidth=2, label=class_name)
    
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    class_names: List[str],
    output_dir: Optional[str] = None
) -> Dict[str, float]:
    """Comprehensive model evaluation with visualizations."""
    model.eval()
    metrics_calc = MetricsCalculator(num_classes=len(class_names))
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = probs.argmax(dim=1)
            
            metrics_calc.update(labels, preds, probs)
    
    metrics = metrics_calc.compute()
    y_true, y_pred, y_prob = metrics_calc.get_arrays()
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        plot_confusion_matrix(y_true, y_pred, class_names, save_path=output_dir / 'confusion_matrix.png')
        
        if y_prob is not None:
            plot_roc_curve(y_true, y_prob, class_names, save_path=output_dir / 'roc_curve.png')
            plot_precision_recall_curve(y_true, y_prob, class_names, save_path=output_dir / 'pr_curve.png')
        
        with open(output_dir / 'metrics.txt', 'w') as f:
            f.write("Evaluation Metrics\n" + "=" * 40 + "\n")
            for name, value in metrics.items():
                f.write(f"{name}: {value:.4f}\n")
    
    plt.close('all')
    return metrics


class ClinicalValidator:
    """Clinical validation utilities."""
    
    def __init__(self, class_names: List[str]):
        self.class_names = class_names
    
    def compute_inter_rater_agreement(
        self, model_preds: np.ndarray, expert_preds: np.ndarray
    ) -> Dict[str, float]:
        """Compute agreement metrics between model and expert."""
        return {
            'cohen_kappa': cohen_kappa_score(expert_preds, model_preds),
            'agreement_rate': np.mean(model_preds == expert_preds),
            'accuracy': accuracy_score(expert_preds, model_preds)
        }
    
    def reader_study_analysis(
        self,
        diagnoses_without_ai: np.ndarray,
        diagnoses_with_ai: np.ndarray,
        ground_truth: np.ndarray
    ) -> Dict[str, float]:
        """Analyze if AI assistance improves clinician accuracy."""
        acc_without = accuracy_score(ground_truth, diagnoses_without_ai)
        acc_with = accuracy_score(ground_truth, diagnoses_with_ai)
        
        return {
            'accuracy_without_ai': acc_without,
            'accuracy_with_ai': acc_with,
            'improvement': acc_with - acc_without,
            'relative_improvement': (acc_with - acc_without) / acc_without * 100
        }


class FairnessEvaluator:
    """Evaluate model fairness across demographic groups."""
    
    def __init__(self, class_names: List[str]):
        self.class_names = class_names
    
    def evaluate_demographic_parity(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        demographics: np.ndarray,
        demographic_groups: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate model performance across demographic groups."""
        results = {}
        
        for group in demographic_groups:
            mask = demographics == group
            if mask.sum() == 0:
                continue
            
            group_true = y_true[mask]
            group_pred = y_pred[mask]
            
            results[group] = {
                'accuracy': accuracy_score(group_true, group_pred),
                'precision': precision_score(group_true, group_pred, average='weighted', zero_division=0),
                'recall': recall_score(group_true, group_pred, average='weighted', zero_division=0),
                'f1': f1_score(group_true, group_pred, average='weighted', zero_division=0),
                'n_samples': int(mask.sum())
            }
        
        if len(results) >= 2:
            accuracies = [r['accuracy'] for r in results.values()]
            results['_disparity'] = {
                'max_accuracy_gap': max(accuracies) - min(accuracies),
                'std_accuracy': np.std(accuracies)
            }
        
        return results


if __name__ == "__main__":
    np.random.seed(42)
    
    n_samples = 100
    y_true = np.random.randint(0, 2, n_samples)
    y_prob = np.random.rand(n_samples, 2)
    y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)
    y_pred = y_prob.argmax(axis=1)
    
    metrics = compute_metrics(y_true, y_pred, y_prob)
    print("Metrics:", metrics)
    
    class_names = ['NORMAL', 'PNEUMONIA']
    fig = plot_confusion_matrix(y_true, y_pred, class_names)
    plt.savefig('test_confusion_matrix.png')
    print("Saved test confusion matrix")
