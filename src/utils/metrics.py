"""Evaluation metrics for DTI prediction models."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    matthews_corrcoef,
    roc_auc_score
)
from imblearn.metrics import specificity_score
from typing import Dict, Tuple, List


def calculate_metrics(
    labels: List,
    predictions: List,
    probabilities: List = None
) -> Dict[str, float]:
    """
    Calculate comprehensive evaluation metrics.

    Args:
        labels: True labels
        predictions: Predicted labels
        probabilities: Prediction probabilities (optional, required for AUC)

    Returns:
        Dictionary containing all metrics
    """
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()

    # Basic metrics
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, zero_division=0)
    recall = recall_score(labels, predictions, zero_division=0)  # Sensitivity
    specificity = specificity_score(labels, predictions, average='binary', pos_label=1)
    mcc = matthews_corrcoef(labels, predictions)

    metrics = {
        'accuracy': round(accuracy, 4),
        'precision': round(precision, 4),
        'sensitivity': round(recall, 4),
        'specificity': round(specificity, 4),
        'mcc': round(mcc, 4),
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn)
    }

    # AUC if probabilities provided
    if probabilities is not None:
        try:
            auc = roc_auc_score(labels, probabilities)
            metrics['auc'] = round(auc, 4)
        except ValueError:
            metrics['auc'] = 0.5
            print("Warning: Could not calculate AUC")

    return metrics


def evaluate_predictions(df: pd.DataFrame) -> Dict[str, float]:
    """
    Evaluate predictions from a DataFrame.

    Args:
        df: DataFrame with 'Label', 'Predictions', and optionally 'Predictions_Proba' columns

    Returns:
        Dictionary containing all metrics
    """
    labels = df['Label'].tolist()
    predictions = df['Predictions'].tolist()

    probabilities = None
    if 'Predictions_Proba' in df.columns:
        probabilities = df['Predictions_Proba'].tolist()

    return calculate_metrics(labels, predictions, probabilities)


def print_metrics(metrics: Dict[str, float], dataset_name: str = ""):
    """
    Print metrics in a formatted way.

    Args:
        metrics: Dictionary of metrics
        dataset_name: Name of the dataset (for display)
    """
    if dataset_name:
        print(f"\nMetrics for {dataset_name}")

    print(f"Accuracy:    {metrics['accuracy']:.4f}")
    print(f"Precision:   {metrics['precision']:.4f}")
    print(f"Sensitivity: {metrics['sensitivity']:.4f}")
    print(f"Specificity: {metrics['specificity']:.4f}")
    print(f"MCC:         {metrics['mcc']:.4f}")
    if 'auc' in metrics:
        print(f"AUC:         {metrics['auc']:.4f}")

    print(f"\nConfusion Matrix:")
    print(f"  TP: {metrics['tp']:5d}  |  FP: {metrics['fp']:5d}")
    print(f"  FN: {metrics['fn']:5d}  |  TN: {metrics['tn']:5d}")


def report_mean_std(all_metrics: Dict[str, List[Dict[str, float]]]):
    """
    Report mean ± std across seeds, matching manuscript table format.

    Args:
        all_metrics: Dict mapping dataset names to lists of metric dicts (one per seed)
    """
    metric_keys = ['accuracy', 'sensitivity', 'specificity', 'mcc', 'auc']
    metric_labels = {
        'accuracy': 'Accuracy',
        'sensitivity': 'Sensitivity',
        'specificity': 'Specificity',
        'mcc': 'MCC',
        'auc': 'AUROC'
    }

    for dataset_name, metrics_list in all_metrics.items():
        n_seeds = len(metrics_list)
        print(f"\n{dataset_name} (mean ± std, {n_seeds} seeds):")
        for key in metric_keys:
            if key in metrics_list[0]:
                values = [m[key] for m in metrics_list]
                mean = np.mean(values)
                std = np.std(values)
                print(f"  {metric_labels.get(key, key):15s}: {mean:.3f} ± {std:.3f}")


def compare_models(results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """
    Compare metrics across multiple models.

    Args:
        results: Dictionary mapping model names to their metrics

    Returns:
        DataFrame comparing all models
    """
    comparison_df = pd.DataFrame(results).T
    return comparison_df.round(4)
