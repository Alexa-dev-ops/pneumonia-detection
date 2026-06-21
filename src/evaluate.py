import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, f1_score, accuracy_score, precision_score, recall_score
from src.config import OUTPUT_DIR, THRESHOLD, CLASS_NAMES

def evaluate_individual_models(test_pred_matrix, y_test, arch_names):
    results = {}
    for i, arch in enumerate(arch_names):
        p_bin = (test_pred_matrix[:, i] >= THRESHOLD).astype(int)
        acc = accuracy_score(y_test, p_bin)
        pre = precision_score(y_test, p_bin, zero_division=0)
        rec = recall_score(y_test, p_bin, zero_division=0)
        f1 = f1_score(y_test, p_bin, zero_division=0)
        fpr, tpr, _ = roc_curve(y_test, test_pred_matrix[:, i])
        roc_auc = auc(fpr, tpr)
        results[arch] = {'accuracy': acc, 'precision': pre, 'recall': rec, 'f1': f1, 'auc': roc_auc}
    return results

def evaluate_ensemble(test_ensemble_prob, y_test):
    test_ensemble_pred = (test_ensemble_prob >= THRESHOLD).astype(int)
    acc = accuracy_score(y_test, test_ensemble_pred)
    pre = precision_score(y_test, test_ensemble_pred, zero_division=0)
    rec = recall_score(y_test, test_ensemble_pred, zero_division=0)
    f1 = f1_score(y_test, test_ensemble_pred, zero_division=0)
    fpr, tpr, _ = roc_curve(y_test, test_ensemble_prob)
    roc_auc = auc(fpr, tpr)
    return {'accuracy': acc, 'precision': pre, 'recall': rec, 'f1': f1, 'auc': roc_auc}, test_ensemble_pred

def plot_confusion_matrix(y_test, test_ensemble_pred):
    cm = confusion_matrix(y_test, test_ensemble_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    ax.set_title('Confusion Matrix — Weighted Ensemble')
    plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
    plt.close()

def save_results_json(ensemble_weights, ens_perf, ind_perf, arch_names):
    results = {
        'ensemble_weights': {arch: float(w) for arch, w in zip(arch_names, ensemble_weights)},
        'ensemble_test_performance': {k: float(v) for k, v in ens_perf.items()},
        'individual_model_performance': {arch: {k: float(v) for k, v in res.items()} for arch, res in ind_perf.items()}
    }
    with open(os.path.join(OUTPUT_DIR, 'results_summary.json'), 'w') as f:
        json.dump(results, f, indent=2)