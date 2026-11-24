%%writefile /kaggle/working/dipthesis/src/evaluation/metrics.py
import os
import time
import tracemalloc
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score, precision_score, recall_score

try:
    import torch
except ImportError:
    torch = None


def find_optimal_f1_threshold(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """Finds the threshold maximizing F1-score on validation/reference scores."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
    f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-8)
    best_idx = np.argmax(f1_scores[:-1]) if len(thresholds) > 0 else 0
    return float(thresholds[best_idx]) if len(thresholds) > 0 else 0.5


def compute_image_metrics(y_true: np.ndarray, y_scores: np.ndarray, threshold: float = None) -> dict:
    """
    Computes Image-Level AUROC, Precision, Recall, F1, and False-Positive Rate (FPR).
    y_true: binary array (0 = normal, 1 = anomalous)
    y_scores: continuous anomaly scores
    """
    auroc = float(roc_auc_score(y_true, y_scores))
    if threshold is None:
        threshold = find_optimal_f1_threshold(y_true, y_scores)

    y_pred = (y_scores >= threshold).astype(int)
    fp = np.sum((y_pred == 1) & (y_true == 0))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fpr = float(fp / (fp + tn + 1e-8))

    return {
        "image_auroc": round(auroc, 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "fpr": round(fpr, 4),
        "threshold": round(threshold, 6),
    }


def compute_pixel_metrics(gt_masks: np.ndarray, anomaly_maps: np.ndarray, threshold: float = None) -> dict:
    """
    Computes Pixel-Level AUROC and IoU on test_public ground-truth masks.
    gt_masks: binary array of shape (N, H, W)
    anomaly_maps: continuous heatmap array of shape (N, H, W)
    """
    gt_flat = (gt_masks > 0).astype(np.uint8).ravel()
    pred_flat = anomaly_maps.ravel()

    pixel_auroc = float(roc_auc_score(gt_flat, pred_flat))
    if threshold is None:
        threshold = find_optimal_f1_threshold(gt_flat, pred_flat)

    pred_bin = (pred_flat >= threshold).astype(np.uint8)
    intersection = np.sum((pred_bin == 1) & (gt_flat == 1))
    union = np.sum((pred_bin == 1) | (gt_flat == 1))
    iou = float(intersection / (union + 1e-8))

    return {
        "pixel_auroc": round(pixel_auroc, 4),
        "pixel_iou": round(iou, 4),
    }


def profile_inference(predict_fn, sample_batch, warmup: int = 5) -> dict:
    """
    Measures per-image inference latency (ms) and peak memory (MB) across CPU or GPU.
    predict_fn: callable accepting sample_batch
    """
    for _ in range(warmup):
        _ = predict_fn(sample_batch)

    use_cuda = torch is not None and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    tracemalloc.start()
    start_time = time.perf_counter()

    _ = predict_fn(sample_batch)

    if use_cuda:
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    _, peak_cpu_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    n_images = len(sample_batch)
    latency_per_img_ms = elapsed_ms / max(n_images, 1)
    peak_vram_mb = (torch.cuda.max_memory_allocated() / (1024 ** 2)) if use_cuda else 0.0
    peak_ram_mb = peak_cpu_bytes / (1024 ** 2)

    return {
        "latency_ms_per_image": round(latency_per_img_ms, 2),
        "peak_ram_mb": round(peak_ram_mb, 2),
        "peak_vram_mb": round(peak_vram_mb, 2),
    }


def get_model_size_mb(filepath: str) -> float:
    """Returns serialized model checkpoint size in megabytes."""
    if not os.path.exists(filepath):
        return 0.0
    return round(os.path.getsize(filepath) / (1024 ** 2), 2)