import os
import matplotlib.pyplot as plt
import numpy as np

from src.preprocessing.preprocessing import apply_lighting_shift, preprocess_image


def save_preprocessing_grid(sample_image: np.ndarray, save_path: str, category: str = "sheet_metal"):
    """
    Generates a 4x4 visual grid comparing lighting perturbations (rows)
    against classical enhancement filters (columns) for IEEE paper evidence.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    lightings = ["original", "low_light", "overexposed", "uneven"]
    preps = ["none", "clahe", "bilateral", "clahe_bilateral"]

    fig, axes = plt.subplots(len(lightings), len(preps), figsize=(12, 12))
    for r, light in enumerate(lightings):
        shifted = apply_lighting_shift(sample_image, light)
        for c, prep in enumerate(preps):
            enhanced = preprocess_image(shifted, prep)
            axes[r, c].imshow(enhanced)
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
            if r == 0:
                axes[r, c].set_title(f"Prep: {prep}", fontsize=11, fontweight="bold")
            if c == 0:
                axes[r, c].set_ylabel(f"Light: {light}", fontsize=11, fontweight="bold")

    plt.suptitle(f"Illumination Perturbation & Enhancement Effects ({category})", fontsize=14, fontweight="bold", y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_localization_comparison(
    test_images: np.ndarray,
    gt_masks: np.ndarray,
    hog_maps: np.ndarray,
    padim_maps: np.ndarray,
    y_true: np.ndarray,
    save_path: str,
    title: str = "",
    num_samples: int = 3,
):
    """
    Saves a 4-column visual evidence figure:
    [Processed Input | Ground Truth Mask | HOG-OCSVM Heatmap | PaDiM Heatmap]
    for defective test samples.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    defect_indices = np.where(y_true == 1)[0]
    if len(defect_indices) == 0:
        return

    chosen = defect_indices[:min(num_samples, len(defect_indices))]
    fig, axes = plt.subplots(len(chosen), 4, figsize=(13, 3.3 * len(chosen)))
    if len(chosen) == 1:
        axes = np.expand_dims(axes, axis=0)

    col_titles = ["Processed Input", "Ground-Truth Defect", "HOG-OCSVM Heatmap", "PaDiM Heatmap"]

    for row_idx, img_idx in enumerate(chosen):
        img = test_images[img_idx]
        gt = gt_masks[img_idx]
        h_map = hog_maps[img_idx]
        p_map = padim_maps[img_idx]

        # Normalize heatmaps to [0, 1] for consistent visual comparison
        h_norm = (h_map - h_map.min()) / (h_map.max() - h_map.min() + 1e-8)
        p_norm = (p_map - p_map.min()) / (p_map.max() - p_map.min() + 1e-8)

        axes[row_idx, 0].imshow(img)
        axes[row_idx, 1].imshow(gt, cmap="gray")

        axes[row_idx, 2].imshow(img)
        axes[row_idx, 2].imshow(h_norm, cmap="jet", alpha=0.5)

        axes[row_idx, 3].imshow(img)
        axes[row_idx, 3].imshow(p_norm, cmap="jet", alpha=0.5)

        for col_idx in range(4):
            axes[row_idx, col_idx].set_xticks([])
            axes[row_idx, col_idx].set_yticks([])
            if row_idx == 0:
                axes[row_idx, col_idx].set_title(col_titles[col_idx], fontsize=11, fontweight="bold")

    if title:
        plt.suptitle(title, fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)