import os
import glob
import cv2
import numpy as np
from src.preprocessing.preprocessing import apply_lighting_shift, preprocess_image


def load_rgb_image(path: str, image_size: tuple) -> np.ndarray:
    """Loads an image as RGB uint8 and resizes to (width, height)."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = image_size
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


def load_binary_mask(path: str, image_size: tuple) -> np.ndarray:
    """Loads a ground-truth defect mask as binary uint8 (0 or 1)."""
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not read mask: {path}")
    h, w = image_size
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    return (mask > 0).astype(np.uint8)


def load_mvtec_ad2_category(cfg: dict) -> dict:
    """
    Loads train, validation, and test_public splits for one MVTec AD 2 category.
    Lighting perturbations and adaptive preprocessing are applied to test images
    while keeping ground-truth masks unchanged.
    """
    data_root = cfg["paths"]["data_root"]
    category = cfg["experiment"]["category"]
    image_size = tuple(cfg["experiment"]["image_size"])  # (H, W)

    prep_mode = cfg["preprocessing"]["mode"]
    prep_cfg = cfg["preprocessing"]
    light_cfg = cfg["lighting_shift"]

    cat_dir = os.path.join(data_root, category)

    # 1. Train & Validation (normal images only, preprocessed under training lighting)
    train_paths = sorted(glob.glob(os.path.join(cat_dir, "train", "good", "*.*")))
    val_paths = sorted(glob.glob(os.path.join(cat_dir, "validation", "good", "*.*")))

    train_imgs = np.array([
        preprocess_image(load_rgb_image(p, image_size), prep_mode, prep_cfg)
        for p in train_paths
    ])
    val_imgs = np.array([
        preprocess_image(load_rgb_image(p, image_size), prep_mode, prep_cfg)
        for p in val_paths
    ])

    # 2. Test Public (normal + anomalous images with ground-truth masks)
    test_imgs, y_true, gt_masks = [], [], []
    h, w = image_size

    # Normal test images
    for p in sorted(glob.glob(os.path.join(cat_dir, "test_public", "good", "*.*"))):
        img = load_rgb_image(p, image_size)
        img = apply_lighting_shift(
            img,
            condition=light_cfg["condition"],
            low_gamma=light_cfg["low_light_gamma"],
            high_gamma=light_cfg["overexposed_gamma"],
        )
        img = preprocess_image(img, prep_mode, prep_cfg)
        test_imgs.append(img)
        y_true.append(0)
        gt_masks.append(np.zeros((h, w), dtype=np.uint8))

    # Defective test images + masks
    bad_dir = os.path.join(cat_dir, "test_public", "bad")
    gt_dir = os.path.join(cat_dir, "test_public", "ground_truth", "bad")
    for p in sorted(glob.glob(os.path.join(bad_dir, "*.*"))):
        img = load_rgb_image(p, image_size)
        img = apply_lighting_shift(
            img,
            condition=light_cfg["condition"],
            low_gamma=light_cfg["low_light_gamma"],
            high_gamma=light_cfg["overexposed_gamma"],
        )
        img = preprocess_image(img, prep_mode, prep_cfg)
        test_imgs.append(img)
        y_true.append(1)

        stem = os.path.splitext(os.path.basename(p))[0]
        mask_matches = glob.glob(os.path.join(gt_dir, f"{stem}*.*"))
        if not mask_matches:
            raise FileNotFoundError(f"Missing ground-truth mask for {p} in {gt_dir}")
        gt_masks.append(load_binary_mask(mask_matches[0], image_size))

    return {
        "train_images": train_imgs,
        "val_images": val_imgs,
        "test_images": np.array(test_imgs),
        "y_true": np.array(y_true, dtype=np.int64),
        "gt_masks": np.array(gt_masks, dtype=np.uint8),
    }