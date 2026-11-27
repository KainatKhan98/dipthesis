import glob
import os
import cv2
import numpy as np

from src.preprocessing.preprocessing import apply_lighting_shift, preprocess_image


def _load_and_resize(filepath: str, img_size: tuple, is_mask: bool = False) -> np.ndarray:
    if is_mask:
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Could not read mask: {filepath}")
        img = cv2.resize(img, (img_size[1], img_size[0]), interpolation=cv2.INTER_NEAREST)
        return (img > 0).astype(np.uint8)
    else:
        img = cv2.imread(filepath, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {filepath}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (img_size[1], img_size[0]), interpolation=cv2.INTER_AREA)
        return img


def load_mvtec_ad2_category(cfg: dict) -> dict:
    """
    Loads normal train images and public test images + ground truth masks for an MVTec AD 2 category.
    Applies synthetic lighting shifts ONLY to test images, and applies the configured
    preprocessing mode consistently to BOTH train and test images.
    """
    data_root = cfg["paths"]["data_root"]
    category = cfg["experiment"]["category"]
    img_size = tuple(cfg["experiment"]["image_size"])  # (H, W)
    light_cond = cfg["lighting_shift"]["condition"]
    prep_mode = cfg["preprocessing"]["mode"]

    cat_dir = os.path.join(data_root, category)
    train_dir = os.path.join(cat_dir, "train", "good")
    test_dir = os.path.join(cat_dir, "test_public")
    gt_dir = os.path.join(test_dir, "ground_truth")

    # 1. Load and preprocess normal training images
    train_paths = sorted(glob.glob(os.path.join(train_dir, "*.*")))
    if not train_paths:
        raise FileNotFoundError(f"No training images found in {train_dir}")

    train_images = np.array([
        preprocess_image(_load_and_resize(p, img_size, is_mask=False), prep_mode)
        for p in train_paths
    ], dtype=np.uint8)

    # 2. Load test_public subsets ('good' and defect subfolders)
    test_images = []
    y_true = []
    gt_masks = []

    subfolders = sorted([
        d for d in os.listdir(test_dir)
        if os.path.isdir(os.path.join(test_dir, d)) and d != "ground_truth"
    ])

    for sub in subfolders:
        sub_path = os.path.join(test_dir, sub)
        img_paths = sorted(glob.glob(os.path.join(sub_path, "*.*")))
        is_normal = (sub == "good")

        for p in img_paths:
            raw_img = _load_and_resize(p, img_size, is_mask=False)
            shifted_img = apply_lighting_shift(raw_img, light_cond)
            proc_img = preprocess_image(shifted_img, prep_mode)
            test_images.append(proc_img)

            if is_normal:
                y_true.append(0)
                gt_masks.append(np.zeros(img_size, dtype=np.uint8))
            else:
                y_true.append(1)
                stem = os.path.splitext(os.path.basename(p))[0]
                mask_candidates = glob.glob(os.path.join(gt_dir, sub, f"{stem}*.*"))
                if mask_candidates:
                    mask = _load_and_resize(mask_candidates[0], img_size, is_mask=True)
                else:
                    mask = np.zeros(img_size, dtype=np.uint8)
                gt_masks.append(mask)

    return {
        "train_images": train_images,
        "test_images": np.array(test_images, dtype=np.uint8),
        "y_true": np.array(y_true, dtype=np.int32),
        "gt_masks": np.array(gt_masks, dtype=np.uint8),
    }