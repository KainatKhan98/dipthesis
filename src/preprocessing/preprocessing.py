import cv2
import numpy as np


def apply_lighting_shift(image: np.ndarray, condition: str = "original",
                         low_gamma: float = 2.2, high_gamma: float = 0.45) -> np.ndarray:
    """
    Simulates unseen factory illumination on an RGB uint8 image (H, W, 3).
    Ground-truth binary defect masks remain unchanged.
    """
    if condition == "original":
        return image.copy()

    img_float = image.astype(np.float32) / 255.0

    if condition == "low_light":
        shifted = np.power(np.clip(img_float, 1e-6, 1.0), low_gamma)
    elif condition == "overexposed":
        shifted = np.power(np.clip(img_float, 1e-6, 1.0), high_gamma)
    elif condition == "uneven":
        # Horizontal illumination gradient (bright left -> dim right)
        h, w, _ = img_float.shape
        gradient = np.linspace(1.25, 0.35, w, dtype=np.float32)[None, :, None]
        shifted = np.clip(img_float * gradient, 0.0, 1.0)
    else:
        raise ValueError(f"Unknown lighting condition: {condition}")

    return (shifted * 255.0).astype(np.uint8)


def apply_clahe(image: np.ndarray, clip_limit: float = 2.0,
                tile_grid_size: tuple = (8, 8)) -> np.ndarray:
    """Applies CLAHE strictly to the L (luminance) channel in LAB space."""
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_eq = clahe.apply(l_chan)
    merged = cv2.merge((l_eq, a_chan, b_chan))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)


def apply_bilateral(image: np.ndarray, d: int = 5,
                    sigma_color: float = 50, sigma_space: float = 50) -> np.ndarray:
    """Reduces sensor noise under low-light amplification while preserving defect edges."""
    return cv2.bilateralFilter(image, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)


def preprocess_image(image: np.ndarray, mode: str = "none", cfg: dict = None) -> np.ndarray:
    """
    Unified preprocessing entry point for the 5-step ablation study:
    Modes: 'none', 'clahe', 'bilateral', 'clahe_bilateral'
    """
    cfg = cfg or {}
    clahe_cfg = cfg.get("clahe", {})
    bilat_cfg = cfg.get("bilateral", {})

    if mode == "none":
        return image
    elif mode == "clahe":
        return apply_clahe(
            image,
            clip_limit=clahe_cfg.get("clip_limit", 2.0),
            tile_grid_size=tuple(clahe_cfg.get("tile_grid_size", (8, 8))),
        )
    elif mode == "bilateral":
        return apply_bilateral(
            image,
            d=bilat_cfg.get("d", 5),
            sigma_color=bilat_cfg.get("sigma_color", 50),
            sigma_space=bilat_cfg.get("sigma_space", 50),
        )
    elif mode == "clahe_bilateral":
        enhanced = apply_clahe(
            image,
            clip_limit=clahe_cfg.get("clip_limit", 2.0),
            tile_grid_size=tuple(clahe_cfg.get("tile_grid_size", (8, 8))),
        )
        return apply_bilateral(
            enhanced,
            d=bilat_cfg.get("d", 5),
            sigma_color=bilat_cfg.get("sigma_color", 50),
            sigma_space=bilat_cfg.get("sigma_space", 50),
        )
    else:
        raise ValueError(f"Unsupported preprocessing mode: {mode}")