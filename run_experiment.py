import argparse
import csv
import os
import random
import numpy as np
import torch
import yaml


from src.evaluation.visualize import save_localization_comparison, save_preprocessing_grid
from src.data.data import load_mvtec_ad2_category
from src.evaluation.metrics import (
    compute_image_metrics,
    compute_pixel_metrics,
    find_optimal_f1_threshold,
    get_model_size_mb,
    profile_inference,
)
from src.models.hog_ocsvm import HOGOneClassSVM
from src.models.padim import PaDiM


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run(cfg: dict, model_name: str):
    set_seed(cfg["experiment"]["seed"])
    os.makedirs(cfg["paths"]["results_dir"], exist_ok=True)

    category = cfg["experiment"]["category"]
    light_cond = cfg["lighting_shift"]["condition"]
    prep_mode = cfg["preprocessing"]["mode"]
    print(f"\n=== Running [{model_name.upper()}] | Category: {category} | Lighting: {light_cond} | Prep: {prep_mode} ===")

    data = load_mvtec_ad2_category(cfg)
    train_imgs = data["train_images"]
    test_imgs = data["test_images"]
    y_true = data["y_true"]
    gt_masks = data["gt_masks"]

    # Initialize & fit model on normal training images
    if model_name == "hog_ocsvm":
        m_cfg = cfg["models"]["hog_ocsvm"]
        detector = HOGOneClassSVM(
            orientations=m_cfg["orientations"],
            pixels_per_cell=tuple(m_cfg["pixels_per_cell"]),
            cells_per_block=tuple(m_cfg["cells_per_block"]),
            nu=m_cfg["nu"],
            kernel=m_cfg["kernel"],
        )
        ckpt_path = os.path.join(cfg["paths"]["results_dir"], f"hog_ocsvm_{category}_{prep_mode}.joblib")
    elif model_name == "padim":
        m_cfg = cfg["models"]["padim"]
        detector = PaDiM(
            backbone=m_cfg["backbone"],
            layers=m_cfg["layers"],
            n_features=m_cfg["n_features"],
            seed=cfg["experiment"]["seed"],
        )
        ckpt_path = os.path.join(cfg["paths"]["results_dir"], f"padim_{category}_{prep_mode}.npz")
    else:
        raise ValueError(f"Unknown model: {model_name}")

    detector.fit(train_imgs)
    detector.save(ckpt_path)

    # Profile computational efficiency on a sample batch
    sample_batch = test_imgs[:min(8, len(test_imgs))]
    perf_stats = profile_inference(lambda x: detector.predict(x), sample_batch, warmup=2)
    model_size_mb = get_model_size_mb(ckpt_path)

    # Full test evaluation
    image_scores, anomaly_maps = detector.predict(test_imgs)
    img_metrics = compute_image_metrics(y_true, image_scores)
    pix_metrics = compute_pixel_metrics(gt_masks, anomaly_maps)

    row = {
        "model": model_name,
        "category": category,
        "lighting": light_cond,
        "preprocessing": prep_mode,
        **img_metrics,
        **pix_metrics,
        **perf_stats,
        "model_size_mb": model_size_mb,
    }

    for k, v in row.items():
        print(f"  {k}: {v}")

    # Append results to CSV
    csv_path = os.path.join(cfg["paths"]["results_dir"], "ablation_results.csv")
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
        
    # Save 4x4 lighting vs. preprocessing grid once per category
    fig_dir = os.path.join(cfg["paths"]["results_dir"], "figures")
    grid_path = os.path.join(fig_dir, f"preprocessing_grid_{category}.png")
    if not os.path.exists(grid_path):
        defect_idx = np.where(y_true == 1)[0]
        sample_idx = defect_idx[0] if len(defect_idx) > 0 else 0
        # Load raw unshifted sample from test_imgs when lighting=='original' & prep=='none'
        if light_cond == "original" and prep_mode == "none":
            save_preprocessing_grid(test_imgs[sample_idx], grid_path, category=category)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--model", choices=["hog_ocsvm", "padim", "both"], default="both")
    parser.add_argument("--data_root", default=None, help="Override paths.data_root")
    parser.add_argument("--category", default=None, help="Override experiment.category")
    parser.add_argument("--lighting", default=None, help="Override lighting_shift.condition")
    parser.add_argument("--prep", default=None, help="Override preprocessing.mode")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    if args.data_root:
        cfg["paths"]["data_root"] = args.data_root
    if args.category:
        cfg["experiment"]["category"] = args.category
    if args.lighting:
        cfg["lighting_shift"]["condition"] = args.lighting
    if args.prep:
        cfg["preprocessing"]["mode"] = args.prep

    models_to_run = ["hog_ocsvm", "padim"] if args.model == "both" else [args.model]
    for m in models_to_run:
        run(cfg, m)