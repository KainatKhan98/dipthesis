import random
import numpy as np
from scipy.ndimage import gaussian_filter
import torch
import torch.nn.functional as F
from torchvision import models, transforms


class PaDiM:
    """
    Lightweight Patch Distribution Modeling (PaDiM) using a frozen ResNet18 backbone
    with random feature subsampling and top-1% patch aggregation.
    """

    def __init__(self, backbone: str = "resnet18", layers: list = None,
                 n_features: int = 100, seed: int = 42, device: str = None):
        self.layers = layers or ["layer1", "layer2", "layer3"]
        self.n_features = n_features
        self.seed = seed
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        weights = models.ResNet18_Weights.DEFAULT
        self.model = models.resnet18(weights=weights).to(self.device)
        self.model.eval()

        self._features = {}
        for layer_name in self.layers:
            layer = getattr(self.model, layer_name)
            layer.register_forward_hook(self._make_hook(layer_name))

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        self.selected_indices = None
        self.means = None
        self.inv_covs = None

    def _make_hook(self, name: str):
        def hook(_, __, output):
            self._features[name] = output.detach()
        return hook

    def _preprocess_batch(self, images: np.ndarray) -> torch.Tensor:
        tensor = torch.from_numpy(images).permute(0, 3, 1, 2).float() / 255.0
        return self.normalize(tensor).to(self.device)

    @torch.no_grad()
    def _extract_embeddings(self, images: np.ndarray, batch_size: int = 16) -> torch.Tensor:
        all_embeddings = []
        for i in range(0, len(images), batch_size):
            batch = self._preprocess_batch(images[i:i + batch_size])
            self._features.clear()
            _ = self.model(batch)

            target_h, target_w = self._features[self.layers[0]].shape[-2:]
            resized_maps = []
            for layer_name in self.layers:
                fmap = self._features[layer_name]
                if fmap.shape[-2:] != (target_h, target_w):
                    fmap = F.interpolate(fmap, size=(target_h, target_w), mode="bilinear", align_corners=False)
                resized_maps.append(fmap)

            concat = torch.cat(resized_maps, dim=1)

            if self.selected_indices is None:
                total_c = concat.shape[1]
                rng = random.Random(self.seed)
                k = min(self.n_features, total_c)
                self.selected_indices = torch.tensor(sorted(rng.sample(range(total_c), k)), device=self.device)

            concat = torch.index_select(concat, dim=1, index=self.selected_indices)
            all_embeddings.append(concat.cpu())

        return torch.cat(all_embeddings, dim=0)

    def fit(self, train_images: np.ndarray, batch_size: int = 16):
        embeddings = self._extract_embeddings(train_images, batch_size=batch_size)
        n, c, h_p, w_p = embeddings.shape
        embeddings = embeddings.view(n, c, h_p * w_p).numpy()

        self.means = np.mean(embeddings, axis=0)
        self.inv_covs = np.zeros((h_p * w_p, c, c), dtype=np.float32)
        reg = 0.01 * np.eye(c, dtype=np.float32)

        for p in range(h_p * w_p):
            cov = np.cov(embeddings[:, :, p], rowvar=False) + reg
            self.inv_covs[p] = np.linalg.inv(cov)

        self.patch_grid_shape = (h_p, w_p)
        return self

    def predict(self, test_images: np.ndarray, batch_size: int = 16) -> tuple:
        """
        Computes Mahalanobis distance maps upsampled to input resolution.
        Uses top-1% spatial patch mean for image-level scoring to suppress isolated glare spikes.
        """
        h_img, w_img = test_images.shape[1], test_images.shape[2]
        embeddings = self._extract_embeddings(test_images, batch_size=batch_size)
        n, c, h_p, w_p = embeddings.shape
        embeddings = embeddings.view(n, c, h_p * w_p).numpy()

        dist_maps = np.zeros((n, h_p * w_p), dtype=np.float32)
        for p in range(h_p * w_p):
            delta = embeddings[:, :, p] - self.means[:, p]
            dist_sq = np.sum((delta @ self.inv_covs[p]) * delta, axis=1)
            dist_maps[:, p] = np.sqrt(np.maximum(dist_sq, 0.0))

        dist_maps = dist_maps.reshape(n, h_p, w_p)
        dist_tensor = torch.from_numpy(dist_maps).unsqueeze(1)
        upsampled = F.interpolate(dist_tensor, size=(h_img, w_img), mode="bilinear", align_corners=False)
        upsampled = upsampled.squeeze(1).numpy()

        anomaly_maps = np.array([gaussian_filter(m, sigma=4) for m in upsampled], dtype=np.float32)

        # Top-1% pixel aggregation instead of single noisy maximum
        flat_maps = anomaly_maps.reshape(n, -1)
        top_k = max(1, int(0.01 * flat_maps.shape[1]))
        image_scores = np.mean(np.sort(flat_maps, axis=1)[:, -top_k:], axis=1)

        return image_scores, anomaly_maps

    def save(self, filepath: str):
        np.savez_compressed(
            filepath,
            means=self.means,
            inv_covs=self.inv_covs,
            selected_indices=self.selected_indices.cpu().numpy(),
            patch_grid_shape=np.array(self.patch_grid_shape),
        )