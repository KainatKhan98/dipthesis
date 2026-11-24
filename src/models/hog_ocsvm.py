import joblib
import numpy as np
from skimage.color import rgb2gray
from skimage.feature import hog
from skimage.transform import resize
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM


class HOGOneClassSVM:
    """
    Classical visual anomaly detector combining HOG feature extraction
    with a One-Class SVM trained solely on normal images.
    """

    def __init__(self, orientations: int = 9, pixels_per_cell: tuple = (16, 16),
                 cells_per_block: tuple = (2, 2), nu: float = 0.05, kernel: str = "rbf"):
        self.orientations = orientations
        self.pixels_per_cell = tuple(pixels_per_cell)
        self.cells_per_block = tuple(cells_per_block)
        self.scaler = StandardScaler()
        self.model = OneClassSVM(nu=nu, kernel=kernel, gamma="scale")

    def _extract_features(self, images: np.ndarray) -> np.ndarray:
        """Extracts flattened HOG descriptors for a batch of RGB uint8 images (N, H, W, 3)."""
        features = []
        for img in images:
            gray = rgb2gray(img)
            feat = hog(
                gray,
                orientations=self.orientations,
                pixels_per_cell=self.pixels_per_cell,
                cells_per_block=self.cells_per_block,
                block_norm="L2-Hys",
                feature_vector=True,
            )
            features.append(feat)
        return np.asarray(features, dtype=np.float32)

    def _extract_patch_features(self, images: np.ndarray) -> tuple:
        """
        Extracts unflattened HOG block grids (N, grid_h, grid_w, feat_dim)
        to build spatial anomaly heatmaps for pixel-level localization.
        """
        patch_grids = []
        for img in images:
            gray = rgb2gray(img)
            grid = hog(
                gray,
                orientations=self.orientations,
                pixels_per_cell=self.pixels_per_cell,
                cells_per_block=self.cells_per_block,
                block_norm="L2-Hys",
                feature_vector=False,
            )
            # grid shape: (n_blocks_y, n_blocks_x, cy, cx, orientations)
            gy, gx = grid.shape[0], grid.shape[1]
            patch_grids.append(grid.reshape(gy, gx, -1))
        return np.asarray(patch_grids, dtype=np.float32)

    def fit(self, train_images: np.ndarray):
        """Fits One-Class SVM on normal training images and stores patch-level reference mean."""
        X_train = self._extract_features(train_images)
        X_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_scaled)

        # Store normal patch template for spatial localization map
        train_patches = self._extract_patch_features(train_images)
        self.patch_mean_ = np.mean(train_patches, axis=0)
        self.patch_std_ = np.std(train_patches, axis=0) + 1e-6
        return self

    def predict(self, test_images: np.ndarray) -> tuple:
        """
        Returns:
            image_scores: 1D array (N,) where higher values indicate anomalies
            anomaly_maps: 3D array (N, H, W) resized to input image dimensions
        """
        X_test = self._extract_features(test_images)
        X_scaled = self.scaler.transform(X_test)
        # decision_function returns positive for inliers, negative for outliers -> negate
        image_scores = -self.model.decision_function(X_scaled)

        # Build spatial anomaly maps from patch deviation
        h, w = test_images.shape[1], test_images.shape[2]
        test_patches = self._extract_patch_features(test_images)
        z_scores = np.linalg.norm((test_patches - self.patch_mean_) / self.patch_std_, axis=-1)

        anomaly_maps = np.array([
            resize(z_map, (h, w), order=1, mode="reflect", anti_aliasing=True)
            for z_map in z_scores
        ], dtype=np.float32)

        return image_scores, anomaly_maps

    def save(self, filepath: str):
        joblib.dump(self, filepath)