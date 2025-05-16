import numpy as np
import cv2
import matplotlib.pyplot as plt
import tifffile
from skimage.filters import frangi
from skimage.exposure import rescale_intensity
from tqdm import trange
from joblib import Parallel, delayed


class FocusAnalyzer:
    def __init__(self, tiff_path):
        self.tiff_path = tiff_path
        self.stack = self._load_tiff()
        self.laplacian_variances = []
        self.masked_stack = None

    def _load_tiff(self):
        try:
            stack = tifffile.imread(self.tiff_path)
            if stack.ndim != 3:
                raise ValueError("TIFF must be a 3D (t, y, x) stack.")
            return stack
        except Exception as e:
            raise RuntimeError(f"Failed to load TIFF: {e}")

    def apply_curvilinear_filter(self, threshold=0.1, n_jobs=-1):
        """Applies Frangi filter in parallel to enhance curvilinear structures and mask the image stack."""

        def process_frame(frame):
            frame = frame.astype(np.float32)
            frangi_response = frangi(frame)
            frangi_norm = rescale_intensity(frangi_response, out_range=(0, 1))
            mask = frangi_norm > threshold
            masked = np.where(mask, frame, 0)
            return masked

        print("Applying Frangi filter in parallel...")
        results = Parallel(n_jobs=n_jobs)(
            delayed(process_frame)(self.stack[t]) for t in range(self.stack.shape[0])
        )

        self.masked_stack = np.array(results, dtype=np.float32)
        print("Curvilinear structure filtering complete.")
        return self.masked_stack


    def compute_laplacian_variance(self, use_masked=False):
        """Computes Laplacian variance for each frame in the stack or masked stack."""
        self.laplacian_variances = []
        if use_masked:
            self.apply_curvilinear_filter()
        stack_to_use = self.masked_stack if self.masked_stack is not None else self.stack

        for t in range(stack_to_use.shape[0]):
            frame = stack_to_use[t]
            laplacian = cv2.Laplacian(frame, cv2.CV_64F)
            variance = laplacian.var()
            self.laplacian_variances.append(variance)
        return self.laplacian_variances

    def plot_focus_metric(self):
        if not self.laplacian_variances:
            raise RuntimeError("Laplacian variance not computed. Run compute_laplacian_variance() first.")
        
        plt.figure(figsize=(10, 4))
        plt.plot(self.laplacian_variances, marker='o')
        plt.title("Laplacian Variance Over Time")
        plt.xlabel("Frame Index (Time)")
        plt.ylabel("Laplacian Variance (Focus Metric)")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def show_best_focus_frame(self, use_masked=False):
        if not self.laplacian_variances:
            raise RuntimeError("Laplacian variance not computed. Run compute_laplacian_variance() first.")

        best_idx = np.argmax(self.laplacian_variances)
        stack_to_use = self.masked_stack if use_masked and self.masked_stack is not None else self.stack
        best_frame = stack_to_use[best_idx]

        plt.figure(figsize=(6, 6))
        plt.imshow(best_frame, cmap='gray')
        plt.title(f"Best Focus Frame: {best_idx}")
        plt.axis('off')
        plt.tight_layout()
        plt.show()
