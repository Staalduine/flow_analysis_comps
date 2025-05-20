import numpy as np
import cv2
import matplotlib.pyplot as plt
import tifffile
from skimage.filters import frangi
from skimage.exposure import rescale_intensity
from tqdm import trange
from joblib import Parallel, delayed
from skimage.util import img_as_float
from sklearn.preprocessing import MinMaxScaler

class FocusAnalyzer:
    def __init__(self, tiff_path, frangi_sigmas=[30]):
        self.tiff_path = tiff_path
        self.stack = self._load_tiff()
        self.laplacian_variances = []
        self.masked_stack = None
        self.frangi_sigmas = frangi_sigmas
        self.lap_norm  = []
        self.fft_norm  = []
        # self.auto_norm = []
    def _load_tiff(self):
        try:
            stack = tifffile.imread(self.tiff_path)
            if stack.ndim != 3:
                raise ValueError("TIFF must be a 3D (t, y, x) stack.")
            return stack
        except Exception as e:
            raise RuntimeError(f"Failed to load TIFF: {e}")
        
    def _power_spectrum_energy(self, frame, radius=30):
        f = np.fft.fft2(frame)
        fshift = np.fft.fftshift(f)
        magnitude = np.abs(fshift)

        h, w = frame.shape
        cy, cx = h // 2, w // 2
        Y, X = np.ogrid[:h, :w]
        dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)

        high_pass = magnitude[dist > radius]
        return np.mean(high_pass)
    
    def compute_combined_focus_metrics(self, use_masked=False, fft_radius=30):
        """Computes focus scores using Laplacian, Power Spectrum, and Autocorrelation."""
        stack = self.masked_stack if use_masked and self.masked_stack is not None else self.stack
        laplacian_scores = []
        fft_scores = []
        autocorr_scores = []

        for frame in stack:
            frame = img_as_float(frame)
            lap = cv2.Laplacian(frame, cv2.CV_32F).var()
            fft = self._power_spectrum_energy(frame, radius=fft_radius)
            # auto = self._autocorrelation_peak(frame)
            
            laplacian_scores.append(lap)
            fft_scores.append(fft)
            # autocorr_scores.append(auto)

        # Normalize each score list to [0, 1]
        scaler = MinMaxScaler()
        self.lap_norm = scaler.fit_transform(np.array(laplacian_scores).reshape(-1, 1)).flatten()
        self.fft_norm = scaler.fit_transform(np.array(fft_scores).reshape(-1, 1)).flatten()
        # self.auto_norm = scaler.fit_transform(1 - np.array(autocorr_scores).reshape(-1, 1)).flatten()  # inverse = sharper

        # Combined score (equal weights)
        self.combined_focus_score = (self.lap_norm + self.fft_norm) / 2
        self.laplacian_variances = laplacian_scores  # Store in case needed elsewhere

        return self.combined_focus_score

    # def _autocorrelation_peak(self, frame):
    #     normed = (frame - np.mean(frame)) / (np.std(frame) + 1e-8)
    #     corr = cv2.matchTemplate(normed, normed, method=cv2.TM_CCORR_NORMED)
    #     return np.max(corr)

    def apply_curvilinear_filter(self, threshold=0.01, n_jobs=-1):
        """Applies Frangi filter in parallel to enhance curvilinear structures and mask the image stack."""
        results = np.zeros_like(self.stack, dtype=np.float32)

        stack_avg = self.stack.mean(axis=0)
        frame = stack_avg.astype(np.float32)
        frangi_response = frangi(frame, sigmas=self.frangi_sigmas, scale_range=None, scale_step=None)
        frangi_norm = rescale_intensity(frangi_response, out_range=(0, 1))
        mask = frangi_norm > threshold
        for i, frame in enumerate(self.stack):
            masked = np.where(mask, frame, 0)
            results[i] = masked

        self.masked_stack = np.array(results, dtype=np.float32)
        print("Curvilinear structure filtering complete.")
        return self.masked_stack


    def compute_laplacian_variance(self, use_masked=False):
        """Computes Laplacian variance for each frame in the stack or masked stack."""
        self.laplacian_variances = []
        if use_masked and self.masked_stack is None:
            self.apply_curvilinear_filter()
        stack_to_use = self.masked_stack if self.masked_stack is not None else self.stack

        for t in range(stack_to_use.shape[0]):
            frame = stack_to_use[t]
            laplacian = cv2.Laplacian(frame, cv2.CV_32F)
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

    def plot_combined_focus(self):
        if not hasattr(self, "combined_focus_score"):
            raise RuntimeError("Run compute_combined_focus_metrics() first.")

        plt.figure(figsize=(10, 4))
        plt.plot(self.combined_focus_score, label="Combined Focus Score", color='purple')
        plt.plot(self.fft_norm, label="FFT Score", color='red')
        plt.plot(self.lap_norm, label="Laplacian Score", color='green')
        # plt.plot(self.auto_norm, label="Autocorr Score", color='blue')
        plt.title("Combined Focus Metric Over Time")
        plt.xlabel("Frame Index")
        plt.ylabel("Normalized Score")
        plt.grid(True)
        plt.tight_layout()
        plt.legend()
        plt.show()

    def show_best_focus_frame_combined(self, use_masked=False):
        if not hasattr(self, "combined_focus_score"):
            raise RuntimeError("Run compute_combined_focus_metrics() first.")
        
        best_idx = np.argmax(self.combined_focus_score)
        best_frame = self.stack[best_idx]

        plt.figure(figsize=(6, 6))
        plt.imshow(best_frame, cmap='gray')
        plt.title(f"Best Focus Frame (Combined): {best_idx}")
        plt.axis('off')
        plt.tight_layout()
        plt.show()