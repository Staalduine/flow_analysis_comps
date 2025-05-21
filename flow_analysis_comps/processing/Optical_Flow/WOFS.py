import numpy as np
import pywt
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import minimize
from scipy.ndimage import sobel, laplace

class WaveletOpticalFlow:
    def __init__(self, I0, I1, wavelet='bior4.4', sigma=1.0, alpha=0.1, num_scales=3):
        """
        I0, I1: input grayscale images (2D numpy arrays)
        wavelet: wavelet name, biorthogonal 9/7 -> 'bior4.4'
        sigma: noise scale for Lorentzian data term
        alpha: regularization weight
        num_scales: number of multiscale levels
        """
        self.I0 = I0.astype(np.float32)
        self.I1 = I1.astype(np.float32)
        self.wavelet = wavelet
        self.sigma = sigma
        self.alpha = alpha
        self.num_scales = num_scales
        self.H, self.W = I0.shape
        self.v = np.zeros((self.H, self.W, 2), dtype=np.float32)  # velocity init
        
    # --- Wavelet transforms ---
    def forward_wavelet_transform(self, v):
        coeffs_x = pywt.wavedec2(v[..., 0], self.wavelet, level=None)
        coeffs_y = pywt.wavedec2(v[..., 1], self.wavelet, level=None)
        return coeffs_x, coeffs_y

    def inverse_wavelet_transform(self, coeffs_x, coeffs_y):
        v_x = pywt.waverec2(coeffs_x, self.wavelet)
        v_y = pywt.waverec2(coeffs_y, self.wavelet)
        shape = (self.H, self.W, 2)
        v = np.stack([v_x, v_y], axis=-1)
        return v[:shape[0], :shape[1], :]
    
    # --- Bicubic spline warping ---
    def warp_image(self, I, flow):
        grid_y, grid_x = np.meshgrid(np.arange(self.H), np.arange(self.W), indexing='ij')
        coords_x = grid_x + flow[..., 0]
        coords_y = grid_y + flow[..., 1]
        spline = RectBivariateSpline(np.arange(self.H), np.arange(self.W), I, kx=3, ky=3, s=0)
        warped = spline.ev(coords_y.ravel(), coords_x.ravel())
        return warped.reshape(self.H, self.W)
    
    # --- Regularization penalty and gradient ---
    def regularization_penalty(self, v):
        grad_x = np.gradient(v[..., 0])
        grad_y = np.gradient(v[..., 1])
        penalty = np.sum(grad_x[0]**2 + grad_x[1]**2 + grad_y[0]**2 + grad_y[1]**2)
        return penalty

    def regularization_gradient(self, v):
        grad_x = laplace(v[..., 0])
        grad_y = laplace(v[..., 1])
        return np.stack([grad_x, grad_y], axis=-1)
    
    # --- Lorentzian data term and gradients ---
    def lorentzian_data_term(self, I0, I1, flow):
        I1_warped = self.warp_image(I1, flow)
        diff = (I0 - I1_warped) / self.sigma
        loss = np.log(1 + 0.5 * diff**2)
        return np.sum(loss), I1_warped, diff

    def image_gradient(self, I):
        grad_x = sobel(I, axis=1, mode='reflect') / 8.0
        grad_y = sobel(I, axis=0, mode='reflect') / 8.0
        return grad_x, grad_y

    def lorentzian_gradient_data_term(self, I0, I1, flow):
        I1_warped = self.warp_image(I1, flow)
        diff = I0 - I1_warped

        grad_x, grad_y = self.image_gradient(I1)
        
        # Warp gradients to flow coordinates
        def warp(img):
            spline = RectBivariateSpline(np.arange(self.H), np.arange(self.W), img, kx=3, ky=3, s=0)
            coords_x = np.arange(self.W) + flow[..., 0]
            coords_y = np.arange(self.H) + flow[..., 1]
            grid_x, grid_y = np.meshgrid(coords_x, coords_y)
            warped_vals = spline.ev(grid_y.ravel(), grid_x.ravel())
            return warped_vals.reshape(self.H, self.W)

        grad_x_warped = warp(grad_x)
        grad_y_warped = warp(grad_y)

        denominator = 2 * self.sigma**2 + diff**2
        scalar = -diff / denominator

        grad = np.zeros_like(flow)
        grad[..., 0] = scalar * grad_x_warped
        grad[..., 1] = scalar * grad_y_warped
        
        return grad
    
    # --- Helper functions for optimizer vectorization ---
    def flatten_coeffs(self, coeffs_x, coeffs_y):
        flat_x = np.concatenate([c.ravel() for c in coeffs_x])
        flat_y = np.concatenate([c.ravel() for c in coeffs_y])
        return np.concatenate([flat_x, flat_y])

    def unflatten_coeffs(self, flat, coeffs_x_template, coeffs_y_template):
        coeffs_x = []
        coeffs_y = []
        idx = 0
        for c in coeffs_x_template:
            size = c.size
            coeffs_x.append(flat[idx:idx+size].reshape(c.shape))
            idx += size
        for c in coeffs_y_template:
            size = c.size
            coeffs_y.append(flat[idx:idx+size].reshape(c.shape))
            idx += size
        return coeffs_x, coeffs_y

    # --- Objective and gradient function for optimizer ---
    def objective_and_grad(self, psi_coeffs):
        coeffs_x, coeffs_y = psi_coeffs
        v = self.inverse_wavelet_transform(coeffs_x, coeffs_y)
        
        JD, _, _ = self.lorentzian_data_term(self.I0, self.I1, v)
        JR = self.regularization_penalty(v)
        J = JD + self.alpha * JR
        
        grad_JD = self.lorentzian_gradient_data_term(self.I0, self.I1, v)
        grad_JR = self.regularization_gradient(v)
        grad_v = grad_JD + self.alpha * grad_JR
        
        grad_coeffs_x = pywt.wavedec2(grad_v[..., 0], self.wavelet)
        grad_coeffs_y = pywt.wavedec2(grad_v[..., 1], self.wavelet)
        
        return J, (grad_coeffs_x, grad_coeffs_y)
    
    # --- Optimization wrapper ---
    def optimize_at_scale(self, psi_init_coeffs):
        x0 = self.flatten_coeffs(*psi_init_coeffs)

        def func(x):
            coeffs_x, coeffs_y = self.unflatten_coeffs(x, psi_init_coeffs[0], psi_init_coeffs[1])
            val, grad = self.objective_and_grad((coeffs_x, coeffs_y))
            grad_flat = self.flatten_coeffs(*grad)
            return val, grad_flat

        result = minimize(func, x0, method='L-BFGS-B', jac=True, options={'maxiter': 100})
        coeffs_x_opt, coeffs_y_opt = self.unflatten_coeffs(result.x, psi_init_coeffs[0], psi_init_coeffs[1])
        return coeffs_x_opt, coeffs_y_opt

    # --- Multiscale processing ---
    def run_multiscale(self):
        v = self.v.copy()
        for s in range(self.num_scales):
            # Optional: downscale images & velocity for pyramid - not shown here
            
            psi_init_coeffs = self.forward_wavelet_transform(v)
            psi_opt_coeffs = self.optimize_at_scale(psi_init_coeffs)
            v = self.inverse_wavelet_transform(*psi_opt_coeffs)
            self.v = v
            # Optional: upsample velocity for next scale
            
        return v
