import numpy as np

def mirror_pad_with_exponential_fade(img: np.ndarray, pad: int, alpha: float = 3.0) -> np.ndarray:
    """Pad image with mirror padding and exponential fade to black at the edges. Removes ringing in Fourier analysis."""
    # Mirror pad
    padded = np.pad(img, pad_width=pad, mode='reflect')
    h, w = padded.shape

    # Create exponential fade mask
    # y = np.linspace(-1, 1, h)
    # x = np.linspace(-1, 1, w)

    y = np.concatenate([
        np.linspace(-1, 0, pad),
        np.zeros(img.shape[0]),
        np.linspace(0, 1, pad)
    ])
    x = np.concatenate([
        np.linspace(-1, 0, pad),
        np.zeros(img.shape[1]),
        np.linspace(0, 1, pad)
    ])



    xv, yv = np.meshgrid(x, y)
    dist = np.sqrt(xv**2 + yv**2)
    # Create mask of original image location
    # mask = np.zeros_like(padded, dtype=bool)
    # mask[pad:-pad, pad:-pad] = True
    fade = np.exp(-alpha * dist)  # Exponential fade

    # Apply fade
    faded = padded * fade
    return faded.astype(img.dtype)