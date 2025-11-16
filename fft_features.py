import torch
import torch.fft as fft

def fft_features(image):
    """
    Compute grayscale FFT magnitude per image.
    Input:  image tensor of shape (B, C, H, W)
    Output: magnitude tensor of shape (B, 1, H, W)
    """
    # 2D FFT over spatial dims
    fft_image = fft.fft2(image, dim=(-2, -1))
    fft_image = fft.fftshift(fft_image, dim=(-2, -1))
    mag = torch.abs(fft_image)              # (B, C, H, W)
    mag = mag.mean(dim=1, keepdim=True)     # collapse channels -> (B, 1, H, W)
    return mag
