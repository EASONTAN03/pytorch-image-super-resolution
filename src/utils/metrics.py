import torch
import torch.nn as nn
import torch.nn.functional as F

class PSNR(nn.Module):
    def __init__(self, max_val: float = 1.0, crop_border: int = 0):
        super().__init__()
        self.max_val = max_val
        self.crop_border = crop_border

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        if img1.shape != img2.shape:
            raise ValueError(f"Image shapes must match. Got {img1.shape} and {img2.shape}")
        if img1.device != img2.device:
            img2 = img2.to(img1.device)
        if self.crop_border > 0:
            img1 = img1[..., self.crop_border:-self.crop_border, self.crop_border:-self.crop_border]
            img2 = img2[..., self.crop_border:-self.crop_border, self.crop_border:-self.crop_border]
        mse = F.mse_loss(img1, img2, reduction='mean')
        if mse == 0:
            return torch.tensor(float('inf'))
        psnr = 10 * torch.log10(self.max_val ** 2 / mse)
        return psnr

class SSIM(nn.Module):
    def __init__(self, window_size: int = 11, channels: int = 1, max_val: float = 1.0, crop_border: int = 0):
        super().__init__()
        self.window_size = window_size
        self.channels = channels
        self.max_val = max_val
        self.crop_border = crop_border
        self.C1 = (0.01 * self.max_val) ** 2
        self.C2 = (0.03 * self.max_val) ** 2
        self.gaussian_window = self._create_gaussian_window(window_size, channels)

    def _create_gaussian_window(self, window_size, channels):
        _1D_gaussian = torch.exp(-(torch.arange(window_size, dtype=torch.float32) - window_size // 2).pow(2) / float(window_size // 2)**2 / 2)
        _1D_gaussian = _1D_gaussian / _1D_gaussian.sum()
        _2D_gaussian = _1D_gaussian.unsqueeze(0) * _1D_gaussian.unsqueeze(1)
        window = _2D_gaussian.expand(channels, 1, window_size, window_size).contiguous()
        return window

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        if img1.shape != img2.shape:
            raise ValueError(f"Image shapes must match. Got {img1.shape} and {img2.shape}")
        if img1.device != img2.device:
            img2 = img2.to(img1.device)
        window = self.gaussian_window.to(img1.device)
        if self.crop_border > 0:
            img1 = img1[..., self.crop_border:-self.crop_border, self.crop_border:-self.crop_border]
            img2 = img2[..., self.crop_border:-self.crop_border, self.crop_border:-self.crop_border]

        mu1 = F.conv2d(img1, window, padding=0, groups=self.channels, stride=1)
        mu2 = F.conv2d(img2, window, padding=0, groups=self.channels, stride=1)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=0, groups=self.channels, stride=1) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=0, groups=self.channels, stride=1) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=0, groups=self.channels, stride=1) - mu1_mu2

        numerator = (2 * mu1_mu2 + self.C1) * (2 * sigma12 + self.C2)
        denominator = (mu1_sq + mu2_sq + self.C1) * (sigma1_sq + sigma2_sq + self.C2)
        ssim_map = numerator / denominator
        return ssim_map.mean()