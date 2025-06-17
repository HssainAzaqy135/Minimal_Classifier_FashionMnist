import torch

def compute_mean_std(train_loader):
    total_sum = 0.0
    total_squared = 0.0
    total_pixels = 0

    for images, _ in train_loader:
        # images shape: [batch_size, 1, height, width]
        total_sum += images.sum().item()
        total_squared += (images ** 2).sum().item()
        total_pixels += images.numel()

    mean = total_sum / total_pixels
    var = total_squared / total_pixels - mean ** 2
    std = var ** 0.5

    return mean, std
