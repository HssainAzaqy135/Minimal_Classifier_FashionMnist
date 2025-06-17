import torch
import matplotlib.pyplot as plt

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

def show_images(data_loader, n=6, class_names=None):
    """
    Display n images from a PyTorch DataLoader.

    Args:
        data_loader: PyTorch DataLoader (e.g., train_loader)
        n: Number of images to show
        class_names: Optional list mapping label indices to class names
    """
    images, labels = next(iter(data_loader))# First batch

    # Ensure n doesn't exceed batch size
    n = min(n, images.size(0))

    # Squeeze channel and convert to numpy
    images = images.squeeze().numpy()

    # Plot
    fig, axes = plt.subplots(1, n, figsize=(n * 2, 2))
    for i in range(n):
        ax = axes[i] if n > 1 else axes
        ax.imshow(images[i], cmap='gray')
        label = labels[i].item()
        if class_names:
            ax.set_title(class_names[label])
        else:
            ax.set_title(f"Label: {label}")
        ax.axis('off')
    plt.tight_layout()
    plt.show()
