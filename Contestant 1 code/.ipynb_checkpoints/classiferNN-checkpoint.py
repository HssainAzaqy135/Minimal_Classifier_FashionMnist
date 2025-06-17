import torch
import torch.nn as nn
import torch.nn.functional as F
import time

class BasicNN(nn.Module):
    def __init__(self,dim,hidden_dims=[64,32]):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(dim*dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], 10)
        )

    def forward(self, x):
        return self.network(x)

    def get_num_params(self):
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total trainable parameters: {total_params}")


class CNN(nn.Module):
    def __init__(self, dim, conv_channels=[8, 16], fc_dims=[64], kernel_size=3, stride=2, padding=None, max_pooling=False):
        super().__init__()

        num_layers = len(conv_channels)

        # Normalize kernel_size and stride
        kernel_sizes = [kernel_size] * num_layers if isinstance(kernel_size, int) else kernel_size
        strides = [stride] * num_layers if isinstance(stride, int) else stride

        # Handle padding: if None, default to 'same' (i.e., kernel_size // 2)
        if padding is None:
            paddings = [k // 2 for k in kernel_sizes]
        else:
            paddings = [padding] * num_layers if isinstance(padding, int) else padding

        assert len(kernel_sizes) == num_layers, "kernel_size list must match conv_channels"
        assert len(strides) == num_layers, "stride list must match conv_channels"
        assert len(paddings) == num_layers, "padding list must match conv_channels"

        layers = []
        in_channels = 1
        current_dim = dim

        for i in range(num_layers):
            k = kernel_sizes[i]
            s = strides[i]
            p = paddings[i]

            layers.append(nn.Conv2d(in_channels, conv_channels[i], kernel_size=k, stride=s, padding=p))
            layers.append(nn.ReLU())
            in_channels = conv_channels[i]

            current_dim = (current_dim + 2 * p - k) // s + 1  # after conv

            if max_pooling:
                layers.append(nn.MaxPool2d(2))
                current_dim = current_dim // 2

        self.conv = nn.Sequential(*layers)
        conv_output_dim = conv_channels[-1] * current_dim * current_dim

        # Fully connected layers
        fc_layers = []
        in_dim = conv_output_dim
        for h in fc_dims:
            fc_layers.append(nn.Linear(in_dim, h))
            fc_layers.append(nn.ReLU())
            in_dim = h
        fc_layers.append(nn.Linear(in_dim, 10))

        self.fc = nn.Sequential(*fc_layers)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def get_num_params(self):
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total trainable parameters: {total_params}")




# ---------------------------------------------- Training code ----------------------------------------------
def train_model(model, train_loader, val_loader, optimizer, criterion, num_epochs, device, val_every_k=5):
    model.to(device)

    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        total_loss = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        end_time = time.time()
        epoch_duration = end_time - start_time

        print(f"Epoch {epoch+1}/{num_epochs} completed in {epoch_duration:.2f}s.")

        # Validate every k epochs
        if (epoch + 1) % val_every_k == 0 or epoch == num_epochs - 1:
            val_acc = evaluate_model(model, val_loader, device)
            print(f"Validation Accuracy after epoch {epoch+1}: {val_acc:.2f}%")



def evaluate_model(model, data_loader, device):
    model.to(device)
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return 100 * correct / total


