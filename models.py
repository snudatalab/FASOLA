import torch
import torch.nn as nn
import torch.nn.functional as F

class CPResNet(nn.Module):
    """
    CP-ResNet model architecture for audio classification.
    """
    def __init__(self, num_classes=10, rho=24, input_channels=1):
        super().__init__()
        
        self.conv1 = nn.Conv2d(input_channels, rho, kernel_size=5, stride=2, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(rho)
        self.relu = nn.ReLU(inplace=True)
        
        self.layer1 = self._make_layer(rho, rho, kernel_size=3)
        self.layer2 = self._make_layer(rho, 2*rho, kernel_size=3)
        self.layer3 = self._make_layer(2*rho, 4*rho, kernel_size=3)
        self.layer4 = self._make_layer(4*rho, 4*rho, kernel_size=3)
        
        # Classification Head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(4*rho, num_classes)
        
    def _make_layer(self, in_channels, out_channels, kernel_size):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=kernel_size//2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2) 
        )

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
            
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def get_dcase_model(model_name="cp-resnet", num_classes=10, input_dim=128):
    """
    Factory function to initialize DCASE models.

    Args:
        model_name (str): Name of the model (default: "cp-resnet").
        num_classes (int): Number of output classes.
        input_dim (int): Input dimension (not used for CPResNet but kept for interface).

    Returns:
        nn.Module: Initialized model.
    """
    if model_name.lower() == "cp-resnet":
        return CPResNet(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")
