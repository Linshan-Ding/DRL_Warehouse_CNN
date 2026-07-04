import torch.nn as nn


def _group_norm(num_channels: int) -> nn.Module:

    preferred = [32, 16, 8, 4, 2, 1]
    for g in preferred:
        if num_channels % g == 0:
            return nn.GroupNorm(g, num_channels)
    return nn.GroupNorm(1, num_channels)

class CNNFeatureExtractor(nn.Module):
    """
    输入:  [batch_size, 4, H, W]
    输出:  [batch_size, feature_dim]
    """

    def __init__(self, input_channels: int = 4, feature_dim: int = 256):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            _group_norm(64),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            _group_norm(128),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            _group_norm(256),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            _group_norm(512),
            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, feature_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x
