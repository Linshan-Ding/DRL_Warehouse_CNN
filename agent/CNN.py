import torch.nn as nn

class CNNFeatureExtractor(nn.Module):
    """
    深层卷积特征提取器 - 无区域划分版本
    输入: [batch_size, 4, H, W]
    输出: [batch_size, feature_dim]
    """

    def __init__(self, input_channels=4, feature_dim=256):
        super(CNNFeatureExtractor, self).__init__()

        # 深层卷积主干
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),  # -> 尺寸减半

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(2),  # -> 再次减半

            nn.AdaptiveAvgPool2d((1, 1))  # 全局平均池化
        )

        # 全连接层
        self.fc_layers = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # 展平
        x = self.fc_layers(x)
        return x
