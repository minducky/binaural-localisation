import torch.nn as nn


class Dummy(nn.Module):
    """Dummy Conv2D baseline. Input: (B, 44100, 2) → Output: (B, num_classes)"""
    def __init__(self, config):
        super().__init__()
        num_classes = config['NUM_CLASSES']
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(64, 2), stride=(16, 1)),   # (B, 16, 2753, 1)
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=(16, 1), stride=(8, 1)),   # (B, 32, 342, 1)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(8, 1), stride=(4, 1)),    # (B, 64, 84, 1)
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((16, 1)),                            # (B, 64, 16, 1)
        )
        self.fc = nn.Linear(64 * 16, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)        # (B, 1, 44100, 2)
        x = self.conv(x)          # (B, 64, 16, 1)
        x = x.flatten(1)          # (B, 1024)
        return self.fc(x)         # (B, num_classes)


class Dummy1M(nn.Module):
    """~1M param Conv2D model. Input: (B, 44100, 2) → Output: (B, num_classes)
    Conv: 1→64→128→256→512, GlobalAvgPool → FC
    Approx params: 926K (conv) + 258K (fc) ≈ 1.18M
    """
    def __init__(self, config):
        super().__init__()
        num_classes = config['NUM_CLASSES']
        self.conv = nn.Sequential(
            nn.Conv2d(1,   64,  kernel_size=(64, 2), stride=(16, 1)),  # (B, 64,  2753, 1)
            nn.ReLU(),
            nn.Conv2d(64,  128, kernel_size=(16, 1), stride=(8,  1)),  # (B, 128, 343,  1)
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=(8,  1), stride=(4,  1)),  # (B, 256, 84,   1)
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=(4,  1), stride=(2,  1)),  # (B, 512, 41,   1)
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),                               # (B, 512, 1,    1)
        )
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)        # (B, 1, 44100, 2)
        x = self.conv(x)          # (B, 512, 1, 1)
        x = x.flatten(1)          # (B, 512)
        return self.fc(x)         # (B, num_classes)


class Dummy5M(nn.Module):
    """~5M param Conv2D model. Input: (B, 44100, 2) → Output: (B, num_classes)
    Conv: 1→128→256→512→1024, Pool(2,1) → FC
    Approx params: 3.69M (conv) + 1.03M (fc) ≈ 4.72M
    """
    def __init__(self, config):
        super().__init__()
        num_classes = config['NUM_CLASSES']
        self.conv = nn.Sequential(
            nn.Conv2d(1,   128,  kernel_size=(64, 2), stride=(16, 1)),  # (B, 128,  2753, 1)
            nn.ReLU(),
            nn.Conv2d(128, 256,  kernel_size=(16, 1), stride=(8,  1)),  # (B, 256,  343,  1)
            nn.ReLU(),
            nn.Conv2d(256, 512,  kernel_size=(8,  1), stride=(4,  1)),  # (B, 512,  84,   1)
            nn.ReLU(),
            nn.Conv2d(512, 1024, kernel_size=(4,  1), stride=(2,  1)),  # (B, 1024, 41,   1)
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 1)),                                # (B, 1024, 2,    1)
        )
        self.fc = nn.Linear(1024 * 2, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)        # (B, 1, 44100, 2)
        x = self.conv(x)          # (B, 1024, 2, 1)
        x = x.flatten(1)          # (B, 2048)
        return self.fc(x)         # (B, num_classes)


class Dummy10M(nn.Module):
    """~10M param Conv2D model. Input: (B, 44100, 2) → Output: (B, num_classes)
    Conv: 1→128→256→512→1024→1024, GlobalAvgPool → FC(1024→1024) → FC → num_classes
    Approx params: 7.88M (conv) + 1.05M (fc1) + 0.52M (fc2) ≈ 9.45M
    """
    def __init__(self, config):
        super().__init__()
        num_classes = config['NUM_CLASSES']
        self.conv = nn.Sequential(
            nn.Conv2d(1,    128,  kernel_size=(64, 2), stride=(16, 1)),  # (B, 128,  2753, 1)
            nn.ReLU(),
            nn.Conv2d(128,  256,  kernel_size=(16, 1), stride=(8,  1)),  # (B, 256,  343,  1)
            nn.ReLU(),
            nn.Conv2d(256,  512,  kernel_size=(8,  1), stride=(4,  1)),  # (B, 512,  84,   1)
            nn.ReLU(),
            nn.Conv2d(512,  1024, kernel_size=(4,  1), stride=(2,  1)),  # (B, 1024, 41,   1)
            nn.ReLU(),
            nn.Conv2d(1024, 1024, kernel_size=(4,  1), stride=(2,  1)),  # (B, 1024, 19,   1)
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),                                 # (B, 1024, 1,    1)
        )
        self.fc = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        x = x.unsqueeze(1)        # (B, 1, 44100, 2)
        x = self.conv(x)          # (B, 1024, 1, 1)
        x = x.flatten(1)          # (B, 1024)
        return self.fc(x)         # (B, num_classes)
