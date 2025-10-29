from torch import nn

class DefaultCNN(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 10):
        super(DefaultCNN, self).__init__()
        
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            nn.Flatten(),
            nn.Linear(9216, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, out_channels),
        )

    def forward(self, x):
        return self.net(x)