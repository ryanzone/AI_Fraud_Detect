import torch
import torch.nn as nn

class AudioCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(AudioCNN, self).__init__()
        # Input shape: [1, 64, ~94] (1 channel, 64 mel bins, 94 time frames)
        
        self.conv_layers = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # Block 2
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # Block 3
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # Block 4
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        
        # Global Average Pooling to handle variable time length robustly
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.fc_layers = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1) # Flatten
        x = self.fc_layers(x)
        return x

def get_model(num_classes=2):
    return AudioCNN(num_classes=num_classes)

if __name__ == "__main__":
    model = get_model()
    # Dummy input: batch_size=1, channels=1, freq=64, time=94
    dummy_input = torch.randn(1, 1, 64, 94)
    output = model(dummy_input)
    print("Output shape:", output.shape)
