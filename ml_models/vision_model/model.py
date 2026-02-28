import torch
import torch.nn as nn
import torchvision.models as models

def get_model(num_classes=2, freeze=True):

    # Load pretrained ResNet50
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

    # Freeze all layers if required
    if freeze:
        for param in model.parameters():
            param.requires_grad = False

    # Replace final fully connected layer
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model


if __name__ == "__main__":
    model = get_model()

    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)

    print("Output shape:", output.shape)