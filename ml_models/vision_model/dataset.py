import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ImageNet normalization values (required for pretrained models)
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# Transforms for training
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

# Transforms for validation
val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

# Load datasets
train_dataset = datasets.ImageFolder(
    root="Dataset/Train",
    transform=train_transforms
)

val_dataset = datasets.ImageFolder(
    root="Dataset/Validation",
    transform=val_transforms
)

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

if __name__ == "__main__":
    print("Classes:", train_dataset.classes)
    print("Class to index:", train_dataset.class_to_idx)

    images, labels = next(iter(train_loader))
    print("Batch image shape:", images.shape)
    print("Batch label shape:", labels.shape)