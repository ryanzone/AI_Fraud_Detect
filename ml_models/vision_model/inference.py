import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from vision_model.model import get_model


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import os

base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, "saved_models", "best_model.pth")

# Load model
model = get_model(num_classes=2, freeze=False)
model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
model.to(device)
model.eval()

# Same normalization used during training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        probs = F.softmax(output, dim=1)

    fake_prob = probs[0][0].item()   # depends on class index
    real_prob = probs[0][1].item()

    return fake_prob


if __name__ == "__main__":
    result1 = predict_image("./Dataset/Test/Real/real_1.jpg")
    result2 = predict_image("./Dataset/Test/Fake/fake_1.jpg")
    print(result1)
    print(result2)