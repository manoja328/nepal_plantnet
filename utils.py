import os
import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torch import nn, optim
from PIL import Image
from pathlib import Path

# Preprocessing
preprocess = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

LABELS = ["healthy", "diseased"]
NUM_CLASSES = len(LABELS)

def get_squeezenet_model(weight_path: str = None):
    # Load a pre-trained model and modify it for binary classification
    model = models.squeezenet1_1(weights="DEFAULT")
    model.classifier[1] = nn.Conv2d(512, NUM_CLASSES, kernel_size=(1, 1))
    model.num_classes = NUM_CLASSES
    if weight_path and Path(weight_path).exists():
        model.load_state_dict(torch.load(weight_path, map_location="cpu"))
    return model

def classify(model, img_path: Path):
    """
    Classify an image using the provided model.
    Args:
        model: The PyTorch model to use for classification.
        img_path: Path to the image file.
    Returns:
        A tuple containing the predicted label and its confidence score.
    """
    if not img_path.exists():
        raise FileNotFoundError(f"Image file {img_path} does not exist.")
    if not isinstance(model, nn.Module):
        raise ValueError("The model must be a PyTorch nn.Module instance.")
    if not img_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
        raise ValueError("Unsupported image format. Please use .jpg, .jpeg, or .png.")
    if not torch.cuda.is_available():
        print("Warning: No GPU detected, running on CPU. Performance may be slow.")
        torch.set_num_threads(1)
    if not isinstance(img_path, Path):
        img_path = Path(img_path)
    if not img_path.is_file():
        raise ValueError(f"Provided path {img_path} is not a valid file.")
    img = Image.open(img_path).convert("RGB")
    model.eval()  # Ensure the model is in evaluation mode
    with torch.no_grad():
        x = preprocess(img).unsqueeze(0)
        prob = torch.softmax(model(x), dim=1)[0]
    label_idx = int(prob.argmax())
    return LABELS[label_idx], float(prob[label_idx])