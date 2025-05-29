import os
import torch
import torchvision
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

# LABELS = ["healthy", "diseased"]
# NUM_CLASSES = len(LABELS)


class2idx = {'Apple___Apple_scab': 0, 'Apple___Black_rot': 1, 'Apple___Cedar_apple_rust': 2, 'Apple___healthy': 3, 'Blueberry___healthy': 4, 'Cherry_(including_sour)___Powdery_mildew': 5, 'Cherry_(including_sour)___healthy': 6, 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': 7, 'Corn_(maize)___Common_rust_': 8, 'Corn_(maize)___Northern_Leaf_Blight': 9, 'Corn_(maize)___healthy': 10, 'Grape___Black_rot': 11, 'Grape___Esca_(Black_Measles)': 12, 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': 13, 'Grape___healthy': 14, 'Orange___Haunglongbing_(Citrus_greening)': 15, 'Peach___Bacterial_spot': 16, 'Peach___healthy': 17, 'Pepper,_bell___Bacterial_spot': 18, 'Pepper,_bell___healthy': 19, 'Potato___Early_blight': 20, 'Potato___Late_blight': 21, 'Potato___healthy': 22, 'Raspberry___healthy': 23, 'Soybean___healthy': 24, 'Squash___Powdery_mildew': 25, 'Strawberry___Leaf_scorch': 26, 'Strawberry___healthy': 27, 'Tomato___Bacterial_spot': 28, 'Tomato___Early_blight': 29, 'Tomato___Late_blight': 30, 'Tomato___Leaf_Mold': 31, 'Tomato___Septoria_leaf_spot': 32, 'Tomato___Spider_mites Two-spotted_spider_mite': 33, 'Tomato___Target_Spot': 34, 'Tomato___Tomato_Yellow_Leaf_Curl_Virus': 35, 'Tomato___Tomato_mosaic_virus': 36, 'Tomato___healthy': 37}
LABELS = list(class2idx.keys())
NUM_CLASSES = len(LABELS)

# Model definition
class PlantDiseaseModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Load pretrained SqueezeNet
        squeezenet = torchvision.models.squeezenet1_1(weights='DEFAULT')
        # Replace final classifier
        squeezenet.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=1)
        self.model = squeezenet
        # Add dropout for regularization
        self.dropout = nn.Dropout(p=0.5)
    
    def forward(self, x):
        x = self.model(x)
        x = self.dropout(x)
        return torch.squeeze(x)


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
        output = model(x)
        prob = torch.softmax(output, dim=-1).flatten()
    label_idx = int(prob.argmax())
    return LABELS[label_idx], float(prob[label_idx])