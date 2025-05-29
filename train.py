import os
import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torch import nn, optim
from utils import *

# Paths
DATA_DIR = "dataset"
MODEL_SAVE_PATH = "leaf_model_quantized.pt"

# Model setup
if not torch.cuda.is_available():
    print("Warning: No GPU detected, running on CPU. Performance may be slow.")
    torch.set_num_threads(1)  # Limit to single thread for CPU inference


def train( output_dir: str = MODEL_SAVE_PATH,
              data_dir: str = DATA_DIR,
              batch_size: int = 32,
              num_epochs: int = 10):
    """
    Train the model.
    """

    # Load dataset
    train_data = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=preprocess)
    val_data = datasets.ImageFolder(os.path.join(data_dir, "val"), transform=preprocess)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    # Training setup
    model = get_squeezenet_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    
    for epoch in range(num_epochs):  # Adjust epochs as needed
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")

        # Validation loop
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f"Validation Accuracy: {100 * correct / total:.2f}%")

    # Save the model
    torch.save(model.state_dict(), output_dir)
    print(f"Model saved to {output_dir}")


if __name__ == "__main__":
    import fire
    fire.Fire(train)