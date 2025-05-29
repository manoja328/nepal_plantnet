import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm
from utils import PlantDiseaseModel

# Set random seed
torch.manual_seed(42)


# Data transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
])

# Replace the existing transform with:
# transform = transforms.Compose([
#     transforms.RandomResizedCrop(224),
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomRotation(15),
#     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], 
#                        std=[0.229, 0.224, 0.225])
# ])

# Add separate transform for validation/test
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
])


# Dataset and DataLoader
def get_data_loaders(train_dir, test_dir=None, test_size=0.2, batch_size=32):
    """
    Create data loaders ensuring consistent class mappings between train and test sets
    Args:
        train_dir: Directory containing training data
        test_dir: Optional directory containing test data
        test_size: Fraction of train_dir to use as validation (ignored if test_dir provided)
        batch_size: Batch size for DataLoader
    """
    # Load training dataset first to establish class mapping
    train_dataset = ImageFolder(root=train_dir, transform=transform)
    class_to_idx = train_dataset.class_to_idx
    test_loader = None

    # Split training data for validation
    train_size = int((1 - test_size) * len(train_dataset))
    val_size = len(train_dataset) - train_size
    
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42))
    
    train_loader = DataLoader(train_subset, batch_size=batch_size,
                            shuffle=True, num_workers=4)
    val_loader = DataLoader(val_subset, batch_size=batch_size,
                            shuffle=False, num_workers=4)
    
    print(f"Training samples: {len(train_subset)}")
    print(f"Validation samples: {len(val_subset)}")

    if test_dir:
        # Use same class mapping for test dataset
        test_dataset = ImageFolder(root=test_dir, transform=transform)
        test_dataset.classes = train_dataset.classes
        test_dataset.class_to_idx = class_to_idx
        # Update targets to match new class_to_idx
        test_dataset.targets = [class_to_idx[test_dataset.classes[t]] for t in test_dataset.targets]
        
        test_loader = DataLoader(test_dataset, batch_size=batch_size,
                               shuffle=False, num_workers=4)
        
        print(f"Test samples: {len(test_dataset)}")

    print(f"Number of classes: {len(train_dataset.classes)}")
    print("Class mapping:", train_dataset.class_to_idx)
    
    return train_loader, val_loader, test_loader, train_dataset.classes


def evaluate_model(model, data_loader, criterion):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    val_loss /= len(data_loader)
    accuracy = 100. * correct / total
    return val_loss, accuracy


# Training function
def train_model(model, train_loader, val_loader, num_epochs=50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.001)
    scheduler = ExponentialLR(optimizer, gamma=0.9)
    
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        # Validation
        model.eval()
      
        val_loss, accuracy = evaluate_model(model, val_loader, criterion)
        print(f'Training Loss: {running_loss/len(train_loader):.4f}')
        print(f'Validation Loss: {val_loss:.4f}')
        print(f'Accuracy: {accuracy:.2f}%')
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
            print("Best model saved")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break
        
        scheduler.step()

# Main execution
def main( evaluate: bool = False):
    print("Loading datasets...")
    train_loader, val_loader, test_loader, classes = get_data_loaders(
        train_dir="dataset/train",
        test_dir="dataset/test",
        test_size=0.2,  # Use 20% of train data for validation
        batch_size=32,
    )

    model = PlantDiseaseModel(num_classes=len(classes))
    if not evaluate:
        train_model(model, train_loader, val_loader)
        print("Training complete. Best model saved as 'best_model.pth'.")
    else:
        print("Evaluating on test set...")
        model.load_state_dict(torch.load('best_model.pth', weights_only=True))
        test_loss, accuracy = evaluate_model(model, val_loader, nn.CrossEntropyLoss())
        print(f'Test Loss: {test_loss:.4f}')
        print(f'Accuracy: {accuracy:.2f}%')

if __name__ == "__main__":
    import fire
    fire.Fire(main)