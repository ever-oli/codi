SOLUTION = """
# CNN IMAGE CLASSIFICATION: CATS VS DOGS + RESNET18 TRANSFER LEARNING

# !pip install datasets torch torchvision matplotlib numpy -q

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from datasets import load_dataset
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on device: {device}\\n")


# 1. Data Loading via Hugging Face

print("Loading microsoft/cats_vs_dogs dataset...")
dataset = load_dataset("microsoft/cats_vs_dogs", split="train")
dataset = dataset.train_test_split(test_size=0.2, seed=42)
train_data = dataset['train']
val_data = dataset['test']

print(f"Training samples: {len(train_data)} | Validation samples: {len(val_data)}\\n")


# 2. Data Augmentation and Pre-processing

train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(128, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class HFVisionDataset(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.hf_dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        item = self.hf_dataset[idx]
        image = item['image'].convert("RGB")
        label = item['labels']
        if self.transform:
            image = self.transform(image)
        return image, label

train_dataset = HFVisionDataset(train_data, transform=train_transforms)
val_dataset = HFVisionDataset(val_data, transform=val_transforms)


# 3. Handling Class Imbalances

print("Calculating class weights for imbalance handling...")
train_labels = [item['labels'] for item in train_data]
class_counts = Counter(train_labels)
num_samples = len(train_labels)
class_weights = {cls: num_samples / (len(class_counts) * count) for cls, count in class_counts.items()}
print(f"Class Weights: {class_weights}")

weights_tensor = torch.tensor([class_weights[0], class_weights[1]], dtype=torch.float32).to(device)


# 4. DataLoaders

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)


# 5. CNN Architecture

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2, 2)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 16 * 16, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 2)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        x = self.flatten(x)
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


def run_training_loop(model, criterion, optimizer, train_loader, val_loader, epochs, title):
    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    print(f"\\nStarting {title} Training...\\n")
    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_train_loss = running_loss / total
        epoch_train_acc = correct / total
        train_losses.append(epoch_train_loss)
        train_accs.append(epoch_train_acc)

        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        epoch_val_loss = val_loss / total
        epoch_val_acc = correct / total
        val_losses.append(epoch_val_loss)
        val_accs.append(epoch_val_acc)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.4f} | Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.4f}")

    return train_losses, val_losses, train_accs, val_accs


# 6. Train Custom CNN

cnn_model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss(weight=weights_tensor)
optimizer = optim.Adam(cnn_model.parameters(), lr=0.001)

epochs = 5
cnn_train_losses, cnn_val_losses, cnn_train_accs, cnn_val_accs = run_training_loop(
    cnn_model, criterion, optimizer, train_loader, val_loader, epochs, "Custom CNN"
)


# 7. Transfer Learning Architecture (ResNet18)

print("\\nLoading pre-trained ResNet18 model...")
weights = models.ResNet18_Weights.DEFAULT
resnet_model = models.resnet18(weights=weights)

for param in resnet_model.parameters():
    param.requires_grad = False

num_ftrs = resnet_model.fc.in_features
resnet_model.fc = nn.Linear(num_ftrs, 2)
resnet_model = resnet_model.to(device)

criterion = nn.CrossEntropyLoss(weight=weights_tensor)
optimizer = optim.Adam(resnet_model.fc.parameters(), lr=0.001)

print("Architecture updated to ResNet18. Ready for training.")

resnet_train_losses, resnet_val_losses, resnet_train_accs, resnet_val_accs = run_training_loop(
    resnet_model, criterion, optimizer, train_loader, val_loader, epochs, "ResNet18 Transfer Learning"
)


# 8. Plot Results

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].plot(range(1, epochs+1), cnn_train_losses, label='Train', color='#e74c3c', linewidth=2)
axes[0, 0].plot(range(1, epochs+1), cnn_val_losses, label='Validation', color='#2ecc71', linewidth=2)
axes[0, 0].set_title('Custom CNN: Loss')
axes[0, 0].legend()

axes[0, 1].plot(range(1, epochs+1), cnn_train_accs, label='Train', color='#e74c3c', linewidth=2)
axes[0, 1].plot(range(1, epochs+1), cnn_val_accs, label='Validation', color='#2ecc71', linewidth=2)
axes[0, 1].set_title('Custom CNN: Accuracy')
axes[0, 1].legend()

axes[1, 0].plot(range(1, epochs+1), resnet_train_losses, label='Train', color='#2c3e50', linewidth=2)
axes[1, 0].plot(range(1, epochs+1), resnet_val_losses, label='Validation', color='#e74c3c', linewidth=2, linestyle='--')
axes[1, 0].set_title('ResNet18: Loss')
axes[1, 0].legend()

axes[1, 1].plot(range(1, epochs+1), resnet_train_accs, label='Train', color='#2c3e50', linewidth=2)
axes[1, 1].plot(range(1, epochs+1), resnet_val_accs, label='Validation', color='#e74c3c', linewidth=2, linestyle='--')
axes[1, 1].set_title('ResNet18: Accuracy')
axes[1, 1].legend()

plt.tight_layout()
plt.show()
""".strip()

DESCRIPTION = "Train a custom CNN and fine-tune ResNet18 via transfer learning for binary cats vs dogs image classification."
