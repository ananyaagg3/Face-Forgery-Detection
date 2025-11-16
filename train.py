from HybridXRayModel import HybridXRayModel  # Import the model class from the correct file

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, random_split
from torchvision import transforms
import os  # Add this line
from PIL import Image  # Add this line to import the Image module
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


# Create your custom dataset loader
class FaceForgeryDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.image_paths = []
        self.labels = []

        real_dir = os.path.join(root_dir, "real")
        for f in os.listdir(real_dir):
            self.image_paths.append(os.path.join(real_dir, f))
            self.labels.append(0)

        fake_dir = os.path.join(root_dir, "fake")
        for f in os.listdir(fake_dir):
            self.image_paths.append(os.path.join(fake_dir, f))
            self.labels.append(1)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# Transformations for the images
# transform = transforms.Compose([
#     transforms.Resize((256, 256)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Load dataset and split into training and test sets
dataset = FaceForgeryDataset(root_dir="dataset", transform=transform)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Initialize model, optimizer, and loss function
model = HybridXRayModel()  # This works now without circular import
optimizer = optim.Adam(model.parameters(), lr=0.0005)
criterion = nn.CrossEntropyLoss()

# Training loop
for epoch in range(10):  # 10 epochs for training
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/10], Loss: {running_loss / len(train_loader)}")


def evaluate_model(model, test_loader):
    model.eval()
    all_labels = []
    all_logits = []

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)          # shape [B, 2] logits
            all_labels.append(labels.numpy())
            all_logits.append(outputs.numpy())

    all_labels = np.concatenate(all_labels, axis=0)        # shape [N]
    all_logits = np.concatenate(all_logits, axis=0)        # shape [N, 2]

    # convert logits -> probabilities for class 1 ("fake")
    probs_fake = torch.softmax(torch.tensor(all_logits), dim=1)[:, 1].numpy()

    auc = roc_auc_score(all_labels, probs_fake)
    ap = average_precision_score(all_labels, probs_fake)

    print(f"\nTest AUC: {auc:.4f}")
    print(f"Test AP : {ap:.4f}")

# run eval
evaluate_model(model, test_loader)

torch.save(model.state_dict(), "face_forgery_model.pth")
print("\nSaved model -> face_forgery_model.pth")
