import os
from PIL import Image
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np

# --- Step 1: Dataset Definition ---
class FaceForgeryDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # Real images -> label 0
        real_dir = os.path.join(root_dir, "real")
        for f in os.listdir(real_dir):
            self.image_paths.append(os.path.join(real_dir, f))
            self.labels.append(0)

        # Fake images -> label 1
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

# --- Step 2: Transformations ---
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Pretrained ResNet normalization
])

# --- Step 3: Data Loading ---
dataset = FaceForgeryDataset(root_dir="dataset", transform=transform)

# Split dataset into training and test sets (80-20 split)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# DataLoader for training and testing
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# --- Step 4: Model Setup ---
# from hybrid_xray_model import HybridXRayModel  # Ensure this file is available and imported
model = HybridXRayModel()

# --- Step 5: Optimizer & Loss ---
optimizer = optim.Adam(model.parameters(), lr=0.0002)
criterion = nn.CrossEntropyLoss()  # Binary classification (real vs forged)

# --- Step 6: Training Loop ---
num_epochs = 5  # Number of epochs (you can increase this)
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(train_loader)}")

# --- Step 7: Evaluation Function ---
def evaluate_model(model, test_loader):
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            all_labels.append(labels.cpu().numpy())
            all_preds.append(outputs.cpu().numpy())

    all_labels = np.concatenate(all_labels, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)

    auc = roc_auc_score(all_labels, all_preds[:, 1])  # AUC for fake class
    ap = average_precision_score(all_labels, all_preds[:, 1])  # Average Precision
    print(f"Test AUC: {auc}, AP: {ap}")

# --- Step 8: Evaluate the Model ---
evaluate_model(model, test_loader)

# --- Step 9: Save the Model ---
torch.save(model.state_dict(), 'face_forgery_model.pth')

# --- Step 10: Load the Model (for inference or resuming training) ---
# model.load_state_dict(torch.load('face_forgery_model.pth'))
