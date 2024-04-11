# !gdown https://drive.google.com/uc?id=137RyRjvTBkBiIfeYBNZBtViDHQ6_Ewsp
# !tar -xzvf 101_ObjectCategories.tar.gz

import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from tqdm.notebook import tqdm
import numpy as np
from sklearn import svm
import torch.hub
from PIL import Image
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score
from pathlib import Path
import os
import gdown
import tarfile

# Download the dataset
url = 'https://drive.google.com/uc?id=137RyRjvTBkBiIfeYBNZBtViDHQ6_Ewsp'
output = '101_ObjectCategories.tar.gz'
gdown.download(url, output, quiet=False)

# Extract the dataset
if tarfile.is_tarfile(output):
    with tarfile.open(output, "r:gz") as tar_ref:
        tar_ref.extractall()
    print("File extracted successfully.")
else:
    print("Downloaded file is not a tar file.")


transform_image = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(256),       
    transforms.CenterCrop(224),  
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  
])

dataset_path = './101_ObjectCategories'
original_dataset = datasets.ImageFolder(dataset_path)
filtered_data = [(img, label) for img, label in original_dataset.imgs if "BACKGROUND_Google" not in img]

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path, label = self.data[index]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label - int(label > original_dataset.class_to_idx['BACKGROUND_Google'])


filtered_dataset = CustomDataset(filtered_data, transform=transform_image)


total_size = len(filtered_dataset)
val_size = test_size = int(0.1 * total_size)  
train_size = total_size - val_size - test_size  

# Split
train_dataset, test_val_dataset = random_split(filtered_dataset, [train_size, val_size + test_size])
val_dataset, test_dataset = random_split(test_val_dataset, [val_size, test_size])

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print(f"Training set size: {len(train_dataset)}")
print(f"Validation set size: {len(val_dataset)}")
print(f"Test set size: {len(test_dataset)}")

# DinoV2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dinov2_vits14 = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(device)

# Embeddings
def compute_embeddings(data_loader):
    dinov2_vits14.eval()
    all_embeddings = []
    labels = []
    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="Computing embeddings"):
            images = images.to(device)
            embeddings = dinov2_vits14(images)
            all_embeddings.append(embeddings.cpu().numpy())
            labels.extend(targets.numpy())
    all_embeddings = np.vstack(all_embeddings)
    return all_embeddings, labels
train_embeddings, train_labels = compute_embeddings(train_loader)
val_embeddings, val_labels = compute_embeddings(val_loader)
test_embeddings, test_labels = compute_embeddings(test_loader)

# Classifier
class EmbeddingClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(EmbeddingClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

input_size = 384  # Feature dimension of DINOv2 embeddings
hidden_size = 512 
num_classes = 101

classifier = EmbeddingClassifier(input_size, hidden_size, num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(classifier.parameters(), lr=0.001)

train_embeddings_tensor = torch.tensor(train_embeddings).float().to(device)
train_labels_tensor = torch.tensor(train_labels).long().to(device)

val_embeddings_tensor = torch.tensor(val_embeddings).float().to(device)
val_labels_tensor = torch.tensor(val_labels).long().to(device)

model_dir = Path('teacher_model')
model_dir.mkdir(parents=True, exist_ok=True) 

epochs = 10
batch_size = 32

num_train_batches = len(train_embeddings_tensor) // batch_size
num_val_batches = len(val_embeddings_tensor) // batch_size

best_val_loss = float('inf') 

# Train
for epoch in range(epochs):
    classifier.train()
    train_loss = 0.0
    for i in tqdm(range(num_train_batches), desc=f'Epoch {epoch+1}/{epochs}, Training'):
        batch_start = i * batch_size
        batch_end = (i + 1) * batch_size
        
        images = train_embeddings_tensor[batch_start:batch_end]
        labels = train_labels_tensor[batch_start:batch_end]
        
        optimizer.zero_grad()
        outputs = classifier(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    avg_train_loss = train_loss / num_train_batches
    tqdm.write(f'Epoch [{epoch+1}/{epochs}], Training Loss: {avg_train_loss:.4f}')
    
    # Validation phase
    classifier.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for i in tqdm(range(num_val_batches), desc=f'Epoch {epoch+1}/{epochs}, Validating'):
            batch_start = i * batch_size
            batch_end = (i + 1) * batch_size
            
            images = val_embeddings_tensor[batch_start:batch_end]
            labels = val_labels_tensor[batch_start:batch_end]
            
            outputs = classifier(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_val_loss = val_loss / num_val_batches
    val_accuracy = correct / total * 100
    tqdm.write(f'Epoch [{epoch+1}/{epochs}], Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')
    
    # Save
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_model_path = model_dir / f'best_model_epoch_{epoch+1}.pth'
        torch.save(classifier.state_dict(), best_model_path)
        tqdm.write(f"Best model saved to {best_model_path} with Validation Loss: {best_val_loss:.4f}")

# Test
test_embeddings_tensor = torch.tensor(test_embeddings).float().to(device)
test_labels_tensor = torch.tensor(test_labels).long().to(device)

classifier.eval()
test_loss = 0.0
correct = 0
total = 0

with torch.no_grad():
    outputs = classifier(test_embeddings_tensor)  # Forward pass
    _, predicted = torch.max(outputs, 1)  
    total = test_labels_tensor.size(0) 
    correct = (predicted == test_labels_tensor).sum().item()  

# Accuracy
test_accuracy = 100 * correct / total
print(f'Test Accuracy: {test_accuracy:.2f}%')