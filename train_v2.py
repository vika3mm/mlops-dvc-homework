import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import f1_score, accuracy_score
import json

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

class ImageDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = Path(root_dir)
        self.transform = transform
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        img_path = self.root_dir / self.data.iloc[idx]['file_name']
        image = Image.open(img_path).convert('RGB')
        label = self.data.iloc[idx]['label']
        if self.transform:
            image = self.transform(image)
        return image, label

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = ImageDataset(
    csv_file='/data/shared_ml/vmoskalenko/ai-vs-human-generated-dataset-hw/Train_2/train.csv',
    root_dir='/data/shared_ml/vmoskalenko/ai-vs-human-generated-dataset-hw/Train_2',
    transform=train_transform
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)

# Загружаем модель v1
model = models.resnet18(pretrained=False)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)
model.load_state_dict(torch.load('model_v1.pth'))
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Меньший lr для дообучения

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_f1 = f1_score(all_labels, all_preds, average='weighted')
    return epoch_loss, epoch_acc, epoch_f1

num_epochs = 5
for epoch in range(num_epochs):
    train_loss, train_acc, train_f1 = train_epoch(model, train_loader, criterion, optimizer, device)
    print(f'Epoch {epoch+1}: Loss={train_loss:.4f}, Acc={train_acc:.4f}, F1={train_f1:.4f}')

torch.save(model.state_dict(), 'model_v2.pth')
print('Model saved as model_v2.pth')

with open('metrics_v2.json', 'w') as f:
    json.dump({'train_loss': train_loss, 'train_acc': train_acc, 'train_f1': train_f1}, f)
