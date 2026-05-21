import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import json

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

class ImageDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = Path(root_dir)
        self.transform = transform
        # Fix for Test_1: replace train_data with test_data
        if 'test_data' in str(self.root_dir):
            self.data['file_name'] = self.data['file_name'].str.replace('train_data/', 'test_data/')
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        img_path = self.root_dir / self.data.iloc[idx]['file_name']
        image = Image.open(img_path).convert('RGB')
        label = self.data.iloc[idx]['label']
        if self.transform:
            image = self.transform(image)
        return image, label

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def evaluate_model(model_path, test_loader, model_name):
    model = models.resnet18(pretrained=False)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()
    
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    prec = precision_score(all_labels, all_preds, average='weighted')
    rec = recall_score(all_labels, all_preds, average='weighted')
    
    print(f'{model_name}: Acc={acc:.4f}, F1={f1:.4f}, Prec={prec:.4f}, Rec={rec:.4f}')
    return {'accuracy': acc, 'f1': f1, 'precision': prec, 'recall': rec}

# Test_1
test1_dataset = ImageDataset(
    csv_file='/data/shared_ml/vmoskalenko/ai-vs-human-generated-dataset-hw/Test_1/test.csv',
    root_dir='/data/shared_ml/vmoskalenko/ai-vs-human-generated-dataset-hw/Test_1',
    transform=test_transform
)
test1_loader = DataLoader(test1_dataset, batch_size=32, shuffle=False, num_workers=4)

# Test_2
test2_dataset = ImageDataset(
    csv_file='/data/shared_ml/vmoskalenko/ai-vs-human-generated-dataset-hw/Test_2/test.csv',
    root_dir='/data/shared_ml/vmoskalenko/ai-vs-human-generated-dataset-hw/Test_2',
    transform=test_transform
)
test2_loader = DataLoader(test2_dataset, batch_size=32, shuffle=False, num_workers=4)

# Оценка
results = {}
results['v1_on_test1'] = evaluate_model('model_v1.pth', test1_loader, 'V1 on Test_1')
results['v1_on_test2'] = evaluate_model('model_v1.pth', test2_loader, 'V1 on Test_2')
results['v2_on_test2'] = evaluate_model('model_v2.pth', test2_loader, 'V2 on Test_2')

# Сохраняем результаты
with open('test_results.json', 'w') as f:
    json.dump(results, f, indent=2)
