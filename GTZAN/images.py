import os
import glob
import random
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torchvision.io import read_image
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

## Dataset Prep ##

class GTZANDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_path = self.df.iloc[idx, 0]
        label = self.df.iloc[idx, 1]
        if isinstance(label, str):
            label = label_to_index[label]
        image = read_image(file_path).float() / 255.0  # Normalize the image
        if image.shape[0] == 4:  # Check if the image has 4 channels
            image = image[:3]  # Discard the alpha channel
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)


data_dir = 'GTZAN/images_original'

# List all .png files
png_files = glob.glob(os.path.join(data_dir, '*.png'))

# Extract labels and file paths
data = []
for png_file in png_files:
    base_name = os.path.basename(png_file)
    label = ''.join(filter(str.isalpha, base_name.split('.')[0]))
    data.append((png_file, label))

df = pd.DataFrame(data, columns=['file_path', 'label'])
random.seed(42)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)

label_to_index = {label: idx for idx, label in enumerate(df['label'].unique())}
index_to_label = {idx: label for label, idx in label_to_index.items()}

transform = transforms.Compose([
    transforms.Resize((128, 128)),
])

train_dataset = GTZANDataset(train_df, transform=transform)
val_dataset = GTZANDataset(val_df, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

## CNN Definition ##

class MusicGenreCNN(nn.Module):
    def __init__(self):
        super(MusicGenreCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(128 * 16 * 16, 256)  
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 16 * 16)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

## Training ##

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

        evaluate_model(model, val_loader, criterion)

def evaluate_model(model, val_loader, criterion):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    val_loss /= len(val_loader.dataset)
    accuracy = correct / total
    print(f"Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}")

    print(classification_report(all_labels, all_preds, target_names=[index_to_label[i] for i in range(10)]))
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=[index_to_label[i] for i in range(10)], yticklabels=[index_to_label[i] for i in range(10)])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

model = MusicGenreCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=20)

