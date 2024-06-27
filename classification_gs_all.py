import os
import random
import torch
import torchaudio
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from datasets import load_dataset, Dataset, Audio
import pyarrow as pa
import pyarrow.parquet as pq
import librosa
import sys

train_dir = '/work/tc062/tc062/manishav/huggingface_cache/datasets/speechcolab___gigaspeech/xs/0.0.0/0db31224ad43470c71b459deb2f2b40956b3a4edfde5fb313aaec69ec7b50d3c/gigaspeech-train.arrow'
val_dir = '/work/tc062/tc062/manishav/huggingface_cache/datasets/speechcolab___gigaspeech/xs/0.0.0/0db31224ad43470c71b459deb2f2b40956b3a4edfde5fb313aaec69ec7b50d3c/gigaspeech-validation.arrow'
test_dir = '/work/tc062/tc062/manishav/huggingface_cache/datasets/speechcolab___gigaspeech/xs/0.0.0/0db31224ad43470c71b459deb2f2b40956b3a4edfde5fb313aaec69ec7b50d3c/gigaspeech-test.arrow'

# Load the full datasets
train_dataset = Dataset.from_file(train_dir)
train_dataset = train_dataset.cast_column("audio", Audio(sampling_rate=16000))

val_dataset = Dataset.from_file(val_dir)
val_dataset = val_dataset.cast_column("audio", Audio(sampling_rate=16000))

test_dataset = Dataset.from_file(test_dir)
test_dataset = test_dataset.cast_column("audio", Audio(sampling_rate=16000))

class AudioUtil():
    @staticmethod
    def open(audio_file):
        sig, sr = torchaudio.load(str(audio_file))
        return (sig, sr)

    @staticmethod
    def rechannel(aud, new_channel):
        sig, sr = aud
        if sig.shape[0] == new_channel:
            return aud
        if new_channel == 1:
            sig = sig.mean(dim=0, keepdim=True)
        else:
            sig = sig.expand(new_channel, -1)
        return (sig, sr)

    @staticmethod
    def resample(aud, new_sr):
        sig, sr = aud
        if sr == new_sr:
            return aud
        num_channels = sig.shape[0]
        resig = torchaudio.transforms.Resample(sr, new_sr)(sig[:1, :])
        if num_channels > 1:
            retwo = torchaudio.transforms.Resample(sr, new_sr)(sig[1:, :])
            resig = torch.cat([resig, retwo])
        return (resig, new_sr)

    @staticmethod
    def pad_trunc(aud, max_ms):
        sig, sr = aud
        num_rows, sig_len = sig.shape
        max_len = sr // 1000 * max_ms
        if sig_len > max_len:
            sig = sig[:, :max_len]
        elif sig_len < max_len:
            pad_begin_len = random.randint(0, max_len - sig_len)
            pad_end_len = max_len - sig_len - pad_begin_len
            pad_begin = torch.zeros((num_rows, pad_begin_len))
            pad_end = torch.zeros((num_rows, pad_end_len))
            sig = torch.cat((pad_begin, sig, pad_end), 1)
        return (sig, sr)

    @staticmethod
    def spectro_gram(aud, n_mels=64, n_fft=1024, hop_len=None):
        sig, sr = aud
        top_db = 80
        spec = torchaudio.transforms.MelSpectrogram(
            sr, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)(sig)
        spec = torchaudio.transforms.AmplitudeToDB(top_db=top_db)(spec)
        return spec.squeeze(0)  # Remove the channel dimension

class GenreDataset(Dataset):
    def __init__(self, dataset, duration=5000, sr=16000, transform=None):
        self.dataset = dataset
        self.duration = duration
        self.sr = sr
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        audio_data = item['audio']
        
        # Assuming audio_data is a list with a single dictionary
        if isinstance(audio_data, list) and len(audio_data) > 0:
            audio_data = audio_data[0]
        
        # Now audio_data should be a dictionary
        if 'array' in audio_data:
            if isinstance(audio_data['array'], np.ndarray):
                sig = torch.from_numpy(audio_data['array']).float()
            else:
                # If it's not a numpy array, it might be a list, so convert it
                sig = torch.tensor(audio_data['array']).float()
        else:
            # If 'array' is not present, try to load from 'path'
            audio_path = audio_data.get('path')
            if audio_path:
                sig, sr = torchaudio.load(audio_path)
            else:
                raise ValueError(f"Cannot load audio data for item {idx}")

        sr = audio_data.get('sampling_rate', self.sr)

        # Ensure the signal is 2D (add channel dimension if necessary)
        if sig.dim() == 1:
            sig = sig.unsqueeze(0)

        # Resample audio to ensure uniform sampling rate
        if sr != self.sr:
            sig = torchaudio.transforms.Resample(sr, self.sr)(sig)

        label = item['category']

        aud = (sig, self.sr)  # Ensure uniform sampling rate
        aud = AudioUtil.rechannel(aud, 1)
        aud = AudioUtil.pad_trunc(aud, self.duration)
        sgram = AudioUtil.spectro_gram(aud, n_mels=64, n_fft=1024, hop_len=None)
        
        # Remove the channel dimension if it exists
        if sgram.dim() == 3:
            sgram = sgram.squeeze(0)

        if self.transform:
            sgram = self.transform(sgram)

        return sgram, torch.tensor(label, dtype=torch.long)

# Create dataset objects
train_dataset = GenreDataset(train_dataset)
val_dataset = GenreDataset(val_dataset)
test_dataset = GenreDataset(test_dataset)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

class AudioClassifier(nn.Module):
    def __init__(self):
        super(AudioClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 4 * 13, 128)
        self.fc2 = nn.Linear(128, 23)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        # Add a channel dimension if it's missing
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = x.view(x.size(0), -1)
        embeddings = F.relu(self.fc1(x))
        x = self.dropout(embeddings)
        x = self.fc2(x)
        return x, embeddings

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AudioClassifier().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=20):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            # Ensure inputs have the correct shape (batch_size, 1, n_mels, time)
            if inputs.dim() == 3:
                inputs = inputs.unsqueeze(1)
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs, _ = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch {epoch}/{num_epochs - 1}, Loss: {epoch_loss:.4f}')
        
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs, _ = model(inputs)
                _, preds = torch.max(outputs, 1)
                correct += torch.sum(preds == labels.data)
                total += labels.size(0)
        val_acc = correct.double() / total
        print(f'Validation Accuracy: {val_acc:.4f}')

train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=20)

# Extract embeddings
embeddings = []
labels = []

model.eval()
with torch.no_grad():
    for inputs, label in train_loader:
        inputs = inputs.to(device)
        _, emb = model(inputs)
        embeddings.append(emb.cpu().numpy())
        labels.append(label.numpy())

embeddings = np.concatenate(embeddings)
labels = np.concatenate(labels)

# T-SNE
tsne = TSNE(n_components=2, random_state=42)
tsne_results = tsne.fit_transform(embeddings)

plt.figure(figsize=(10, 10))
scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels, cmap='viridis')
plt.colorbar(scatter)
plt.title('T-SNE of Audio Embeddings')
plt.show()