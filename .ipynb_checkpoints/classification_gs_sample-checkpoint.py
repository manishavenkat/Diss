import os
import random
import torch
import torchaudio
import pandas as pd
# from torch.utils.data import Dataset, DataLoader
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as DatasetTorch
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

# Load the dataset from the .arrow file
train_dataset = Dataset.from_file(train_dir) ##.arrow file; looks exactly like the dict on HF
train_dataset = train_dataset.cast_column("audio", Audio(sampling_rate=16000))

val_dataset = Dataset.from_file(val_dir)
val_dataset = val_dataset.cast_column("audio", Audio(sampling_rate=16000))

test_dataset = Dataset.from_file(test_dir)
test_dataset = test_dataset.cast_column("audio", Audio(sampling_rate=16000))

def sample_dataset(dataset, num_samples=25):
    category_names = dataset.features['category'].names
    grouped_data = {category: [] for category in category_names}

    for example in dataset:
        category_id = example['category']
        category_name = category_names[category_id]
        grouped_data[category_name].append(example)
    
    sampled_data = []
    for category, examples in grouped_data.items():
        if len(examples) >= num_samples:
            sampled_data.extend(random.sample(examples, num_samples))
        else:
            sampled_data.extend(examples)  # If less than num_samples, take all
    
    return Dataset.from_pandas(pd.DataFrame(sampled_data)) #retaining .arrow Dataset original format after sampling

## each of the following is a large dict (see HF data structure) with keys like segment_id and audio. Some keys are lists. ##
## audio is a list but EACH item in that last is a dict on its own, with keys like path, sr and decoded signal ##
train_sampled_dataset = sample_dataset(train_dataset)
val_sampled_dataset = sample_dataset(val_dataset)
test_sampled_dataset = sample_dataset(test_dataset)

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

class GenreDataset(DatasetTorch):
    def __init__(self, dataset, duration=5000, sr=16000, transform=None):
        self.dataset = dataset
        self.duration = duration
        self.sr = sr
        self.transform = transform
        self.flat_dataset = self._flatten_dataset()

    def _flatten_dataset(self):
        flat = []
        for i in range(len(self.dataset['audio'])):
            flat.append({
                'audio': self.dataset['audio'][i],
                'category': self.dataset['category'][i]
            })
        return flat

    def __len__(self):
        return len(self.flat_dataset)

    def __getitem__(self, idx):
        item = self.flat_dataset[idx]
        
        audio_data = item['audio']
        audio_path = audio_data['path']
        
        # Load audio using torchaudio
        sig, sr = torchaudio.load(audio_path)
        
        # Resample audio to ensure uniform sampling rate
        sig = torchaudio.transforms.Resample(sr, self.sr)(sig)

        label = item['category']

        aud = (sig, self.sr)  # Ensure uniform sampling rate
        aud = AudioUtil.rechannel(aud, 1)
        aud = AudioUtil.pad_trunc(aud, self.duration)
        sgram = AudioUtil.spectro_gram(aud, n_mels=64, n_fft=1024, hop_len=None)
        # print("Shape after spectrogram creation:", sgram.shape)
        
        # Remove the channel dimension if it exists
        if sgram.dim() == 3:
            sgram = sgram.squeeze(0)

        if self.transform:
            sgram = self.transform(sgram)

        return sgram, torch.tensor(label, dtype=torch.long)

    def __getitems__(self, indices): ## this supersedes __getitem__ above when dataloader is called
        return [self.__getitem__(idx) for idx in indices]
    
        #this was returning a typle of lists like ([spectrogram1, spectrogram2, ...], [label1, label2, ...])
        ## dataloader is expecting each tuple to be (sgram,label) sample and not an embedded list of samples
        ## DataLoader tries to batch these items, but it can't handle the list structure you're returning

train_dataset = GenreDataset(train_sampled_dataset)
val_dataset = GenreDataset(val_sampled_dataset)
test_dataset = GenreDataset(test_sampled_dataset)

# # To print the first two items:
# print(train_dataset[0])  # This will print a single (spectrogram, label) pair
# print(train_dataset[1])  # This will print another single (spectrogram, label) pair

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

class AudioClassifier(nn.Module):
    def __init__(self, num_classes=29):
        super(AudioClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = None  # We'll define this in the forward pass
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # print("Input shape to model:", x.shape)
        if x.dim() == 3:
            x = x.unsqueeze(1)
        # print("Shape after potential unsqueeze:", x.shape)
        x = self.pool(F.relu(self.conv1(x)))
        # print("Shape after first conv and pool:", x.shape)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        
        # Flatten the output
        x = x.view(x.size(0), -1)
        # print("Shape after flattening:", x.shape)
        
        # Dynamically create fc1 if it doesn't exist
        if self.fc1 is None:
            self.fc1 = nn.Linear(x.shape[1], 128).to(x.device)
        
        embeddings = F.relu(self.fc1(x))
        x = self.dropout(embeddings)
        x = self.fc2(x)
        return x, embeddings

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AudioClassifier().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=20):
    train_losses = []
    val_accuracies = []
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
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
        train_losses.append(epoch_loss)
        print(f'Epoch {epoch}/{num_epochs - 1}, Loss: {epoch_loss:.4f}')
        
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                if inputs.dim() == 3:
                    inputs = inputs.unsqueeze(1)
                inputs, labels = inputs.to(device), labels.to(device)
                outputs, _ = model(inputs)
                _, preds = torch.max(outputs, 1)
                correct += torch.sum(preds == labels.data)
                total += labels.size(0)
        val_acc = correct.double() / total
        val_accuracies.append(val_acc.item())
        print(f'Validation Accuracy: {val_acc:.4f}')
    
    # Save the loss plot
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.legend()
    plt.savefig(os.path.join("gs-embeddings", "loss_plot.png"))
    plt.show()

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


# Ensure the directory exists
os.makedirs("gs-embeddings", exist_ok=True)

# Save the plot
plt.figure(figsize=(10, 10))
scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels, cmap='viridis')
plt.colorbar(scatter)
plt.title('T-SNE of Audio Embeddings Sample')
plt.savefig(os.path.join("gs-embeddings", "tsne_gs_sample_mv50.png"))
plt.show()