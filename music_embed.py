import os
import glob
import random
import torch
import torchaudio
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
from sklearn.model_selection import train_test_split
import soundfile as sf
from sklearn.manifold import TSNE
import plotly.express as px
import IPython.display as ipd

data_dir = 'Data/genres_original'

# List all .wav files
wav_files = glob.glob(os.path.join(data_dir, '*.wav'))

data = []

for root, dirs, files in os.walk(data_dir):
    for file in files:
        if file.endswith('.wav'):
            wav_file = os.path.join(root, file)
            genre, _ = os.path.splitext(file)
            genre = genre.split('.')[0]
            data.append((wav_file, genre))

# Convert to DataFrame
df = pd.DataFrame(data, columns=['file_path', 'label'])

# Mapping of labels to indices
label_to_index = {label: idx for idx, label in enumerate(sorted(df['label'].unique()))}
index_to_label = {idx: label for label, idx in label_to_index.items()}

# Convert labels to indices
df['label'] = df['label'].map(label_to_index)

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
        sgram = torchaudio.transforms.MelSpectrogram(
            sr, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)(sig)
        sgram = torchaudio.transforms.AmplitudeToDB(top_db=top_db)(sgram)
        return sgram

class GenreDataset(Dataset):
    def __init__(self, df, duration=5000, sr=22050, transform=None):
        self.df = df
        self.duration = duration
        self.sr = sr
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_path = self.df.iloc[idx, 0]
        label = self.df.iloc[idx, 1]
        aud = AudioUtil.open(file_path)
        aud = AudioUtil.resample(aud, self.sr)
        aud = AudioUtil.rechannel(aud, 1)
        aud = AudioUtil.pad_trunc(aud, self.duration)
        sgram = AudioUtil.spectro_gram(aud, n_mels=64, n_fft=1024, hop_len=None)
        if self.transform:
            sgram = self.transform(sgram)
        return sgram, torch.tensor(label, dtype=torch.long), file_path

# Ensure reproducibility
random.seed(42)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Split into train and validation sets (80% train, 20% validation)
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)

train_dataset = GenreDataset(train_df)
val_dataset = GenreDataset(val_df)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, drop_last=True)

class AudioClassifier(nn.Module):
    def __init__(self):
        super(AudioClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 4 * 13, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = x.view(x.size(0), -1) 
        x = F.relu(self.fc1(x))
        embeddings = self.dropout(x)
        x = self.fc2(embeddings)
        return x, embeddings

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AudioClassifier().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=20):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels, _ in train_loader:
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
            for inputs, labels, _ in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs, _ = model(inputs)
                _, preds = torch.max(outputs, 1)
                correct += torch.sum(preds == labels.data)
                total += labels.size(0)
        val_acc = correct.double() / total
        print(f'Validation Accuracy: {val_acc:.4f}')

def extract_embeddings(model, data_loader):
    model.eval()
    embeddings = []
    labels = []
    file_paths = []
    with torch.no_grad():
        for inputs, lbls, paths in data_loader:
            inputs = inputs.to(device)
            outputs, embs = model(inputs)
            embeddings.append(embs.cpu())
            labels.append(lbls)
            file_paths.extend(paths)
    return torch.cat(embeddings), torch.cat(labels), file_paths

train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=20)

# Extract embeddings
train_embeddings, train_labels, train_paths = extract_embeddings(model, train_loader)
val_embeddings, val_labels, val_paths = extract_embeddings(model, val_loader)

def compute_tsne(embeddings):
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(embeddings)
    return tsne_results

# Compute t-SNE
train_tsne_results = compute_tsne(train_embeddings)
val_tsne_results = compute_tsne(val_embeddings)

# Function to plot the t-SNE results
def plot_tsne(tsne_results, labels, paths, title="t-SNE"):
    df = pd.DataFrame(tsne_results, columns=['x', 'y'])
    df['label'] = labels
    df['file_path'] = paths
    fig = px.scatter(df, x='x', y='y', color='label', hover_data=['file_path'])
    fig.update_layout(title=title)
    
    def click_callback(trace, points, state):
        ind = points.point_inds[0]
        file_path = df.iloc[ind]['file_path']
        print(f"Playing: {file_path}")
        ipd.display(ipd.Audio(file_path))
        
    fig.data[0].on_click(click_callback)
    fig.show()

# Plot t-SNE results
plot_tsne(train_tsne_results, train_labels, train_paths, title="Train t-SNE")
plot_tsne(val_tsne_results, val_labels, val_paths, title="Validation t-SNE")
