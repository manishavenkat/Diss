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
from sklearn.model_selection import train_test_split

# Path to the directory containing .wav files
data_dir = 'Data/genres_original'

# List all .wav files
wav_files = glob.glob(os.path.join(data_dir, '*.wav'))
# print(wav_files)

# Extract labels and file paths
data = []
for root, dirs, files in os.walk(data_dir):
    for file in files:
        if file.endswith('.wav'):
            wav_file = os.path.join(root, file)
            genre, _ = os.path.splitext(file)
            genre = genre.split('.')[0]
            data.append((file, genre))

print(data)

# Convert to DataFrame
df = pd.DataFrame(data, columns=['file_path', 'label'])

# Mapping of labels to indices
label_to_index = {label: idx for idx, label in enumerate(sorted(df['label'].unique()))}
index_to_label = {idx: label for label, idx in label_to_index.items()}

# Convert labels to indices
df['label'] = df['label'].map(label_to_index)

# print("Genre to int mapping:")
# print(label_to_index)

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
        # Convert the label to an integer if it's a string
        if isinstance(label, str):
            label = genre_to_int[label]
            # Split the file path to get the file name and extension separately
        file_dir, file_name = os.path.split(file_path)
        file_name_parts = file_name.split('.')
        # Assume the last part is the extension
        file_ext = file_name_parts[-1]
        # Reconstruct the file path with the correct extension
        corrected_file_path = os.path.join(file_dir, '.'.join(file_name_parts[:-1]) + '.' + file_ext)
        try:
            aud = AudioUtil.open(corrected_file_path)
        except Exception as e:
            print(f"Error opening file {corrected_file_path}: {e}")
            return None, None
        if aud is None:
            return None, None
        aud = AudioUtil.resample(aud, self.sr)
        aud = AudioUtil.rechannel(aud, 1)
        aud = AudioUtil.pad_trunc(aud, self.duration)
        sgram = AudioUtil.spectro_gram(aud, n_mels=64, n_fft=1024, hop_len=None)
        if self.transform:
            sgram = self.transform(sgram)
        return sgram, torch.tensor(label, dtype=torch.long)

# Ensure reproducibility
random.seed(42)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Split into train and validation sets (80% train, 20% validation)
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)

train_dataset = GenreDataset(train_df)
val_dataset = GenreDataset(val_df)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=32, drop_last=True)

class AudioClassifier(nn.Module):
    def __init__(self):
        super(AudioClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AudioClassifier().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=20):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            if outputs.size(0) != labels.size(0):
                print(f"Mismatch in batch sizes: outputs={outputs.size(0)}, labels={labels.size(0)}")
                print(f"Outputs shape: {outputs.shape}")
                print(f"Labels shape: {labels.shape}")
                continue
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
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                correct += torch.sum(preds == labels.data)
                total += labels.size(0)
        val_acc = correct.double() / total
        print(f'Validation Accuracy: {val_acc:.4f}')

train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=20)