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
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
from sklearn.manifold import TSNE
import numpy as np
from tqdm import tqdm

# Path to the directory containing .wav files
data_dir = 'GTZAN/genres_original'

# List all .wav files
wav_files = glob.glob(os.path.join(data_dir, '*.wav'))
#print(wav_files[:5]) #OUT: 'Data/genres_original/blues.00002.wav'

# Extract labels and file paths
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
        file_path = self.df.iloc[idx, 0] #OUT: jazz.00065.wav
        label = self.df.iloc[idx, 1] #OUT: jazz
        aud = AudioUtil.open(file_path)
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
print('Splitting Data into 80-20 Test-Val Split')
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)

print('Processing waveform inputs')
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
    
    def forward(self, x, return_embeddings=False):
        # print(f"Input shape: {x.shape}")  
        x = self.pool(F.relu(self.conv1(x)))
        # print(f"After conv1 and pool: {x.shape}")  
        x = self.pool(F.relu(self.conv2(x)))
        # print(f"After conv2 and pool: {x.shape}")  
        x = self.pool(F.relu(self.conv3(x)))
        # print(f"After conv3 and pool: {x.shape}")  
        x = self.pool(F.relu(self.conv4(x)))
        # print(f"After conv4 and pool: {x.shape}")  
        x = x.view(x.size(0), -1) 
        # print(f"After view: {x.shape}")  
        embeddings = F.relu(self.fc1(x))
        # print(f"After fc1: {x.shape}") 
        x = self.dropout(embeddings)
        x = self.fc2(x)
        # print(f"Output shape: {x.shape}")  
        if return_embeddings:
            return x, embeddings
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AudioClassifier().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_model(model, train_loader, val_loader, criterion, optimizer, max_epochs=1000, patience=10):
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    best_val_loss = float('inf')
    best_model = None
    epochs_without_improvement = 0
    
    for epoch in range(max_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                correct += torch.sum(preds == labels.data)
                total += labels.size(0)
        
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)
        val_acc = correct.double() / total
        val_accuracies.append(val_acc)
        
        print(f'Epoch {epoch}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}')
        
        # Check if this is the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        
        # Early stopping
        if epochs_without_improvement >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Load the best model
    model.load_state_dict(best_model)
    
    # Plot losses
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('loss_plot_trainval.png')
    plt.close()
    
    return model, train_losses, val_losses, val_accuracies

# Train the model
print('Starting Training')
model, train_losses, val_losses, val_accuracies = train_model(model, train_loader, val_loader, criterion, optimizer, patience=10)

# Evaluation function
def evaluate_model(model, dataloader, classes):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('confusion_matrix_music_trainval.png')
    plt.close()
    
    # Compute metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    
    print(f"Overall Accuracy: {accuracy:.4f}")
    print(f"Overall Precision: {precision:.4f}")
    print(f"Overall Recall: {recall:.4f}")
    print(f"Overall F1-score: {f1:.4f}")
    
    # Per-class breakdown
    print("\nPer-class breakdown:")
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average=None)
    for i, class_name in enumerate(classes):
        print(f"{class_name}:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision[i]:.4f}")
        print(f"  Recall: {recall[i]:.4f}")
        print(f"  F1-score: {f1[i]:.4f}")

# Evaluate the model
classes = list(index_to_label.values())
print('Evaluating Model on Val Set')
evaluate_model(model, val_loader, classes)

# TSNE #

def create_tsne_plot(model, dataloader, classes):
    model.eval()
    embeddings = []
    labels = []
    
    with torch.no_grad():
        for inputs, batch_labels in dataloader:
            inputs = inputs.to(device)
            _, batch_embeddings = model(inputs, return_embeddings=True)
            embeddings.append(batch_embeddings.cpu().numpy())
            labels.extend(batch_labels.numpy())
    
    embeddings = np.vstack(embeddings)
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Plot
    plt.figure(figsize=(10, 8))
    for i, class_name in enumerate(classes):
        mask = np.array(labels) == i
        plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], label=class_name, alpha=0.7)
    
    plt.legend()
    plt.title('t-SNE plot of audio embeddings')
    plt.xlabel('t-SNE feature 1')
    plt.ylabel('t-SNE feature 2')
    plt.savefig('tsne_plot_music_trainval.png')
    plt.close()

print('Making T-SNE')
create_tsne_plot(model, val_loader, classes)
print('Training, Validation, Evaluation and Visualization Complete')