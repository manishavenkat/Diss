import os
import random
import torch
import torchaudio
import pandas as pd
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
from tqdm import tqdm
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import colorsys
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, confusion_matrix


train_dir = '/work/tc062/tc062/manishav/huggingface_cache/datasets/speechcolab___gigaspeech/xs/0.0.0/0db31224ad43470c71b459deb2f2b40956b3a4edfde5fb313aaec69ec7b50d3c/gigaspeech-train.arrow'

# Load the full datasets
train_dataset = Dataset.from_file(train_dir)
train_dataset = train_dataset.cast_column("audio", Audio(sampling_rate=16000))

print('getting category names')

category_names = train_dataset.features['category'].names
# Create a dictionary mapping indices to category names
category_dict = {i: name for i, name in enumerate(category_names)}
all_categories = list(category_dict.values())

print('loaded dataset')

def stratified_train_test_split(dataset, test_size=0.2, random_state=42):
    df = dataset.to_pandas()
    train_df, test_df = train_test_split(df, test_size=test_size, stratify=df['category'], random_state=random_state)
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)
    return train_dataset, test_dataset

# Split the original training data
train_dataset, new_test_dataset = stratified_train_test_split(train_dataset)

print('split data')

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

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        audio_data = item['audio']

        # Get the file path
        file_path = audio_data.get('path', '')
        
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

        return sgram, torch.tensor(label, dtype=torch.long), file_path

# Create dataset objects
train_dataset = GenreDataset(train_dataset)
test_dataset = GenreDataset(new_test_dataset)  # Use the new test set

print('starting dataloader')
# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, collate_fn=lambda x: tuple(zip(*x)))
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, collate_fn=lambda x: tuple(zip(*x)))
print('ending dataloader')

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

def train_model(model, train_loader, criterion, optimizer, num_epochs=2000):
    train_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch in tqdm(train_loader):
            inputs, labels, _ = batch
            inputs = torch.stack(inputs)  # Stack the inputs into a single tensor
            labels = torch.stack(labels)
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
    
    # Save the loss plot
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.legend()
    plt.savefig(os.path.join("gs-embeddings", f"loss_plot_{num_epochs}epochs.png"))
    plt.close()

# Call the function
train_model(model, train_loader, criterion, optimizer, num_epochs=2000)

# Extract embeddings from test set
model.eval()
embeddings = []
labels_list = []
file_paths = []
predictions = []

with torch.no_grad():
    for batch in test_loader:
        inputs, labels, paths = batch
        inputs = torch.stack(inputs)
        labels = torch.stack(labels)
        if inputs.dim() == 3:
            inputs = inputs.unsqueeze(1)
        inputs = inputs.to(device)
        outputs, emb = model(inputs)
        _, preds = torch.max(outputs, 1)
        
        embeddings.append(emb.cpu().numpy())
        labels_list.append(labels.cpu().numpy())
        file_paths.extend(paths)
        predictions.append(preds.cpu().numpy())

embeddings = np.concatenate(embeddings)
labels = np.concatenate(labels_list)
predictions = np.concatenate(predictions)

# Print shapes for debugging
print("Embeddings shape:", embeddings.shape)
print("Labels shape:", labels.shape)
print("Predictions shape:", predictions.shape)
print("Number of file paths:", len(file_paths))

# After running predictions on the test set

accuracy = accuracy_score(labels, predictions)
f1 = f1_score(labels, predictions, average='weighted')
precision = precision_score(labels, predictions, average='weighted')

print(f"Test Set Accuracy: {accuracy:.4f}")
print(f"Test Set F1 Score: {f1:.4f}")
print(f"Test Set Precision: {precision:.4f}")

# Confusion Matrix
cm = confusion_matrix(labels, predictions)

# Plot confusion matrix
plt.figure(figsize=(25, 20))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=all_categories,
            yticklabels=all_categories)
plt.xlabel('Predicted', fontsize=14)
plt.ylabel('True', fontsize=14)
plt.title('Confusion Matrix', fontsize=16)
plt.xticks(rotation=90, fontsize=10)
plt.yticks(rotation=0, fontsize=10)
plt.tight_layout()
plt.savefig('confusion_matrix_test_2kepochs.png', dpi=300)
plt.close()

print("Confusion matrix saved as 'confusion_matrix_2kepochs_test.png'")

from sklearn.manifold import TSNE
import plotly.express as px

# T-SNE
tsne = TSNE(n_components=2, random_state=42)
tsne_results = tsne.fit_transform(embeddings)

# Generate distinct colors
def generate_distinct_colors(n):
    HSV_tuples = [(x * 1.0 / n, 0.5, 0.5) for x in range(n)]
    RGB_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)
    return ['rgb'+str(tuple(int(255*x) for x in rgb)) for rgb in RGB_tuples]

# Generate distinct colors for each category
colors = generate_distinct_colors(len(category_dict))
color_map = {name: color for name, color in zip(category_dict.values(), colors)}

# Create a DataFrame for plotting
df = pd.DataFrame({
    'x': tsne_results[:, 0],
    'y': tsne_results[:, 1],
    'label': labels,
    'predicted': predictions,
    'category': [category_dict[l] for l in labels],
    'predicted_category': [category_dict[p] for p in predictions],
    'file_path': file_paths
})

# Create the interactive plot
fig = px.scatter(df, x='x', y='y', color='category', 
                 symbol='predicted_category',
                 hover_data=['category', 'predicted_category', 'file_path'], 
                 color_discrete_map=color_map, 
                 title='T-SNE of Speech Embeddings (Test Set)')

# Update the legend
fig.update_layout(
    legend_title_text='True Categories',
    legend=dict(
        itemsizing='constant',
        title_font_family='Arial',
        font=dict(family='Arial', size=10),
        itemwidth=30
    )
)

# Adjust marker size and opacity
fig.update_traces(marker=dict(size=5, opacity=0.7))

output_dir = 'gs-embeddings'

# Save the plot as an HTML file
plot_file = os.path.join(output_dir, "tsne_embeddings_test_set.html")
fig.write_html(plot_file)
print(f"Saved interactive plot to {plot_file}")

# Show the plot in a browser
fig.show()

