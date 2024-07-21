print('starting imports')
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
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix

print('finished imports')

def preprocess_and_save(dataset, output_dir, max_ms=30000, n_mels=64, n_fft=1024, hop_len=None):
    os.makedirs(output_dir, exist_ok=True)
    
    for idx in tqdm(range(len(dataset)), desc="Pre-processing audio"):
        item = dataset[idx]
        audio_file = item['audio']['path']
        
        # Open and process the audio
        aud = AudioUtil.open(audio_file)
        reaud = AudioUtil.resample(aud, 16000)
        rechan = AudioUtil.rechannel(reaud, 1)
        dur_aud = AudioUtil.pad_trunc(rechan, max_ms)
        spect = AudioUtil.spectro_gram(dur_aud, n_mels=n_mels, n_fft=n_fft, hop_len=hop_len)
        spect = spect.unsqueeze(0)
        
        # Save the processed spectrogram
        output_file = os.path.join(output_dir, f"{idx}.pt")
        torch.save({
            'spectrogram': spect,
            'category': item['category'],
            'title': item['title'],
            'segment_id':item['segment_id']
        }, output_file)


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
    def __init__(self, preprocessed_dir, indices):
        self.preprocessed_dir = preprocessed_dir
        self.file_list = sorted([f for f in os.listdir(preprocessed_dir) if f.endswith('.pt')])
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        file_idx = self.indices[idx]
        file_path = os.path.join(self.preprocessed_dir, self.file_list[file_idx])
        data = torch.load(file_path)
        return data['spectrogram'], data['category'], data['title'], data['segment_id']


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

## Stratified train-test split on ONE preprocessed folder by fraction of data ##

train_dir = '/work/tc062/tc062/manishav/huggingface_cache/datasets/speechcolab___gigaspeech/s/0.0.0/0db31224ad43470c71b459deb2f2b40956b3a4edfde5fb313aaec69ec7b50d3c/gigaspeech-train.arrow'

# Load the full datasets
full_dataset = Dataset.from_file(train_dir)
full_dataset = full_dataset.cast_column("audio", Audio(sampling_rate=16000))

print('getting category names')

category_names = full_dataset.features['category'].names
# Create a dictionary mapping indices to category names
category_dict = {i: name for i, name in enumerate(category_names)}
print(f"Category Mapping: {category_dict}")

def stratified_train_test_split(preprocessed_dir, test_size=0.2, fraction=0.25, random_state=42):
    print('starting to sort')
    file_list = sorted([f for f in os.listdir(preprocessed_dir) if f.endswith('.pt')])
    print('finished sorting')

    # Load all data to get categories and titles
    all_data = [torch.load(os.path.join(preprocessed_dir, f)) for f in file_list]

    # Create a dictionary to group files by category and title
    grouped_files = {}
    for idx, data in tqdm(enumerate(all_data)):
        key = (data['category'], data['title'])
        if key not in grouped_files:
            grouped_files[key] = []
        grouped_files[key].append(idx)
    
    # Perform stratified sampling
    random.seed(random_state)
    train_indices = []
    test_indices = []

    for group in tqdm(grouped_files.values()):
        # Calculate how many samples to take from this group
        n_samples = max(1, int(len(group) * fraction))
        sampled_group = random.sample(group, n_samples)
        
        n_test = max(1, int(len(sampled_group) * test_size))
        n_train = len(sampled_group) - n_test
        
        # Ensure at least one sample in each set if possible
        if n_train == 0 and len(sampled_group) > 1:
            n_train = 1
            n_test = len(sampled_group) - 1
        
        group_test = random.sample(sampled_group, n_test)
        group_train = [idx for idx in sampled_group if idx not in group_test]
        
        train_indices.extend(group_train)
        test_indices.extend(group_test)
    
    return train_indices, test_indices


preprocessed_dir = 'preprocessed_data'
train_indices, test_indices = stratified_train_test_split(preprocessed_dir, test_size=0.2, fraction=0.25)
print('done stratifying')

# Create DataLoaders with preprocessed data
train_dataset = GenreDataset(preprocessed_dir, train_indices)
test_dataset = GenreDataset(preprocessed_dir, test_indices)

print(f'Number of training samples: {len(train_dataset)}')
print(f'Number of testing samples: {len(test_dataset)}')

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print('finished loading')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AudioClassifier().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_model(model, train_loader, criterion, optimizer, num_epochs=100):
    train_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch in tqdm(train_loader):
            inputs, labels, titles, segment_ids = batch

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
        
        print(f'Epoch {epoch}/{num_epochs - 1}, Train Loss: {epoch_loss:.4f}')
    
    # Save the loss plot
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.legend()
    plt.savefig(os.path.join("outputs/loss", f"loss_plot_{num_epochs}epochs_exp1_sl_25%.png"))
    plt.close()

# Train the model
train_model(model, train_loader, criterion, optimizer, num_epochs=100)

# Evaluation on test dataset
model.eval()
test_embeddings = []
test_labels = []
test_predictions = []
# test_file_paths = []
test_titles = []
test_segment_ids = []

with torch.no_grad():
    for batch in test_loader:
        inputs, labels, titles, segment_ids = batch

        if inputs.dim() == 3:
            inputs = inputs.unsqueeze(1)
        inputs = inputs.to(device)
        outputs, emb = model(inputs)
        _, preds = torch.max(outputs, 1)
        
        test_embeddings.append(emb.cpu().numpy())
        test_labels.append(labels.cpu().numpy())
        test_predictions.append(preds.cpu().numpy())
        # test_file_paths.extend(paths)
        test_titles.extend(titles)
        test_segment_ids.extend(segment_ids)

test_embeddings = np.concatenate(test_embeddings)
test_labels = np.concatenate(test_labels)
test_predictions = np.concatenate(test_predictions)


## FULL EVALUATION OF TEST SET ##

# Get unique labels
unique_labels = sorted(set(test_labels))

# Calculate metrics per category
print("Metrics per category:")
for label in unique_labels:
    mask = test_labels == label
    category_accuracy = accuracy_score(test_labels[mask], test_predictions[mask])
    category_f1 = f1_score(test_labels[mask], test_predictions[mask], average='weighted')
    category_precision = precision_score(test_labels[mask], test_predictions[mask], average='weighted')
    category_recall = recall_score(test_labels[mask], test_predictions[mask], average='weighted')
    
    print(f"\nCategory: {category_dict[label]}")
    print(f"  Accuracy: {category_accuracy:.4f}")
    print(f"  F1 Score: {category_f1:.4f}")
    print(f"  Precision: {category_precision:.4f}")
    print(f"  Recall: {category_recall:.4f}")

# Calculate overall metrics
print("\nOverall Metrics:")

# Weighted average
accuracy = accuracy_score(test_labels, test_predictions)
f1_weighted = f1_score(test_labels, test_predictions, average='weighted')
precision_weighted = precision_score(test_labels, test_predictions, average='weighted')
recall_weighted = recall_score(test_labels, test_predictions, average='weighted')

print("Weighted Average:")
print(f"  Accuracy: {accuracy:.4f}")
print(f"  F1 Score: {f1_weighted:.4f}")
print(f"  Precision: {precision_weighted:.4f}")
print(f"  Recall: {recall_weighted:.4f}")

# Macro average
f1_macro = f1_score(test_labels, test_predictions, average='macro')
precision_macro = precision_score(test_labels, test_predictions, average='macro')
recall_macro = recall_score(test_labels, test_predictions, average='macro')

print("\nMacro Average:")
print(f"  F1 Score: {f1_macro:.4f}")
print(f"  Precision: {precision_macro:.4f}")
print(f"  Recall: {recall_macro:.4f}")

# Micro average
f1_micro = f1_score(test_labels, test_predictions, average='micro')
precision_micro = precision_score(test_labels, test_predictions, average='micro')
recall_micro = recall_score(test_labels, test_predictions, average='micro')

print("\nMicro Average:")
print(f"  F1 Score: {f1_micro:.4f}")
print(f"  Precision: {precision_micro:.4f}")
print(f"  Recall: {recall_micro:.4f}")

# Classification report
print("\nClassification Report:")
print(classification_report(test_labels, test_predictions, target_names=[category_dict[i] for i in unique_labels]))

# Confusion Matrix
cm = confusion_matrix(test_labels, test_predictions)
all_categories = list(category_dict.values())

plt.figure(figsize=(25, 20))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=all_categories,
            yticklabels=all_categories)
plt.xlabel('Predicted', fontsize=14)
plt.ylabel('True', fontsize=14)
plt.title('Confusion Matrix for Test Set', fontsize=16)
plt.xticks(rotation=90, fontsize=10)
plt.yticks(rotation=0, fontsize=10)
plt.tight_layout()
plt.savefig('confusion_matrix_exp1_sl_100epochs_25%.png', dpi=300)
plt.close()

print("saved confusion_matrix_exp1_sl.png")

# T-SNE visualization
tsne = TSNE(n_components=2, random_state=42)
tsne_results = tsne.fit_transform(test_embeddings)

def generate_distinct_colors(n):
    HSV_tuples = [(x * 1.0 / n, 0.5, 0.5) for x in range(n)]
    RGB_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)
    return ['rgb'+str(tuple(int(255*x) for x in rgb)) for rgb in RGB_tuples]

colors = generate_distinct_colors(len(category_dict))
color_map = {name: color for name, color in zip(category_dict.values(), colors)}

df = pd.DataFrame({
    'x': tsne_results[:, 0],
    'y': tsne_results[:, 1],
    'label': test_labels,
    'predicted': test_predictions,
    'category': [category_dict[l] for l in test_labels],
    'predicted_category': [category_dict[p] for p in test_predictions],
    'title': test_titles,
    'segment_id': test_segment_ids
})

fig = px.scatter(df, x='x', y='y', color='category', 
                 symbol='predicted_category',
                 hover_data=['category', 'predicted_category', 'title', 'segment_id'], 
                 color_discrete_map=color_map, 
                 title='T-SNE of Test Set Embeddings')

fig.update_layout(
    legend_title_text='True Categories',
    legend=dict(
        itemsizing='constant',
        title_font_family='Arial',
        font=dict(family='Arial', size=10),
        itemwidth=30
    )
)

fig.update_traces(marker=dict(size=5, opacity=0.7))

output_dir = 'outputs/tsne'
plot_file = os.path.join(output_dir, "tsne_exp1_sl_50epochs_25%.html")
fig.write_html(plot_file)
print(f"Saved interactive plot to {plot_file}")

fig.show()
