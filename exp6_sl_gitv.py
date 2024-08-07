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
import csv
import json

print('finished imports')

categories_to_include = set([21, 6, 12, 17, 15, 0, 25, 11])

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)  

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
    def __init__(self, preprocessed_dir, filtered_metadata, indices, category_mapping):
        self.preprocessed_dir = preprocessed_dir
        self.filtered_metadata = filtered_metadata
        self.indices = indices
        self.category_mapping = category_mapping

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        metadata = self.filtered_metadata[self.indices[idx]]
        file_path = os.path.join(self.preprocessed_dir, metadata['filename'])
        data = torch.load(file_path)
        original_category = int(metadata['category'])
        mapped_category = self.category_mapping[original_category]
        return data['spectrogram'], mapped_category, metadata['title'], metadata['segment_id']


class AudioClassifier(nn.Module):
    def __init__(self):
        super(AudioClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 4, 128)  # Corrected dimension
        self.fc2 = nn.Linear(128, 8)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        # print(f"Input shape: {x.shape}")          ##Input shape: torch.Size([32, 1, 64, 938])
        x = self.pool(F.relu(self.conv1(x)))
        # print(f"After conv1 and pool: {x.shape}")  ##torch.Size([32, 8, 32, 469])
        x = self.pool(F.relu(self.conv2(x)))
        # print(f"After conv2 and pool: {x.shape}")  ##([32, 16, 16, 234])
        x = self.pool(F.relu(self.conv3(x)))
        # print(f"After conv3 and pool: {x.shape}")  ##torch.Size([32, 32, 8, 117])
        x = self.pool(F.relu(self.conv4(x)))
        # print(f"After conv4 and pool: {x.shape}")  ##torch.Size([32, 64, 4, 58])
        # x = x.view(x.size(0), -1)  ##Flatten 
        # print(f"After view: {x.shape}")  ##torch.Size([32, 14848])
        x = torch.mean(x, dim=-1) ## -1 bc its dim of audio 
        # print(f"After mean: {x.shape}")
        x = x.view(x.size(0), -1)  ##Flatten 
        # print(f"After view: {x.shape}")  ##torch.Size([32, 14848])
        x = F.relu(self.fc1(x))
        # print(f"After fc1: {x.shape}")
        # x = self.dropout(x)
        # x = self.fc2(x)

        embeddings = x
        # print("Shape of embeddings:", embeddings.shape)
        x = self.dropout(embeddings)
        x = self.fc2(x)
        # print(f"Output shape: {x.shape}")
        return x, embeddings

## Stratified train-test split on ONE preprocessed folder by fraction of data ##

def stratified_train_test_split(metadata_file, fraction=1.0, test_size=0.20, random_state=42):
    # Specify the categories to include
    split_file = 'exp6_sl_train_test.json'
    global categories_to_include
    print(f"Categories to include: {categories_to_include}")
    category_mapping = {old_id: new_id for new_id, old_id in enumerate(sorted(categories_to_include))}
    reverse_category_mapping = {new_id: old_id for old_id, new_id in category_mapping.items()}

    if os.path.exists(split_file):
        with open(split_file, 'r') as f:
            split_data = json.load(f)
        return (split_data['train_indices'], split_data['test_indices'], 
                split_data['category_mapping'], split_data['reverse_category_mapping'])
    else:
        # Load metadata
        with open(metadata_file, 'r') as f:
            reader = csv.DictReader(f)
            filtered_metadata = [item for item in reader if int(item['category']) in categories_to_include]

        print(f"Total items after category filtering: {len(filtered_metadata)}")

        if len(filtered_metadata) == 0:
            print("No items match the specified categories. Check your category IDs.")
            return [], []

        # Create a dictionary to group files by category and title
        grouped_files = {}
        for idx, item in enumerate(filtered_metadata):
            key = (int(item['category']), item['title'])
            if key not in grouped_files:
                grouped_files[key] = []
            grouped_files[key].append(idx)
        
        print(f"Unique (category, title) pairs: {len(grouped_files)}")

        # Group titles by category
        category_titles = {}
        for (category, title), indices in grouped_files.items():
            if category not in category_titles:
                category_titles[category] = set()
            category_titles[category].add(title)
        
        print(f"Categories found in data: {list(category_titles.keys())}")

        # Perform stratified sampling of titles
        random.seed(random_state)
        train_indices = []
        test_indices = []

        for category, titles in category_titles.items():
            # Sample 25% of titles for this category
            n_titles_to_use = max(1, int(len(titles) * fraction))
            titles_to_use = set(random.sample(titles, n_titles_to_use))
            
            # Split these titles into train (80%) and test (20%)
            n_test = max(1, int(len(titles_to_use) * test_size))
            test_titles = set(random.sample(titles_to_use, n_test))
            train_titles = titles_to_use - test_titles

            for (cat, title), indices in grouped_files.items():
                if cat == category and title in titles_to_use:
                    if title in train_titles:
                        train_indices.extend(indices)
                    else:
                        test_indices.extend(indices)
        
        print(f"Train indices: {len(train_indices)}")
        print(f"Test indices: {len(test_indices)}")

        # calculate total durations
        train_duration = sum(float(filtered_metadata[i]['duration']) for i in train_indices)
        test_duration = sum(float(filtered_metadata[i]['duration']) for i in test_indices)

        print(f"Total duration of train set: {train_duration:.2f} seconds")
        print(f"Total duration of test set: {test_duration:.2f} seconds")

        # Save the split data
        split_data = {
            'train_indices': train_indices,
            'test_indices': test_indices,
            'category_mapping': category_mapping,
            'reverse_category_mapping': reverse_category_mapping,
            'train_duration': train_duration,
            'test_duration': test_duration
        }
        with open(split_file, 'w') as f:
            json.dump(split_data, f)
    
        return train_indices, test_indices, category_mapping, reverse_category_mapping, filtered_metadata

# Usage
preprocessed_dir = 'preprocessed_data'
set_seed(42)
train_indices, test_indices, category_mapping, reverse_category_mapping, filtered_metadata = stratified_train_test_split('metadata_longaudio.csv', test_size=0.2, fraction=1.0, random_state=42)

print('done stratifying')

# Usage
train_dataset = GenreDataset('preprocessed_data', filtered_metadata, train_indices, category_mapping)
test_dataset = GenreDataset('preprocessed_data', filtered_metadata, test_indices, category_mapping)


print(f'Number of training samples: {len(train_dataset)}')
print(f'Number of testing samples: {len(test_dataset)}')

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print('finished loading')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AudioClassifier().to(device)
# model.load_state_dict(torch.load('SL/exp3_sl_200e.pt', map_location=device))

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_model(model, train_loader, criterion, optimizer, num_epochs=500):
    train_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        ## uncomment this to checkpoint ##

        # if epoch % 50 == 0: 
        #     model_path = f'SL/exp3_sl_{num_epochs}.pt'
        #     torch.save(model.state_dict(), model_path)

        for batch in tqdm(train_loader):
            inputs, labels, titles, segment_ids = batch

            if inputs.dim() == 3:
                inputs = inputs.unsqueeze(1)
            
            inputs, labels = inputs.to(device), labels.to(device)
            # optimizer.zero_grad()
            outputs, _ = model(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
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
    plt.savefig(os.path.join("outputs/loss", f"loss_plot_{num_epochs}epochs_exp6_sl.png"))
    plt.close()

# Train the model
train_model(model, train_loader, criterion, optimizer, num_epochs=500)

print(model)

# Print model's state_dict
print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

# Print optimizer's state_dict
print("Optimizer's state_dict:")
for var_name in optimizer.state_dict():
    print(var_name, "\t", optimizer.state_dict()[var_name])

# Define the file path where the model will be saved
model_path = 'SL/exp6_sl.pt'
# Save the model's state dictionary to the specified file path
torch.save(model.state_dict(), model_path)

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

# Map back to original categories
original_test_labels = np.array([reverse_category_mapping[label] for label in test_labels])
original_test_predictions = np.array([reverse_category_mapping[pred] for pred in test_predictions])

## FULL EVALUATION OF TEST SET ##

# Get unique labels
unique_labels = sorted(set(original_test_labels))

category_dict = {
    0: 'People  and  Blogs', 1: 'Business', 2: 'Nonprofits  and  Activism', 3: 'Crime', 
    4: 'History', 5: 'Pets  and  Animals', 6: 'News and Politics', 7: 'Travel and Events', 
    8: 'Kids and Family', 9: 'Leisure', 10: 'N/A', 11: 'Comedy', 12: 'News  and  Politics', 
    13: 'Sports', 14: 'Arts', 15: 'Science  and  Technology', 16: 'Autos  and  Vehicles', 
    17: 'Science and Technology', 18: 'People and Blogs', 19: 'Music', 20: 'Society and Culture', 
    21: 'Education', 22: 'Howto  and  Style', 23: 'Film  and  Animation', 24: 'Gaming', 
    25: 'Entertainment', 26: 'Travel  and  Events', 27: 'Health and Fitness', 28: 'audiobook'
}

# Calculate metrics per category
print("Metrics per category:")
for label in unique_labels:
    mask = original_test_labels == label
    category_accuracy = accuracy_score(original_test_labels[mask], original_test_predictions[mask])
    category_f1 = f1_score(original_test_labels[mask], original_test_predictions[mask], average='weighted')
    category_precision = precision_score(original_test_labels[mask], original_test_predictions[mask], average='weighted')
    category_recall = recall_score(original_test_labels[mask], original_test_predictions[mask], average='weighted')
    
    print(f"\nCategory: {category_dict[label]}")
    print(f"  Accuracy: {category_accuracy:.4f}")
    print(f"  F1 Score: {category_f1:.4f}")
    print(f"  Precision: {category_precision:.4f}")
    print(f"  Recall: {category_recall:.4f}")

# Calculate overall metrics
print("\nOverall Metrics:")

# Weighted average
accuracy = accuracy_score(original_test_labels, original_test_predictions)
f1_weighted = f1_score(original_test_labels, original_test_predictions, average='weighted')
precision_weighted = precision_score(original_test_labels, original_test_predictions, average='weighted')
recall_weighted = recall_score(original_test_labels, original_test_predictions, average='weighted')

print("Weighted Average:")
print(f"  Accuracy: {accuracy:.4f}")
print(f"  F1 Score: {f1_weighted:.4f}")
print(f"  Precision: {precision_weighted:.4f}")
print(f"  Recall: {recall_weighted:.4f}")

# Macro average
f1_macro = f1_score(original_test_labels, original_test_predictions, average='macro')
precision_macro = precision_score(original_test_labels, original_test_predictions, average='macro')
recall_macro = recall_score(original_test_labels, original_test_predictions, average='macro')

print("\nMacro Average:")
print(f"  F1 Score: {f1_macro:.4f}")
print(f"  Precision: {precision_macro:.4f}")
print(f"  Recall: {recall_macro:.4f}")

# Micro average
f1_micro = f1_score(original_test_labels, original_test_predictions, average='micro')
precision_micro = precision_score(original_test_labels, original_test_predictions, average='micro')
recall_micro = recall_score(original_test_labels, original_test_predictions, average='micro')

print("\nMicro Average:")
print(f"  F1 Score: {f1_micro:.4f}")
print(f"  Precision: {precision_micro:.4f}")
print(f"  Recall: {recall_micro:.4f}")

# Classification report
print("\nClassification Report:")
print(classification_report(original_test_labels, original_test_predictions, target_names=[category_dict[i] for i in unique_labels]))

# Confusion Matrix
cm = confusion_matrix(original_test_labels, original_test_predictions)
# Get category names for these unique classes
class_names = [category_dict[cls] for cls in unique_labels]

plt.figure(figsize=(25, 20))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names)
plt.xlabel('Predicted', fontsize=14)
plt.ylabel('True', fontsize=14)
plt.title('Confusion Matrix for Test Set', fontsize=16)
plt.xticks(rotation=90, fontsize=10)
plt.yticks(rotation=0, fontsize=10)
plt.tight_layout()
plt.savefig('confusion_matrix_exp6_sl.png', dpi=300)
plt.close()

print("saved confusion_matrix_exp6_sl.png")

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
    'label': original_test_labels,
    'predicted': original_test_predictions,
    'category': [category_dict[l] for l in original_test_labels],
    'predicted_category': [category_dict[p] for p in original_test_predictions],
    'title': test_titles,
    'segment_id': test_segment_ids
})

fig = px.scatter(df, x='x', y='y', color='category', 
                 symbol='predicted_category',
                 hover_data=['category', 'predicted_category', 'title', 'segment_id'], 
                 color_discrete_map=color_map, 
                 title='T-SNE of Test Set Embeddings (Larger points: 10-20 sec duration)')

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
plot_file = os.path.join(output_dir, "tsne_exp6_sl.html")
fig.write_html(plot_file)
print(f"Saved interactive plot to {plot_file}")

fig.show()

print('finished script')
