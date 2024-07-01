import os
import random
import torch
import torchaudio
import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as DatasetTorch
import torch.nn.functional as F
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from datasets import load_dataset, Dataset, Audio
from transformers import Wav2Vec2Processor, HubertForCTC
from tqdm import tqdm
import plotly.express as px
import colorsys

os.environ['HF_HOME'] = "/work/tc062/tc062/manishav/huggingface_cache"

# Load your datasets
train_dir = '/work/tc062/tc062/manishav/huggingface_cache/datasets/speechcolab___gigaspeech/xs/0.0.0/0db31224ad43470c71b459deb2f2b40956b3a4edfde5fb313aaec69ec7b50d3c/gigaspeech-train.arrow'
# val_dir = '/work/tc062/tc062/manishav/huggingface_cache/datasets/speechcolab___gigaspeech/xs/0.0.0/0db31224ad43470c71b459deb2f2b40956b3a4edfde5fb313aaec69ec7b50d3c/gigaspeech-validation.arrow'
# test_dir = '/work/tc062/tc062/manishav/huggingface_cache/datasets/speechcolab___gigaspeech/xs/0.0.0/0db31224ad43470c71b459deb2f2b40956b3a4edfde5fb313aaec69ec7b50d3c/gigaspeech-test.arrow'


train_dataset = Dataset.from_file(train_dir).cast_column("audio", Audio(sampling_rate=16000))
# val_dataset = Dataset.from_file(val_dir).cast_column("audio", Audio(sampling_rate=16000))
# test_dataset = Dataset.from_file(test_dir).cast_column("audio", Audio(sampling_rate=16000))

# Define your dataset class
class GenreDataset(DatasetTorch):
    def __init__(self, dataset, sr=16000, transform=None):
        self.dataset = dataset
        self.sr = sr
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        audio_data = item['audio']
        if isinstance(audio_data['array'], np.ndarray):
            sig = torch.from_numpy(audio_data['array']).float()
        else:
            sig = torch.tensor(audio_data['array']).float()
        
        sr = audio_data.get('sampling_rate', self.sr)

        if sig.dim() == 1:
            sig = sig.unsqueeze(0)

        label = item['category']
        file_path = audio_data.get('path', '')

        return sig, torch.tensor(label, dtype=torch.long), file_path

category_names = train_dataset.features['category'].names
category_dict = {i: name for i, name in enumerate(category_names)}

train_dataset = GenreDataset(train_dataset)
# val_dataset = GenreDataset(val_dataset)
# test_dataset = GenreDataset(test_dataset)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, collate_fn=lambda x: tuple(zip(*x)))
# val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, collate_fn=lambda x: tuple(zip(*x)))
# test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, collate_fn=lambda x: tuple(zip(*x)))

# Load the HuBERT model and processor
processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")
model = HubertForCTC.from_pretrained("facebook/hubert-large-ls960-ft").to(device)

def extract_hubert_embeddings(loader):
    embeddings = []
    labels_list = []
    file_paths = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader):
            inputs, labels, paths = batch
            inputs = [processor(audio.numpy(), sampling_rate=16000, return_tensors="pt", padding=True).input_values for audio in inputs]
            inputs = torch.cat(inputs).to(device)
            
            outputs = model(inputs).last_hidden_state.mean(dim=1).cpu().numpy()

            embeddings.append(outputs)
            labels_list.append([label.item() for label in labels])
            file_paths.extend(paths)

    embeddings = np.concatenate(embeddings)
    labels = np.concatenate(labels_list)

    return embeddings, labels, file_paths

train_embeddings, train_labels, train_paths = extract_hubert_embeddings(train_loader)

# T-SNE
tsne = TSNE(n_components=2, random_state=42)
tsne_results = tsne.fit_transform(train_embeddings)

# Generate distinct colors
def generate_distinct_colors(n):
    HSV_tuples = [(x * 1.0 / n, 0.5, 0.5) for x in range(n)]
    RGB_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)
    return ['rgb'+str(tuple(int(255*x) for x in rgb)) for rgb in RGB_tuples]

colors = generate_distinct_colors(len(category_dict))
color_map = {name: color for name, color in zip(category_dict.values(), colors)}

df = pd.DataFrame({
    'x': tsne_results[:, 0],
    'y': tsne_results[:, 1],
    'label': train_labels,
    'category': [category_dict[l] for l in train_labels],
    'file_path': train_paths
})

fig = px.scatter(df, x='x', y='y', color='category', hover_data=['category', 'file_path'], color_discrete_map=color_map, title='T-SNE of HuBERT Embeddings')

fig.update_layout(
    legend_title_text='Categories',
    legend=dict(
        itemsizing='constant',
        title_font_family='Arial',
        font=dict(family='Arial', size=10),
        itemwidth=30
    )
)

fig.update_traces(marker=dict(size=5, opacity=0.7))

output_dir = "gs-embeddings"

plot_file = os.path.join(output_dir, "tsne_hubert_embeddings.html")
fig.write_html(plot_file)
print(f"Saved interactive plot to {plot_file}")

fig.show()
