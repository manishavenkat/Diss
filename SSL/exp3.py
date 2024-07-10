import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as DatasetTorch
# from torch.profiler import profile, ProfilerActivity
import torch.nn.functional as F
import numpy as np
from sklearn.manifold import TSNE
import plotly.graph_objects as go
from datasets import load_dataset, Dataset, Audio
from sklearn.model_selection import train_test_split
import torchaudio
import random
from tqdm import tqdm
import plotly.graph_objs as go
import colorsys
import os
import matplotlib.pyplot as plt
import plotly.express as px
from plotly.subplots import make_subplots

print('loading train_dir')

train_dir = '/work/tc062/tc062/manishav/huggingface_cache/datasets/speechcolab___gigaspeech/xs/0.0.0/0db31224ad43470c71b459deb2f2b40956b3a4edfde5fb313aaec69ec7b50d3c/gigaspeech-train.arrow'

# Load the full datasets
train_data = Dataset.from_file(train_dir)
train_data = train_data.cast_column("audio", Audio(sampling_rate=16000))

# all_categories = []
# category_names = train_data.features['category'].names
# category_dict = {i: name for i, name in enumerate(category_names)}
# all_category_names = [category_dict[cat] for cat in all_categories]

print('loaded dataset')

def stratified_train_test_split(dataset, test_size=0.2, random_state=42):
    df = dataset.to_pandas()
    train_df, test_df = train_test_split(df, test_size=test_size, stratify=df['category'], random_state=random_state)
    train_data = Dataset.from_pandas(train_df)
    test_data = Dataset.from_pandas(test_df)
    return train_data, test_data

# Split the original training data
train_data, test_data = stratified_train_test_split(train_data)

print('split data')

print('running AudioUtil')

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

print('running AudioDataset')

class AudioDataset(DatasetTorch):
    def __init__(self, data, max_ms=30000, n_mels=64, n_fft=1024, hop_len=None):
        self.data = data
        self.max_ms = max_ms
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_len = hop_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        audio_file = item['audio']['path']
        
        # Open the audio file
        aud = AudioUtil.open(audio_file)
        
        # Resample
        reaud = AudioUtil.resample(aud, 16000)

                # Rechannel
        rechan = AudioUtil.rechannel(reaud, 1)
        
        # Pad or truncate
        dur_aud = AudioUtil.pad_trunc(rechan, self.max_ms)
        
        # Create spectrogram
        spect = AudioUtil.spectro_gram(dur_aud, n_mels=self.n_mels, n_fft=self.n_fft, hop_len=self.hop_len)
        
        # Add channel dimension for the model
        spect = spect.unsqueeze(0)

        category = item['category']
        file_path = item['audio']['path']
        title = item['title']
        return spect, category, file_path, title

print('running ConvAutoencoder')

class PrintLayer(nn.Module):
    def __init__(self, message):
        super(PrintLayer, self).__init__()
        self.message = message
    
    def forward(self, x):
        print(self.message, x.shape)
        return x

class ConvAutoencoder(nn.Module):
    def __init__(self, num_clusters, in_channels=64):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),  
            nn.ReLU(),
            # PrintLayer("After first conv2d"),
            # nn.MaxPool2d(2, 2),  #reduce spatial dimensions and retain local features
            # PrintLayer("After first MaxPool"),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  
            nn.ReLU(),
            # PrintLayer("After second conv2d"),
            # nn.MaxPool2d(2, 2), 
            # PrintLayer("After second MaxPool"),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), #get a globally representative feature vectors for clustering
            # PrintLayer("After AdaptiveAvgPool2d")
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            # PrintLayer("After first ConvTranspose2d"),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            # PrintLayer("After second ConvTranspose2d"),
            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            # PrintLayer("After third ConvTranspose2d"),
            nn.ConvTranspose2d(8, 4, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            # PrintLayer("After fourth ConvTranspose2d"),
            nn.ConvTranspose2d(4, 2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            # PrintLayer("After fifth ConvTranspose2d"),
            nn.ConvTranspose2d(2, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            # PrintLayer("After final ConvTranspose2d")
        )

        # Clustering head
        self.clustering_head = nn.Linear(64, num_clusters)
        
    def forward(self, x):
        encoded = self.encoder(x)
        latent = encoded.squeeze(-1).squeeze(-1)  # Remove spatial dimensions
        # print(f"latent shape: {latent.shape}")
        decoded = self.decoder(encoded)
        # print(f"decoded shape: {decoded.shape}")
        cluster_logits = self.clustering_head(latent)
        return decoded, latent, cluster_logits

# Print the architecture to check the changes
model = ConvAutoencoder(num_clusters=30)
# print(model)

print('running TotalLoss')

## No contrastive loss ##

class TotalLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=0.1):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.mse = nn.MSELoss()
        self.ce = nn.CrossEntropyLoss()
    
    def forward(self, decoded, original, latent, cluster_logits):
        # Resize the original to match the decoded dimensions
        original_resized = F.interpolate(original, size=decoded.shape[2:])
        
        # Reconstruction loss
        rec_loss = self.mse(decoded, original_resized)
        
        # Clustering loss (entropy minimization)
        # cluster_probs = torch.softmax(cluster_logits, dim=1)
        # cluster_loss = -(cluster_probs * torch.log(cluster_probs + 1e-10)).sum(dim=1).mean()
        cluster_loss = self.ce(cluster_logits, torch.argmax(cluster_logits, dim=1))
        
        return self.alpha * rec_loss + self.beta * cluster_loss 

print('loading train and test data')

# Create DataLoaders
train_dataset = AudioDataset(train_data)
test_dataset = AudioDataset(test_data)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print('finished loading')

# Initialize model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ConvAutoencoder(num_clusters=30).to(device)
criterion = TotalLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

print('starting training')

loss_values = []
# Manisha Training loop (slow)
num_epochs = 40

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for batch in tqdm(train_loader):
        audio, _, _, _ = batch
        audio = audio.to(device)  
        optimizer.zero_grad()
        decoded, latent, cluster_logits = model(audio)
        loss = criterion(decoded, audio, latent, cluster_logits)
        # loss_values.append(loss.item())
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    avg_epoch_loss = epoch_loss / len(train_loader)
    loss_values.append(avg_epoch_loss)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_epoch_loss:.4f}")

## Andrea Training Loop

#this substitutes the training loop

# with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
#              with_stack=True) as prof:

#     model.train()
#     total_loss = 0
#     for batch in tqdm(train_loader):
#         audio, _, _, _ = batch
#         audio = audio.to(device)  
#         optimizer.zero_grad()
#         decoded, latent, cluster_logits = model(audio)
#         loss = criterion(decoded, audio, latent, cluster_logits)
#         loss_values.append(loss.item())
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()
#         break
# print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=10))
# print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
# print(prof.key_averages(group_by_stack_n=5).table(sort_by="self_cuda_time_total", row_limit=5))
# exit()

epochs = list(range(1, num_epochs + 1))
plt.figure(figsize=(10, 6))
plt.plot(epochs, loss_values, marker='o', linestyle='-', color='b', label='Loss')
plt.title('Training Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Average Loss')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

loss_output_dir = 'outputs/loss'
plot_file = os.path.join(loss_output_dir, "loss_exp3_40epochs.png")
plt.savefig(plot_file)
print(f"Saved loss plot to {plot_file}")

print('starting eval')

# Evaluation and embedding extraction
model.eval()
all_embeddings = []
all_file_paths = []
all_titles = []
all_categories = []

with torch.no_grad():
    for batch in tqdm(test_loader):
        audio, category, file_path, title = batch
        audio = audio.to(device)
        _, latent, _ = model(audio)
        all_embeddings.append(latent.cpu().numpy())
        all_categories.extend(category)
        all_file_paths.extend(file_path)
        all_titles.extend(title)

all_embeddings = np.concatenate(all_embeddings, axis=0)

print('starting tsne')

# # Perform T-SNE in batches
# tsne = TSNE(n_components=2, random_state=42)
# batch_size = 1000
# embeddings_2d = []

# for i in range(0, len(all_embeddings), batch_size):
#     batch = all_embeddings[i:i+batch_size]
#     embeddings_2d.append(tsne.fit_transform(batch))

# embeddings_2d = np.concatenate(embeddings_2d, axis=0)

# Perform T-SNE on the entire dataset
tsne = TSNE(n_components=2, random_state=42)
embeddings_2d = tsne.fit_transform(all_embeddings)

print(f"Shape of embeddings_2d: {embeddings_2d.shape}")
print(f"Type of embeddings_2d: {type(embeddings_2d)}")

print('starting plotting')

# Convert PyTorch tensors to Python integers (if not already done)
all_categories_int = [cat.item() if isinstance(cat, torch.Tensor) else cat for cat in all_categories]

# Create a color scale
unique_categories = sorted(set(all_categories_int))
num_categories = len(unique_categories)
print(f"num_categories: {num_categories}")
color_scale = px.colors.qualitative.Plotly[:num_categories]

# Create a mapping of categories to colors
category_to_color = {cat: color_scale[i % len(color_scale)] for i, cat in enumerate(unique_categories)}
marker_colors = [category_to_color[cat] for cat in all_categories_int]

# Create the figure
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=embeddings_2d[:, 0],
    y=embeddings_2d[:, 1],
    mode='markers',
    marker=dict(
        size=5,
        color=marker_colors,
        showscale=False
    ),
    text=[f"Category: {cat}<br>File: {path}<br>Title: {title}" 
          for cat, path, title in zip(all_categories_int, all_file_paths, all_titles)],
    hoverinfo='text',
    showlegend=False
))

# Create a custom legend
legend_traces = []
for category, color in category_to_color.items():
    legend_traces.append(
        go.Scatter(
            x=[None], y=[None],  # No data, just for legend
            mode='markers',
            marker=dict(size=10, color=color),
            name=f'Category {category}',
            showlegend=True
        )
    )

# Add legend traces to the figure
for trace in legend_traces:
    fig.add_trace(trace)

# Update layout
fig.update_layout(
    title='T-SNE Visualization of Audio Categories',
    xaxis_title='T-SNE Dimension 1',
    yaxis_title='T-SNE Dimension 2',
    legend_title='Categories',
    hovermode='closest'
)

## TSNE W/O legend: tsne_exp1.html output ##

# # Now use this in your scatter plot
# fig = go.Figure(data=go.Scatter(
#     x=embeddings_2d[:, 0],
#     y=embeddings_2d[:, 1],
#     mode='markers',
#     marker=dict(
#         size=5,
#         color=marker_colors,  # Use the mapped colors
#         showscale=False  
#     ),
#     text=[f"Category: {cat}<br>File: {path}<br>Title: {title}" for cat, path, title in zip(all_categories_int, all_file_paths, all_titles)],
#     hoverinfo='text'
# ))

# # Add a color bar to show category mapping
# fig.update_layout(
#     coloraxis_colorbar=dict(
#         title="Categories",
#         tickvals=list(category_to_color.keys()),
#         ticktext=list(category_to_color.keys()),
#         lenmode="pixels", len=300,
#     )
# )

output_dir = 'outputs/tsne'

# Save the plot as an HTML file
plot_file = os.path.join(output_dir, "tsne_exp3_40epochs.html")
fig.write_html(plot_file)
print(f"Saved interactive plot to {plot_file}")

fig.update_layout(title='T-SNE Exp1')
fig.show()