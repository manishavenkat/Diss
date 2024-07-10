import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as DatasetTorch
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
from sklearn.decomposition import PCA
import seaborn as sns

## PCA AND HEATMAP TO UNDERSTAND WAVEFORM-LIKE TSNE PLOT ##

print('loading train_dir')

train_dir = '/work/tc062/tc062/manishav/huggingface_cache/datasets/speechcolab___gigaspeech/xs/0.0.0/0db31224ad43470c71b459deb2f2b40956b3a4edfde5fb313aaec69ec7b50d3c/gigaspeech-train.arrow'

# Load the full datasets
train_data = Dataset.from_file(train_dir)
train_data = train_data.cast_column("audio", Audio(sampling_rate=16000))

all_categories = []
category_names = train_data.features['category'].names
category_dict = {i: name for i, name in enumerate(category_names)}
all_category_names = [category_dict[cat] for cat in all_categories]

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

class TotalLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=0.1, gamma=0.01):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.mse = nn.MSELoss()
        self.ce = nn.CrossEntropyLoss()
    
    def forward(self, decoded, original, latent, cluster_logits):
        # Resize the original to match the decoded dimensions
        original_resized = F.interpolate(original, size=decoded.shape[2:])
        
        # Reconstruction loss
        rec_loss = self.mse(decoded, original_resized)
        
        # Clustering loss (entropy minimization)
        cluster_probs = torch.softmax(cluster_logits, dim=1)
        cluster_loss = -(cluster_probs * torch.log(cluster_probs + 1e-10)).sum(dim=1).mean()
        
        # Contrastive loss
        norm_latent = torch.nn.functional.normalize(latent, p=2, dim=1)
        sim_matrix = torch.matmul(norm_latent, norm_latent.t())
        temperature = 0.5
        sim_matrix /= temperature
        labels = torch.arange(sim_matrix.size(0)).to(sim_matrix.device)
        contrastive_loss = self.ce(sim_matrix, labels)
        
        return self.alpha * rec_loss + self.beta * cluster_loss + self.gamma * contrastive_loss

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

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader):
        audio, _, _, _ = batch
        audio = audio.to(device)  
        optimizer.zero_grad()
        decoded, latent, cluster_logits = model(audio)
        loss = criterion(decoded, audio, latent, cluster_logits)
        loss_values.append(loss.item())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}")

epochs = list(range(1, len(loss_values) + 1))
plt.figure(figsize=(10, 6))
plt.plot(epochs, loss_values, marker='o', linestyle='-', color='b', label='Loss')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()
plt.tight_layout()

loss_output_dir = 'outputs/loss'
plot_file = os.path.join(loss_output_dir, "loss_exp2_100epochs.png")
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
print(all_embeddings[0])

print('starting tsne')

print('Creating heatmap and PCA')

output_dir = 'outputs/pca-heatmaps'

# Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(all_embeddings[:100, :100], cmap='viridis')  # Adjust size as needed
plt.title('Heatmap of Embeddings (First 100x100)')
plt.tight_layout()
heatmap_file = os.path.join(output_dir, "heatmap_exp2.png")
plt.savefig(heatmap_file)
print(f"Saved heatmap to {heatmap_file}")
plt.close()

# PCA
pca = PCA(n_components=2)
embeddings_2d_pca = pca.fit_transform(all_embeddings)

# Create interactive PCA plot
fig = go.Figure(data=go.Scatter(
    x=embeddings_2d_pca[:, 0],
    y=embeddings_2d_pca[:, 1],
    mode='markers',
    marker=dict(
        size=5,
        color=embeddings_2d_pca[:, 0],  # Color by first PCA component
        colorscale='Viridis',
        showscale=True
    ),
    text=[f"Category: {cat}<br>File: {path}<br>Title: {title}" for cat, path, title in zip(all_categories, all_file_paths, all_titles)],
    hoverinfo='text'
))

fig.update_layout(
    title='PCA of Embeddings',
    xaxis_title='First Principal Component',
    yaxis_title='Second Principal Component'
)

# Save the PCA plot as an HTML file
pca_plot_file = os.path.join(output_dir, "pca_exp2_100epochs.html")
fig.write_html(pca_plot_file)
print(f"Saved interactive PCA plot to {pca_plot_file}")

fig.show()