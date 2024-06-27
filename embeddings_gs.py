import os
import torchaudio
import torch
import re
import numpy as np
import matplotlib.pyplot as plt
from speechbrain.inference.classifiers import EncoderClassifier
from sklearn.decomposition import PCA
import datasets
from datasets import Dataset, Audio
import soundfile as sf
import librosa
from sklearn.manifold import TSNE
import pandas as pd 
import plotly.express as px

# Path to the .arrow file
arrow_file = "/work/tc062/tc062/manishav/huggingface_cache/datasets/speechcolab___gigaspeech/xs/0.0.0/0db31224ad43470c71b459deb2f2b40956b3a4edfde5fb313aaec69ec7b50d3c/gigaspeech-train.arrow"
output_dir = "gs-embeddings"

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Load the dataset from the .arrow file
dataset = Dataset.from_file(arrow_file)
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000)) #cast_column(feature name, feature we want to convert it to)

# Slice the dataset to get only the top 300 rows
dataset = dataset.select(range(300))

# Load the pre-trained ECAPA-TDNN model from SpeechBrain
classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="tmpdir")

all_embeddings = []
labels = []
file_paths = []

# Get the mapping of category IDs to category names
category_names = dataset.features['category'].names

# Process each speech file in the dataset
for i, example in enumerate(dataset):
    audio = example['audio']['array']
    sample_rate = example['audio']['sampling_rate']

    # Convert the audio data to a tensor
    signal = torch.tensor(audio).unsqueeze(0)  # Add channel dimension

    # Ensure the signal is mono
    if signal.shape[0] > 1:
        signal = signal.mean(dim=0, keepdim=True)

    # Obtain embeddings
    embeddings = classifier.encode_batch(signal)

    # Convert the embeddings to numpy and flatten them if necessary
    embeddings = embeddings.squeeze().detach().numpy()  # Converting tensor to a numpy array

    # Save embeddings to a file
    embedding_file = os.path.join(output_dir, f"embedding_{i}.pt")
    torch.save(embeddings, embedding_file)
    print(f"Saved embeddings for example {i} to {embedding_file}")

    # Collect all embeddings
    all_embeddings.append(embeddings)

    # Extract category label
    category_id = example['category']
    category_label = category_names[category_id]
    labels.append(category_label)

    # Save file path
    file_paths.append(example['audio']['path'])

# Convert list of embeddings to a 2D numpy array
all_embeddings = np.array(all_embeddings)
all_embeddings = all_embeddings.reshape(all_embeddings.shape[0], -1)  # Flatten to 2D array

## PCA PLOT ##

# # Apply PCA for dimensionality reduction
# pca = PCA(n_components=2)
# principalComponents = pca.fit_transform(all_embeddings)

# # Map labels to colors
# unique_labels = list(set(labels))
# colors = plt.cm.jet(np.linspace(0, 1, len(unique_labels)))
# label_to_color = {label: colors[i] for i, label in enumerate(unique_labels)}

# # Plot PCA
# plt.figure(figsize=(12, 8))
# for i, (x, y) in enumerate(principalComponents):
#     plt.scatter(x, y, color=label_to_color[labels[i]], label=labels[i], s=100)  # Larger points
#     plt.text(x + 0.01, y + 0.01, labels[i], fontsize=9)  # Offset the text slightly for better visibility

# plt.title('PCA of Speech Embeddings')
# plt.xlabel('Principal Component 1')
# plt.ylabel('Principal Component 2')

# # Save the plot as an image file
# plot_file = os.path.join(output_dir, "pca_embeddings_plot.png")
# plt.savefig(plot_file)
# print(f"Saved plot to {plot_file}")

## t-SNE ##

# Apply T-SNE for dimensionality reduction
tsne = TSNE(n_components=2, random_state=0)
tsne_results = tsne.fit_transform(all_embeddings)

# Create a DataFrame for plotting
df = pd.DataFrame({
    'x': tsne_results[:, 0],
    'y': tsne_results[:, 1],
    'label': labels,
    'file_path': file_paths
})

# Create the interactive plot
fig = px.scatter(df, x='x', y='y', color='label', hover_data=['file_path'], title='T-SNE of Speech Embeddings by Audio ID')

# Save the plot as an HTML file
plot_file = os.path.join(output_dir, "tsne_embeddings_audioid_plot.html")
fig.write_html(plot_file)
print(f"Saved interactive plot to {plot_file}")

# Show the plot in a browser
fig.show()