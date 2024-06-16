import os
import torchaudio 
import torch
import speechbrain
from speechbrain import inference
from speechbrain.inference import classifiers
from speechbrain.inference.classifiers import EncoderClassifier 
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import re 

# Path to the directory containing your speech files
speech_dir = "Data/genres_original"
output_dir = "music-embeddings"

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Load the pre-trained ECAPA-TDNN model from SpeechBrain
classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="tmpdir")

all_embeddings = []
labels = []
# Process each speech file in the directory
for file_name in os.listdir(speech_dir):
    if file_name.endswith(".wav"):  # Process only .wav files
        file_path = os.path.join(speech_dir, file_name)
        
        # Load the audio file
        signal, fs = torchaudio.load(file_path)
        
        # Ensure the signal is mono
        if signal.shape[0] > 1:
            signal = signal.mean(dim=0, keepdim=True)
        
        # Obtain embeddings
        embeddings = classifier.encode_batch(signal)

        # Convert the embeddings to numpy and flatten them if necessary
        embeddings = embeddings.squeeze().detach().numpy() #converting tensor to a numpy array 
        
        # Save embeddings to a file
        embedding_file = os.path.join(output_dir, f"{os.path.splitext(file_name)[0]}_embedding.pt")
        torch.save(embeddings, embedding_file)
        print(f"Saved embeddings for {file_name} to {embedding_file}")

        # Collect all embeddings
        all_embeddings.append(embeddings)

        # Extract speaker label from the file name
        speaker_label = re.match(r'[a-zA-Z]+', file_name).group()
        labels.append(speaker_label)

# Convert list of embeddings to a 2D numpy array
all_embeddings = np.array(all_embeddings)
all_embeddings = all_embeddings.reshape(all_embeddings.shape[0], -1)  # Flatten to 2D array

# Apply PCA for dimensionality reduction
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(all_embeddings)

# Map labels to colors
unique_labels = list(set(labels))
colors = plt.cm.jet(np.linspace(0, 1, len(unique_labels)))
label_to_color = {label: colors[i] for i, label in enumerate(unique_labels)}

#plot pca 
plt.figure(figsize=(12, 8))
for i, (x, y) in enumerate(principalComponents):
    plt.scatter(x, y, color=label_to_color[labels[i]], label=labels[i], s=100)  # Larger points
    plt.text(x + 0.01, y + 0.01, labels[i], fontsize=9)  # Offset the text slightly for better visibility


plt.title('PCA of Speech Embeddings')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

# Save the plot as an image file
plot_file = os.path.join(output_dir, "pca_embeddings_plot.png")
plt.savefig(plot_file)
print(f"Saved plot to {plot_file}")