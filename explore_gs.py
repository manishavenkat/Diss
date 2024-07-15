from datasets import load_dataset, Dataset, Audio
import pandas as pd
import numpy as np

train_dir = '/work/tc062/tc062/manishav/huggingface_cache/datasets/speechcolab___gigaspeech/s/0.0.0/0db31224ad43470c71b459deb2f2b40956b3a4edfde5fb313aaec69ec7b50d3c/gigaspeech-train-00000-of-00002.arrow'

# Load the full datasets
dataset = Dataset.from_file(train_dir)
# train_dataset = train_dataset.cast_column("audio", Audio(sampling_rate=16000))

category_names = dataset.features['category'].names
# Create a dictionary mapping indices to category names
category_dict = {i: name for i, name in enumerate(category_names)}
all_categories = list(category_dict.values())
print(all_categories)

# Average and total length of audio
audio_lengths = [item['end_time'] - item['begin_time'] for item in dataset]
avg_length = np.mean(audio_lengths)
total_length = np.sum(audio_lengths)

print(f"Average audio length: {avg_length:.2f} seconds")
print(f"   Total audio length: {total_length:.2f} seconds")

# Unique categories and their labels
unique_categories = set(dataset.features['category'])
print("\n Unique categories and their labels:")
for category in unique_categories:
    category_data = [item for item in dataset if item['category'] == category]
    if category_data:
        print(f"   {category}: {category_data[0]['title']}")

# Unique sources per category
print("\n Unique sources per category:")
for category in unique_categories:
    category_data = [item for item in dataset if item['category'] == category]
    unique_sources = set(item['source'] for item in category_data)
    if category_data:
        print(f"   Category {category} ({category_data[0]['title']}): {len(unique_sources)} unique sources")

# Distribution of categories
category_counts = pd.Series(dataset.features['category']).value_counts(normalize=True) * 100
print("\n4. Distribution of categories:")
print(category_counts)

# Distribution of top 10 titles within each category
print("\n5. Distribution of titles within each category:")
for category in unique_categories:
    category_data = [item for item in dataset if item['category'] == category]
    title_counts = pd.Series([item['title'] for item in category_data]).value_counts(normalize=True) * 100
    print(f"\n   Category {category}:")
    print(title_counts.head(10))  