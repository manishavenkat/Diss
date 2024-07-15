from datasets import load_dataset, Dataset, Audio
import pandas as pd
import numpy as np
from tqdm import tqdm

train_dir = '/work/tc062/tc062/manishav/huggingface_cache/datasets/speechcolab___gigaspeech/s/0.0.0/0db31224ad43470c71b459deb2f2b40956b3a4edfde5fb313aaec69ec7b50d3c/gigaspeech-train-00000-of-00002.arrow'

# Load the full datasets
dataset = Dataset.from_file(train_dir)
# train_dataset = train_dataset.cast_column("audio", Audio(sampling_rate=16000))

category_names = dataset.features['category'].names
# Create a dictionary mapping indices to category names
category_dict = {i: name for i, name in enumerate(category_names)}
all_categories = list(category_dict.values())
print(all_categories)

# # Average and total length of audio
# audio_lengths = [item['end_time'] - item['begin_time'] for item in tqdm(dataset, desc="Calculating audio lengths")]
# avg_length = np.mean(audio_lengths)
# total_length = np.sum(audio_lengths)

# print(f"Average audio length: {avg_length:.2f} seconds")
# print(f"   Total audio length: {total_length:.2f} seconds")

# # Unique categories and their labels
# print("\n Unique categories and their labels:")
# for category_id, category_name in tqdm(enumerate(category_names), desc="Processing categories", total=len(category_names)):
#     category_data = dataset.filter(lambda example: example['category'] == category_id)
#     if len(category_data) > 0:
#         print(f"   {category_name}: {category_data[0]['title']}")

# # Unique sources per category
# print("\n3. Unique sources per category:")
# for category_id, category_name in tqdm(enumerate(category_names), desc="Counting unique sources", total=len(category_names)):
#     category_data = dataset.filter(lambda example: example['category'] == category_id)
#     unique_sources = set(item['source'] for item in category_data)
#     if len(category_data) > 0:
#         print(f"Category {category_name}: {len(unique_sources)} unique sources")
#         print("Unique sources:")
#         for source in unique_sources:
#             print(f"  - {source}")
#         print()  # Add an empty line for better readability between categories

# Distribution of categories
print("\n4. Calculating distribution of categories...")
category_counts = pd.Series([category_dict[item['category']] for item in tqdm(dataset, desc="Counting categories")]).value_counts(normalize=True) * 100
print(category_counts)

# Distribution of top 10 titles within each category
print("\n5. Distribution of titles within each category:")
for category_id, category_name in tqdm(enumerate(category_names), desc="Processing title distributions", total=len(category_names)):
    category_data = dataset.filter(lambda example: example['category'] == category_id)
    title_counts = pd.Series([item['title'] for item in category_data]).value_counts(normalize=True) * 100
    print(f"\nCategory {category_name}:")
    
    if len(title_counts) > 0:
        for title, percentage in title_counts.items():
            print(f"  {title}: {percentage:.2f}%")
    else:
        print("  No titles found in this category.")
    
    print() 