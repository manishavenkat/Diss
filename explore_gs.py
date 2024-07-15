from datasets import load_dataset
import pandas as pd
import numpy as np

# Load the dataset
dataset = '/work/tc062/tc062/manishav/huggingface_cache'

# Average and total length of audio
audio_lengths = dataset['end_time'] - dataset['begin_time']
avg_length = np.mean(audio_lengths)
total_length = np.sum(audio_lengths)

print(f"1. Average audio length: {avg_length:.2f} seconds")
print(f"   Total audio length: {total_length:.2f} seconds")

# Unique categories and their labels
unique_categories = set(dataset['category'])
print("\n2. Unique categories and their labels:")
for category in unique_categories:
    category_data = [item for item in dataset if item['category'] == category]
    if category_data:
        print(f"   {category}: {category_data[0]['title']}")

# Unique sources per category
print("\n3. Unique sources per category:")
for category in unique_categories:
    category_data = [item for item in dataset if item['category'] == category]
    unique_sources = set(item['source'] for item in category_data)
    if category_data:
        print(f"   Category {category} ({category_data[0]['title']}): {len(unique_sources)} unique sources")

# Distribution of categories
category_counts = pd.Series(dataset['category']).value_counts(normalize=True) * 100
print("\n4. Distribution of categories:")
print(category_counts)

# Distribution of titles within each category
print("\n5. Distribution of titles within each category:")
for category in unique_categories:
    category_data = [item for item in dataset if item['category'] == category]
    title_counts = pd.Series([item['title'] for item in category_data]).value_counts(normalize=True) * 100
    print(f"\n   Category {category}:")
    print(title_counts.head(10))  # Showing top 10 titles to keep output manageable