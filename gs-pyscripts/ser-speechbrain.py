from speechbrain.inference.interfaces import foreign_class
import torch
from datasets import Dataset
from tqdm import tqdm
from ffmpeg import FFmpeg

# Load the emotion classifier
classifier = foreign_class(source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP", 
                           pymodule_file="custom_interface.py", 
                           classname="CustomEncoderWav2vec2Classifier")

# Load the GigaSpeech dataset
train_dir = '/work/tc062/tc062/manishav/huggingface_cache/datasets/speechcolab___gigaspeech/xs/0.0.0/0db31224ad43470c71b459deb2f2b40956b3a4edfde5fb313aaec69ec7b50d3c/gigaspeech-train.arrow'
train_dataset = Dataset.from_file(train_dir).cast_column("audio", Audio(sampling_rate=16000))

# Function to get emotion from audio array
def get_emotion(audio_array, sampling_rate):
    # Convert to torch tensor if it's not already
    if not isinstance(audio_array, torch.Tensor):
        audio_tensor = torch.tensor(audio_array)
    else:
        audio_tensor = audio_array
    
    # Ensure the tensor is on CPU and detached from any computation graph
    audio_tensor = audio_tensor.cpu().detach()
    
    # Classify the audio
    out_prob, score, index, text_lab = classifier.classify_batch(audio_tensor)
    return text_lab[0]

# Process each audio sample
results = []
for sample in tqdm(train_dataset):
    try:
        emotion = get_emotion(sample['audio']['array'], sample['audio']['sampling_rate'])
        results.append({
            'segment_id': sample['segment_id'],
            'text': sample['text'],
            'emotion': emotion
        })
    except Exception as e:
        print(f"Error processing {sample['segment_id']}: {str(e)}")

# Print some results
for result in results[:5]:
    print(f"Segment ID: {result['segment_id']}")
    print(f"Text: {result['text']}")
    print(f"Predicted Emotion: {result['emotion']}")
    print("---")