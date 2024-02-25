import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

import numpy as np
import sys
import json

# Access command-line arguments
arguments = sys.argv[1:]

# Create an empty dictionary to store key-value pairs
args_dict = {}

# Parse arguments with keys
for arg in arguments:
    key, value = arg.split('=')  # Split the argument at '=' to separate key and value
    args_dict[key] = value

# Define the key of interest
audio_file_path = 'audio_file_path'

# Check if the desired key exists in the dictionary
if audio_file_path in args_dict:
    audio_file_path_value = args_dict[audio_file_path]
    print(f"Value for key '{audio_file_path}': {audio_file_path_value}")
else:
    print(f"Key '{audio_file_path}' not found in the arguments.")

# Check device availability
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Determine the torch data type based on device availability
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Define the model ID - https://huggingface.co/openai/whisper-large-v3
model_id = "openai/whisper-large-v3"

# Load the speech-to-text model
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, 
    torch_dtype=torch_dtype, 
    low_cpu_mem_usage=True, 
    use_safetensors=True
)
model.to(device)

# Load the processor
processor = AutoProcessor.from_pretrained(model_id)

# Create a pipeline for automatic speech recognition
pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=30,
    batch_size=16,
    return_timestamps=True,
    torch_dtype=torch_dtype,
    device=device,
)

# Perform automatic speech recognition
resultTranscription = pipe(audio_file_path_value, generate_kwargs={"language": None})
print("\n",resultTranscription["text"])

# Perform translation
resultTranslation = pipe(audio_file_path_value, generate_kwargs={"task": "translate"})
print("\n",resultTranslation["text"])

# Load the sentiment analysis classifier
classifier = pipeline("sentiment-analysis",
                      model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
                      revision ="af0f99b")

# Perform sentiment analysis on the translated text
resultClassifier = classifier(resultTranslation["text"])
print("\n",resultClassifier)

# Create a Python dictionary
data = {
    "resultTranscription": resultTranscription,
    "resultTranslation": resultTranslation,
    "resultClassifier": resultClassifier
}

# Convert the dictionary to a JSON string
json_string = json.dumps(data)

# Print the JSON string
print("\n",json_string)