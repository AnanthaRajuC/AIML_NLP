import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset
import speech_recognition as sr
import numpy as np
import sys
import json

# Access command-line arguments
arguments = sys.argv[1:]

# Print the arguments passed from Java
for arg in arguments:
    print("Argument:", arg)

# device = "cuda:0" if torch.cuda.is_available() else "cpu"
device = "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

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


resultTranscription = pipe("/home/anantha/PCF/AIML/SpeechToText/HIN_M_AbhishekS.mp3", generate_kwargs={"language": None})
print("")
print(resultTranscription["text"])


resultTranslation = pipe("/home/anantha/PCF/AIML/SpeechToText/HIN_M_AbhishekS.mp3", generate_kwargs={"task": "translate"})
print("")
print(resultTranslation["text"])


classifier = pipeline("sentiment-analysis",model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",revision ="af0f99b")
resultClassifier = classifier(resultTranslation["text"])
print("")
print(resultClassifier)

# Create a Python dictionary
data = {
    "resultTranscription": resultTranscription,
    "resultTranslation": resultTranslation,
    "resultClassifier": resultClassifier
}

# Convert the dictionary to a JSON string
json_string = json.dumps(data)

# Print the JSON string
print(json_string)