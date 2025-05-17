import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer
from app.model.classifier import EmotionClassifier
import json
import numpy as np

app = FastAPI()

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = EmotionClassifier(n_classes=28)
model.load_state_dict(torch.load('app/model/emotion_model.pt', map_location=device))
model.to(device)
model.eval()

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

# Load label mapping
with open('label_mapping.json', 'r') as f:
    label_mapping = json.load(f)
    label_mapping = {int(k): v for k, v in label_mapping.items()}  # keys to int

class TextInput(BaseModel):
    text: str

@app.post("/predict")
async def predict_emotions(input_data: TextInput):
    encoding = tokenizer(
        input_data.text,
        return_tensors='pt',
        truncation=True,
        padding='max_length',
        max_length=128
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        probs = outputs.squeeze().cpu().numpy()

    # Filter emotions with probability > 0.3 and convert to percentage
    top_indices = np.where(probs > 0.3)[0]
    top_emotions = {
        label_mapping[i]: f"{round(probs[i] * 100)}%" for i in top_indices
    }

    return {
        "text": input_data.text,
        "emotions": top_emotions
    }
