import torch
from transformers import BertTokenizer, BertForSequenceClassification
import json

# Load model and tokenizer
model = BertForSequenceClassification.from_pretrained("./model")
tokenizer = BertTokenizer.from_pretrained("./model")

# Load mappings
with open("label_to_tag.json") as f:
    label_to_tag = json.load(f)

def predict_intent(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=32)
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_label = torch.argmax(probs, dim=-1).item()
    return label_to_tag[str(predicted_label)]
