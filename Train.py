import json
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

# Load intents.json
with open("intents.json") as file:
    intents = json.load(file)

# Prepare data
texts = []
labels = []
label_to_tag = {}
tag_to_label = {}

for i, intent in enumerate(intents):
    tag = intent["tag"]
    for pattern in intent["patterns"]:
        texts.append(pattern)
        labels.append(i)
    label_to_tag[i] = tag
    tag_to_label[tag] = i

# Split data
train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Dataset class
class IntentDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encodings = tokenizer(self.texts[idx], truncation=True, padding="max_length", max_length=32, return_tensors="pt")
        item = {key: val.squeeze(0) for key, val in encodings.items()}
        item["labels"] = self.labels[idx]
        return item

# Initialize tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(label_to_tag))

# Create datasets
train_dataset = IntentDataset(train_texts, train_labels)
val_dataset = IntentDataset(val_texts, val_labels)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=10,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Train and save the model
trainer.train()
model.save_pretrained("./Model")
tokenizer.save_pretrained("./Model")

# Save mappings
with open("label_to_tag.json", "w") as f:
    json.dump(label_to_tag, f)
