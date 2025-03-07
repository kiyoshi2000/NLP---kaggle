import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import os
from tqdm import tqdm
import time
from safetensors.torch import load_file

# Constants
DATA_PATH = "train_submission.csv"
MODEL_NAME = "bert-base-multilingual-cased"
OUTPUT_DIR = "./results"
CHECKPOINT_PATH = "./results/checkpoint-64329"
LOGGING_DIR = "./logs"
MODEL_SAVE_PATH = "./language_classifier"
MAX_LENGTH = 128
BATCH_SIZE = 8
PREVIOUS_EPOCHS = 3  # Previous training epochs
ADDITIONAL_EPOCHS = 2  # We want +2 epochs
TOTAL_EPOCHS = PREVIOUS_EPOCHS + ADDITIONAL_EPOCHS  # New total epochs
RANDOM_STATE = 42
TEST_SIZE = 0.1

# Ensure output directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOGGING_DIR, exist_ok=True)
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

# Load dataset
print("Loading dataset...")
df = pd.read_csv(DATA_PATH)
print(f"Dataset loaded successfully with {len(df)} rows.")

# Encode labels
print("Encoding labels...")
df["Label"] = df["Label"].astype(str)
labels = df["Label"].unique().tolist()
label2id = {label: idx for idx, label in enumerate(labels)}
id2label = {idx: label for label, idx in label2id.items()}
df["Label_ID"] = df["Label"].map(label2id)

# Split data into training and validation sets
print("Splitting data into training and validation sets...")
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["Text"].tolist(), df["Label_ID"].tolist(), test_size=TEST_SIZE, random_state=RANDOM_STATE
)

# Load tokenizer
print("Loading tokenizer...")
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

# Tokenize the data
class LangDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=MAX_LENGTH):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }

# Create datasets
print("Creating PyTorch datasets...")
train_dataset = LangDataset(train_texts, train_labels, tokenizer)
val_dataset = LangDataset(val_texts, val_labels, tokenizer)

# Load Model from Checkpoint
print(f"Loading model from checkpoint: {CHECKPOINT_PATH}...")
safetensor_model_path = os.path.join(CHECKPOINT_PATH, "model.safetensors")
pytorch_model_path = os.path.join(CHECKPOINT_PATH, "pytorch_model.bin")

# Convert `safetensors` to `pytorch_model.bin` if necessary
if not os.path.exists(pytorch_model_path) and os.path.exists(safetensor_model_path):
    print("Converting `model.safetensors` to `pytorch_model.bin`...")
    state_dict = load_file(safetensor_model_path)
    torch.save(state_dict, pytorch_model_path)
    print("Conversion completed.")

# Load Model
model = BertForSequenceClassification.from_pretrained(CHECKPOINT_PATH, num_labels=len(labels))
print("Model loaded successfully.")

# Define compute metrics function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="weighted", zero_division=0)
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

# Set Training Arguments
print("Setting training arguments...")
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir=LOGGING_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=TOTAL_EPOCHS,  
    weight_decay=0.01,
    save_total_limit=2,
    load_best_model_at_end=True,
    logging_steps=100,
    report_to="none"  # Disable external logging services
)
print("Training arguments set.")

# Initialize Trainer
print("Initializing Trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)
print("Trainer initialized successfully.")

# Ensure checkpoint exists before resuming training
if not os.path.isfile(os.path.join(CHECKPOINT_PATH, "trainer_state.json")):
    print(f"Incomplete checkpoint or `trainer_state.json` missing in {CHECKPOINT_PATH}. Training cannot be resumed.")
else:
    print(f"Resuming training for 2 more epochs (total {TOTAL_EPOCHS})...")
    with tqdm(total=ADDITIONAL_EPOCHS, desc="Training", unit="epoch") as pbar:
        trainer.train(resume_from_checkpoint=CHECKPOINT_PATH)  
        pbar.update(ADDITIONAL_EPOCHS)
    print("Training resumed and completed.")

# Save the fine-tuned model
print("Saving the fine-tuned model...")
model.save_pretrained(MODEL_SAVE_PATH)
tokenizer.save_pretrained(MODEL_SAVE_PATH)
print("Model saved successfully.")
