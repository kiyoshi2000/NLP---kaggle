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

# Constants for paths and model configuration
DATA_PATH = "train_submission.csv"  # Adjust if necessary
MODEL_NAME = "bert-base-multilingual-cased"
OUTPUT_DIR = "./results"
LOGGING_DIR = "./logs"
MODEL_SAVE_PATH = "./language_classifier"
MAX_LENGTH = 128
BATCH_SIZE = 8
NUM_EPOCHS = 3
RANDOM_STATE = 42
TEST_SIZE = 0.1

# Ensure output directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOGGING_DIR, exist_ok=True)
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

# Load dataset
print("Loading dataset...")
start_time = time.time()
df = pd.read_csv(DATA_PATH)
print(f"Dataset loaded successfully with {len(df)} rows. Time taken: {time.time() - start_time:.2f} seconds.")

# Encode labels (including "NaN" as a valid category)
print("Encoding labels...")
start_time = time.time()
df["Label"] = df["Label"].astype(str)  # Convert to string to handle NaN
labels = df["Label"].unique().tolist()  # List all categories, including "NaN"
label2id = {label: idx for idx, label in enumerate(labels)}
id2label = {idx: label for label, idx in label2id.items()}
df["Label_ID"] = df["Label"].map(label2id)
print(f"Label encoding completed. Time taken: {time.time() - start_time:.2f} seconds.")

# Split data into train and validation sets
print("Splitting data into training and validation sets...")
start_time = time.time()
train_texts, val_texts, train_labels, val_labels = train_test_split(
        df["Text"].tolist(), df["Label_ID"].tolist(), test_size=TEST_SIZE, random_state=RANDOM_STATE
)
print(f"Training set size: {len(train_texts)}, Validation set size: {len(val_texts)}. Time taken: {time.time() - start_time:.2f} seconds.")

# Load tokenizer
print("Loading tokenizer...")
start_time = time.time()
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
print(f"Tokenizer loaded successfully. Time taken: {time.time() - start_time:.2f} seconds.")

# Tokenize the data
print("Tokenizing data...")

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

print("Creating PyTorch datasets...")
start_time = time.time()
train_dataset = LangDataset(train_texts, train_labels, tokenizer)
val_dataset = LangDataset(val_texts, val_labels, tokenizer)
print(f"Datasets created successfully. Time taken: {time.time() - start_time:.2f} seconds.")

# Load BERT model for classification
print("Loading BERT model...")
start_time = time.time()
model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=len(labels))
print(f"Model loaded successfully. Time taken: {time.time() - start_time:.2f} seconds.")

# Define compute metrics function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="weighted", zero_division=0)
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

# Set training arguments
print("Setting training arguments...")
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir=LOGGING_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=NUM_EPOCHS,
    weight_decay=0.01,
    save_total_limit=2,
    load_best_model_at_end=True,
    logging_steps=100,
    report_to="none"  # Disable external logging services like Weights & Biases
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

# Train the model with tqdm progress bar
print("Starting model training...")
start_time = time.time()
with tqdm(total=NUM_EPOCHS, desc="Training", unit="epoch") as pbar:
    trainer.train()
    pbar.update(NUM_EPOCHS)
print(f"Training completed. Time taken: {time.time() - start_time:.2f} seconds.")

# Save the fine-tuned model
print("Saving the fine-tuned model...")
start_time = time.time()
model.save_pretrained(MODEL_SAVE_PATH)
tokenizer.save_pretrained(MODEL_SAVE_PATH)
print(f"Model saved successfully. Time taken: {time.time() - start_time:.2f} seconds.")
