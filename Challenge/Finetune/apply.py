import torch
import pandas as pd
import json
import logging
from tqdm import tqdm
from transformers import BertTokenizer, BertForSequenceClassification

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Force "Eager" mode in case torch.compile() does not work
import torch._dynamo
torch._dynamo.config.suppress_errors = True

# Define the device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

# File paths
MODEL_SAVE_PATH = "./language_classifier"
TRAIN_DATA_PATH = "train_submission.csv"
TEST_DATA_PATH = "test_without_labels.csv"
OUTPUT_PATH = "./test_predictions.csv"
LABEL_MAPPING_PATH = "id2label.json"

# Load tokenizer and trained model
logging.info("Loading tokenizer and model...")
tokenizer = BertTokenizer.from_pretrained(MODEL_SAVE_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_SAVE_PATH).to(device)

# Check if the GPU supports torch.compile() (Triton requires Compute Capability >= 7.0)
use_torch_compile = False
if torch.__version__ >= "2.0" and torch.cuda.is_available():
    major_cc, _ = torch.cuda.get_device_capability()
    if major_cc >= 7:
        use_torch_compile = True

# Apply torch.compile() only if the GPU supports it
if use_torch_compile:
    logging.info("Optimizing model with torch.compile()...")
    model = torch.compile(model)
else:
    logging.info("torch.compile() disabled - GPU not supported. Running in normal mode.")

# Set model to evaluation mode
model.eval()
logging.info("Model loaded and ready for inference.")

# Load training data to reconstruct the label mapping
logging.info("Loading training data to create the label mapping...")
df_train = pd.read_csv(TRAIN_DATA_PATH)

if "Label" not in df_train.columns:
    raise ValueError("The 'Label' column was not found in the training file.")

# Create label dictionary
labels = df_train["Label"].astype(str).unique().tolist()
label2id = {label: idx for idx, label in enumerate(labels)}
id2label = {idx: label for label, idx in label2id.items()}

# Save mapping for future use
with open(LABEL_MAPPING_PATH, "w") as f:
    json.dump(id2label, f)

logging.info(f"Label mapping saved in {LABEL_MAPPING_PATH}")

# Load test data
logging.info("Loading test data...")
df_test = pd.read_csv(TEST_DATA_PATH)

if "Text" not in df_test.columns:
    raise ValueError("The test file must contain a 'Text' column.")

# Extract texts for prediction
test_texts = df_test["Text"].astype(str).tolist()

# Batch size configuration (adjust as needed)
BATCH_SIZE = 64 if torch.cuda.is_available() else 32  # Use a larger batch size on GPU
predicted_labels = []

logging.info(f"Starting tokenization and prediction in batches of {BATCH_SIZE}...")

# Process in batches to avoid memory overflow
for i in tqdm(range(0, len(test_texts), BATCH_SIZE), desc="Processing Batches"):
    batch_texts = test_texts[i : i + BATCH_SIZE]

    # Tokenize batch
    encodings = tokenizer(
        batch_texts,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt"
    )

    # Move tensors to GPU (if available)
    input_ids = encodings["input_ids"].to(device)
    attention_mask = encodings["attention_mask"].to(device)

    # Make prediction
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        batch_predictions = torch.argmax(outputs.logits, dim=1).cpu().numpy()  # Send to CPU

    # Convert prediction IDs to labels
    predicted_labels.extend([id2label[pred] for pred in batch_predictions])

# Create DataFrame with results
df_results = pd.DataFrame({
    "ID": range(1, len(predicted_labels) + 1),
    "Label": predicted_labels
})

# Save results to CSV
df_results.to_csv(OUTPUT_PATH, index=False)
logging.info(f"Predictions saved in {OUTPUT_PATH}")
