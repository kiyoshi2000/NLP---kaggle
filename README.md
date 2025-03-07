# Fine-Tuning BERT for Language Classification

## Overview
This project fine-tunes the multilingual BERT (`bert-base-multilingual-cased`) model for language classification. The dataset contains text samples labeled with their respective languages. The goal is to create a robust model capable of accurately predicting the language of a given text.

## Features
- Fine-tunes BERT for sequence classification
- Handles imbalanced datasets and missing labels (`NaN` treated as a category)
- Supports training continuation from checkpoints
- Evaluates model performance using accuracy, precision, recall, and F1-score
- Applies the trained model to make predictions on new text samples

## Repository Structure
```
├── fine_tunning.py    # Script for initial model training
├── continue.py        # Script for continuing training from a checkpoint
├── apply.py           # Script for making predictions with the trained model
├── train_submission.csv  # Training dataset (not included in repo)
├── test_without_labels.csv  # Test dataset (not included in repo)
├── README.md          # Project documentation
```

## Setup
### 1. Install Dependencies
```bash
pip install transformers torch pandas scikit-learn tqdm safetensors
```

### 2. Prepare Data
Ensure `train_submission.csv` (training data) is present in the project directory. The CSV should have the following format:
```
ID,Text,Label
1,"Hello, world!",English
2,"Bonjour le monde!",French
...
```

## Training the Model
### Initial Fine-Tuning
Run the following command to fine-tune BERT from scratch:
```bash
python fine_tunning.py
```
This script:
- Loads the dataset and tokenizes the text
- Splits the data into training and validation sets
- Fine-tunes BERT for sequence classification
- Saves the trained model to `./language_classifier`

### Continue Training from a Checkpoint
If training was interrupted or needs additional epochs, run:
```bash
python continue.py
```
This script:
- Loads the last saved checkpoint
- Resumes training for additional epochs

## Making Predictions
To classify text from `test_without_labels.csv`, run:
```bash
python apply.py
```
This script:
- Loads the trained model
- Tokenizes test data
- Predicts labels for each text entry
- Saves predictions in `test_predictions.csv`

## Model Evaluation
During training, performance metrics (accuracy, precision, recall, F1-score) are logged. The model automatically saves the best checkpoint based on validation accuracy.

## Saving & Loading Model
The fine-tuned model and tokenizer are saved in `./language_classifier`. To use it later:
```python
from transformers import BertTokenizer, BertForSequenceClassification
import torch

tokenizer = BertTokenizer.from_pretrained("./language_classifier")
model = BertForSequenceClassification.from_pretrained("./language_classifier")
model.eval()
```

## Acknowledgments
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [PyTorch](https://pytorch.org/)

## License
This project is open-source and available under the MIT License.