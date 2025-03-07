import re
from bs4 import BeautifulSoup
import pandas as pd
import time

DATA_PATH = "train_submission.csv"  # Adjust the path to your dataset if needed
# Load dataset
print("Loading dataset...")
start_time = time.time()
df = pd.read_csv(DATA_PATH)
print(f"Dataset loaded successfully with {len(df)} rows. Time taken: {time.time() - start_time:.2f} seconds.")

# 1. Define a function to clean the text
def clean_text(text):
    # Remove non-printable characters (like \x00)
    text = ''.join(ch for ch in text if ch.isprintable())
    
    # Remove HTML tags using BeautifulSoup
    text = BeautifulSoup(text, "html.parser").get_text()

    # Remove any JavaScript or CSS (e.g., <script>...</script>, <style>...</style>)
    text = re.sub(r'<script.*?>.*?</script>', '', text, flags=re.DOTALL)
    text = re.sub(r'<style.*?>.*?</style>', '', text, flags=re.DOTALL)
    
    # Normalize white spaces (convert multiple spaces to one and remove leading/trailing spaces)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# Apply the cleaning function to the dataset before splitting
print("Cleaning dataset text...")
start_time = time.time()
df["Text"] = df["Text"].apply(clean_text)
print(f"Text cleaning completed. Time taken: {time.time() - start_time:.2f} seconds.")

# Save the cleaned dataset as a new CSV file
cleaned_data_path = "train_submission_clean.csv"  # Save the cleaned data
df.to_csv(cleaned_data_path, index=False)

print(f"Cleaned dataset saved as {cleaned_data_path}")
