import pandas as pd
from transformers import AutoTokenizer
import torch

max_tokens_per_chunk = 510

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Mam specify the path of CSV file containing HADM_id, label, and text
original_file_path = "testing.csv"
df = pd.read_csv(original_file_path)

new_rows = []

def split_text_into_chunks(text):
    tokens = tokenizer.tokenize(text)
    chunked_tokens = []
    for i in range(0, len(tokens), max_tokens_per_chunk):
        chunk = tokens[i:i+max_tokens_per_chunk]
        chunked_tokens.append(chunk)
    return chunked_tokens

for index, row in df.iterrows():
    hadm_id = row['HADM_id']
    label = row['label']
    text = row['text']
    
    chunks = split_text_into_chunks(text)
    
    for i, chunk in enumerate(chunks):
        row_data = {
            'row': index + i + 1,  
            'HADM_id': hadm_id,
            'label': label,
            'split_text': ' '.join(chunk)
        }
        new_rows.append(row_data)

new_df = pd.DataFrame(new_rows)

new_file_path = "split_text.csv"
new_df.to_csv(new_file_path, index=False)

print(f"Split text saved to {new_file_path}")
