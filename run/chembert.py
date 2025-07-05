from transformers import RobertaTokenizer, RobertaModel
import torch
import pandas as pd


data = pd.read_csv("interactions_DB.csv", index_col = 0)

print(data.shape)

# Check for missing or empty values
missing_pfam = data["pfam_id"].isna() | (data["pfam_id"] == "")
missing_smiles = data["SMILES"].isna() | (data["SMILES"] == "")

# Count how many rows were removed due to each column
filtered_by_pfam = missing_pfam.sum()
filtered_by_smiles = missing_smiles.sum()
filtered_by_both = (missing_pfam & missing_smiles).sum()

# Filter the DataFrame
filtered_data = data[~(missing_pfam | missing_smiles)]

# Output stats
print(f"Rows filtered due to missing 'pfam_id': {filtered_by_pfam}")
print(f"Rows filtered due to missing 'SMILES': {filtered_by_smiles}")
print(f"Rows filtered due to missing both: {filtered_by_both}")
print(f"Remaining rows: {len(filtered_data)}")

smiles_set = list(set(data["SMILES"]))
smiles_set = [smile for smile in smiles_set if type(smile)==str]

#print(smiles_set)
# Example list of SMILES
smiles_list = ['CCO', 'CCC(=O)O', 'c1ccccc1']
# Load ChemBERTa model and tokenizer
tokenizer = RobertaTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
model = RobertaModel.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")

# Make sure model is in evaluation mode
model.eval()

# Tokenize SMILES
inputs = tokenizer(smiles_set, return_tensors='pt', padding=True, truncation=True)


# Get embeddings
with torch.no_grad():
    outputs = model(**inputs)
    print(outputs.last_hidden_state)
    # Use CLS token embedding (first token) as representation
    embeddings = outputs.last_hidden_state[:, 0, :]

# Convert to numpy or DataFrame if needed
embedding_df = pd.DataFrame(embeddings.numpy())
#print(embedding_df.shape)