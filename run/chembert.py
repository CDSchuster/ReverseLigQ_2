from transformers import RobertaTokenizer, RobertaModel
import torch
import pandas as pd


data = pd.read_csv("interactions_DB.csv")

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