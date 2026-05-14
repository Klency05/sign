import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb

# Load dataset
df = pd.read_csv("data/train.csv")

# Load embedding model
print("Loading embedding model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# Create ChromaDB client
client = chromadb.PersistentClient(path="medical_db")

# Create collection
collection = client.create_collection(name="medical_qa")

print("Creating embeddings...")

# Loop through dataset
for i, row in df.iterrows():

    # Combine question and answer
    text = f"Question: {row['Question']} Answer: {row['Answer']}"

    # Generate embedding
    embedding = model.encode(text)

    # Store in ChromaDB
    collection.add(
        documents=[text],
        embeddings=[embedding.tolist()],
        ids=[str(i)]
    )

    # Progress update
    if i % 100 == 0:
        print(f"Processed {i} rows")

print("Done!")