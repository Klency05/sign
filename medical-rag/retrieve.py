from sentence_transformers import SentenceTransformer
import chromadb

# Load embedding model
print("Loading model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load persistent database
client = chromadb.PersistentClient(path="medical_db")

# Load collection
collection = client.get_collection(name="medical_qa")

print("Medical RAG Ready!")

while True:

    # User input
    query = input("\nEnter symptoms/question: ")

    # Exit condition
    if query.lower() == "exit":
        break

    # Convert query to embedding
    query_embedding = model.encode(query)

    # Search similar documents
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=3
    )

    print("\nTop Medical Results:\n")

    # Print retrieved documents
    for i, doc in enumerate(results['documents'][0]):

        print(f"Result {i+1}:")
        print(doc)
        print("-" * 50)