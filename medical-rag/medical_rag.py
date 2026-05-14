from sentence_transformers import SentenceTransformer
import chromadb
import ollama

# Load embedding model
print("Loading embedding model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load vector database
client_db = chromadb.PersistentClient(path="medical_db")

collection = client_db.get_collection(name="medical_qa")

print("Medical RAG Assistant Ready!")

while True:

    # User query
    query = input("\nEnter symptoms/question: ")

    # Exit condition
    if query.lower() == "exit":
        break

    # Convert query into embedding
    query_embedding = model.encode(query)

    # Retrieve similar documents
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=3
    )

    # Combine retrieved context
    retrieved_docs = "\n".join(results['documents'][0])

    # Prompt for Mistral
    prompt = f"""
You are a helpful medical assistant.

Use the medical context below to answer the user safely.

Medical Context:
{retrieved_docs}

User Symptoms:
{query}

Rules:
- Do not provide dangerous diagnoses
- Suggest doctor consultation when needed
- Keep response concise and helpful
"""

    # Generate response from Mistral
    response = ollama.chat(
        model='mistral',
        messages=[
            {
                'role': 'user',
                'content': prompt
            }
        ]
    )

    # Print AI response
    print("\nAI Medical Response:\n")

    print(response['message']['content'])