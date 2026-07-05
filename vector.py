import os
import google.generativeai as genai
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from pypdf import PdfReader
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure Gemini API
api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
if not api_key:
    print("⚠️ GOOGLE_API_KEY environment variable not set. Please set it in your .env file.")
    # For testing, you could hardcode it, but environment variables are safer.

genai.configure(api_key=api_key)

# We use gemini-embedding-2 for vectors and gemini-3.1-flash-lite for answering
embedding_model = "models/gemini-embedding-2"
generative_model = genai.GenerativeModel("models/gemini-3.1-flash-lite")

# Initialize Qdrant and the AI Embedding Model
# This creates a persistent local database in the 'qdrant_db' folder
client = QdrantClient(path="./qdrant_db")

# Google's gemini-embedding-2 creates vectors with 3072 dimensions
if not client.collection_exists("my_documents"):
    client.create_collection(
        collection_name="my_documents",
        vectors_config=VectorParams(size=3072, distance=Distance.COSINE),
    )


def search_document_by_text(search_query, top_k=3):
    """
    Searches Qdrant and returns the matching text chunks.
    """
    print(f"\n🔍 Searching for: '{search_query}'")
    
    # 1. Turn the user's question into a vector using Google GenAI
    result = genai.embed_content(
        model=embedding_model,
        content=search_query
    )
    query_vector = result['embedding']
    
    # 2. Search Qdrant for the closest matches
    search_result = client.query_points(
        collection_name="my_documents",
        query=query_vector,
        limit=top_k
    )
    
    # 3. Extract and return the contexts
    contexts = []
    for match in search_result.points:
        contexts.append({
            "file_name": match.payload["file_name"],
            "text_content": match.payload["text_content"]
        })
    
    return contexts

def answer_question_with_flash(question, contexts):
    """
    Uses Gemini 3.1 Flash Lite to answer the question based on the contexts.
    """
    print(f"\n🤖 Asking Gemini 3.1 Flash Lite to generate an answer...")
    
    context_text = ""
    for ctx in contexts:
        context_text += f"Document: {ctx['file_name']}\nText: {ctx['text_content']}\n\n---\n\n"
    
    prompt = f'''
You are a helpful assistant. Use the following context to answer the user's question.
If the answer is not in the context, just say "I don't know based on the provided documents."

Context:
{context_text}

Question:
{question}
'''
    response = generative_model.generate_content(prompt)
    return response.text

# ==========================================
# 🚀 TESTING THE WORKFLOW
# ==========================================

if __name__ == "__main__":
    if not api_key:
        print("❌ Cannot run test without GOOGLE_API_KEY.")
    else:
        # Ask a question about the documents
        question = "Mrs. Green gladly accepted" 
        
        contexts = search_document_by_text(question)
        if contexts:
            answer = answer_question_with_flash(question, contexts)
            print(f"\n✨ Answer:\n{answer}")
            
            # Print the path of the files that were retrieved to answer this
            print("\n📚 Sources used:")
            unique_files = set(ctx['file_name'] for ctx in contexts)
            for file_name in unique_files:
                print(f"PDF path: /documents/{file_name}")
        else:
            print("\n❌ No relevant context found.")
            
        # Close the client to prevent exceptions during Python shutdown
        client.close()
