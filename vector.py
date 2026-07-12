import os
import google.generativeai as genai
from qdrant_client import QdrantClient
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure Gemini API
api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
if not api_key:
    print("Warning: GOOGLE_API_KEY environment variable not set. Please set it in your .env file.")

genai.configure(api_key=api_key)

embedding_model = "models/gemini-embedding-2"

client = QdrantClient(path="./qdrant_db")

def search_document_semantic(search_query, top_k=5):
    """
    Searches Qdrant using Semantic Search (Dense).
    """
    print(f"\nSemantic Searching for: '{search_query}'")
    
    # 1. Turn the user's question into a Dense Vector
    result = genai.embed_content(
        model=embedding_model,
        content=search_query
    )
    dense_query_vector = result['embedding']
    
    # 2. Search Qdrant
    search_result = client.query_points(
        collection_name="semantic_documents",
        query=dense_query_vector,
        limit=top_k,
        with_payload=True
    )
    
    # 3. Extract and return the full payload and scores
    results = []
    for match in search_result.points:
        text_content = match.payload.get("text_content", "")
        results.append({
            "document_id": match.payload.get("document_id"),
            "original_name": match.payload.get("original_name", match.payload.get("file_name", "unknown")),
            "user_id": match.payload.get("user_id"),
            "text_content": text_content,
            "score": match.score
        })
        
    return results

# ==========================================
# TESTING THE WORKFLOW
# ==========================================

if __name__ == "__main__":
    if not api_key:
        print("Error: Cannot run test without GOOGLE_API_KEY.")
    else:
        question = "One evening, as the sun dipped below the horizon"
        
        matches = search_document_semantic(question, top_k=5)
        
        if matches:
            print(f"\nTop Source Matches:")
            for i, match in enumerate(matches, 1):
                if match['score'] > 0.0:
                    print(f"{i}. Document: {match['original_name']} (ID: {match['document_id']}, User: {match['user_id']})")
                    print(f"   Cosine Score: {match['score']:.4f}")
                    print(f"   Snippet: {match['text_content'][:150] if match['text_content'] else 'No content'}...\n")
        else:
            print("\nNo relevant documents found.")
            
        client.close()
