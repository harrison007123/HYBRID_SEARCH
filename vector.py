import os
import google.generativeai as genai
from qdrant_client import QdrantClient
from qdrant_client.models import Prefetch, SparseVector, FusionQuery, Fusion
from dotenv import load_dotenv
from fastembed import SparseTextEmbedding

# Load environment variables from .env file
load_dotenv()

# Configure Gemini API
api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
if not api_key:
    print("Warning: GOOGLE_API_KEY environment variable not set. Please set it in your .env file.")

genai.configure(api_key=api_key)

embedding_model = "models/gemini-embedding-2"

client = QdrantClient(path="./qdrant_db")
sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")

def search_document_hybrid(search_query, top_k=5):
    """
    Searches Qdrant using Hybrid Search (Reciprocal Rank Fusion of Dense + Sparse).
    """
    print(f"\nHybrid Searching for: '{search_query}'")
    
    # 1. Turn the user's question into a Dense Vector
    result = genai.embed_content(
        model=embedding_model,
        content=search_query
    )
    dense_query_vector = result['embedding']
    
    # 2. Turn the user's question into a Sparse Vector
    sparse_result = list(sparse_model.query_embed(search_query))[0]
    sparse_query_vector = SparseVector(
        indices=sparse_result.indices,
        values=sparse_result.values
    )
    
    # 3. Search Qdrant using RRF (Reciprocal Rank Fusion)
    search_result = client.query_points(
        collection_name="my_documents",
        prefetch=[
            Prefetch(
                query=dense_query_vector,
                using="text-dense",
                limit=top_k
            ),
            Prefetch(
                query=sparse_query_vector,
                using="text-sparse",
                limit=top_k
            )
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=top_k,
        with_payload=True
    )
    
    # 4. Extract and return the full payload and scores
    results = []
    for match in search_result.points:
        results.append({
            "document_id": match.payload.get("document_id"),
            "original_name": match.payload.get("original_name", match.payload.get("file_name", "unknown")),
            "user_id": match.payload.get("user_id"),
            "text_content": match.payload.get("text_content"),
            "score": match.score
        })
    
    return results


# ==========================================
# TESTING THE WORKFLOW
# ==========================================

if __name__ == "__main__":
    Score=0.5
    if not api_key:
        print("Error: Cannot run test without GOOGLE_API_KEY.")
    else:
        question = "One evening, as the sun dipped below the horizon"
        
        matches = search_document_hybrid(question, top_k=5)
        
        if matches:
            print(f"\nTop Match Results:")
            for i, match in enumerate(matches, 1):
                if(match['score']>0.0):
                    print(f"{i}. Document: {match['original_name']} (ID: {match['document_id']}, User: {match['user_id']})")
                    print(f"   Fusion Score: {match['score']:.4f}")
                    print(f"   Snippet: {match['text_content'][:150] if match['text_content'] else 'No content'}...\n")
        else:
            print("\nNo relevant documents found.")
            
        client.close()
