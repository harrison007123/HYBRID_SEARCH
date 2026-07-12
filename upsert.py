import os
import uuid
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

genai.configure(api_key=api_key)
embedding_model = "models/gemini-embedding-2"

# Initialize Qdrant
client = QdrantClient(path="./qdrant_db")

# Drop existing collection to apply schema changes
if client.collection_exists("semantic_documents"):
    print("🗑️ Dropping old collection to reset schema...")
    client.delete_collection("semantic_documents")

# Create collection with semantic search schema
client.create_collection(
    collection_name="semantic_documents",
    vectors_config=VectorParams(size=3072, distance=Distance.COSINE)
)

def extract_text_from_pdf(pdf_path):
    """Reads a PDF and returns its text."""
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

def chunk_text(text, chunk_size=1000, overlap=100):
    """Splits long text into smaller chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def upload_document_to_qdrant(doc_id, original_name, text_chunks):
    """
    Turns chunks into Gemini dense vectors and saves them to Qdrant.
    """
    print(f"📥 Processing: {original_name} (ID: {doc_id}) ({len(text_chunks)} chunks)")

    points = []
    for i, chunk in enumerate(text_chunks):
        # 1. Generate Dense Vector (Gemini)
        result = genai.embed_content(
            model=embedding_model,
            content=chunk
        )
        dense_vector = result['embedding']
        
        point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{doc_id}_{i}")) 
        
        point = PointStruct(
            id=point_id,
            vector=dense_vector,
            payload={
                "document_id": doc_id,
                "original_name": original_name,
                "user_id": "user_123",
                "text_content": chunk
            }
        )
        points.append(point)
        
    client.upsert(collection_name="semantic_documents", points=points)
    print(f"✅ Saved {len(text_chunks)} chunks from '{original_name}' to Qdrant.")

if __name__ == "__main__":
    docs_dir = "documents"
    
    if not os.path.exists(docs_dir):
        print(f"❌ Could not find {docs_dir} directory.")
    elif not api_key:
        print("❌ Cannot run upload without GOOGLE_API_KEY.")
    else:
        print("🚀 Starting PDF upload process (Semantic Schema)...")
        for filename in os.listdir(docs_dir):
            if filename.lower().endswith(".pdf"):
                pdf_path = os.path.join(docs_dir, filename)
                
                is_processed = False
                if len(filename) > 37 and filename[36] == '_':
                    try:
                        parsed_uuid = uuid.UUID(filename[:36])
                        if str(parsed_uuid) == filename[:36]:
                            is_processed = True
                            doc_id = filename[:36]
                            original_name = filename[37:]
                    except ValueError:
                        pass
                
                if not is_processed:
                    doc_id = str(uuid.uuid4())
                    original_name = filename
                    new_filename = f"{doc_id}_{original_name}"
                    new_pdf_path = os.path.join(docs_dir, new_filename)
                    os.rename(pdf_path, new_pdf_path)
                    pdf_path = new_pdf_path
                    print(f"\nRenamed {filename} to {new_filename}")
                else:
                    print(f"\nFound existing processed file: {filename}")

                print(f"\n📄 Extracting text from {pdf_path}...")
                full_text = extract_text_from_pdf(pdf_path)
                chunks = chunk_text(full_text)
                
                upload_document_to_qdrant(doc_id, original_name, chunks)
                
        print("\n🎉 All documents uploaded successfully!")
        client.close()
