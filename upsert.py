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

genai.configure(api_key=api_key)
embedding_model = "models/gemini-embedding-2"

# Initialize Qdrant
client = QdrantClient(path="./qdrant_db")

# Create collection if it doesn't exist
if not client.collection_exists("my_documents"):
    client.create_collection(
        collection_name="my_documents",
        vectors_config=VectorParams(size=3072, distance=Distance.COSINE),
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

def upload_document_to_qdrant(file_name, text_chunks):
    """
    Turns chunks into Google GenAI vectors and saves them to Qdrant.
    """
    print(f"📥 Processing: {file_name} ({len(text_chunks)} chunks)")
    
    points = []
    for i, chunk in enumerate(text_chunks):
        result = genai.embed_content(
            model=embedding_model,
            content=chunk
        )
        vector = result['embedding']
        
        point_id = hash(f"{file_name}_{i}") % (10**8) 
        
        point = PointStruct(
            id=point_id,
            vector=vector,
            payload={
                "file_name": file_name,
                "text_content": chunk
            }
        )
        points.append(point)
        
    client.upsert(collection_name="my_documents", points=points)
    print(f"✅ Saved {len(text_chunks)} chunks from '{file_name}' to Qdrant.")

if __name__ == "__main__":
    docs_dir = "documents"
    
    if not os.path.exists(docs_dir):
        print(f"❌ Could not find {docs_dir} directory.")
    elif not api_key:
        print("❌ Cannot run upload without GOOGLE_API_KEY.")
    else:
        print("🚀 Starting PDF upload process...")
        for filename in os.listdir(docs_dir):
            if filename.lower().endswith(".pdf"):
                pdf_path = os.path.join(docs_dir, filename)
                print(f"\n📄 Extracting text from {pdf_path}...")
                full_text = extract_text_from_pdf(pdf_path)
                chunks = chunk_text(full_text)
                
                upload_document_to_qdrant(filename, chunks)
                
        print("\n🎉 All documents uploaded successfully!")
        client.close()
