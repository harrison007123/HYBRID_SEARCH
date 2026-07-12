import os
import uuid
import google.generativeai as genai
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from pypdf import PdfReader
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Gemini API
api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
if not api_key:
    print("⚠️ GOOGLE_API_KEY environment variable not set. Please set it in your .env file.")

genai.configure(api_key=api_key)
embedding_model = "models/gemini-embedding-2"

from contextlib import asynccontextmanager

client = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global client
    # Initialize Qdrant
    client = QdrantClient(path="./qdrant_db")

    # Create collection only if it doesn't exist
    if not client.collection_exists("semantic_documents"):
        print("Creating 'semantic_documents' collection in Qdrant...")
        client.create_collection(
            collection_name="semantic_documents",
            vectors_config=VectorParams(size=3072, distance=Distance.COSINE)
        )
    yield
    client.close()

# Setup upload directory
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Helper functions
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

def chunk_text(text, chunk_size=1000, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

# Initialize FastAPI app
app = FastAPI(title="Local RAG API (Semantic)", lifespan=lifespan)

class SearchQuery(BaseModel):
    question: str
    top_k: int = 5

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    
    # 1. Generate unique ID and save file
    doc_id = str(uuid.uuid4())
    original_name = file.filename
    internal_filename = f"{doc_id}_{original_name}"
    file_path = os.path.join(UPLOAD_DIR, internal_filename)
    
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())
        
    # 2. Extract text and chunk
    try:
        full_text = extract_text_from_pdf(file_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read PDF: {str(e)}")
        
    text_chunks = chunk_text(full_text)
    if not text_chunks:
        raise HTTPException(status_code=400, detail="No readable text found in PDF.")
        
    # 3. Embed and upload to Qdrant
    points = []
    
    for i, chunk in enumerate(text_chunks):
        # Dense vector
        result = genai.embed_content(model=embedding_model, content=chunk)
        dense_vector = result['embedding']
        
        point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{doc_id}_{i}")) 
        
        points.append(PointStruct(
            id=point_id,
            vector=dense_vector,
            payload={
                "document_id": doc_id,
                "original_name": original_name,
                "internal_filename": internal_filename,
                "text_content": chunk
            }
        ))
        
    client.upsert(collection_name="semantic_documents", points=points)
    
    return {
        "message": "File successfully uploaded and processed.",
        "document_id": doc_id,
        "original_name": original_name,
        "chunks_processed": len(text_chunks)
    }

@app.post("/search")
async def search_documents(query: SearchQuery):
    search_query = query.question
    top_k = query.top_k
    
    # 1. Dense Query
    result = genai.embed_content(model=embedding_model, content=search_query)
    dense_query_vector = result['embedding']
    
    # 2. Qdrant Semantic Search
    search_result = client.query_points(
        collection_name="semantic_documents",
        query=dense_query_vector,
        limit=top_k,
        with_payload=True
    )
    
    # 3. Format Results
    results = []
    for rank, match in enumerate(search_result.points, 1):
        if match.score > 0.0:
            doc_id = match.payload.get("document_id")
            original_name = match.payload.get("original_name", "unknown.pdf")
            internal_filename = match.payload.get("internal_filename", f"{doc_id}_{original_name}")
            text_content = match.payload.get("text_content", "")
            
            results.append({
                "rank": rank,
                "original_name": original_name,
                "document_id": doc_id,
                "file_url": f"/files/{internal_filename}",
                "snippet": text_content,
                "score": round(match.score, 4)
            })
            
    return results

@app.get("/")
async def read_index():
    """Serves the frontend UI."""
    return FileResponse("index.html")

@app.get("/files/{internal_filename}")
async def get_file(internal_filename: str):
    """Serves the raw PDF file for viewing/downloading."""
    file_path = os.path.join(UPLOAD_DIR, internal_filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(file_path, media_type="application/pdf", headers={"Content-Disposition": "inline"})

if __name__ == "__main__":
    import uvicorn
    print("Starting server... Access Swagger UI at http://localhost:8000/docs")
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
