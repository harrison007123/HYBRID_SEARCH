# Local RAG with Gemini and Qdrant

This is a local Retrieval-Augmented Generation (RAG) application that lets you "chat" with your PDF documents. It uses Google's Gemini models for text embeddings and answer generation, and Qdrant as a local vector database.

## Prerequisites
- Python 3.8+
- A Google Gemini API Key

## Setup

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd VDB
   ```

2. **Create and activate a virtual environment (recommended):**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables:**
   Create a `.env` file in the root directory and add your Google API key:
   ```env
   GOOGLE_API_KEY=your_api_key_here
   ```



