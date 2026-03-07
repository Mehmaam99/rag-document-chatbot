"""
========================================================
PROJECT 1: Company FAQ RAG Chatbot
========================================================
Author      : Syed Muhammad Mehmam
Tech Stack  : FastAPI + LangChain + ChromaDB + Groq + Sentence Transformers
Description : A production-ready RAG (Retrieval-Augmented Generation) chatbot
              that ingests PDF/TXT documents and answers questions using
              semantic search + Groq LLM. Built for portfolio demonstration.
========================================================
"""

import os
import shutil
import stat
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv


load_dotenv()  # Add this at the top

# ── Configuration ────────────────────────────────────────────────────────────

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY", "")   # Optional: LangSmith tracing

# Use absolute paths anchored to this file so the server works from any CWD
BASE_DIR = Path(__file__).resolve().parent.parent
CHROMA_DB_DIR = str(BASE_DIR / "chroma_db")
UPLOAD_DIR = str(BASE_DIR / "uploads")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Free, local embedding model

# Create required directories with explicit write permissions
for _d in [UPLOAD_DIR, CHROMA_DB_DIR]:
    Path(_d).mkdir(parents=True, exist_ok=True)
    os.chmod(_d, stat.S_IRWXU | stat.S_IRWXG | stat.S_IROTH | stat.S_IXOTH)

# ── FastAPI App ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="RAG Chatbot API",
    description="Upload documents and chat with them using Groq LLM + ChromaDB",
    version="1.0.0"
)

app.mount("/static", StaticFiles(directory="static"), name="static")

# ── Global State ─────────────────────────────────────────────────────────────
# In production this would be a database, but for demo purposes we keep it simple

vectorstore = None
qa_chain = None

# ── Pydantic Models ───────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    """Request model for chat endpoint"""
    question: str
    session_id: Optional[str] = "default"

class ChatResponse(BaseModel):
    """Response model for chat endpoint"""
    answer: str
    sources: list[str]
    question: str

# ── Helper Functions ──────────────────────────────────────────────────────────

def load_document(file_path: str) -> list:
    """
    Load a document from file path.
    Supports PDF and TXT formats.
    Returns list of LangChain Document objects.
    """
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".txt"):
        loader = TextLoader(file_path, encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file type: {file_path}")

    documents = loader.load()
    print(f"[INFO] Loaded {len(documents)} pages/chunks from {file_path}")
    return documents


def chunk_documents(documents: list) -> list:
    """
    Split documents into smaller chunks for better retrieval.

    Why chunking?
    - LLMs have token limits — we can't feed entire documents
    - Smaller chunks = more precise retrieval
    - Overlap ensures context isn't lost at chunk boundaries

    Strategy: RecursiveCharacterTextSplitter
    - Tries to split on paragraphs first, then sentences, then words
    - Preserves semantic meaning better than fixed-size splitting
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,        # Each chunk = ~500 characters
        chunk_overlap=50,      # 50 char overlap between chunks to preserve context
        separators=["\n\n", "\n", ".", " ", ""]  # Priority order for splitting
    )
    chunks = splitter.split_documents(documents)
    print(f"[INFO] Split into {len(chunks)} chunks")
    return chunks


def build_vectorstore(chunks: list) -> Chroma:
    """
    Create embeddings and store in ChromaDB.

    Why ChromaDB?
    - Persistent local vector database
    - Simple Python API
    - Supports metadata filtering
    - Great for small-to-medium scale demos

    Why sentence-transformers?
    - Free, runs locally — no API cost
    - High quality embeddings for semantic search
    """
    print("[INFO] Building embeddings with sentence-transformers...")

    # Load embedding model (runs locally, no API needed)
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"}
    )

    # Clear existing DB and rebuild
    if Path(CHROMA_DB_DIR).exists():
        shutil.rmtree(CHROMA_DB_DIR)

    # Create vector store from document chunks
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DB_DIR
    )

    print(f"[INFO] Stored {len(chunks)} chunks in ChromaDB")
    return vectorstore


def build_qa_chain(vectorstore: Chroma) -> RetrievalQA:
    """
    Build the RAG chain: Retriever → Prompt → Groq LLM → Answer

    Pipeline:
    1. User query → embed query
    2. Search ChromaDB for top-3 similar chunks
    3. Inject chunks into prompt template
    4. Send to Groq LLM for answer generation
    5. Return answer + source references
    """

    # Initialize Groq LLM — fast inference, free tier available
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama-3.1-8b-instant",   # Fast and capable model on Groq
        temperature=0.1,                # Low temperature = more factual, less creative
        max_tokens=1024
    )

    # Custom prompt — instructs LLM to only use provided context
    prompt_template = """You are a helpful assistant that answers questions based on the provided document context.

STRICT GROUNDING RULES:
- Answer using **only** information explicitly present in the CONTEXT below. Never add external knowledge, assumptions, personal opinions, or fabricated details.
- If the CONTEXT lacks sufficient information for a complete, accurate answer, reply **exactly** with: "I don't have sufficient information in the provided document to answer this question."
- Be concise, professional, and factual.
- Always include inline citations to specific parts of the document (e.g., [Experience section, bullet 3], [Skills list], [Summary paragraph], [page 1, line 5-8]).
- For questions asking for analysis, evaluation, suggestions, strengths/weaknesses, or suitability (e.g., "Is this good for an AI role?", "Give suggestions"), base your response **solely** on facts from the CONTEXT. Infer reasonable conclusions from listed experience, skills, projects, and achievements — but do not speculate beyond what's stated. Structure such answers with clear sections (e.g., Strengths, Areas for Improvement) and support every point with citations.
CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    # Build retrieval QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",                        # "stuff" = concatenate all chunks into one prompt
        retriever=vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}                 # Retrieve top 3 most relevant chunks
        ),
        return_source_documents=True,              # Return which chunks were used
        chain_type_kwargs={"prompt": prompt}
    )

    print("[INFO] QA chain built successfully")
    return qa_chain

# ── API Endpoints ─────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the frontend HTML"""
    with open("static/index.html") as f:
        return f.read()


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a PDF or TXT document to the chatbot.
    This triggers the full ingestion pipeline:
    Load → Chunk → Embed → Store in ChromaDB
    """
    global vectorstore, qa_chain

    # Validate file type
    if not file.filename.endswith((".pdf", ".txt")):
        raise HTTPException(status_code=400, detail="Only PDF and TXT files are supported")

    # Save uploaded file
    file_path = f"{UPLOAD_DIR}/{file.filename}"
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)

    try:
        # Run ingestion pipeline
        documents = load_document(file_path)
        chunks = chunk_documents(documents)
        vectorstore = build_vectorstore(chunks)
        qa_chain = build_qa_chain(vectorstore)

        return JSONResponse({
            "status": "success",
            "message": f"Document '{file.filename}' ingested successfully",
            "chunks_created": len(chunks),
            "pages_loaded": len(documents)
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat with the uploaded document.
    Runs the full RAG pipeline:
    Query → Embed → Retrieve → Generate → Return
    """
    global qa_chain

    if qa_chain is None:
        raise HTTPException(
            status_code=400,
            detail="No document uploaded yet. Please upload a PDF or TXT file first."
        )

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        # Run RAG chain
        result = qa_chain({"query": request.question})

        # Extract source document references
        sources = []
        if "source_documents" in result:
            for doc in result["source_documents"]:
                source = doc.metadata.get("source", "Unknown")
                page = doc.metadata.get("page", "")
                sources.append(f"{source} (page {page})" if page else source)

        return ChatResponse(
            answer=result["result"],
            sources=list(set(sources)),   # Deduplicate sources
            question=request.question
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating answer: {str(e)}")


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "document_loaded": qa_chain is not None,
        "vectorstore_ready": vectorstore is not None
    }


@app.delete("/reset")
async def reset():
    """Reset the chatbot — clear all uploaded documents and vector store"""
    global vectorstore, qa_chain
    vectorstore = None
    qa_chain = None
    if Path(CHROMA_DB_DIR).exists():
        shutil.rmtree(CHROMA_DB_DIR)
    return {"status": "reset", "message": "Chatbot cleared. Upload a new document to start."}
