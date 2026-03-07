# 🤖 RAG Document Chatbot

> Upload any PDF or TXT document and chat with it using Groq LLM + ChromaDB vector search.

Built by **Syed Muhammad Mehmam** — AI Engineer | [LinkedIn](https://linkedin.com/in/muhammad-mehmam) | [GitHub](https://github.com/Mehmaam99)

---

## 🎯 What This Does

This project demonstrates a **production-grade RAG (Retrieval-Augmented Generation) pipeline**:

1. **Upload** a PDF or TXT document
2. **Ingestion pipeline** automatically chunks, embeds, and stores in ChromaDB
3. **Ask questions** — the system retrieves relevant chunks and generates accurate answers using Groq LLM
4. **Sources cited** — every answer shows which document sections were used

## 🏗️ Architecture

```
User Query
    │
    ▼
[Embed Query]  ←── sentence-transformers (local, free)
    │
    ▼
[ChromaDB Similarity Search]  ←── Top 3 relevant chunks
    │
    ▼
[Groq LLM - Llama3]  ←── Context + Query → Answer
    │
    ▼
Answer + Sources
```

## 🛠️ Tech Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| API Framework | FastAPI | Fast, async, auto docs |
| LLM | Groq (Llama3-8b) | Free, ultra-fast inference |
| Vector DB | ChromaDB | Simple, persistent, local |
| Embeddings | sentence-transformers | Free, runs locally |
| Chunking | RecursiveCharacterTextSplitter | Preserves semantic meaning |
| Frontend | Vanilla HTML/JS | Lightweight, no build step |

## 🚀 How to Run

```bash
# 1. Clone the repo
git clone https://github.com/Mehmaam99/rag-chatbot
cd rag-chatbot

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your Groq API key (free at console.groq.com)
export GROQ_API_KEY="your-key-here"

# 4. Run the app
uvicorn app.main:app --reload --port 8000

# 5. Open browser
# http://localhost:8000
```

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Frontend UI |
| POST | `/upload` | Upload PDF/TXT document |
| POST | `/chat` | Ask a question |
| GET | `/health` | Check system status |
| DELETE | `/reset` | Clear all data |

## 💡 Real-World Application

This project is directly inspired by a **production healthcare system** I built at Xloop Digital Services — processing 500+ clinical sessions/month, where clinicians used a similar RAG pipeline to retrieve patient context. This demo shows the same core architecture in an accessible, portfolio-ready format.

## 📁 Project Structure

```
project1_rag_chatbot/
├── app/
│   └── main.py          # FastAPI app + RAG pipeline
├── static/
│   └── index.html       # Frontend UI
├── requirements.txt
└── README.md
```
