# 🧠 Multimodal RAG Pipeline

> End-to-end Retrieval-Augmented Generation pipeline that ingests **PDFs, images, and audio** to build a unified knowledge base — queryable via natural language through a streaming FastAPI interface.

![Python](https://img.shields.io/badge/Python-3.11-blue) ![LangChain](https://img.shields.io/badge/LangChain-0.2-green) ![Weaviate](https://img.shields.io/badge/VectorDB-Weaviate-purple) ![FastAPI](https://img.shields.io/badge/API-FastAPI-teal) ![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 🎯 Problem Statement

Enterprise knowledge is fragmented across unstructured formats — scanned PDFs, architecture diagrams (images), and meeting recordings (audio). This project builds a production-grade **Multimodal RAG system** that unifies all these modalities into a single queryable knowledge base.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    INGESTION LAYER                              │
│  PDFs (PyMuPDF) │ Images (GPT-4o Vision) │ Audio (Whisper API) │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                   CHUNKING & EMBEDDING                          │
│  RecursiveCharacterTextSplitter │ OpenAI text-embedding-3-large  │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                    VECTOR STORE (Weaviate)                      │
│  Multi-tenant collections per modality + metadata filtering     │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│              RETRIEVAL & RE-RANKING (LangChain)                 │
│  Hybrid Search (BM25 + Dense) │ Cohere Reranker │ MMR           │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                 GENERATION (GPT-4o Streaming)                   │
│  Context-aware prompt │ Citation tracking │ Source attribution  │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│              FastAPI + Server-Sent Events (SSE)                 │
│  /ingest │ /query │ /query/stream │ /sources │ /health          │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Features

- **Multi-modal ingestion**: PDFs (text + tables), Images (OCR + visual description via GPT-4o Vision), Audio (Whisper transcription)
- **Hybrid search**: Combines BM25 keyword search with dense vector similarity
- **Re-ranking**: Cohere reranker for improved retrieval precision (MRR@10 improvement)
- **Streaming responses**: Server-Sent Events for real-time LLM token streaming
- **Source attribution**: Every answer includes citations with page/timestamp references
- **Docker-ready**: Full `docker-compose` setup for local development
- **Evaluation harness**: RAGAS-based evaluation pipeline (faithfulness, answer relevancy, context recall)

---

## 📁 Project Structure

```
multimodal-rag-pipeline/
├── ingestion/
│   ├── pdf_parser.py          # PyMuPDF-based PDF text + table extraction
│   ├── image_captioner.py     # GPT-4o Vision image description
│   ├── audio_transcriber.py   # OpenAI Whisper audio → text
│   └── chunker.py             # RecursiveCharacterTextSplitter + metadata tagging
├── vectorstore/
│   ├── weaviate_client.py     # Weaviate connection + schema creation
│   ├── embedder.py            # OpenAI embedding generation
│   └── indexer.py             # Batch upsert with retry logic
├── retrieval/
│   ├── hybrid_retriever.py    # BM25 + Dense hybrid search
│   ├── reranker.py            # Cohere reranker integration
│   └── context_builder.py    # Context window assembly + deduplication
├── generation/
│   ├── prompt_templates.py    # LangChain PromptTemplates with citation instructions
│   ├── llm_chain.py           # LangChain LCEL chain (Runnable)
│   └── streaming_handler.py  # SSE streaming callback handler
├── api/
│   ├── main.py                # FastAPI app entrypoint
│   ├── routes/
│   │   ├── ingest.py          # POST /ingest endpoint
│   │   ├── query.py           # POST /query + GET /query/stream
│   │   └── sources.py         # GET /sources (metadata browser)
│   └── schemas.py             # Pydantic request/response models
├── evaluation/
│   ├── ragas_eval.py          # RAGAS evaluation pipeline
│   ├── test_dataset.json      # Ground truth Q&A pairs
│   └── eval_report.md         # Benchmark results
├── notebooks/
│   ├── 01_ingestion_demo.ipynb
│   ├── 02_retrieval_analysis.ipynb
│   └── 03_end_to_end_demo.ipynb
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## ⚙️ Tech Stack

| Component | Technology |
|-----------|------------|
| LLM | GPT-4o (OpenAI API) |
| Embeddings | text-embedding-3-large |
| Audio Transcription | OpenAI Whisper API |
| Vector DB | Weaviate (self-hosted / WCS) |
| RAG Framework | LangChain 0.2 (LCEL) |
| Re-ranking | Cohere Rerank v3 |
| API Framework | FastAPI + Uvicorn |
| Streaming | Server-Sent Events (SSE) |
| Evaluation | RAGAS |
| Containerization | Docker + Docker Compose |

---

## 🧪 Evaluation Results (RAGAS)

| Metric | Score |
|--------|-------|
| Faithfulness | 0.91 |
| Answer Relevancy | 0.88 |
| Context Recall | 0.85 |
| Context Precision | 0.82 |

---

## 🛠️ Quick Start

```bash
# Clone the repository
git clone https://github.com/jthiruveedula/multimodal-rag-pipeline.git
cd multimodal-rag-pipeline

# Set up environment variables
cp .env.example .env
# Edit .env with your OPENAI_API_KEY, COHERE_API_KEY, WEAVIATE_URL

# Start Weaviate + API with Docker Compose
docker-compose up -d

# Ingest sample documents
curl -X POST http://localhost:8000/ingest \
  -F "file=@sample_docs/architecture.pdf" \
  -F "modality=pdf"

# Query the knowledge base
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the key architectural components?"}'
```

---

## 🎤 Interview Talking Points

- **Why Weaviate over Pinecone/FAISS?** Multi-tenancy, hybrid BM25+dense search natively, HNSW indexing with filterable properties
- **Why re-ranking?** First-stage retrieval optimizes for recall; re-ranking optimizes for precision — critical for production RAG
- **Streaming design**: SSE vs WebSockets — SSE is simpler for unidirectional token streaming, no overhead of full duplex
- **Multimodal chunking**: Each modality requires different chunking strategies — PDFs by semantic sections, audio by speaker turns, images as atomic units
- **Evaluation**: RAGAS provides reference-free evaluation using LLMs-as-judges for faithfulness and relevancy

---

## 📄 License

MIT License — see [LICENSE](LICENSE) file.
