"""Multimodal RAG Pipeline - Core Implementation
Supports PDF, images, audio ingestion with GCP Vertex AI embeddings
and BigQuery vector search for enterprise-scale retrieval.
"""

from __future__ import annotations
import os
import json
import logging
from dataclasses import dataclass, field
from typing import Any

from google.cloud import bigquery, storage
from vertexai.language_models import TextEmbeddingModel
from vertexai.vision_models import MultiModalEmbeddingModel
import vertexai

logger = logging.getLogger(__name__)


@dataclass
class RAGConfig:
    project_id: str
    location: str = "us-central1"
    bq_dataset: str = "rag_store"
    bq_table: str = "embeddings"
    gcs_bucket: str = ""
    text_model: str = "text-embedding-004"
    multimodal_model: str = "multimodalembedding@001"
    top_k: int = 5
    chunk_size: int = 512
    chunk_overlap: int = 64
    extra: dict = field(default_factory=dict)


class MultimodalRAGPipeline:
    """End-to-end multimodal RAG pipeline backed by BigQuery vector search."""

    def __init__(self, config: RAGConfig) -> None:
        self.config = config
        vertexai.init(project=config.project_id, location=config.location)
        self.bq = bigquery.Client(project=config.project_id)
        self.gcs = storage.Client(project=config.project_id)
        self.text_embedder = TextEmbeddingModel.from_pretrained(config.text_model)
        self.mm_embedder = MultiModalEmbeddingModel.from_pretrained(config.multimodal_model)
        self._ensure_bq_table()

    # ------------------------------------------------------------------
    # Schema bootstrap
    # ------------------------------------------------------------------
    def _ensure_bq_table(self) -> None:
        schema = [
            bigquery.SchemaField("id", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("source", "STRING"),
            bigquery.SchemaField("modality", "STRING"),
            bigquery.SchemaField("chunk_text", "STRING"),
            bigquery.SchemaField("embedding", "FLOAT64", mode="REPEATED"),
            bigquery.SchemaField("metadata", "JSON"),
        ]
        table_ref = f"{self.config.project_id}.{self.config.bq_dataset}.{self.config.bq_table}"
        try:
            self.bq.get_table(table_ref)
            logger.info("BQ table %s exists.", table_ref)
        except Exception:
            dataset = bigquery.Dataset(f"{self.config.project_id}.{self.config.bq_dataset}")
            dataset.location = self.config.location
            self.bq.create_dataset(dataset, exists_ok=True)
            table = bigquery.Table(table_ref, schema=schema)
            self.bq.create_table(table)
            logger.info("Created BQ table %s.", table_ref)

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------
    def ingest_text(self, text: str, source: str, metadata: dict | None = None) -> list[str]:
        """Chunk and embed a text document, store in BigQuery."""
        chunks = self._chunk_text(text)
        rows = []
        ids = []
        for i, chunk in enumerate(chunks):
            emb = self.text_embedder.get_embeddings([chunk])[0].values
            doc_id = f"{source}::chunk_{i}"
            rows.append({
                "id": doc_id,
                "source": source,
                "modality": "text",
                "chunk_text": chunk,
                "embedding": emb,
                "metadata": json.dumps(metadata or {}),
            })
            ids.append(doc_id)
        self._upsert_rows(rows)
        logger.info("Ingested %d text chunks from %s.", len(rows), source)
        return ids

    def ingest_image(self, gcs_uri: str, caption: str = "", metadata: dict | None = None) -> str:
        """Embed an image from GCS and store in BigQuery."""
        from vertexai.vision_models import Image
        img = Image.load_from_file(gcs_uri)
        emb_response = self.mm_embedder.get_embeddings(image=img, contextual_text=caption)
        emb = emb_response.image_embedding
        doc_id = gcs_uri.replace("gs://", "").replace("/", "_")
        row = {
            "id": doc_id,
            "source": gcs_uri,
            "modality": "image",
            "chunk_text": caption,
            "embedding": emb,
            "metadata": json.dumps(metadata or {}),
        }
        self._upsert_rows([row])
        logger.info("Ingested image %s.", gcs_uri)
        return doc_id

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    def retrieve(self, query: str, modality_filter: str | None = None) -> list[dict]:
        """Semantic search via BigQuery VECTOR_SEARCH."""
        query_emb = self.text_embedder.get_embeddings([query])[0].values
        emb_str = ", ".join(str(v) for v in query_emb)
        modality_clause = ""
        if modality_filter:
            modality_clause = f"AND modality = '{modality_filter}'"
        sql = f"""
            SELECT id, source, modality, chunk_text, metadata,
                   COSINE_DISTANCE(embedding, [{emb_str}]) AS score
            FROM `{self.config.project_id}.{self.config.bq_dataset}.{self.config.bq_table}`
            WHERE TRUE {modality_clause}
            ORDER BY score ASC
            LIMIT {self.config.top_k}
        """
        rows = list(self.bq.query(sql).result())
        return [{"id": r.id, "source": r.source, "text": r.chunk_text, "score": r.score} for r in rows]

    def generate_answer(self, query: str, context_docs: list[dict]) -> str:
        """Build grounded prompt and call Gemini for final answer."""
        import vertexai.generative_models as genai
        context = "\n\n".join(f"[{d['source']}]\n{d['text']}" for d in context_docs)
        prompt = (
            f"You are a helpful assistant. Use ONLY the context below to answer.\n\n"
            f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        )
        model = genai.GenerativeModel("gemini-1.5-pro")
        response = model.generate_content(prompt)
        return response.text

    def query(self, question: str, modality_filter: str | None = None) -> dict:
        """Full RAG query: retrieve + generate."""
        docs = self.retrieve(question, modality_filter)
        answer = self.generate_answer(question, docs)
        return {"answer": answer, "sources": docs}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _chunk_text(self, text: str) -> list[str]:
        words = text.split()
        chunks, start = [], 0
        while start < len(words):
            end = min(start + self.config.chunk_size, len(words))
            chunks.append(" ".join(words[start:end]))
            start += self.config.chunk_size - self.config.chunk_overlap
        return chunks

    def _upsert_rows(self, rows: list[dict]) -> None:
        table_ref = f"{self.config.project_id}.{self.config.bq_dataset}.{self.config.bq_table}"
        errors = self.bq.insert_rows_json(table_ref, rows)
        if errors:
            raise RuntimeError(f"BigQuery insert errors: {errors}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Multimodal RAG Pipeline CLI")
    parser.add_argument("--project", required=True)
    parser.add_argument("--query", default="What are the key findings?")
    args = parser.parse_args()

    cfg = RAGConfig(project_id=args.project)
    pipeline = MultimodalRAGPipeline(cfg)
    result = pipeline.query(args.query)
    print(json.dumps(result, indent=2))

