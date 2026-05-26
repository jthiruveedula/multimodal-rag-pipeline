import unittest
from unittest.mock import MagicMock, patch
import sys

# Completely mock the infrastructure
class MockBigQuery:
    def Client(self, project=None):
        mock = MagicMock()
        mock.insert_rows_json.return_value = [] # Return empty list to signal success
        return mock
    @property
    def SchemaField(self): return MagicMock()
    @property
    def Dataset(self): return MagicMock()
    @property
    def Table(self): return MagicMock()

class MockVertexAI:
    def init(self, project=None, location=None): pass
    class language_models:
        class TextEmbeddingModel:
            @classmethod
            def from_pretrained(cls, name): return MagicMock()
    class vision_models:
        class MultiModalEmbeddingModel:
            @classmethod
            def from_pretrained(cls, name): return MagicMock()
        class Image:
            @classmethod
            def load_from_file(cls, uri): return MagicMock()
    class generative_models:
        class GenerativeModel:
            def __init__(self, name): pass
            def generate_content(self, prompt): return MagicMock()

mbq = MockBigQuery()
sys.modules['google'] = MagicMock()
sys.modules['google.cloud'] = MagicMock()
sys.modules['google.cloud.bigquery'] = mbq
sys.modules['google.cloud.storage'] = MagicMock()
sys.modules['vertexai'] = MockVertexAI()
sys.modules['vertexai.language_models'] = MockVertexAI.language_models
sys.modules['vertexai.vision_models'] = MockVertexAI.vision_models
sys.modules['vertexai.generative_models'] = MockVertexAI.generative_models

from src.multimodal_rag import MultimodalRAGPipeline, RAGConfig

class TestRAGOptimization(unittest.TestCase):
    def setUp(self):
        self.config = RAGConfig(project_id="test-project")

        with patch('vertexai.language_models.TextEmbeddingModel.from_pretrained') as mock_text_model, \
             patch('vertexai.generative_models.GenerativeModel') as mock_gen_model, \
             patch('src.multimodal_rag.bigquery.Client') as mock_bq_client:

            self.mock_text_embedder = MagicMock()
            mock_text_model.return_value = self.mock_text_embedder

            self.mock_gen_model_instance = MagicMock()
            mock_gen_model.return_value = self.mock_gen_model_instance

            self.mock_bq = MagicMock()
            self.mock_bq.insert_rows_json.return_value = []
            mock_bq_client.return_value = self.mock_bq

            self.pipeline = MultimodalRAGPipeline(self.config)
            self.mock_gen_model_class = mock_gen_model

    def test_ingest_text_is_batched(self):
        mock_emb1 = MagicMock()
        mock_emb1.values = [0.1, 0.2]
        mock_emb2 = MagicMock()
        mock_emb2.values = [0.3, 0.4]
        self.mock_text_embedder.get_embeddings.return_value = [mock_emb1, mock_emb2]

        text = "word " * 600
        source = "test.txt"

        self.pipeline.ingest_text(text, source)

        self.assertEqual(self.mock_text_embedder.get_embeddings.call_count, 1)
        args, _ = self.mock_text_embedder.get_embeddings.call_args
        self.assertEqual(len(args[0]), 2)

    def test_generate_answer_uses_preinstantiated_model(self):
        mock_response = MagicMock()
        mock_response.text = "Grounded answer"
        self.mock_gen_model_instance.generate_content.return_value = mock_response

        context = [{"source": "s1", "text": "t1"}]
        self.pipeline.generate_answer("q1", context)
        self.pipeline.generate_answer("q2", context)

        self.assertEqual(self.mock_gen_model_class.call_count, 1)
        self.assertEqual(self.mock_gen_model_instance.generate_content.call_count, 2)

if __name__ == '__main__':
    unittest.main()
