import sys
import types
from sentence_transformers import SentenceTransformer
import chromadb

# Workaround for Windows Device Guard / Code Integrity blocking unsigned cygrpc DLL in grpcio
if "opentelemetry.exporter.otlp.proto.grpc.trace_exporter" not in sys.modules:
    try:
        import grpc  # noqa: F401
    except (ImportError, OSError):
        dummy_grpc_exporter = types.ModuleType("opentelemetry.exporter.otlp.proto.grpc.trace_exporter")
        dummy_grpc_exporter.OTLPSpanExporter = None
        sys.modules["opentelemetry.exporter.otlp.proto.grpc.trace_exporter"] = dummy_grpc_exporter

chroma_client = chromadb.PersistentClient(
    path="./chromadb"
)

collection = chroma_client.get_or_create_collection(
    name="hrms_documents",
    metadata={"hnsw:space": "cosine"}
)

# Shared embedding model for both ingestion and retrieval
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


def generate_embeddings(texts):
    if isinstance(texts, str):
        texts = [texts]
    embeddings = embedding_model.encode(
        texts,
        normalize_embeddings=True
    )
    return embeddings.tolist()


def count_tokens(text: str) -> int:
    """Accurately count tokens using the model's tokenizer."""
    if not text:
        return 0
    return len(embedding_model.tokenizer.encode(text, add_special_tokens=False))


# Pipeline configuration defaults
DEFAULT_TOP_K = 3
DEFAULT_CONTEXT_TOKEN_BUDGET = 1000
SCORE_THRESHOLD = 0.85  # Max cosine distance threshold for relevance
