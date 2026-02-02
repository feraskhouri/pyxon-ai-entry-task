"""Configuration for retrieval-optimized settings."""

# Chunking: larger chunks = fewer chunks = less competition in retrieval
CHUNK_SIZE = 1024
CHUNK_OVERLAP = 128

# Embedding models (sentence-transformers)
# Default: good multilingual, Arabic support
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
# Alternative: BGE-M3 - often better for RAG, multilingual
BGE_M3_MODEL = "BAAI/bge-m3"
