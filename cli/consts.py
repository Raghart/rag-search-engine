import os

BM25_K1 = 1.5
BM25_B = 0.75
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "movies.json")
STOPWORDS_PATH = os.path.join(PROJECT_ROOT, "data", "stopwords.txt")
IDX_PATH = os.path.join(PROJECT_ROOT, "cache", "index.pkl")
DOCMAP_PATH = os.path.join(PROJECT_ROOT, "cache", "docmap.pkl")
TERM_PATH = os.path.join(PROJECT_ROOT, "cache", "term_frequencies.pkl")
DOC_LENGTH_PATH = os.path.join(PROJECT_ROOT, "cache", "doc_lengths.pkl")
EMBEDDINGS_PATH = os.path.join(PROJECT_ROOT, "cache", "movie_embeddings.npy")
CHUNK_EMBEDDINGS_PATH = os.path.join(PROJECT_ROOT, "cache", "chunk_embeddings.npy")
CHUNK_METADATA_PATH = os.path.join(PROJECT_ROOT, "cache", "chunk_metadata.json")
TEST_TEXT = "Why is Boot.dev such a great place to learn about RAG? Use one paragraph maximum."