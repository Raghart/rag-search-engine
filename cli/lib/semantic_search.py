from sentence_transformers import SentenceTransformer
from consts import EMBEDDINGS_PATH, DATA_PATH, CHUNK_EMBEDDINGS_PATH, CHUNK_METADATA_PATH
import numpy as np
import os, json, re

class SemanticSearch:
    def __init__(self, model = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model)
        self.embeddings = None
        self.documents = None
        self.document_map = {}

    def build_embeddings(self, documents):
        self.documents = documents
        movie_list = []
        for doc in self.documents:
            self.document_map[doc["id"]] = doc
            movie_list.append(f"{doc['title']}: {doc['description']}")
        
        self.embeddings = self.model.encode(movie_list, show_progress_bar=True)
        with open(EMBEDDINGS_PATH, "wb") as f:
            np.save(f, self.embeddings)
        return self.embeddings
    
    def search(self, query: str, limit: int):
        if self.embeddings is None:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")
        
        query_embedding = self.generate_embedding(query)
        result_slice = []
        for idx, doc_embedding in enumerate(self.embeddings, 0):
            cos_score = cosine_similarity(query_embedding, doc_embedding)
            result_slice.append((cos_score, self.documents[idx]))
        
        sorted_list = list(sorted(result_slice, key=lambda x: x[0], reverse=True))
        return sorted_list[:limit]
    
    def load_or_create_embeddings(self, documents):
        self.documents = documents
        for doc in self.documents:
            self.document_map[doc["id"]] = doc
        
        if os.path.exists(EMBEDDINGS_PATH):
            with open(EMBEDDINGS_PATH, "rb") as f:
                self.embeddings = np.load(f)
            if len(self.embeddings) == len(documents):
                return self.embeddings
            
        return self.build_embeddings(documents)

    def generate_embedding(self, text: str):
        if text.strip() == "":
            raise ValueError("the message to embbed musn't be empty!")

        embedding = self.model.encode([text])
        return embedding[0]

class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name = "all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None

    def build_chunk_embeddings(self, documents):
        self.documents = documents
        all_chunks = []
        chunks_metadata = []
        for doc in documents:
            self.document_map[doc["id"]] = doc
            
            if len(doc["description"]) == 0:
                continue
            
            chunk_arr = semantic_chunk(doc["description"], 4, 1)
            
            for idx, chunk in enumerate(chunk_arr):
                chunk_dict = {}
                all_chunks.append(chunk)
                chunk_dict["movie_idx"] = doc["id"]
                chunk_dict["chunk_idx"] = idx
                chunk_dict["total_chunks"] = len(chunk_arr)
                chunks_metadata.append(chunk_dict)

        self.chunk_embeddings = self.model.encode(all_chunks, show_progress_bar=True)
        self.chunk_metadata = chunks_metadata

        with open(CHUNK_EMBEDDINGS_PATH, "wb") as f:
            np.save(f, self.chunk_embeddings)

        with open(CHUNK_METADATA_PATH, "w") as f:
            json.dump({"chunks": chunks_metadata, "total_chunks": len(all_chunks)}, f, indent=2)

        return self.chunk_embeddings
    
    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        for doc in documents:
            self.document_map[doc["id"]] = doc
        
        if os.path.exists(CHUNK_EMBEDDINGS_PATH) and os.path.exists(CHUNK_METADATA_PATH):
            with open(CHUNK_EMBEDDINGS_PATH, "rb") as f:
                self.chunk_embeddings = np.load(f)
            
            with open(CHUNK_METADATA_PATH, "r") as f:
                self.chunk_metadata = json.load(f)["chunks"]
            
            return self.chunk_embeddings 

        return self.build_chunk_embeddings(documents)

def verify_model():
    sem_search = SemanticSearch()
    print(f"Model loaded: {sem_search.model}")
    print(f"Max sequence length: {sem_search.model.max_seq_length}")

def embed_text(text):
    sem_search = SemanticSearch()
    embedding = sem_search.generate_embedding(text)
    
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

def verify_embeddings():
    sem_search = SemanticSearch()
    with open(DATA_PATH, "rb") as f:
        documents = json.load(f)

    movie_list = documents["movies"]
    embeddings = sem_search.load_or_create_embeddings(movie_list)
    print(f"Number of docs:   {len(documents)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)

def embed_query_text(query):
    sem_search = SemanticSearch()
    embedding = sem_search.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")

def semantic_search(query, limit=5):
    sem_search = SemanticSearch()
    with open(DATA_PATH, "rb") as f:
        movie_data = json.load(f)
    
    movies_arr = movie_data["movies"]

    sem_search.load_or_create_embeddings(movies_arr)
    return sem_search.search(query, limit)

def chunk_text(text: str, size: int, overlap: int):
    print(f"Chunking {len(text)} characters")
    word_list = text.split()
    chunk_num = 1
    current_chunk = []
    
    for _, word in enumerate(word_list):
        current_chunk.append(word)
        if len(current_chunk) >= size:
            text_chunked = " ".join(current_chunk)
            print(f"{chunk_num}. {text_chunked}")
            chunk_num += 1

            if overlap == 0:
                current_chunk.clear()
            else:
                current_chunk = current_chunk[-overlap:]

    if (len(current_chunk) > 1 and overlap == 0) or (overlap >= 1 and len(current_chunk) > overlap):
        final_chunk = " ".join(current_chunk)
        print(f"{chunk_num}. {final_chunk}")

def semantic_chunk(text: str, max_chunk_size: int, overlap: int):
    print(f"Semantically chunking {len(text)} characters")
    text_arr = re.split(r"(?<=[.!?])\s+", text)
    result_arr = []
    current_chunk = []

    for idx, sentence in enumerate(text_arr, 1):
        current_chunk.append(sentence)
        if idx == len(text_arr):
            final_chunk = " ".join(current_chunk)
            result_arr.append(final_chunk)
            break

        if len(current_chunk) == max_chunk_size:
            text_chunked = " ".join(current_chunk)
            result_arr.append(text_chunked)
            if overlap == 0:
                current_chunk.clear()
            else:
                current_chunk = current_chunk[-overlap:]

    return result_arr

def embed_movie_chunks():
    with open(DATA_PATH, "rb") as f:
        movie_documents = json.load(f)["movies"]
    chunk_sem_search = ChunkedSemanticSearch()
    return chunk_sem_search.load_or_create_chunk_embeddings(movie_documents)