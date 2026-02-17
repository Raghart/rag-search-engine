from sentence_transformers import SentenceTransformer
from consts import EMBEDDINGS_PATH, DATA_PATH
import numpy as np
import os, json

class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
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