from PIL import Image
from sentence_transformers import SentenceTransformer
import os, json
import numpy as np
from consts import DATA_PATH

class MultimodalSearch:
    def __init__(self, documents: list, model_name="clip-ViT-B-32"):
        self.model = SentenceTransformer(model_name)
        self.documents = documents
        self.texts = []

        for _, doc in enumerate(self.documents):
            self.texts.append(f"{doc['title']}: {doc['description']}")
        self.text_embeddings = self.model.encode(self.texts, show_progress_bar=True)
    
    def embed_image(self, img_path):
        img_data = Image.open(img_path)
        return self.model.encode([img_data])[0]
    
    def search_with_image(self, img_path: str):
        img_embedding = self.embed_image(img_path)
        search_results = []
        for idx, text_embed in enumerate(self.text_embeddings):
            cos_score = cosine_similarity(img_embedding, text_embed)
            search_results.append({
                "id": self.documents[idx]["id"],
                "title": self.documents[idx]["title"],
                "description": self.documents[idx]["description"],
                "cos_score": cos_score,
            })
        
        return list(sorted(search_results, key=lambda x: x["cos_score"], reverse=True))[:5]
    
def search_movie_by_img(img_path: str):
    movie_data = load_movies()
    multimodal = MultimodalSearch(movie_data)
    return multimodal.search_with_image(img_path)

def verify_image_embedding(img_path: str):
    multi_modal = MultimodalSearch()
    if not os.path.exists(img_path):
        raise ValueError(f"Invalid path, img path doesn't exist: {img_path}")
    
    embedding = multi_modal.embed_image(img_path)
    print(f"Embedding shape: {embedding.shape[0]} dimensions")

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)

def load_movies():
    with open(DATA_PATH, "rb") as f:
        return json.load(f)["movies"]