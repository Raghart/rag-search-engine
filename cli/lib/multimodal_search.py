from PIL import Image
from sentence_transformers import SentenceTransformer
import os

class MultimodalSearch:
    def __init__(self, model_name="clip-ViT-B-32"):
        self.model = SentenceTransformer(model_name)
    
    def embed_image(self, img_path):
        img_data = Image.open(img_path)
        return self.model.encode([img_data])[0]
    
def verify_image_embedding(img_path: str):
    multi_modal = MultimodalSearch()
    if not os.path.exists(img_path):
        raise ValueError(f"Invalid path, img path doesn't exist: {img_path}")
    
    embedding = multi_modal.embed_image(img_path)
    print(f"Embedding shape: {embedding.shape[0]} dimensions")