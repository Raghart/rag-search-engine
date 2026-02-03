import pickle, os
from utils import tokenize_text, load_movies, PROJECT_ROOT
from collections import defaultdict

IDX_PATH = os.path.join(PROJECT_ROOT, "cache", "index.pkl")
DOCMAP_PATH = os.path.join(PROJECT_ROOT, "cache", "docmap.pkl")

class InvertedIndex:
    def __init__(self):
        self.index = defaultdict(set)
        self.docmap = {}
    
    def __add_document(self, doc_id, text):
        tokenized_arr = tokenize_text(text)
        for token in tokenized_arr:
            self.index[token].add(doc_id)
    
    def get_documents(self, term):
        id_set = self.index[term.lower()]
        return sorted(list(id_set))
    
    def build(self):
        movies = load_movies()
        for movie in movies:
            self.__add_document(movie["id"], f"{movie['title']} {movie['description']}")
            self.docmap[movie["id"]] = movie
    
    def save(self):
        if not os.path.exists("cache"):
            os.makedirs("cache")
        with open(IDX_PATH, "wb") as f:
            pickle.dump(self.index, f)
        with open(DOCMAP_PATH, "wb") as f:
            pickle.dump(self.docmap, f)

def build_inverted_idx():
    inverted_idx = InvertedIndex()
    inverted_idx.build()
    inverted_idx.save()

    id_set = inverted_idx.get_documents('merida')
    print(f"First document for token 'merida' = {id_set[0]}")