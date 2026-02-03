import pickle, os
from utils import tokenize_text, PROJECT_ROOT
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
            if token in self.index:
                self.index[token].add(doc_id)
            else:
                self.index[token] = {doc_id}
    
    def get_documents(self, term):
        id_set = self.index[term.lower()]
        return sorted(id_set)
    
    def build(self, movies):
        for movie in movies:
            self.__add_document(movie["id"], f"{movie['title']} {movie['description']}")
            self.docmap[movie["id"]] = movie
    
    def save(self):
        if not os.path.exists("cache"):
            os.makedirs("cache")
        pickle.dump(self.index, open(IDX_PATH, "wb"))
        pickle.dump(self.docmap, open(DOCMAP_PATH, "wb"))