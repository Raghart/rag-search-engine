import pickle, os
from utils import tokenize_text, load_movies, PROJECT_ROOT
from collections import defaultdict, Counter

IDX_PATH = os.path.join(PROJECT_ROOT, "cache", "index.pkl")
DOCMAP_PATH = os.path.join(PROJECT_ROOT, "cache", "docmap.pkl")
TERM_PATH = os.path.join(PROJECT_ROOT, "cache", "term_frequencies.pkl")

class InvertedIndex:
    def __init__(self):
        self.index = defaultdict(set)
        self.docmap: dict[int, dict] = {}
        self.term_frequencies = defaultdict(Counter)
    
    def __add_document(self, doc_id, text):
        tokenized_arr = tokenize_text(text)
        for token in tokenized_arr:
            self.index[token].add(doc_id)
            self.term_frequencies[doc_id][token] += 1
            
    def get_documents(self, term):
        id_set = self.index[term.lower()]
        return sorted(list(id_set))
    
    def get_tf(self, doc_id: int, term: str):
        parsed_term = tokenize_text(term)
        if len(parsed_term) == 0 or len(parsed_term) > 1:
            raise Exception("the term to search must only be one word")
        return self.term_frequencies[doc_id][parsed_term[0]]
    
    def build(self):
        movies = load_movies()
        for movie in movies:
            self.__add_document(movie["id"], f"{movie['title']} {movie['description']}")
            self.docmap[movie["id"]] = movie

    def load(self):
        if not os.path.exists(IDX_PATH) or not os.path.exists(DOCMAP_PATH):
            raise Exception("Invalid loading, first you have to use the build code to generate the cache!")
        
        with open(IDX_PATH, "rb") as f:
            self.index = pickle.load(f)
        
        with open(DOCMAP_PATH, "rb") as f:
            self.docmap = pickle.load(f)
        
        with open(TERM_PATH, "rb") as f:
            self.term_frequencies = pickle.load(f)

    def save(self):
        if not os.path.exists("cache"):
            os.makedirs("cache")
        with open(IDX_PATH, "wb") as f:
            pickle.dump(self.index, f)
        with open(DOCMAP_PATH, "wb") as f:
            pickle.dump(self.docmap, f)
        with open(TERM_PATH, "wb") as f:
            pickle.dump(self.term_frequencies, f)

def build_inverted_idx():
    inverted_idx = InvertedIndex()
    inverted_idx.build()
    inverted_idx.save()

def search_movies(query):
    result = []
    idx = InvertedIndex()
    idx.load()
    tokenized_query = tokenize_text(query)

    for token in tokenized_query:
        id_set = idx.get_documents(token)
        for id in id_set:
            result.append(idx.docmap[id])
            if len(result) >= 5:
                break
        if len(result) >= 5:
            break
            
    return result

def search_term_frequencies(id, term):
    idx = InvertedIndex()
    idx.load()
    return idx.get_tf(int(id), term)