import pickle, os, math
from utils import tokenize_text, load_movies
from collections import defaultdict, Counter
from consts import BM25_K1, DOCMAP_PATH, IDX_PATH, TERM_PATH, DOC_LENGTH_PATH, BM25_B

class InvertedIndex:
    def __init__(self):
        self.index = defaultdict(set)
        self.docmap: dict[int, dict] = {}
        self.term_frequencies = defaultdict(Counter)
        self.doc_lengths = {}
    
    def __add_document(self, doc_id: int, text: str):
        tokenized_arr = tokenize_text(text)
        self.doc_lengths[doc_id] = len(tokenized_arr)
        for token in tokenized_arr:
            self.index[token].add(doc_id)
            self.term_frequencies[doc_id][token] += 1
            
    def get_documents(self, term: str):
        id_set = self.index[term.lower()]
        return sorted(list(id_set))
    
    def get_tf(self, doc_id: int, term: str):
        parsed_term = tokenize_text(term)
        if len(parsed_term) == 0 or len(parsed_term) > 1:
            raise Exception("the term to search must only be one word")
        return self.term_frequencies[doc_id][parsed_term[0]]
    
    def bm25(self, doc_id: int, term: str):
        return self.get_bm25_tf(doc_id, term) * self.get_bm25_idf(term)
    
    def bm25_search(self, query: str, limit: int):
        tokenized_query = tokenize_text(query)
        score_dict = {}
        
        for token in tokenized_query:
            id_set = self.index[token]
            for doc_id in id_set:
                score_dict[doc_id] = score_dict.get(doc_id, 0) + self.bm25(doc_id, token)
        
        documents_array = []
        for doc_id, score in score_dict.items():
            documents_array.append((self.docmap[doc_id], score))
        
        sorted_array = list(sorted(documents_array, key=lambda x: x[1], reverse=True)[:limit])
        return sorted_array
    
    def get_bm25_idf(self, term: str) -> float:
        tokenized_slice = tokenize_text(term)
        if len(tokenized_slice) != 1:
            raise Exception("the term to search must only be one word")
        N = len(self.docmap)
        df = len(self.index[tokenized_slice[0]])
        return math.log((N - df + 0.5)/(df+0.5)+1)
    
    def get_bm25_tf(self, doc_id: int, term: str, k1: float=BM25_K1, b: float=BM25_B):
        tf = self.get_tf(doc_id, term)
        length_norm = 1 - b + b * (self.doc_lengths[doc_id] / self.__get_avg_doc_length())
        return (tf * (k1+1)) / (tf+k1 * length_norm)
    
    def __get_avg_doc_length(self) -> float:
        if len(self.doc_lengths) == 0:
            return 0.00
        
        total_length = 0
        for _, value in self.doc_lengths.items():
            total_length += value
        return total_length / len(self.doc_lengths)
    
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

        with open(DOC_LENGTH_PATH, "rb") as f:
            self.doc_lengths = pickle.load(f)

    def save(self):
        if not os.path.exists("cache"):
            os.makedirs("cache")
        with open(IDX_PATH, "wb") as f:
            pickle.dump(self.index, f)
        with open(DOCMAP_PATH, "wb") as f:
            pickle.dump(self.docmap, f)
        with open(TERM_PATH, "wb") as f:
            pickle.dump(self.term_frequencies, f)
        with open(DOC_LENGTH_PATH, "wb") as f:
            pickle.dump(self.doc_lengths, f)

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

def search_term_frequencies(id: int, term: str):
    idx = InvertedIndex()
    idx.load()
    return idx.get_tf(int(id), term)

def calculate_idf(term: str):
    idx = InvertedIndex()
    idx.load()
    tokenized_list = tokenize_text(term)
    if len(tokenized_list) != 1:
        raise Exception("Can only calculate the idf of one word at a time")
    
    total_doc_count = len(idx.docmap)
    term_match_doc_count = len(idx.index[tokenized_list[0]])
    return math.log((total_doc_count + 1) / (term_match_doc_count + 1))

def calculate_tfidf(id: int, term: str):
    idx = InvertedIndex()
    idx.load()
    tf_score = idx.get_tf(id, term)
    idf_score = calculate_idf(term)
    return tf_score * idf_score

def calculate_bm25_idf(term: str):
    idx = InvertedIndex()
    idx.load()
    return idx.get_bm25_idf(term)

def calculate_bm25_tf(doc_id: int, term: str, k1: float, b: float) -> float:
    idx = InvertedIndex()
    idx.load()
    return idx.get_bm25_tf(doc_id, term, k1, b)

def search_bm25(query: str):
    idx = InvertedIndex()
    idx.load()
    return idx.bm25_search(query, 5)