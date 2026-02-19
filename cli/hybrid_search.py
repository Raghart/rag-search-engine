import os
from inverted_index import InvertedIndex
from lib.semantic_search import ChunkedSemanticSearch
from consts import IDX_PATH


class HybridSearch:
    def __init__(self, documents):
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(IDX_PATH):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query, limit):
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query, alpha, limit=5):
        raise NotImplementedError("Weighted hybrid search is not implemented yet.")

    def rrf_search(self, query, k, limit=10):
        raise NotImplementedError("RRF hybrid search is not implemented yet.")

def normalize_data(array_num: list):
    if len(array_num) == 0:
        return []
    
    max_num = max(array_num)
    min_num = min(array_num)

    if max_num == min_num:
        return [1.0]
    
    result_arr = []
    for _, num in enumerate(array_num):
        result_arr.append((num - min_num) / (max_num - min_num))
        
    return result_arr