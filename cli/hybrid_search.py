import os, time
from inverted_index import InvertedIndex
from lib.semantic_search import ChunkedSemanticSearch
from consts import IDX_PATH
from lib.semantic_search import load_movies
from dotenv import load_dotenv
from google import genai
from google.genai import types
from sentence_transformers.cross_encoder import CrossEncoder

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
    
    def _hybrid_score(self, bm25_score: float, semantic_score: float, alpha=0.5):
        return alpha * bm25_score + (1-alpha) * semantic_score
    
    def _build_query_prompt(self, query:str, enhance: str):
        if enhance == "spell":
            return f"""Fix any spelling errors in this movie search query.

            Only correct obvious typos. Don't change correctly spelled words.

            Query: "{query}"

            If no errors, return the original query.
            Corrected:"""
        if enhance == "rewrite":
            return f"""Rewrite this movie search query to be more specific and searchable.

            Original: "{query}"

            Consider:
            - Common movie knowledge (famous actors, popular films)
            - Genre conventions (horror = scary, animation = cartoon)
            - Keep it concise (under 10 words)
            - It should be a google style search query that's very specific
            - Don't use boolean logic

            Examples:

            - "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
            - "movie about bear in london with marmalade" -> "Paddington London marmalade"
            - "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

            Rewritten query:"""
        if enhance == "expand":
            return f"""Expand this movie search query with related terms.

            Add synonyms and related concepts that might appear in movie descriptions.
            Keep expansions relevant and focused.
            This will be appended to the original query.

            Examples:

            - "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
            - "action movie with bear" -> "action thriller bear chase fight adventure"
            - "comedy with bear" -> "comedy funny bear humor lighthearted"

            Query: "{query}"
            """
        raise ValueError("Enhance is not a valid option")

    def weighted_search(self, query, alpha, limit=5):
        bm25_results = self._bm25_search(query, limit*500)
        chunk_search_results = self.semantic_search.search_chunks(query, limit*500)

        bm25_scores = []
        chunk_scores = []
        for bm25_data in bm25_results:
            bm25_scores.append(bm25_data[1])
        
        for chunk_result in chunk_search_results:
            chunk_scores.append(chunk_result["score"])
        
        normalized_bm25_scores = normalize_data(bm25_scores)
        normalized_chunk_scores = normalize_data(chunk_scores)

        hybrid_map = {}
        for idx, data in enumerate(bm25_results):
            hybrid_map[data[0]["id"]] = {
                "doc": data[0],
                "bm25_score": normalized_bm25_scores[idx],
                "semantic_score": 0
            }
        
        for idx, sem_data in enumerate(chunk_search_results):
            if sem_data["id"] in hybrid_map:
                data_dict = hybrid_map[sem_data["id"]]
                data_dict["semantic_score"] = normalized_chunk_scores[idx]
            else:
                hybrid_map[sem_data["id"]] = {
                    "doc": {
                        "id": sem_data["id"],
                        "title": sem_data["title"],
                        "description": sem_data["document"]
                    },
                    "semantic_score": normalized_chunk_scores[idx],
                    "bm25_score": 0
                }

        hybrid_results = []
        
        for idx, hybrid_data in enumerate(hybrid_map.values()):
            bm25_score = hybrid_data["bm25_score"]
            sem_score = hybrid_data["semantic_score"]
            hybrid_data["hybrid_score"] = self._hybrid_score(bm25_score, sem_score, alpha)
            hybrid_results.append(hybrid_data)

        return list(sorted(hybrid_results, key=lambda x: x["hybrid_score"], reverse=True))[:limit]
    
    def _get_rrf_score(self, rank: int, k:int=60) -> float:
        return 1 / (k+rank)

    def process_query(self, query:str, enhance: str):
        api_key = os.environ.get("rag-gemini-key")
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model='gemini-2.5-flash', 
            contents=self._build_query_prompt(query, enhance)
        )

        if enhance == "spell":
            return response.text.replace("Corrected: ", "")
        
        if enhance == "rewrite":
            return response.text.replace("Rewritten query:","")
        
        if enhance == "expand":
            return response.text.replace("Query: ","")
        
    def rrf_search(self, query:str, k:int, limit:int, enhance:str, rerank_method: str):
        search_query = query
        load_dotenv()
        if enhance is not None:
            search_query = self.process_query(query, enhance)
            print(f"Enhanced query ({enhance}): '{query}' -> '{search_query}'")
        
        number_searches = limit*500
        if rerank_method is not None:
            number_searches = limit*5
            
        bm25_results = self._bm25_search(search_query, number_searches)
        semantic_results = self.semantic_search.search_chunks(search_query, number_searches)

        search_map = {}
        for bm25_rank, bm25_data in enumerate(bm25_results, 1):
            doc = bm25_data[0]
            search_map[doc["id"]] = {
                "id": doc["id"],
                "title": doc["title"],
                "description": doc["description"],
                "bm25_rank": bm25_rank,
                "sem_rank": 0,
                "rrf_score": self._get_rrf_score(bm25_rank, k)
            }
        
        for sem_rank, sem_data in enumerate(semantic_results, 1):
            doc_id = sem_data["id"]
            if doc_id in search_map:
                search_data = search_map[doc_id]
                search_data["sem_rank"] = sem_rank
                search_data["rrf_score"] += self._get_rrf_score(sem_rank, k)
            else:
                search_map[doc_id] = {
                    "id": doc_id,
                    "title": sem_data["title"],
                    "description": sem_data["document"],
                    "sem_rank": sem_rank,
                    "bm25_rank": 0,
                    "rrf_score": self._get_rrf_score(sem_rank, k)
                }
        
        rrf_results = list(
            sorted(search_map.values(), 
                   key=lambda x: x["rrf_score"], 
                   reverse=True)) [:number_searches]
        
        if rerank_method is not None:
            print(f"Reranking top {limit} results using cross_encoder method...")
            return self.rerank_results(rrf_results, search_query, rerank_method, limit)  
            
        return rrf_results [:limit]

    def rerank_results(self, rrf_results: list, query:str, rerank_method: str, limit: int):
        api_key = os.environ.get("rag-gemini-key")
        client = genai.Client(api_key=api_key)

        if rerank_method == "cross_encoder":
            pairs = []
            for _, doc in enumerate(rrf_results):
                pairs.append([query, f"{doc.get('title','')} - {doc.get('document','')}"])
            cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2")
            scores = cross_encoder.predict(pairs)

            for idx, score in enumerate(scores):
                rrf_results[idx]["cross_score"] = score
            
            return list(sorted(rrf_results, key=lambda x: x["cross_score"], reverse=True)) [:limit]


        if rerank_method == "individual":
            for _, rrf_data in enumerate(rrf_results):
                response = client.models.generate_content(
                    model='gemini-2.5-flash', 
                    contents=self._build_rerank_prompt(rrf_data, query, rerank_method),
                    config=types.GenerateContentConfig(
                        safety_settings=[
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                        threshold=types.HarmBlockThreshold.BLOCK_NONE,
                    ),
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                        threshold=types.HarmBlockThreshold.BLOCK_NONE,
                    ),
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                        threshold=types.HarmBlockThreshold.BLOCK_NONE,
                    ),
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                        threshold=types.HarmBlockThreshold.BLOCK_NONE,
                    ),
                    ]
                    )
                )
                print(response)
                print(response.text)
                llm_score = float(response.text.replace("Score:", ""))
                rrf_data["llm_score"] = llm_score
                time.sleep(5)
            
            return list(sorted(rrf_results, key=lambda x: x["llm_score"], reverse=True))
             
    
    def _build_rerank_prompt(self, doc: dict, query: str, rerank_method: str):
        if rerank_method == "individual":
            return f"""Rate how well this movie matches the search query.

            Query: "{query}"
            Movie: {doc.get("title", "")} - {doc.get("description", "")}

            Consider:
            - Direct relevance to query
            - User intent (what they're looking for)
            - Content appropriateness

            Rate 0-10 (10 = perfect match).
            Give me ONLY the number in your response, no other text or explanation.

            Score:"""


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

def weighted_search(text: str, alpha: float, limit):
    movie_documents = load_movies()
    hybrid_search = HybridSearch(movie_documents)
    return hybrid_search.weighted_search(text, alpha, limit)

def rrf_search_query(query: str, k: int, limit: int, enhance: str, rerank_method: str):
    movie_data = load_movies()
    hybrid_search = HybridSearch(movie_data)
    return hybrid_search.rrf_search(query, k, limit, enhance, rerank_method)

def evalute_results(query: str, rrf_results: list):
    api_key = os.environ.get("rag-gemini-key")
    client = genai.Client(api_key=api_key)
    
    formatted_list = []
    for _, data in enumerate(rrf_results):
        formatted_list.append(str(data))

    response = client.models.generate_content(
                    model='gemini-2.5-flash', 
                    contents=f"""Rate how relevant each result is to this query on a 0-3 scale:

                Query: "{query}"

                Results:
                {", ".join(formatted_list)}

                Scale:
                - 3: Highly relevant
                - 2: Relevant
                - 1: Marginally relevant
                - 0: Not relevant

                Do NOT give any numbers out than 0, 1, 2, or 3.

                Return ONLY the scores in the same order you were given the documents. Return a valid JSON list, nothing else. For example:

                [2, 0, 3, 2, 0, 1]""")
    
    llm_parsed_response = response.text.replace("[","").replace("]","").split(",")
    for idx, rrf_data in enumerate(rrf_results,1):
        mov_title = rrf_data.get("title","")
        print(f"{idx}. {mov_title}: {llm_parsed_response[idx-1].strip()}/3")