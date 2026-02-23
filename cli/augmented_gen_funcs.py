from hybrid_search import rrf_search_query
from google import genai
import os

def rag_search(query: str):
    movie_results = rrf_search_query(query, 60, 5, None, None)
    api_key = os.environ.get("rag-gemini-key")
    client = genai.Client(api_key=api_key)
    prompt = f"""Answer the question or provide information based on the provided documents. This should be tailored to Hoopla users. Hoopla is a movie streaming service.

    Query: {query}

    Documents:
    {movie_results}

    Provide a comprehensive answer that addresses the query:"""
    
    response = client.models.generate_content(
        model='gemini-2.5-flash', 
        contents=prompt
    )
    return movie_results, response.text