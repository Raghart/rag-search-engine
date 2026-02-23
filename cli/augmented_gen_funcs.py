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

def sum_search_query(query: str, limit: int):
    search_results = rrf_search_query(query, 60, limit, None, None)
    api_key = os.environ.get("rag-gemini-key")
    client = genai.Client(api_key=api_key)
    prompt = f"""
        Provide information useful to this query by synthesizing information from multiple search results in detail.
        The goal is to provide comprehensive information so that users know what their options are.
        Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.
        This should be tailored to Hoopla users. Hoopla is a movie streaming service.
        Query: {query}
        Search Results:
        {search_results}
        Provide a comprehensive 3–4 sentence answer that combines information from multiple sources:
        """
    
    response = client.models.generate_content(
        model='gemini-2.5-flash', 
        contents=prompt
    )

    return search_results, response.text