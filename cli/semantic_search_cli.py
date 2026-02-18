#!/usr/bin/env python3

import argparse
from lib.semantic_search import verify_model, embed_text, verify_embeddings, embed_query_text
from lib.semantic_search import semantic_search, chunk_text, semantic_chunk, embed_movie_chunks
from lib.semantic_search import search_chunk_text

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="avaible commands")
    subparsers.add_parser("verify", help="verify if the model is working correctly")

    embed_text_parser = subparsers.add_parser("embed_text", help="Command to embed a single string")
    embed_text_parser.add_argument("text", type=str, help="text to be embbeded")

    subparsers.add_parser("verify_embeddings", help="verify the embeddings in the movies.json")

    embed_query_parser = subparsers.add_parser("embedquery", help="command to embed a query")
    embed_query_parser.add_argument("query", type=str, help="query to be embedded")

    semantic_search_parser = subparsers.add_parser("search", help="semantic search a query")
    semantic_search_parser.add_argument("query", type=str, help="query to be searched using semantic search")
    semantic_search_parser.add_argument("--limit", type=int, nargs="?", default=5, help="limit the number of songs received")

    chunk_parser = subparsers.add_parser("chunk", help="chunk a text size into smaller pieces")
    chunk_parser.add_argument("text", type=str, help="text to be chunked")
    chunk_parser.add_argument("--chunk-size", type=int, nargs="?", default=200, help="number of characters to chunk")
    chunk_parser.add_argument("--overlap", type=int, nargs="?", default=0, help="number of words to overlap")

    semantic_chunk_parser = subparsers.add_parser("semantic_chunk", help="semantic chunk a text")
    semantic_chunk_parser.add_argument("text", type=str, help="text to be semantic chunked")
    semantic_chunk_parser.add_argument("--max-chunk-size", type=int, nargs="?", default=4, help="max number of chunks")
    semantic_chunk_parser.add_argument("--overlap", type=int, nargs="?", default=0, help="num of overlap for each chunk")

    subparsers.add_parser("embed_chunks", help="embed the movies.json")

    search_chunk_parser = subparsers.add_parser("search_chunked", help="search movies using chunked")
    search_chunk_parser.add_argument("text", type=str, help="text to search in the DB")
    search_chunk_parser.add_argument("--limit", type=int, nargs="?", default=5, help="set the limit of results")

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        
        case "embed_text":
            print(f"embedding the text: '{args.text}'")
            embed_text(args.text)

        case "verify_embeddings":
            print("verifying the embeddings in the movies.json file...")
            verify_embeddings()
        
        case "embedquery":
            print(f"embedding query for the text: '{args.query}'")
            embed_query_text(args.query)
        
        case "search":
            print(f"searching using semantic search on the query: {args.query}")
            results = semantic_search(args.query, args.limit)
            for num, query_result in enumerate(results, 1):
                print(f"{num}. {query_result[1]['title']} (score: {query_result[0]})")
                print(f"{query_result[1]['description']}\n")

        case "chunk":
            print(f"Starting the chunking process with {args.chunk_size}...")
            chunk_text(args.text, args.chunk_size, args.overlap)
        
        case "semantic_chunk":
            print(f"Starting the semantic chunk process...")
            chunked_arr = semantic_chunk(args.text, args.max_chunk_size , args.overlap)
            for idx, text_chunked in enumerate(chunked_arr, 1):
                print(f"{idx}. {text_chunked}")
        
        case "embed_chunks":
            print("Starting the embedding chunk process...")
            embeddings = embed_movie_chunks()
            print(f"Generated {len(embeddings)} chunked embeddings")

        case "search_chunked":
            print(f"Starting search for the text: {args.text}")
            search_chunk_text(args.text, args.limit)

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()