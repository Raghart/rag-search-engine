#!/usr/bin/env python3

import argparse
from lib.semantic_search import verify_model, embed_text, verify_embeddings, embed_query_text
from lib.semantic_search import semantic_search

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

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()