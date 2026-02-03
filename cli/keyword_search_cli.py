#!/usr/bin/env python3

import argparse
from utils import load_movies, search_movies
from inverted_index import InvertedIndex

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser = subparsers.add_parser("build", help="Build an inversed index of the movies")
    search_parser.add_argument("query", nargs="?", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            data = load_movies()
            movies_result = search_movies(data, args.query)
            for idx, movie in enumerate(movies_result, 1):
                print(f"{idx}. {movie['title']}")
        
        case "build":
            inverted_idx = InvertedIndex()
            movie_data = load_movies()
            inverted_idx.build(movie_data)
            inverted_idx.save()

            id_set = inverted_idx.get_documents('merida')
            print(f"First document for token 'merida' = {id_set[0]}")
            
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()