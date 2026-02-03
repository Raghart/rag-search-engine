#!/usr/bin/env python3

import argparse
from utils import load_movies, search_movies
from inverted_index import build_inverted_idx

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    subparsers = subparsers.add_parser("build", help="Build an inversed index of the movies")
    search_parser.add_argument("query", type=str, help="Search query")
    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            data = load_movies()
            movies_result = search_movies(data, args.query)
            for idx, movie in enumerate(movies_result, 1):
                print(f"{idx}. ({movie['id']}) {movie['title']}")
        
        case "build":
            print("building inverted index...")
            build_inverted_idx()
            print("the building of the inverted index was sucessful!")
            
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()