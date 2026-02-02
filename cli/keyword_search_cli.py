#!/usr/bin/env python3

import argparse
from utils import load_movies, search_movies

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            data = load_movies()
            movies_result = search_movies(data, args.query)
            for idx, movie in enumerate(movies_result, 1):
                print(f"{idx}. {movie['title']}")
            
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()