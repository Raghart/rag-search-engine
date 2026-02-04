#!/usr/bin/env python3

import argparse
from inverted_index import build_inverted_idx, search_movies, search_term_frequencies

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    subparsers.add_parser("build", help="Build an inversed index of the movies")

    tf_parser = subparsers.add_parser("tf", help="count the number of times a term appear")
    tf_parser.add_argument("id", type=int, help="movie ID number")
    tf_parser.add_argument("term", type=str, help="word to search")

    search_parser.add_argument("query", type=str, help="Search query")
    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            arr_result = search_movies(args.query)
            for idx, movie in enumerate(arr_result, 1):
                print(f"{idx}. ({movie['id']}) {movie['title']}")

        case "build":
            print("building inverted index...")
            build_inverted_idx()
            print("the building of the inverted index was sucessful!")
        
        case "tf":
            print(f"getting the number of times {args.term} is repeated...")
            num_times = search_term_frequencies(args.id, args.term)
            print(f"Times repeated: {num_times}")
            
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()