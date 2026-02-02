#!/usr/bin/env python3

import argparse, json

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            result = []

            with open("data/movies.json", "r") as f:
                data = json.load(f)
                
                for movie in data["movies"]:
                    if args.query in movie["title"]:
                        result.append(movie)
            
            result.sort(key=lambda x: x["id"])
            result = result[:5]
            
            for idx in range(len(result)):
                movie_title = result[idx]["title"]
                print(f"{idx + 1}. {movie_title}")
            
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()