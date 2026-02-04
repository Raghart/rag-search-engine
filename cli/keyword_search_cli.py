#!/usr/bin/env python3

import argparse
from inverted_index import build_inverted_idx, search_movies, search_term_frequencies
from inverted_index import calculate_idf, calculate_tfidf, calculate_bm25_idf, calculate_bm25_tf
from consts import BM25_K1

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    subparsers.add_parser("build", help="Build an inversed index of the movies")

    tf_parser = subparsers.add_parser("tf", help="count the number of times a term appear")
    tf_parser.add_argument("doc_id", type=int, help="movie ID number")
    tf_parser.add_argument("term", type=str, help="word to search")

    idf_parser = subparsers.add_parser("idf", help="calculate de inverse document frequency of a word")
    idf_parser.add_argument("term", type=str, help="Word to calculate")

    tfidf_parser = subparsers.add_parser("tfidf", help="calculate the tf-idf of a single word")
    tfidf_parser.add_argument("doc_id", type=int, help="document ID")
    tfidf_parser.add_argument("term", type=str, help="term used for the calculation")

    bm25_idf_parser = subparsers.add_parser("bm25idf", help="Get BM25 IDF score for a given term")
    bm25_idf_parser.add_argument("term", type=str, help="Term to get BM25 IDF score for")

    bm25_tf_parser = subparsers.add_parser("bm25tf", help="Get BM25 TF score for a given document ID and term")
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument("k1", type=float, nargs="?", default=BM25_K1, help="Tunable BM25 K1 parameter")

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
            num_times = search_term_frequencies(args.doc_id, args.term)
            print(f"Times repeated: {num_times}")
        
        case "idf":
            print(f"calculating the idf for the word: '{args.term}'...")
            idf = calculate_idf(args.term)
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")
        
        case "tfidf":
            print(f"calculating the tf-idf for the word: '{args.term}'...")
            tf_idf_score = calculate_tfidf(int(args.doc_id), args.term)
            print(f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf_score:.2f}")
        case "bm25idf":
            print(f"calculating the BM25 IDF score for the word: '{args.term}'")
            bm25_score = calculate_bm25_idf(args.term)
            print(f"BM25 IDF score of '{args.term}': {bm25_score:.2f}")
        
        case "bm25tf":
            print(f"Calculating the BM25 score of {args.term} of the document ({args.doc_id})...")
            bm25_tf_score = calculate_bm25_tf(args.doc_id, args.term, args.k1)
            print(f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25_tf_score:.2f}")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()