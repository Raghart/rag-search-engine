from hybrid_search import normalize_data, weighted_search, rrf_search_query
import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser("normalize", help="normalize a list of floats numbers")
    normalize_parser.add_argument("num_array", type=float, nargs="+", help="list of array to normalize")

    weight_search_parser = subparsers.add_parser("weighted-search", help="Make a hybrid search of a text")
    weight_search_parser.add_argument("text", type=str, help="Text to be searched")
    weight_search_parser.add_argument("--alpha", type=float, nargs="?", default=0.5, help="number that controls the weight")
    weight_search_parser.add_argument("--limit", type=int, nargs="?", default=5, help="num that marks the limit of songs to retrieve")

    rrf_search_parser = subparsers.add_parser("rrf-search", help="make an rrf search of the following text")
    rrf_search_parser.add_argument("query", type=str, help="query to be searched using rrf-search")
    rrf_search_parser.add_argument("-k", type=int, nargs="?", default=60, help="K parameter")
    rrf_search_parser.add_argument("--limit", type=int, nargs="?", default=5, help="limit number of responses")
    rrf_search_parser.add_argument(
        "--enhance", type=str, 
        choices=["spell", "rewrite", "expand"], 
        help="Query enhancement method"
    )

    rrf_search_parser.add_argument(
        "--rerank-method", type=str,
        choices=["individual", "batch"],
        help="Apply rerank method to the search"
    )

    args = parser.parse_args()

    match args.command:
        case "normalize":
            nor_data = normalize_data(args.num_array)
            for score in nor_data:
                print(f"* {score:.4f}")
        
        case "weighted-search":
            print("Starting the weighted search...")
            weighted_results = weighted_search(args.text, args.alpha, args.limit)
            for idx, result in enumerate(weighted_results,1):
                print(f"{idx}. {result['doc']['title']}")
                print(f"BM25: {result['bm25_score']}, Semantic: {result['semantic_score']}")
                print(f"{result['doc']['description'][:100]}...")
        
        case "rrf-search":
            print("Starting the rrf-search...")
            search_results = rrf_search_query(
                args.query, 
                args.k, 
                args.limit, 
                args.enhance, 
                args.rerank_method
            )
            for idx, data in enumerate(search_results, 1):
                print(f"{idx}. {data['title']}")
                print(f"RRF Score: {data['rrf_score']:.4f}")
                print(f"BM25 Rank: {data['bm25_rank']}, Semantic Rank: {data['sem_rank']}")
                print(f"{data['description'][:100]}...\n")
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()