import argparse
from augmented_gen_funcs import rag_search, sum_search_query

def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    rag_sum_parser = subparsers.add_parser(
    "summarize", help="Let the LLM make a summary of a list of movies"
    )
    rag_sum_parser.add_argument("query", type=str, help="Search query for Summarize")
    rag_sum_parser.add_argument("--limit", nargs="?", type=int, default=5, help="Number of search results to retrieve")

    args = parser.parse_args()

    match args.command:
        case "rag":
            query = args.query
            search_results, llm_response = rag_search(query)
            print("Search Results:")
            for _, search_data in enumerate(search_results):
                print(f"    - {search_data['title']}")
            
            print("\nRAG Response:")
            print(f"{llm_response}")
        case "summarize":
            sum_search_results, llm_res = sum_search_query(args.query, args.limit)
            print("Search Results:")
            for _, data in enumerate(sum_search_results):
                print(f"    - {data['title']}")
            
            print("\nLLM Summary:")
            print(f"{llm_res}")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()