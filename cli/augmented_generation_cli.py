import argparse
from augmented_gen_funcs import rag_search

def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

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
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()