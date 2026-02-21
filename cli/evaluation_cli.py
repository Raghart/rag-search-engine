import argparse, json
from consts import GOLDEN_DATASET_PATH
from hybrid_search import rrf_search_query

def main():
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    args = parser.parse_args()
    limit = args.limit

    # run evaluation logic here

    K = 60

    with open(GOLDEN_DATASET_PATH, "r") as f:
        test_cases = json.load(f)["test_cases"]

    print(f"k={K}")
    
    for test in test_cases:
        query = test.get("query","")
        relevant_results = test.get("relevant_docs", [])

        search_titles = []
        search_results = rrf_search_query(query, K, limit, None, None)
        for result in search_results:
            search_titles.append(result.get("title",""))
        
        relevant_retrieved = 0
        for title in search_titles:
            if title in relevant_results:
                relevant_retrieved += 1
        
        precision = relevant_retrieved / len(search_results)
        print(f"- Query: {query}")
        print(f"    - Precision@{limit}: {precision:.4f}")
        print(f"    - Retrieved: {', '.join(search_titles)}")
        print(f"    - Relevant: {', '.join(relevant_results)}")
        print("\n")

if __name__ == "__main__":
    main()