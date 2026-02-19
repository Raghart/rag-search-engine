from hybrid_search import normalize_data
import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser("normalize", help="normalize a list of floats numbers")
    normalize_parser.add_argument("num_array", type=float, nargs="+", help="list of array to normalize")

    args = parser.parse_args()

    match args.command:
        case "normalize":
            nor_data = normalize_data(args.num_array)
            for score in nor_data:
                print(f"* {score:.4f}")
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()