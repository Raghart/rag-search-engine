#!/usr/bin/env python3

import argparse
from lib.semantic_search import verify_model, embed_text

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="avaible commands")
    subparsers.add_parser("verify", help="verify if the model is working correctly")

    embed_text_parser = subparsers.add_parser("embed_text", help="Command to embed a single string")
    embed_text_parser.add_argument("text", type=str, help="text to be embbeded")
    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        
        case "embed_text":
            print(f"embedding the text: '{args.text}'")
            embed_text(args.text)

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()