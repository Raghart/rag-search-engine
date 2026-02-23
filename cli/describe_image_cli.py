import argparse, mimetypes, os
from google import genai
from google.genai import types
from dotenv import load_dotenv

def main():
    parser = argparse.ArgumentParser(description="Photo search CLI")
    load_dotenv()
    parser.add_argument("--image", type=str, help="path to the image to analyze")
    parser.add_argument("--query", type=str, help="query describing the image")
    args = parser.parse_args()
    
    if len(args.image.strip()) == 0 or len(args.query) == 0:
        parser.print_help()
        return
    
    mime, _ = mimetypes.guess_type(args.image)
    mime_type = mime or "image/jpeg"
    with open(args.image, "rb") as f:
        img_data = f.read()
    
    api_key = os.environ.get("rag-gemini-key")
    client = genai.Client(api_key=api_key)
    system_prompt = """
        Given the included image and text query, rewrite the text query to improve search results from a movie database. Make sure to:
        - Synthesize visual and textual information
        - Focus on movie-specific details (actors, scenes, style, etc.)
        - Return only the rewritten query, without any additional commentary
    """
    
    parts = [
        system_prompt,
        types.Part.from_bytes(data=img_data, mime_type=mime_type),
        args.query.strip(),
    ]
    
    response = client.models.generate_content(
        model='gemini-2.5-flash', 
        contents=parts
    )

    print(f"Rewritten query: {response.text.strip()}")
    if response.usage_metadata is not None:
        print(f"Total tokens:    {response.usage_metadata.total_token_count}")

if __name__ == "__main__":
    main()