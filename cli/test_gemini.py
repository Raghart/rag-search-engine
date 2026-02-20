import os
from dotenv import load_dotenv
from google import genai
from consts import TEST_TEXT

load_dotenv()
api_key = os.environ.get("rag-gemini-key")
print(f"Using key {api_key[:6]}...")
client = genai.Client(api_key=api_key)
response = client.models.generate_content(model='gemini-2.5-flash', contents=TEST_TEXT)
print(response.text)
print(f"Prompt Tokens: {response.usage_metadata.prompt_token_count}")
print(f"Response Tokens: {response.usage_metadata.candidates_token_count}")