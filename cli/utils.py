import json, os

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "movies.json")

def load_movies():
    with open(DATA_PATH, "r") as f:
        data = json.load(f)
    return data["movies"]

def search_movies(data, query):
    result = []
    for movie in data:
        if query in movie["title"].lower():
            result.append(movie)
    result.sort(key=lambda x: x["id"])
    return result[:5]