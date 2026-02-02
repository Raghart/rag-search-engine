import json, os, string

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "movies.json")

def load_movies():
    with open(DATA_PATH, "r") as f:
        data = json.load(f)
    return data["movies"]

def search_movies(data, query):
    result = []
    query_slice = query.split()
    filtered_query_slice = list(filter(lambda x: len(x) > 0, query_slice))

    for movie in data:
        for query_word in filtered_query_slice:
            if query_word in parse_movie_title(movie["title"]):
                result.append(movie)
    result.sort(key=lambda x: x["id"])
    return result[:5]

def parse_movie_title(title):
    return title.lower().translate(str.maketrans("", "", string.punctuation))