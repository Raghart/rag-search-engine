import json, os, string
from nltk.stem import PorterStemmer

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "movies.json")
STOPWORDS_PATH = os.path.join(PROJECT_ROOT, "data", "stopwords.txt")


def load_movies():
    with open(DATA_PATH, "r") as f:
        data = json.load(f)
    return data["movies"]

def load_stopwords():
    with open(STOPWORDS_PATH, "r") as f:
        stop_words = f.read().splitlines()
    return stop_words

STOPWORD_LIST = load_stopwords()

def search_movies(data, query):
    result = []
    tokenized_query = tokenize_text(query)

    for movie in data:
        tokenized_title = tokenize_text(movie["title"])
        if has_matching_token(tokenized_query, tokenized_title):
            result.append(movie)

    result.sort(key=lambda x: x["id"])
    return result[:5]

def has_matching_token(query_tokens, movie_tokens):
    for movie_word in movie_tokens:
        for query in query_tokens:
            if query in movie_word:
                return True
    return False

def tokenize_text(title: str):
    parsed_movie = parse_movie_title(title)
    movie_tokens = parsed_movie.split()
    steammer = PorterStemmer()
    filtered_tokens = list(filter(lambda word: word and word not in STOPWORD_LIST, movie_tokens))
    
    stemmed_tokens = []
    for token in filtered_tokens:
        stemmed_tokens.append(steammer.stem(token))
    return stemmed_tokens

def parse_movie_title(title):
    return title.lower().translate(str.maketrans("", "", string.punctuation))