import json, os, string
from nltk.stem import PorterStemmer
from consts import DATA_PATH, STOPWORDS_PATH

def load_movies():
    with open(DATA_PATH, "r") as f:
        data = json.load(f)
    return data["movies"]

def load_stopwords():
    with open(STOPWORDS_PATH, "r") as f:
        stop_words = f.read().splitlines()
    return stop_words

STOPWORD_LIST = load_stopwords()

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