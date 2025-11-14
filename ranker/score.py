from bs4 import BeautifulSoup
import bs4 as BeautifulSoup
import numpy as np
from urllib.request import urlopen
from urllib.parse import urljoin
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import json

stemmer = SnowballStemmer("english")


def stem_tokenizer(text):
    tokens = word_tokenize(text)
    return [stemmer.stem(token) for token in tokens]


# Takes in a website url and recursively crawls until it
# has seen all pages linked on any of the pages.
# Inputs:
#   url: website url to begin crawling on
# Outputs:
#   index: a map from the url to


def indexWebsite(seed_url):

    to_visit = [seed_url]

    url_to_id = {}
    id_to_url = []
    adj = []      # adjacency list: list of sets
    texts = {}    # doc_id -> raw text
    visited = set()

    def get_id(url):
        if url not in url_to_id:
            url_to_id[url] = len(id_to_url)
            id_to_url.append(url)
            adj.append(set())   # allocate adjacency list entry
        return url_to_id[url]

    # initialize seed node
    get_id(seed_url)

    while to_visit:
        url = to_visit.pop()
        doc_id = get_id(url)

        if url in visited:
            continue

        print(f"Visiting url: {url}")

        try:
            html = urlopen(url).read()
            soup = BeautifulSoup.BeautifulSoup(html, "html.parser")
        except Exception as e:
            print(f"Error fetching {url}: {e}")
            visited.add(url)
            continue

        visited.add(url)

        # store text for tf-idf later and tokenize it so we can
        # use it for the query and tf-idf
        texts[doc_id] = stem_tokenizer(soup.get_text().lower())

        # extract outlinks
        outlinks = set()
        for link in soup.find_all("a", href=True):
            target = urljoin(url, link["href"])
            target_id = get_id(target)

            print(f"Adding target_id: {target_id}")
            print(f"Adding outlink: {target_id}")
            outlinks.add(target_id)

            if target not in visited and target not in to_visit:
                to_visit.append(target)

        # Save adjacency list
        adj[doc_id] = outlinks

    return url_to_id, id_to_url, texts, adj


def normalize(x):
    return (x - x.min()) / (x.max() - x.min())


# This works in a bag of words way so repeating
# words in the query doesn't change anything
def vectorize(tokens, index, vocab_size):
    vec = np.zeros(vocab_size, dtype=np.int32)
    for tok in tokens:
        if tok in index:
            vec[index[tok]] = 1
    return vec


def save_query(filepath, url, stemmed_query):
    stemmed = stemmed_query

    try:
        with open(filepath, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        data = {}

    data[url] = list(stemmed)

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)


def load_queries(filepath):
    with open(filepath) as f:
        data = json.load(f)
    return {url: set(tokens) for url, tokens in data.items()}


if __name__ == "__main__":

    # We need to ensure that we have the tokenizer downloaded
    import nltk

    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")

    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab")

    # Query
    query = "repairable car"
    stemmed_query = stem_tokenizer(query.lower())

    # Index the doccuments and get a text info and info for building the tf-idf ranking
    url_to_id, id_to_url, texts, adj = indexWebsite(
        "http://130.215.143.215/doc0.html")

    print(f"url_to_id: {url_to_id}")
    print(f"id_to_url: {id_to_url}")
    print(f"texts: {texts}")
    print(f"adj: {adj}")

    # # TF-IDF COMPUTE
    # # Get tf-idf scores across the doccuments
    # vectorizer = TfidfVectorizer(stop_words='english')
    # tfidf_matrix = vectorizer.fit_transform(texts)
    #
    # # Transform the query using the same fitted vectorizer
    # query_tfidf = vectorizer.transform(query.split(' '))
    #
    # content_scores = cosine_similarity(query_tfidf, tfidf_matrix)
    # # content_scores = cosine_similarity(query_tfidf, tfidf_matrix).flatten()
    # content_scores_norm = normalize(content_scores)
    # print(content_scores)
    #
    # # PAGERANK COMPUTE
    # G = nx.DiGraph()
    #
    # # Adding edges from the indexer's outlinks for pagerank computation
    # for i, outlinks in enumerate(adj):
    #     for j in outlinks:
    #         G.add_edge(i, j)
    #
    # # pagerank_scores = nx.pagerank(G)  # returns dict {doc_index: score}
    # pagerank_scores = nx.pagerank_scipy(G)  # returns dict {doc_index: score}
    # pagerank_scores = np.array([pagerank_scores[i] for i in range(len(texts))])
    #
    # pagerank_scores_norm = normalize(pagerank_scores)

    # Compute our metric. TEMP for now

    # TODO: this assumes that we only get query words within the words in our text space from the scraper
    # Get our text-space
    text_space = set()
    for key in texts:
        for token in texts[key]:
            text_space.add(token)

    vocab = list(text_space)
    index = {word: i for i, word in enumerate(vocab)}

    print(index)

    # Get our current query vector
    query_vec = vectorize(stemmed_query, index, len(vocab))

    # Import our query history
    query_history = load_queries('queries.json')

    # Go through our query history, vectorize all of the
    # queries, and sum them for each URL
    url_query_vects = {}
    for url in query_history:
        combined_vect = np.zeros(len(vocab), dtype=np.int32)
        for query in query_history[url]:
            # We assume the query here (imported) is post-stemming
            combined_vect = combined_vect + vectorize(query, index, len(vocab))

        url_query_vects[url] = combined_vect

    # Now we can go through and compute the score for each URL (we use 1 if
    # no data so we don't hurt things if we haven't seen that users want it
    # for another query type)

    # Now we compute the actual ranking
    for website_id in texts:
        url = id_to_url[website_id]

        if url in url_query_vects:
            custom_method_score = cosine_similarity(query_vec, )[0][0]
        else:
            custom_method_score = 1
