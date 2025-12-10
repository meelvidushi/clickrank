import bs4 as BeautifulSoup
import numpy as np
from urllib.request import urlopen
from urllib.parse import urljoin
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
import networkx as nx
import json
import argparse
import math

stemmer = SnowballStemmer("english")
newWebPages=[]

def stem_tokenizer(text):
    tokens = word_tokenize(text)
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    return " ".join(stemmed_tokens)


# Takes in a website url and recursively crawls until it
# has seen all pages linked on any of the pages.
# Inputs:
#   url: website url to begin crawling on
# Outputs:
#   index: a map from the url to

# Since we have external links we want to drop everything that isn't internal
validIPv4 = "130.215.143.215"


def indexWebsite(seed_url, verbose=False):

    to_visit = [seed_url]

    url_to_id = {}
    id_to_url = []
    adj = []      # adjacency list: list of sets
    texts = []    # text indexed by docid
    visited = set()

    def get_id(url):
        if url not in url_to_id:
            url_to_id[url] = len(id_to_url)
            id_to_url.append(url)
            adj.append(set())   # allocate adjacency list entry
            texts.append("")    # pre-allocate slot for text
        return url_to_id[url]

    # initialize seed node
    get_id(seed_url)

    while to_visit:
        url = to_visit.pop()
        doc_id = get_id(url)

        if url in visited:
            continue

        if verbose:
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

            if validIPv4 not in target:
                if verbose:
                    print(f"Skipping outlink: {target}")
                continue

            target_id = get_id(target)

            if verbose:
                print(f"Adding target_id: {target_id}")
                print(f"Adding outlink: {target}")
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
    try:
        with open(filepath) as f:
            data = json.load(f)
            return {url: set(tokens) for url, tokens in data.items()}
    except FileNotFoundError:
        return {}


if __name__ == "__main__":

    # Parse command line arguments

    parser = argparse.ArgumentParser(
        description="Website rankings for a given query")

    # Required positional argument
    parser.add_argument("query", type=str,
                        help="The query to be used for ranking items")

    # Optional flag
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose output")

    args = parser.parse_args()

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

    # Index the doccuments and get a text info and info for building the tf-idf ranking
    url_to_id, id_to_url, texts, adj = indexWebsite(
        "http://130.215.143.215/", verbose=args.verbose)

    if args.verbose:
        print(f"url_to_id: {url_to_id}")
        print(f"id_to_url: {id_to_url}")
        print(f"texts: {texts}")
        print(f"adj: {adj}")

    # query = "repairable car"
    query = args.query

    # TF-IDF COMPUTE
    # Get tf-idf scores across the doccuments
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(texts)

    stemmed_query = stem_tokenizer(query.lower())
    query_tfidf = vectorizer.transform([stemmed_query])

    # PAGERANK COMPUTE
    G = nx.DiGraph()

    # Adding edges from the indexer's outlinks for pagerank computation
    for i, outlinks in enumerate(adj):
        for j in outlinks:
            G.add_edge(i, j)

    pagerank_scores = nx.pagerank(G)  # returns dict {doc_index: score}

    text_space = set()
    for text in texts:
        for token in text.split(' '):
            text_space.add(token)

    vocab = list(text_space)
    index = {word: i for i, word in enumerate(vocab)}

    if args.verbose:
        print(f"Index {index}")

    # Load a common embedding model
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Get our current query vector
    query_vec = model.encode(query)

    # Import our query history
    query_history = load_queries('queries.json')

    # Go through our query history, vectorize all of the
    # queries, and sum them for each URL
    url_query_vects = {}
    for page in query_history:
        user_embedding = None
        for priorQuery in query_history[page]:
            if user_embedding is None:
                user_embedding = model.encode(priorQuery)
            else:
                # Query here hasn't been stemmed since we need to use semantic understanding
                queryEmbedding = model.encode(priorQuery)
                if args.verbose:
                    print(f"Query: {priorQuery}\nEmbedding: {queryEmbedding}")
                user_embedding = user_embedding + queryEmbedding

        url_query_vects[page] = user_embedding

    # Now we can go through and compute the score for each URL (we use 1 if
    # no data so we don't hurt things if we haven't seen that users want it
    # for another query type)
    normalRanking = {}
    ourRanking = {}

    if args.verbose:
        print("Rankings:")
    
    for website_id in range(len(texts)):
        url = id_to_url[website_id]

        if url.endswith("/"):  # If it's a directory
            continue

        # 1. TF-IDF cosine similarity for this specific document
        doc_tfidf_score = cosine_similarity(
            query_tfidf, tfidf_matrix[website_id])[0][0]

        # 2. PageRank score for this document
        pr_score = pagerank_scores[website_id]

        # 3. Prior query similarity score
        urlPageName = url.split('/')[-1]
        
        if args.verbose:
            print(f"urlPageName: {urlPageName}")            

        if urlPageName not in url_query_vects:
            newWebPages.append(urlPageName)
            custom_method_score = 0
        else:
            custom_method_score = util.cos_sim(
                query_vec, url_query_vects[urlPageName])[0][0]

        normalRanking[url] = doc_tfidf_score*pr_score
        # ourRanking[url] = doc_tfidf_score*pr_score*custom_method_score

        # We want to increase the impact of our prior query score
        alpha = 100
        ourRanking[url] = doc_tfidf_score * \
            pr_score*math.exp(alpha * custom_method_score)

        if args.verbose:
            print(f"{url} score: "
                  f"user-query: {custom_method_score}, "
                  f"tf-idf: {doc_tfidf_score}, "
                  f"pagerank: {pr_score}")



initial_sorting_ourRanking=sorted(ourRanking.items(), key=lambda x: x[1], reverse=True)

if newWebPages:
    thirdRankScore=(initial_sorting_ourRanking[1][1]+initial_sorting_ourRanking[2][1])/2
    if args.verbose:
        print(f"Assigning new pages a score of {thirdRankScore}. Rank 2 score is {initial_sorting_ourRanking[1]}, rank 3 score is {initial_sorting_ourRanking[2]}")
    for newPage in newWebPages:
        if args.verbose:
            print(f"New page detected: {newPage}")
        ourRanking[f"http://130.215.143.215/{newPage}"]=thirdRankScore

if args.verbose:
    print(initial_sorting_ourRanking)

print(f"Query: {query}")
print("Normal ranking:")
for url in sorted(normalRanking, key=normalRanking.get, reverse=True):
    print(f"Score {normalRanking[url]}, Page {url}")

print("Our ranking:")
for url in sorted(ourRanking, key=ourRanking.get, reverse=True):
    print(f"Score {ourRanking[url]}, Page {url}")