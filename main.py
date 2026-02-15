import string
import numpy as np
import nltk
from nltk.tokenize import TreebankWordTokenizer
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# toy documents
docs = [
    '''About us. We deliver Artificial Intelligence & Machine Learning
       solutions to solve business challenges.''',
    '''Contact information. Email [martin davtyan at filament dot ai]
       if you have any questions''',
    '''Filament Chat. A framework for building and maintaining a scalable
       chatbot capability''',
]

# punctuation, tokenization, and stemming setup
REMOVE_PUNCTUATION_TABLE = str.maketrans({x: None for x in string.punctuation})
TOKENIZER = TreebankWordTokenizer()
STEMMER = PorterStemmer()

# funtion to tokenize and stem a document
def tokenize_and_stem(doc):
    return [STEMMER.stem(token) for token in TOKENIZER.tokenize(doc.translate(REMOVE_PUNCTUATION_TABLE))]

# vectorizer setup
vectorizer = TfidfVectorizer(tokenizer=tokenize_and_stem, stop_words='english')
vectorizer.fit(docs)
doc_vectors = vectorizer.transform(docs)


# example usage
example_doc = docs[0]
print("Original document:", example_doc)
print("Processed document:", tokenize_and_stem(example_doc))
print("TF-IDF vector for all documents:", vectorizer.vocabulary_)

query = "contact email to chat martin"
query_vector = vectorizer.transform([query]).todense() # convert query to TF-IDF vector
print(query_vector)

similarity = cosine_similarity(np.asarray(query_vector), doc_vectors) # calculate cosine similarity between query and documents
print(similarity)
ranks = (-similarity).argsort(axis=None) # sort documents by similarity to the query
print(ranks)
print(docs[ranks[0]]) # printing most relevant document to the query