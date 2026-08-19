import string
import numpy as np
import nltk
from nltk.tokenize import TreebankWordTokenizer
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

# example documents
docs = [
    '''About us. We deliver Artificial Intelligence & Machine Learning
       solutions to solve business challenges.''',
    '''Contact information. Email [martin davtyan at filament dot ai]
       if you have any questions''',
    '''Filament Chat. A framework for building and maintaining a scalable
       chatbot capability''',
]

# example feedbacks
feedback = {
        'who makes chatbots': [(2, 0.), (0, 1.), (1, 1.), (0, 1.)],
        'about page': [(0, 1.)]
}

# punctuation, tokenization, and stemming setup
REMOVE_PUNCTUATION_TABLE = str.maketrans({x: None for x in string.punctuation})
TOKENIZER = TreebankWordTokenizer()
STEMMER = PorterStemmer()

# funtion to tokenize and stem a document
def tokenize_and_stem(doc):
    return [STEMMER.stem(token) for token in TOKENIZER.tokenize(doc.translate(REMOVE_PUNCTUATION_TABLE))]


#---------------------------------------------------------------------------------------------------------

# vectorizer setup
vectorizer = TfidfVectorizer(tokenizer=tokenize_and_stem, stop_words='english')
vectorizer.fit(docs)
doc_vectors = vectorizer.transform(docs)


# example usage
example_doc = docs[0]
print("Original document:", example_doc)
print("Processed document:", tokenize_and_stem(example_doc))
print("TF-IDF vector for all documents:", vectorizer.vocabulary_)

# sample query
query = "contact email to chat martin"
query_vector = vectorizer.transform([query]).todense() # convert query to TF-IDF vector
print(query_vector)

similarity = cosine_similarity(np.asarray(query_vector), doc_vectors) # calculate cosine similarity between query and documents
print(similarity)
ranks = (-similarity).argsort(axis=None) # sort documents by similarity to the query
print(ranks)
print(docs[ranks[0]]) # printing most relevant document to the query



similarity = cosine_similarity(vectorizer.transform(['who makes chatbots']), doc_vectors)
ranks = (-similarity).argsort(axis=None)
print(ranks)
print(docs[ranks[0]])


query_1 = "who is making chatbots information"
feedback_queries = list(feedback.keys())

similarity_1 = cosine_similarity(vectorizer.transform([query_1]), vectorizer.transform(feedback_queries))

print(similarity_1)

max_idx = np.argmax(similarity_1)
print(feedback_queries[max_idx])



pos_feedback_doc_idx = [idx for idx, feedback_value in feedback[feedback_queries[max_idx]] if feedback_value == 1.]
print(pos_feedback_doc_idx)

counts = Counter(pos_feedback_doc_idx)
print(counts)

pos_feedback_proportions = {doc_idx: count / sum(counts.values()) for doc_idx, count in counts.items()} #{doc_idx:  for doc_idx, count in counts.items()}
print(pos_feedback_proportions)

nn_similarity = np.max(similarity_1)
pos_feedback_feature = [nn_similarity * pos_feedback_proportions.get(idx, 0.) for idx, _ in enumerate(docs)]
print(pos_feedback_feature)


class Scorer():
    """ Scores documents for a search query based on tf-idf similarity and relevance feedback"""

    def __init__(self, docs):
        """ Initialize a scorer with a collection of documents, fit a vectorizer and list feature functions"""
        
        self.docs = docs
        
        self.vectorizer = TfidfVectorizer(tokenizer=tokenize_and_stem, 
                                          stop_words='english')
        self.doc_tfidf = self.vectorizer.fit_transform(docs)
        
        self.features = [
            self._feature_tfidf,
            self._feature_positive_feedback,
        ]
        self.feature_weights = [
            1.,
            2.,
        ]
        
        self.feedback = {}
        
    def score(self, query):
        """ Generic scoring function: for a query output a numpy array
            of scores aligned with a document list we initialized the
            scorer with
        
        """
        feature_vectors = [feature(query) for feature 
                           in self.features]
        
        feature_vectors_weighted = [feature * weight for feature, weight
                                    in zip(feature_vectors, self.feature_weights)]
        return np.sum(feature_vectors_weighted, axis=0)
    
    def learn_feedback(self, feedback_dict):
        """ Learn feedback in a form of `query` -> (doc index, feedback value).
            In real life it would be an incremental procedure updating the
            feedback object.
        
        """
        self.feedback = feedback_dict
        
    def _feature_tfidf(self, query):
        """ TF-IDF feature. Return a numpy array of cosine similarities
            between TF-IDF vectors of documents and the query
        
        """
        query_vector = vectorizer.transform([query])
        similarity = cosine_similarity(query_vector, self.doc_tfidf)
        return similarity.ravel()
    
    def _feature_positive_feedback(self, query):
        """ Positive feedback feature. Search the feedback dict for a query
            similar to the given one, then assign documents positive values
            if there is positive feedback about them.
        
        """
        if not self.feedback:
            return np.zeros(len(self.docs))
        
        feedback_queries = list(self.feedback.keys())
        similarity = cosine_similarity(self.vectorizer.transform([query]),
                                       self.vectorizer.transform(feedback_queries))
        nn_similarity = np.max(similarity)
        
        nn_idx = np.argmax(similarity)
        pos_feedback_doc_idx = [idx for idx, feedback_value in
                                self.feedback[feedback_queries[nn_idx]]
                                if feedback_value == 1.]
        
        feature_values = {
                doc_idx: nn_similarity * count / sum(counts.values()) 
                for doc_idx, count in Counter(pos_feedback_doc_idx).items()
        }
        return np.array([feature_values.get(doc_idx, 0.) 
                         for doc_idx, _ in enumerate(self.docs)])