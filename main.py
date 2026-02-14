import string
import nltk
from nltk.tokenize import TreebankWordTokenizer
from nltk.stem.porter import PorterStemmer

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

# example usage
example_doc = docs[0]
print("Original document:", example_doc)
print("Processed document:", tokenize_and_stem(example_doc))
