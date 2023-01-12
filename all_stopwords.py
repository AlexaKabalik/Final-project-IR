import re
import nltk
from nltk.corpus import stopwords
# We download all the stopword from library, then we add our corpus stopword from assignment 3 GCP
nltk.download('stopwords')

RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){,24}""", re.UNICODE)

english_stopwords = frozenset(stopwords.words('english'))

corpus_stopwords = ["category", "references", "also", "external", "links",
                    "may", "first", "see", "history", "people", "one", "two",
                    "part", "thumb", "including", "second", "following",
                    "many", "however", "would", "became"]

all_stopwords = english_stopwords.union(corpus_stopwords)
# print(all_stopwords)
