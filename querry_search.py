import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from contextlib import closing
from collections import defaultdict
from inverted_index_gcp import MultiFileReader as MFReader
from inverted_index_gcp import InvertedIndex as InvertedIndex

TUPLE_SIZE = 6
TF_MASK = 2 ** 16 - 1 # Masking the 16 low bits of an integer
CORPUS_SIZE = 6348910 # Corpus size from assignment 3 GCP
BASE_PATH = '/home/chilikoa/'

# We download all the stopword from library, then we add our corpus stopword from assignment 3 GCP
nltk.download('stopwords')

RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){,24}""", re.UNICODE)

english_stopwords = frozenset(stopwords.words('english'))

corpus_stopwords = ["category", "references", "also", "external", "links",
                    "may", "first", "see", "history", "people", "one", "two",
                    "part", "thumb", "including", "second", "following",
                    "many", "however", "would", "became"]

all_stopwords = english_stopwords.union(corpus_stopwords)
print("Sasha allstopwords")
def tokenize(text):
  return [token.group() for token in RE_WORD.finditer(text.lower())]

def tokenize_query(query):
  """
  The function get text from user, tokenized it, filtered the query text from stopword and sort the token in alphabetic order.
  Args:
    query: text from user

  Returns: two sorted lists of tokens without stopwords and uniq tokenized query

  """
  tokenized_query = tokenize(query)
  filtered_sorted_query_tokens = sorted([word for word in tokenized_query if word not in all_stopwords])
  uniq_sorted_tokenized_query = sorted(list(set(filtered_sorted_query_tokens)))
  return filtered_sorted_query_tokens, uniq_sorted_tokenized_query


def read_posting_list(inverted, w, posting_list_path):
  with closing(MFReader()) as reader:
    locs = inverted.posting_locs[w]
    b = reader.read(locs, inverted.df[w] * TUPLE_SIZE, posting_list_path)
    posting_list = []
    for i in range(inverted.df[w]):
      doc_id = int.from_bytes(b[i*TUPLE_SIZE:i*TUPLE_SIZE+4], 'big')
      tf = int.from_bytes(b[i*TUPLE_SIZE+4:(i+1)*TUPLE_SIZE], 'big')
      posting_list.append((doc_id, tf))
    return posting_list

class QueryProcessor:
  def __init__(self):
    self.path = BASE_PATH
    self.body_inverted_index = None
    self.title_inverted_index = None
    self.anchor_inverted_index = None
    self.doc_length_dict = None  # contain a dictionary of {wiki_id: doc_length}
    self.doc_title_dict = None  # contain a dictionary of {wiki_id: doc_title}
    self.doc_norm_factor_dict = None # contain a dictionary of {wiki_id: doc_norm_factor} doc_norm_factor is sum of tf*idf square for each token in doc
    self.page_ranks = None  # converted from csv to dataFrame of two colums: doc_id and page_rank
    self.page_views = None  # dictionary {wiki_id: page_views}
    self.load_pre_calculeted_data()

  def load_pre_calculeted_data(self):
    # load all inverted indexs
    self.body_inverted_index = InvertedIndex.read_index(base_dir= self.path + 'text_postings_gcp',name='text_inverted_index')
    self.title_inverted_index = InvertedIndex.read_index(base_dir= self.path + 'title_postings_gcp',name='title_inverted_index')
    self.anchor_inverted_index = InvertedIndex.read_index(base_dir= self.path + 'anchor_postings_gcp', name='anchor_inverted_index')

    #load all csv and pkl files
    pkl_path = f'{self.path}index/doc_len_dict.pkl'
    self.doc_length_dict = pd.read_pickle(pkl_path)
    pkl_path= f'{self.path}index/doc_to_title.pkl'
    self.doc_title_dict = pd.read_pickle(pkl_path)
    pkl_path = f'{self.path}index/doc_id_nf.pkl'
    self.doc_norm_factor_dict = pd.read_pickle(pkl_path)
    pkl_path= f'{self.path}index/page_views.pkl'
    self.page_views = pd.read_pickle(pkl_path)
    page_rank_path = f'{self.path}page_rank/page_rank.csv.gz'
    self.page_ranks = pd.read_csv(page_rank_path, compression='gzip', header=None).rename({0: 'doc_id', 1: 'order'},
                                                                                          axis=1)


  def get_query_results_by_title(self, uniq_sorted_tokenized_query, is_from_frontend = False):
    """
    The function search if query tokens exist in wiki_doc_title
    Args:
      uniq_sorted_tokenized_query: list of query tokens

    Returns:
      list of [wiki_id, wiki_doc_title...]
    """
    title_inverted_index = self.title_inverted_index
    first_iteration = True
    for token in uniq_sorted_tokenized_query:
      token_posting_list = read_posting_list(inverted=title_inverted_index, w=token, posting_list_path=self.path+'/title_postings_gcp/')
      if first_iteration == True:
        first_iteration = False
        posting_dataframe = pd.DataFrame(token_posting_list).set_index(0).rename(columns={1: token})
      else:
        next_posting_dataframe = pd.DataFrame(token_posting_list).set_index(0).rename(columns={1: token})
        posting_dataframe = posting_dataframe.join(next_posting_dataframe, how='outer') #outer: form union of calling frame’s index (or column if on is specified) with other’s index, and sort it. lexicographically.

    posting_dataframe[~posting_dataframe.isna()] = 1 #replase a number with 1
    posting_dataframe.fillna(0, inplace=True) # replase a NA with 0

    posting_dataframe['order'] = posting_dataframe.sum(axis=1)
    posting_dataframe = posting_dataframe.sort_values(by='order', ascending=False, inplace=False)
    if is_from_frontend:
      ratings = list(posting_dataframe[['order']].itertuples(name=None))
      return ratings,posting_dataframe.index.tolist()
    return posting_dataframe.index.tolist()
      
  def doc_id_with_doc_titles(self, doc_id_lst):
    """
    The function get a list of wikipedia document_ids and return a list of dictionary of {doc_id: title}
    Args:
      doc_id_lst: list of doc_id_lst

    Returns: A list of dictionary of {key=doc_id, value=title}

    """
    doc_id_title_lst = []
    for doc_id in doc_id_lst:
      try:
        title = self.doc_title_dict[doc_id]
        doc_id_title_lst.append((doc_id, title))
      except Exception:
        pass
    return doc_id_title_lst

  def get_query_results_by_anchor(self, uniq_sorted_tokenized_query, is_from_frontend = False):
    """
    The function search if query tokens exist in wiki_doc_anchor
    Args:
      uniq_sorted_tokenized_query: list of query tokens

    Returns:
      list of [wiki_id, wiki_doc_title...]
    """
    anchor_inverted_index = self.anchor_inverted_index
    first_iteration = True
    for token in uniq_sorted_tokenized_query:
      token_posting_list = read_posting_list(inverted=anchor_inverted_index, w=token, posting_list_path=self.path+'/anchor_postings_gcp/')
      if first_iteration == True:
        first_iteration = False
        posting_dataframe = pd.DataFrame(token_posting_list).set_index(0).rename(columns={1: token})
      else:
        next_posting_dataframe = pd.DataFrame(token_posting_list).set_index(0).rename(columns={1: token})
        # outer: form union of calling frame’s index (or column if on is specified) with other’s index, and sort it. lexicographically.
        posting_dataframe = posting_dataframe.join(next_posting_dataframe, how='outer')

    posting_dataframe[~posting_dataframe.isna()] = 1 #replase a number with 1
    posting_dataframe.fillna(0, inplace=True) # replase a NA with 0

    posting_dataframe['order'] = posting_dataframe.sum(axis=1)
    posting_dataframe = posting_dataframe.sort_values(by='order', ascending=False, inplace=False)
    if is_from_frontend:
      ratings = list(posting_dataframe[['order']].itertuples(name=None))
      return ratings, posting_dataframe.index.tolist()
    return posting_dataframe.index.tolist()

  def get_query_results_by_body(self, uniq_sorted_tokenized_query, query_len, top_res = 100, is_from_frontend = False):
    """
    The function search for query tokens in wiki documnets, as we have tf for each token in the posting list
    we can calculate the tf*idf for each token in doc
    Args:
      uniq_sorted_tokenized_query: query uniq words
      query_len: query lengths
      top_res: number of results to return

    Returns: 100 search results, ordered from best to worst based on tf*idf

    """
    query_doc_similarity = defaultdict(int)
    for token in uniq_sorted_tokenized_query:
      token_posting_list = read_posting_list(inverted=self.body_inverted_index, w=token,
                                             posting_list_path=self.path+'/text_postings_gcp/')
      idf_token = np.log10(CORPUS_SIZE/self.body_inverted_index.df[token])
      for doc_id, tf in token_posting_list:
          query_doc_similarity[doc_id] += (tf/self.doc_length_dict[int(doc_id)]*idf_token)
    query_doc_similarity_score = defaultdict(int)
    for doc_id, tf_idf in query_doc_similarity.items():
      query_doc_similarity_score[doc_id] = tf_idf * (1/query_len) * self.doc_norm_factor_dict[doc_id]
    sim_score = sorted(query_doc_similarity_score.items(), key=lambda item: item[1], reverse=True)[:top_res]
    doc_ids = [tup[0] for tup in sim_score]
    if is_from_frontend:
      return sim_score, doc_ids
    return doc_ids


  def get_page_ranks_for_doc_ids(self, doc_id_lst):
    """
    The function get a list of wiki doc_id and return a list of doc_id with her page_rank
    Args:
      doc_id_lst: list of wiki documents

    Returns: list of wiki_ids with page_rank

    """
    doc_id_page_rank_lst = []
    for doc_id in doc_id_lst:
      try:
        page_rank = self.page_ranks[self.page_ranks.doc_id == doc_id]['page_rank'].value[0]
        doc_id_page_rank_lst.append(page_rank)
      except Exception:
        doc_id_page_rank_lst(0)
    return doc_id_page_rank_lst

  def query_search_combination(self, uniq_sorted_tokenized_query, query_len, top_body_res = 400, top_res = 100, body_weight = 0.2,
                               title_weight = 0.8, anchor_weight = 0.6):
    """
    The function combain a search results first by body what return best top_body_res results then it search by title and anchor.
    At the end, function sum the results by the weight of each part.
    Args:
      uniq_sorted_tokenized_query: query uniq words
      query_len: query length
      top_body_res: N best results from search by body
      top_res: N best results from search by title and anchor
      body_weight: body_weight
      title_weight: title_weihgt
      anchor_weight: anchor_weight

    Returns: best top_res, after combination of searching by body, title and anchor

    """
    body_results, body_doc_id_len = self.get_query_results_by_body(uniq_sorted_tokenized_query=uniq_sorted_tokenized_query,
                                                                   query_len=query_len, top_res=top_body_res,
                                                                   is_from_frontend=True)
    title_results, title_doc_id_len = self.get_query_results_by_title(uniq_sorted_tokenized_query=uniq_sorted_tokenized_query,
                                                                      is_from_frontend=True)
    anchor_results, anchor_doc_id_len = self.get_query_results_by_anchor(uniq_sorted_tokenized_query=uniq_sorted_tokenized_query,
                                                                         is_from_frontend=True)
    df_raiting_body = pd.DataFrame(body_results).set_index(0).rename(columns={1: 'body_top_results'})
    df_raiting_title = pd.DataFrame(title_results).set_index(0).rename(columns={1: 'title_top_results'})
    df_raiting_anchor = pd.DataFrame(anchor_results).set_index(0).rename(columns={1: 'anchor_top_results'})
    df_raiting_body = df_raiting_body.join(df_raiting_title, how='outer')
    df_raiting_body = df_raiting_body.join(df_raiting_anchor, how='outer')
    df_raiting_body.fillna(0, inplace=True)
    df_raiting_body['top_results'] = df_raiting_body.apply(lambda row: body_weight*row['body_top_results'] + title_weight*row['title_top_results'] + anchor_weight*row['anchor_top_results'], axis=1)
    return df_raiting_body.sort_values(by='top_results', ascending=False).index.tolist()[:top_res]




