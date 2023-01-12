import pandas as pd

from all_stopwords import RE_WORD, all_stopwords
from contextlib import closing
from inverted_index_gcp import MultiFileReader as MFReader
from inverted_index_gcp import MultiFileWriter as MFWriter
from inverted_index_gcp import InvertedIndex as InvertedIndex

TUPLE_SIZE = 6
TF_MASK = 2 ** 16 - 1 # Masking the 16 low bits of an integer
CORPUS_SIZE = 6348910 # Corpus size from assignment 3 GCP
BASE_PATH='/content/drive/MyDrive/final_project'

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
    self.page_ranks = None  # converted from csv to dataFrame of two colums: doc_id and page_rank
    self.page_views = None  # dictionary {wiki_id: page_views}
    self.load_pre_calculeted_indexs()

  def load_pre_calculeted_indexs(self):
    # temp = self.path + '/title_postings'
    self.title_inverted_index = InvertedIndex.read_index(base_dir= './title_postings', name='inverted_index')
    self.anchor_inverted_index = InvertedIndex.read_index(base_dir= self.path + '/anchor_postings', name='inverted_index')


  def get_query_results_by_title(self, uniq_sorted_tokenized_query):
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
      token_posting_list = read_posting_list(inverted=title_inverted_index, w=token, posting_list_path=self.path+'/title_postings/') #add path from bucket
      if first_iteration == True:
        posting_dataframe = pd.DataFrame(token_posting_list).set_index(0).rename(columns={1: token})
      else:
        next_posting_dataframe = pd.DataFrame(token_posting_list).set_index(0).rename(columns={1: token})
        posting_dataframe = next_posting_dataframe.join(next_posting_dataframe, how='outer') #outer: form union of calling frame’s index (or column if on is specified) with other’s index, and sort it. lexicographically.

    posting_dataframe[~next_posting_dataframe.isna()] = 1 #replase a number with 1
    posting_dataframe.fillna(0, inplace=True) # replase a NA with 0

    posting_dataframe['order'] = posting_dataframe.sum(axis=1)
    posting_dataframe = posting_dataframe.sort_values(by='order', ascending=False, inplace=False)
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

  def get_query_results_by_anchor(self, uniq_sorted_tokenized_query):
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
      token_posting_list = read_posting_list(inverted=anchor_inverted_index, w=token, posting_list_path=self.path+'/anchor_postings/') #add path from bucket
      if first_iteration == True:
        posting_dataframe = pd.DataFrame(token_posting_list).set_index(0).rename(columns={1: token})
      else:
        next_posting_dataframe = pd.DataFrame(token_posting_list).set_index(0).rename(columns={1: token})
        posting_dataframe = next_posting_dataframe.join(next_posting_dataframe, how='outer') #outer: form union of calling frame’s index (or column if on is specified) with other’s index, and sort it. lexicographically.

    posting_dataframe[~next_posting_dataframe.isna()] = 1 #replase a number with 1
    posting_dataframe.fillna(0, inplace=True) # replase a NA with 0

    posting_dataframe['order'] = posting_dataframe.sum(axis=1)
    posting_dataframe = posting_dataframe.sort_values(by='order', ascending=False, inplace=False)
    return posting_dataframe.index.tolist()


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









