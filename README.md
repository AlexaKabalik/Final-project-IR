# Final-project-IR
Team BP - Alexandra Kabalik ID 322159450 Gil Michalovich

This gitHub contains a several files:
1. small_inverted_index_gcp.py - was created for generation of inverted index based on queries_train, the inverted index was builded for body, title and anchor. This was done before working with all wikipedia data.
2. inverted_index_gcp - was created for generation of inverted index for all corpus, also based on body, title and anchor. Also, contain page_rank score and page_view score for queiry retrival. The code in this notebook is based on the assignment 3 with redirection to bucket folders to keep the bin and pkl file for query retrival.
3. run_fronted_gcp - implementation of query retrival, has a seach functions
4. querry_search - cointain all the logic for implementation of search function

liks:

url to our [Bucket](https://console.cloud.google.com/storage/browser/inverted_index_creation;tab=objects?forceOnBucketsSortingFiltering=false&organizationId=536124907474&project=alexandrakabalik&prefix=&forceOnObjectsSortingFiltering=false)
 containing all files and notebooks

url to our search engine http://35.192.154.146:8080/search?query=LinkedIn
