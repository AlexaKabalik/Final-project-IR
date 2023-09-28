
# Overview
This repository contains a Python-based search engine that indexes and searches the entire Wikipedia corpus using Google Cloud Platform (GCP) and PySpark. The search engine allows users to efficiently search for articles and retrieve relevant information from the vast Wikipedia dataset.

## Table of Contents
Features
Getting Started
Prerequisites
Installation
Usage
Architecture
Data Pipeline
Contributing
License
## Features
 Efficient Indexing: Indexes the entire Wikipedia corpus for fast and accurate search results.
 Powerful Search: Provides a robust search capability for querying articles.
 Scalable Architecture: Leverages PySpark and GCP services for scalability.
 Detailed Documentation: Offers clear documentation and code comments for easy understanding.
## Prerequisites
Before you start, ensure you have met the following requirements:

Python 3.x installed.
A Google Cloud Platform (GCP) account with a project set up.
A GCP Storage bucket for storing Wikipedia data.
PySpark and other required libraries installed.
Dependencies listed in requirements.txt.
Getting Started
To get started with this project, follow these steps:

Clone the Repository:

bash
Copy code
git clone https://github.com/AlexaKabalik/Final-project-IR.git
cd Final-project-IR
Set Up GCP:

Create a GCP project.
Set up a GCP Storage bucket for storing Wikipedia data.
Configure Credentials:

Update the configuration file (config.yaml) with your GCP credentials and bucket details.

Download and Prepare Wikipedia Corpus:

Download the Wikipedia corpus or a subset of it.
Preprocess and format the data as needed for indexing.
Indexing:

Use PySpark to run the indexing process to create the search index.

## Deployment:

Deploy the search engine and interact with it.

## Usage
Detailed instructions on how to use the search engine can be found in the project's documentation. You can run queries and retrieve information from the indexed Wikipedia corpus.

bash
Copy code
python search.py "your_query_here"
Architecture
Architecture Diagram

The search engine utilizes a scalable architecture that leverages GCP services for data storage and processing. PySpark is used for efficient indexing, and the search engine itself is deployed on a suitable platform.

## Data Pipeline
The data pipeline consists of multiple stages, including data acquisition, preprocessing, indexing, and querying. These stages are orchestrated to ensure that the Wikipedia corpus is efficiently transformed into a searchable index.
## Code Architechture:
1. small_inverted_index_gcp.py - was created for generation of inverted index based on queries_train, the inverted index was builded for body, title and anchor. This was done before working with all wikipedia data.
2. inverted_index_gcp - was created for generation of inverted index for all corpus, also based on body, title and anchor. Also, contain page_rank score and page_view score for queiry retrival. The code in this notebook is based on the assignment 3 with redirection to bucket folders to keep the bin and pkl file for query retrival.
3. run_fronted_gcp - implementation of query retrival, has a seach functions
4. querry_search - cointain all the logic for implementation of search function
5. search_fronted - implementation of search functions for documents retival

Team BP - Alexandra Kabalik ID 322159450 Gil Michalovich 315041426


## Links:

url to our [Bucket](https://console.cloud.google.com/storage/browser/inverted_index_creation;tab=objects?forceOnBucketsSortingFiltering=false&organizationId=536124907474&project=alexandrakabalik&prefix=&forceOnObjectsSortingFiltering=false)
 containing all files and notebooks

## Team BP - Alexandra Kabalik ID 322159450 Gil Michalovich 315041426
# url to our search engine http://35.192.154.146:8080/search?query=LinkedIn


