# build-search-engine-with-python

## Goal: Implement a search engine that answers 4 types of queries on Wikipedia articles with python 
1. one word query
2. Free text query
3. Phrase query
4. Boolean query

<li> Implement Query Ranking with TI-IDF
<li> Implement PageRank
 
  PageRank should only be used to rank queries if a user types in the following:
  python3 query.py --rank=pagerank …
  
  TF-IDF should be used to rank queries if a user types in either of the following:
  - python3 query.py --rank=tfidf stopwords.dat …
  - python3 query.py stopwords.dat …  
