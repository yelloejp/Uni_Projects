# Description

Key: Graph-based, Pytorch-geometric, nltk, word embedding, GraphSAGE

- The dataset provides a list of text documents and its labels. (numbers of documents = 74496, numbers of authors = 5)
- For Graph-based approach, we build a graph from the dataset. 
    - node = each document
    - edge = the number of common words between two documents 
    - feature of document = sum of word embeddings that a document has 
 - We assume that documents of an author have similar features and we want to aggregate its features from its neighbors documents. 
 - In GraphSAGE CNNs, each document aggregates its features from its random neighbors. 
