# Description

Key: Graph-based, Pytorch-geometric, nltk, word embeddings, GraphSAGE, GraphSAINT, TextGCN, Sampling

Approach 1) Stochastic aggregation
- Instead of aggregating features from all neighbors, it aggregates features from sample nodes. 
- After preprocessing(removing stopwords and less frequent words), it encodes words into pre-trained embeddings as node features. Then, it builds single graph with words and documents as nodes and weights as edges. 
![StochasticAggregation](https://user-images.githubusercontent.com/45250729/85316673-697f7200-b4bd-11ea-9aa9-f42b7c41005e.png)

Approach 2) Ensemble aggregation 
- Instead of training entire single graph, it generates several smaller graphs from a corpus. Preprocessing is same with Approach 1. But, it ensembles several models after training each model. 
![sdf1](https://user-images.githubusercontent.com/45250729/92897046-2a5a2080-f41d-11ea-9942-cfaa41fe067d.jpg)
![result](https://user-images.githubusercontent.com/45250729/92899430-13b4c900-f41f-11ea-9481-a77b34d5ca94.jpg)

Approach 3) Layer level stochastic aggregation 
- Approach 1 and 2 have word and document nodes in single or several graphs. To compress the graph, here uses only documents as nodes. In addtion, it feeds randomly sampled neighbors to aggregate features in CNNs. 
- The dataset provides a list of text documents and its labels. (numbers of documents = 74496, numbers of authors = 5)
- For Graph-based approach, we build a graph from the dataset. 
    - node = each document
    - edge = the number of common words between two documents 
    - feature of document = sum of word embeddings that a document has 
 - We assume that documents of an author have similar features and we want to aggregate its features from its neighbors documents. 
 - In GraphSAGE CNNs, each document aggregates its features from its random neighbors. 


