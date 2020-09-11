# Overview 
## Approach 1) Stochastic aggregation
#### - Instaed of aggregating features from all neighbors, it aggregates features from sample nodes. 
#### - Process
##### 1) Some words occur often are not good representative so it removes stopwords. 2) By using pre-trained embedding model, it generates word features. 3) It builds single graph with words as nodes and weights as edges. 4) During training CNNs, it aggregates features from sample nodes. 5) After training, it uses softmax classifier. 
![StochasticAggregation](https://user-images.githubusercontent.com/45250729/85316673-697f7200-b4bd-11ea-9aa9-f42b7c41005e.png)
##### Reference : "Graph Convolutional Networks for Text Classification. Yao et al. AAAI2019." 

## Approach 2) Subgraph aggregation
#### - Instaed of training entire single graph, it generate several smaller graphs from a corpus. 
#### - Process
##### 1) Some words occur often are not good representative so it removes stopwords. 2) By using pre-trained embedding model, it generates word features. 3) It builds several graphs which are similar 4) After training each graph separately, it ensembles the results. 
![sdf1](https://user-images.githubusercontent.com/45250729/92897046-2a5a2080-f41d-11ea-9942-cfaa41fe067d.jpg)
![result](https://user-images.githubusercontent.com/45250729/92899430-13b4c900-f41f-11ea-9481-a77b34d5ca94.jpg)

