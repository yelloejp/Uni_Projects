# Overview 
## Approach 1) Stochastic aggregation
#### - Instaed of aggregating features from all neighbors, it aggregates features from sample nodes. 
#### - Process
##### 1) Some words occur often are not good representative so it removes stopwords. 2) By using pre-trained embedding model, it generates word features. 3) It builds single graph with words as nodes and weights as edges. 4) During training CNNs, it aggregates features from sample nodes. 5) After training, it uses softmax classifier. 
![StochasticAggregation](https://user-images.githubusercontent.com/45250729/85316673-697f7200-b4bd-11ea-9aa9-f42b7c41005e.png)
##### Reference : "Graph Convolutional Networks for Text Classification. Yao et al. AAAI2019." 

## Approach 2) Subgraph aggregation
#### - Instaed of training entire single graph, it splits the graph into several subgraphs. 
#### - Process
##### 1) Some words occur often are not good representative so it removes stopwords. 2) By using pre-trained embedding model, it generates word features. 3) It builds single graph with words as nodes and weights as edges. 4) It splits the graph into N number of subgraphs. 5) During training, it aggregates features forwarding all subgraphs. 6) After training, it uses softmax classifier. 
