# Overview 
## Approach 1) 
#### - Instaed of aggregating features from all neighbors, it aggregates features from sample nodes. 
#### - Process
##### 1) Some words occur often are not good representative so it removes stopwords. 2) By using pre-trained embedding model, it generates word features. 3) It builds single graph with words as nodes and weights as edges. 4) During training CNNs, it aggregates features from sample nodes. 5) After training, it uses softmax classifier. 
