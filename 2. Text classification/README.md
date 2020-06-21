# Overview 
### Step 1) Pre-processing 
#### It removes stopwords. Why removing stopwords? some words occur often are not good representative and are not relevant. 

### Step 2) Word embedding
#### By using pre-trained model, it generates word embedding. 

### Step 3) Build single graph
#### To build a graph, words are nodes and weights between nodes are the edges. 

### Step 4) Sample subgraph
#### It makes partitions of single graph and trains the model. 

### Step 5) Aggregate global parameters 
#### After each training of subgraph, it updates global parameters for classifier. 
