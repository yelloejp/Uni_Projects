# Description

Key: DTW(Dynamic Time Warping), Regression, Clustering, RNNs

A compaly installed sensors(Y0~Y18) in a city to collect temperature data but the sensors show gaps in the same region depending on the environment such as latitude, distance from rivers, etc.. 
The goal is to build a model to predict temperature of the sensors by using temperature data by sensors and National Weather Service.

For sensor Y0 to Y17, it provides complete data during the whole observation period except Y18. 
The target senor we want to predict is Y18. 

## Approach 1) Vanilar regression
- It divides Y0~Y17 into 5 clusters and then extract relevant features.
- By using Danamic Time Warping distance, it looks for the most relevant cluster for Y18. 
- From the features of the cluster, it predicts temperature of Y18. 

## Approach 2) RNNs 
- By DTW, it looks for clusters which show similar temperature sequence. 
- Comparing partial information which Y18 provides, we assume the most similar cluster that Y18 belongs to. 
