# CFCPred: Advancing circRNAs-encoded Peptides Prediction through Cluster Purity-guided Resampling and Fuzzy Voting

CFCPred is a novel circPEPs prediction tool that integrates cluster purity-guided resampling with fuzzy voting (FV).

To address the scarcity of known circPEPs, a resampling strategy that mitigates class-imbalance in the training data by generating synthetic samples based on relationships among individual samples, their nearest neighbors, and their cluster assignments, is used. 

Additionally, to tackle class overlap in the testing data, FV enables adaptive model selection and dynamic weight adjustment across the ensemble. 

This adaptive ensemble integrates predictions from multiple base models to produce a robust consensus classification.

# Requirement
Python == 3.9.4

biopython == 1.85

numpy == 2.0.1

scikit-learn == 1.6.1

pandas == 2.3.3

# Usage
FeatureDescriptor.py is used for feature extraction

FeatureSelection.py is applied for feature selection but is not used in our paper

HybridSample.py is used for resampling which contains undersampling and oversampling

MLModel.py is used to construct and train machine learning classifiers

EnsembleLearning.py is used for model ensemble based on fuzzy voting, thus predicting

![image](https://github.com/zzssyy/CFCPred/blob/main/Graphical_Abstract.png)
