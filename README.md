# PFE(Probabilistic Face Embeddings) for Recognition
This work show how to realize the training of the uncertainty module for two kinds of data.
The NLS loss is referenced from the Paper "Probabilistic Face Embeddings".

train_uncertainty : train the uncertainty module attached to a small mobilenet model.

confusiontest_mn_uncertainty: get the confusion matrix and the uncertainty for each sample, which are saved in a matfile
and a csv file.
