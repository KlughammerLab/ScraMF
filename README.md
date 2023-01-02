# Name

Name aims to classify cells into males and females based on the count matrix obtained from single cell RNA sequencing experiments.

Even though sequencing experiments have significantly out-paced Moore's law, the cost of single cell sequencing experiments is still quite high.
To circumvent the high epxenses, multiplexing is a very smart and efficient way. 
Multiplexing involves combining multiple samples into one for efficient sequencing, which reduces the overall cost of the experiment.
There are various strategies used for multiplexing such as using cell multiplexing oligos and genetic demultiplexing. However, these strategies are associated with expensive reagents, prior knowledge of the genome to design primers and most important of them all, increased processing time.

Name is based on XGBoost gradient boosting tree algorithm which provides a cost effective, time efficient and computationally inexpensive alternative to the problem. 
The major demultiplexing strategies rely on chemically distinguished signals rather than biological signals.
Males and Females express various genes which are characteristic to the cell. Name exploits these differentially expressed genes to classify cells into males and females. It provieds a convenient and efficient alternative for demultiplexing male and female samples that have been mixed together during library preparation, requiring minimal computational resources compared to alternative approaches.

### QuickStart
Name consits of two packages:
1. The first package consits of two models trained specifically for peripheral blood mononuclear cells (PBMCs) for classification and has the following functions:
- sex_classifier_pbmc
- misclassified
- plot_avg_gene_expression
2. The second package enables the user to train models using the training Anndata, and then use those models to classify cells. It includes the following functions:
- train_model_classifier
- sex_classifier_universal
- misclassified
- plot_avg_gene_expression

The first package works on the Anndata for PBMCs which has gone through the regular filtering steps.
- sex_classifier_pbmc classifies cells into males and females based on the count matrix and adds the prediction column to the test Anndata along with the probabilities of class prediction.
- misclassified function returns a dataframe which tries to gives information about the number of correctly and incorrectly classified cells according to the cell type annotation and the plausible reason for their misclassification.
- plot_avg_gene_expression plots the average gene expression of highly expressed genes separately for males and females across correctly classified and incorrectly classified cells.

The second package requires a test Anndata with Sex Labels for training the models.
- train_model_classifier trains the two models on the training anndata for classifying cells into males and females.
- sex_classifier_universtal classifies the cells into males and females by utilizing the models trained by the previous function and adds the prediction column to the test Anndata along with the probabilities of class prediction.
- misclassified function returns a dataframe which tries to gives information about the number of correctly and incorrectly classified cells according to the cell type annotation and the plausible reason for their misclassification.
- plot_avg_gene_expression plots the average gene expression of highly expressed genes separately for males and females across correctly classified and incorrectly classified cells.

