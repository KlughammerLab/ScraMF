# Name

Name is a tool that uses single cell RNA sequencing (scRNA-Seq) data to classify cells into males and females. 

scRNA-Seq experiments offer a powerful way to study gene expression at the level of individual cells, but they can be resource-intensive in terms of time, effort, and cost. Even though sequencing experiments have significantly out-paced 'Moore's law', the overall cost of scRNA-Seq experiments is still quite high. Multiplexing is a technique that combines multiple samples into one for more efficient processing, which can reduce the overall cost and effort of the experiment. 

scRNA-Seq library preparation involves various steps, including the addition of multiplexing reagents as one of the first steps. There are various strategies used for multiplexing such as using 'cell multiplexing oligos', 'genetic demultiplexing' and 'droplet based multiplexing'. However, multiplexing strategies that involve cell multiplexing oligos often require multiple additional steps including washing, which can result in the loss or death of cells. Doublets are often generated when using droplet based multiplexing and this obscures vital information in crucial experiments. Genetic demultiplexing involves identifying unique genetic signatures, such as single nucleotide polymorphisms, in the samples and requires additional preprocessing to distinguish between them. These signatures can be difficult to distinguish if the samples are closely related such as in case of siblings. Name allows for demultiplexing of such samples because it does not rely on exomes or signatures unique to the samples, unlike other strategies.  Furthermore, efficiency and speed are important considerations in scRNA-seq library preparation in order to minimize unwanted changes in the expression landscape of the cells. Name aims to minimize the time and loss of cells during critical experiments by potentially reducing the number of washing steps required through demultiplexing at the library preparation stage. 

Name is based on the XGBoost gradient boosting tree algorithm, which provides a cost-effective, time-efficient and computational inexpensive solution for demultiplexing male and female samples that have been mixed together during sample preparation. 
Males and Females express various genes which are characteristic to the cell. Name exploits these differentially expressed genes to classify cells into males and females. 

Name could be used to investigate the potential differences in gene expression between males and females in studies with respect to age and intensity of disease during downstream data analysis. Name could potentially also be used to detect doublets in experiments by identifying cells that are classified as approximately 50% male and female. Eventhough Name can only allow demultiplexing of two samples at a time, it is a convenient and efficient alternative that requires minimal computational resources and effort compared to other approaches.


### QuickStart
Name consits of two packages:
1. The first package consits of two models trained specifically for peripheral blood mononuclear cells (PBMCs) for classification. 
2. The second package enables the user to train models using the training Anndata, and then use those models to classify cells.

#### The first package works on the Anndata for PBMCs which has gone through the regular filtering steps. It has the following functions:
- sex_classifier_pbmc classifies cells into males and females based on the count matrix and adds the prediction column to the test Anndata along with the probabilities of class prediction.
```ruby
import Sex_Classifier_PBMC as scp

scp.sex_classifier_pbmc(test_adata, class_prob_cutoff=0.85)
```
- misclassified function returns a dataframe which tries to gives information about the number of correctly and incorrectly classified cells according to the cell type annotation and the plausible reason for their misclassification.
```ruby
df_misclassified = scp.misclassified(adata, min_ncounts=1100, min_genes=300, min_mtfrac=0.04, misclass_cutoff=0.85)
```
- ambiguously_classified function also returns a dataframe but works on test Anndata without sex labels. It classifies cells as 'correct' and 'incorrect'  based on the class probability value specified by the user.
```ruby
df_ambiguously_classified = scu.ambiguously_classified(test_adata, class_prob_cutoff=0.85)
```
- plot_avg_gene_expression plots the average gene expression of highly expressed genes separately for males and females across correctly classified and incorrectly classified cells.
```ruby
plots = scp.plot_avg_gene_expression(test_adata) 
``` 

#### The second package requires a test Anndata with Sex Labels for training the models. It has the following functions:
- train_model_classifier trains the two models on the training anndata for classifying cells into males and females.
```ruby
import Sex_Classifier_Universal as scu

model_1, model_2 = scu.train_model_classifer(adata_training, epochs=20, max_depth=10, eta=0.15, predict=True)
```
- sex_classifier_universtal classifies the cells into males and females by utilizing the models trained by the previous function and adds the prediction column to the test Anndata along with the probabilities of class prediction.
```ruby
scu.sex_classifier_universal(test_adata, model_softmax, model_softprob, class_prob_cutoff=0.85)
```
- misclassified function returns a dataframe which tries to gives information about the number of correctly and incorrectly classified cells according to the cell type annotation and the plausible reason for their misclassification.
```ruby
df_misclassified = scu.misclassified(adata, min_ncounts=1100, min_genes=300, min_mtfrac=0.04, misclass_cutoff=0.85)
```
- ambiguously_classified function also returns a dataframe but works on test Anndata without sex labels. It classifies cells as 'correct' and 'incorrect'  based on the class probability value specified by the user.
```ruby
df_ambiguously_classified = scu.ambiguously_classified(test_adata, class_prob_cutoff=0.85)
```
- plot_avg_gene_expression plots the average gene expression of highly expressed genes separately for males and females across correctly classified and incorrectly classified cells.
```ruby
plots = scu.plot_avg_gene_expression(test_adata) 
``` 


