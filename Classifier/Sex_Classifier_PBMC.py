import warnings
import pickle
import scanpy as sc
import pandas as pd
import numpy as np
import anndata as ad
from scipy import sparse
from anndata import AnnData
import time
import xgboost as xgb
from sklearn.metrics import accuracy_score
import os
import MACA as maca
import scrublet as scr


package_dir = os.path.dirname(__file__)

def sex_classifier(adata):
    start_time = time.time()
    print('Initializing')
    warnings.simplefilter("ignore", UserWarning)
    warnings.simplefilter("ignore", RuntimeWarning)
    warnings.simplefilter("ignore", FutureWarning)
    #Load the models
    model_softmax = xgb.Booster()
    model_softmax.load_model(os.path.join(package_dir, 'model_softmax_final.json'))
    model_softprob = xgb.Booster()
    model_softprob.load_model(os.path.join(package_dir, 'model_softprob_final.json'))
    #Load the gene list
    with open(os.path.join(package_dir, 'genes.txt'), 'rb') as fp:   # Unpickling
        genes_all = pickle.load(fp)
        
    #Create pseudo adata
    var_pseudo = pd.DataFrame({'Gene_Names': genes_all})
    var_pseudo = var_pseudo.set_index(['Gene_Names'])
    obs_pseudo = pd.DataFrame({'Cols':[1]})
    obs_pseudo = obs_pseudo.set_index('Cols')
    matrix_pseudo = np.zeros((1,28517))
    matrix_pseudo = sparse.csr_matrix(matrix_pseudo)
    adata_training = sc.AnnData(matrix_pseudo, obs_pseudo, var_pseudo)
    
    #Slice genes in the test dataset and add genes from training dataset if they are unique to the training dataset. 
    genes_training = [i for i in adata.var.index if i in genes_all]
    adata_test = adata.copy()
    #Rearrange the test dataset to match the index of training.
    adata_test = adata_test[:,genes_training].copy()
    sex_classes = {'F':0, 'M':1}
    
    print('Imported Essentials')
    
    #Merge the datasets 
    adata_test.obs['Set'] = 'Test'
    adata_training.obs['Set'] = 'Training'
    adata_list = [adata_training, adata_test]
    adata_merged = ad.concat(adata_list, join='outer')
    #Split the dataset
    adata_test = adata_merged[adata_merged.obs.Set == 'Test']
    
    #Reindex test dataset
    adata_test.var = adata_test.var.set_index(adata_training.var.index)
    
    print('Adata Modified For The Model')
        
    #Make the test matrix
    if ('Sex' or 'sex') in adata.obs_keys():
        adata_test.obs['Sex_Class'] = adata.obs.Sex.map(sex_classes).astype('category')
        test = xgb.DMatrix(adata_test.X, label=adata_test.obs.Sex_Class)
    else:
        test = xgb.DMatrix(adata_test.X)
    #Predictions
    predictions_softmax = model_softmax.predict(test)
    predictions_softprob = model_softprob.predict(test)
    
    print('Sex Prediction Complete')
    
    #Adding columns to the adata
    adata.obs['Predictions'] =  predictions_softmax
    sex_classes_2 = {0: 'F', 1: 'M'}
    adata.obs['Predictions'] = adata.obs.Predictions.map(sex_classes_2).astype('category')
    adata.obs['Predictions_Probability_Female'] = predictions_softprob.T[0]
    adata.obs['Predictions_Probability_Male'] = predictions_softprob.T[1]
    adata_test.obs['Predictions'] = predictions_softmax
    adata_test.obs['Class_Prediction'] = np.where((adata_test.obs['Predictions'] == 0.0), 
                                                  adata.obs.Predictions_Probability_Female, 
                                                  adata.obs.Predictions_Probability_Male)
    print('Adding Columns to the Adata')
    
    #If loop for adding the class prediction of the provided sex data 
    if ('Sex' or 'sex') in adata.obs_keys():
        conditions = [((adata_test.obs.Sex_Class == adata_test.obs.Predictions) & (adata_test.obs.Class_Prediction > 0.85)),
          ((adata_test.obs.Sex_Class != adata_test.obs.Predictions) & (adata_test.obs.Class_Prediction > 0.85)), 
          ((adata_test.obs.Sex_Class == adata_test.obs.Predictions) & (adata_test.obs.Class_Prediction < 0.85)),
          ((adata_test.obs.Sex_Class != adata_test.obs.Predictions) & (adata_test.obs.Class_Prediction < 0.85))]
        choices = [ "True Positive", 'False Positive', 'False Negative' , 'True Negative']
        adata.obs['Condition'] = np.select(conditions, choices, default=np.nan)
        #Accuracy Score
        adata_test_males = adata_test[adata_test.obs.Sex == 'M']
        acc_score_males = accuracy_score(adata_test_males.obs.Sex_Class, adata_test_males.obs.Predictions)
        print('The accuracy_score for males for universally trained model is {}'.format(acc_score_males))
        adata_test_females = adata_test[adata_test.obs.Sex == 'F']
        acc_score_females = accuracy_score(adata_test_females.obs.Sex_Class, adata_test_females.obs.Predictions)
        print('The accuracy_score for females for universally trained model is {}'.format(acc_score_females))
    else:
        pass
    print('Prediction Completed')
    print("--- %s secs ---" % int((time.time() - start_time)))
    
def misclassified(adata):
    print('Initializing') 
    start_time = time.time()    
    #Annotate all datasets individually and then concat because they have batch effect in between them
    adata_test = adata.copy()
    #Normalize the data to 10000 reads per cell
    sc.pp.normalize_total(adata_test, target_sum=1e4)
    #Log tranform the data
    sc.pp.log1p(adata_test)
    
    #Check which cell type is misclassified
    azimuth_markers = pd.read_csv('/home/sparikh/Classifier_data/azimuth_markers_MACA.csv', index_col=['Unnamed: 0'])
    cells_of_interest_az = azimuth_markers['Expanded Label'].values.tolist()
    
    #Create the lists and dicts
    cell_markers_az = {}
    marker_list_az = []
    for i,j in zip(range(0,93),cells_of_interest_az):
        x = azimuth_markers.iloc[i,:].values.tolist()
        cell_markers_az[j] = x[1:]
        marker_list_az += x[1:]
    marker_list_az = list(set(marker_list_az))
    
    #Slice adata_test for i in marker_list_az if i in adata_test.var.index]
    b = [i for i in marker_list_az if i in adata_test.var.index]
    ad_az = adata_test.copy()
    ad_az = ad_az[:,b].copy()
    ad_az, annotation_az = maca.singleMACA(ad=ad_az, cell_markers=cell_markers_az,res=[1, 1.5, 2.0],n_neis=[3,5,10])
    print('Annotation Complete')
    
    #Add annotation to the original adata
    adata_test.obs['Annotation']=np.array(annotation_az)
    adata_test.obs['n_counts'] = adata_test.X.sum(axis = 1)
    adata_test.obs['n_genes'] = (adata_test.X > 0).sum(axis = 1)
    mt_gene = np.flatnonzero([gene.startswith('MT-') for gene in adata_test.var_names])
    adata_test.obs['mt_frac']= np.sum(adata_test[:, mt_gene].X, axis =1).A1/adata_test.obs['n_counts']
    adata_test = adata_test[adata_test.obs['mt_frac'] < 0.2]
    
    adata_wrong_predictions = adata_test[adata_test.obs.Sex != adata_test.obs.Predictions]

    # filtering/preprocessing parameters:
    min_counts = 2
    min_cells = 3
    vscore_percentile = 85
    n_pc = 50
    # doublet detector parameters:
    expected_doublet_rate = 0.02 
    sim_doublet_ratio = 3
    n_neighbors = 15
    scrub = scr.Scrublet(counts_matrix = adata_wrong_predictions.X,  
                     n_neighbors = n_neighbors,
                     sim_doublet_ratio = sim_doublet_ratio,
                     expected_doublet_rate = expected_doublet_rate)

    doublet_scores, predicted_doublets = scrub.scrub_doublets( 
                    min_counts = min_counts, 
                    min_cells = min_cells, 
                    n_prin_comps = n_pc,
                    use_approx_neighbors = True, 
                    get_doublet_neighbor_parents = False, verbose=False)
    adata_wrong_predictions.obs['doublet_score'] = doublet_scores
    adata_wrong_predictions.obs['doublet'] = predicted_doublets
    
    
    cells_annot = adata_test.obs.Annotation.unique().tolist()
    mis_pred_cells = {}
    total_cells = []
    for i in cells_annot:
        total = len(adata_test.obs[adata_test.obs.Annotation == i])
        abc = adata_test.obs[adata_test.obs.Annotation == i]
        total_cells.append(len(abc))
        mispred = len(abc[abc.Sex != abc.Predictions])
        perc = (mispred/total)*100
        mis_pred_cells[i] = perc
    mis_pred_df = pd.DataFrame.from_dict(mis_pred_cells, orient='index')
    mis_pred_df = mis_pred_df.rename(columns={'index':'Cell_Type', 0:'Perc_Incorrectly_Classified'})
    mis_pred_df['Number_Cells'] = total_cells
    
    total_cells = sum(adata_test.obs.groupby('Annotation').size())
    total_percentage = []
    for i in mis_pred_df.Number_Cells:
        total_percentage.append((i/total_cells)*100)
    mis_pred_df['Perc_Total'] = total_percentage
    mis_pred_df = mis_pred_df[['Number_Cells', 'Perc_Total', 'Perc_Incorrectly_Classified']]
    num_mis = []
    for i in cells_annot:
        total = len(adata_test.obs[adata_test.obs.Annotation == i])
        abc = adata_test.obs[adata_test.obs.Annotation == i]
        mispred = len(abc[abc.Sex != abc.Predictions])
        num_mis.append(mispred)
    mis_pred_df['Number_Cells_Misclass'] = num_mis

    #Some cells could have low gene count + count + doublet and therefore need to be filtered as low quality before individually assigning to columns
    adata_low_quality = adata_wrong_predictions[(adata_wrong_predictions.obs.n_counts < 1100) | (adata_wrong_predictions.obs.n_genes < 300) | 
                                            (adata_wrong_predictions.obs.doublet == True)]
    low_qual = []
    num_doublet = []
    num_low_count_and_genes = []
    num_mt_frac = []
    for i in cells_annot:
        total = len(adata_low_quality.obs[adata_low_quality.obs.Annotation == i])
        low_qual.append(total)
        n_doub = len(adata_low_quality[(adata_low_quality.obs.Annotation == i) & (adata_low_quality.obs.doublet == True)].obs)
        num_doublet.append(n_doub)
        low_count_genes = len(adata_low_quality[(adata_low_quality.obs.Annotation == i) & ((adata_low_quality.obs.n_counts< 1100) | (adata_low_quality.obs.n_genes< 300))].obs)
        num_low_count_and_genes.append(low_count_genes)
        ncells_mt = len(adata_test[(adata_test.obs.Annotation == i) & (adata_test.obs.mt_frac > 0.04)])
        num_mt_frac.append(ncells_mt)
    mis_pred_df['Num_Doublets'] = num_doublet    
    mis_pred_df['NCells_High_MT_Frac'] = ncells_mt
    mis_pred_df['NCells_Low_Count/Genes'] = num_low_count_and_genes
    mis_pred_df['NCells_Explained_Misclass'] = mis_pred_df.Num_Doublets + mis_pred_df['NCells_Low_Count/Genes']
    mis_pred_df['NCells_Unexplained_Misclass'] = mis_pred_df.Number_Cells_Misclass - mis_pred_df.NCells_Explained_Misclass
    mis_pred_df['Perc_Unexplained'] = (mis_pred_df.NCells_Unexplained_Misclass/mis_pred_df.Number_Cells_Misclass)*100
    total_unexplained = mis_pred_df.NCells_Unexplained_Misclass.sum()
    mis_pred_df['Perc_Unexplained_Total_Pop'] = (mis_pred_df.NCells_Unexplained_Misclass/total_unexplained)*100
    mis_pred_df = mis_pred_df.reset_index()
    mis_pred_df = mis_pred_df.rename(columns={'index':'Annotation'})
    mis_pred_df.loc['Total',:]= mis_pred_df.sum(axis=0)   
    mis_pred_df_temp = mis_pred_df.iloc[:-1,:]

    chosen_cell_type = mis_pred_df_temp.iloc[(np.argmax(mis_pred_df_temp['Perc_Unexplained_Total_Pop'])), :].Annotation
    adata_test_chosen = adata_test[adata_test.obs.Annotation == chosen_cell_type].copy()
    adata_test_chosen.obs['Group'] = np.where((adata_test_chosen.obs['Sex'] == adata_test_chosen.obs['Predictions']), 'Correct' , 'Incorrect')
    females = adata_test_chosen[adata_test_chosen.obs.Sex == 'F'].copy()
    males = adata_test_chosen[adata_test_chosen.obs.Sex == 'M'].copy()
    
    print('The most misclassified celltype is {}'.format(chosen_cell_type.upper()))
    print('Low quality cells detected and dataframe created')
    #Differentially expressed genes for females and males
    sc.tl.rank_genes_groups(females, 'Group', method='t-test')
    sc.tl.rank_genes_groups(males, 'Group', method='t-test')

    rcParams['figure.figsize']=(60,15)
    
    #Create female rank genes graph
    females_genes = pd.DataFrame((females.uns['rank_genes_groups']['names'])).head(25)
    females_scores = pd.DataFrame(females.uns['rank_genes_groups']['scores']).head(25)
    females_scores.rename(columns={'Correct': 'Correct_Score', 'Incorrect': 'Incorrect_Score'}, inplace=True)
    females_rank_genes = females_genes.join(females_scores)
    females_rank_genes = females_rank_genes[['Correct','Correct_Score', 'Incorrect', 'Incorrect_Score']]
    
    #Create male rank genes graph
    males_genes = pd.DataFrame((males.uns['rank_genes_groups']['names'])).head(25)
    males_scores = pd.DataFrame(males.uns['rank_genes_groups']['scores']).head(25)
    males_scores.rename(columns={'Correct': 'Correct_Score', 'Incorrect': 'Incorrect_Score'}, inplace=True)
    males_rank_genes = males_genes.join(males_scores)
    males_rank_genes = males_rank_genes[['Correct','Correct_Score', 'Incorrect', 'Incorrect_Score']]
    
    #Plot the figures
    fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(nrows=2, ncols=2)
    fig.subplots_adjust(hspace=0.4, wspace=0.6)
    sb.set(font_scale=1)
    sb.set_style(style='whitegrid')
    sb.set_palette('Dark2')
    p1 = sb.scatterplot(data=females_rank_genes.iloc[:,0:2], x=females_rank_genes.iloc[:,0:2].index,     y=females_rank_genes.iloc[:,0:2].Correct_Score, color='lightblue', 
                        ax=ax1).set(title="Correct Females" )
    p2 = sb.scatterplot(data=females_rank_genes.iloc[:,2:], x=females_rank_genes.iloc[:,2:].index, y=females_rank_genes.iloc[:,2:].Incorrect_Score, color='lightcoral', 
                        ax=ax2).set(title="Incorrect Females" )
    p3 = sb.scatterplot(data=males_rank_genes.iloc[:,0:2], x=males_rank_genes.iloc[:,0:2].index, y=males_rank_genes.iloc[:,0:2].Correct_Score, color='lightblue', 
                        ax=ax3).set(title="Correct Males" )
    p4 = sb.scatterplot(data=males_rank_genes.iloc[:,2:], x=males_rank_genes.iloc[:,2:].index, y=males_rank_genes.iloc[:,2:].Incorrect_Score, color='lightcoral', 
                        ax=ax4).set(title="Incorrect Males",  )
    for i in range(0,females_rank_genes.shape[0]):
        ax1.text(females_rank_genes.iloc[:,0:2].index[i]+0.1, females_rank_genes.iloc[:,0:2].Correct_Score[i], females_rank_genes.iloc[:,0:2].Correct[i], 
                 horizontalalignment='left', size='medium', color='midnightblue', weight='semibold')
        ax2.text(females_rank_genes.iloc[:,2:].index[i]+0.1, females_rank_genes.iloc[:,2:].Incorrect_Score[i], females_rank_genes.iloc[:,2:].Incorrect[i], 
                 horizontalalignment='left', size='medium', color='maroon', weight='semibold')
        ax3.text(males_rank_genes.iloc[:,0:2].index[i]+0.1, males_rank_genes.iloc[:,0:2].Correct_Score[i], males_rank_genes.iloc[:,0:2].Correct[i], 
                 horizontalalignment='left', size='medium', color='midnightblue', weight='semibold')
        ax4.text(males_rank_genes.iloc[:,2:].index[i]+0.1, males_rank_genes.iloc[:,2:].Incorrect_Score[i], males_rank_genes.iloc[:,2:].Incorrect[i], 
                 horizontalalignment='left', size='medium', color='maroon', weight='semibold')
    plt.show()
    plots = fig
    print('Dataframe and Plots Ready')
    print("--- %s mins ---" % int((time.time() - start_time)/60))
    return mis_pred_df, plots

