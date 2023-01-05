import warnings
import pickle
import scanpy as sc
import pandas as pd
import numpy as np
import anndata as ad
from scipy import sparse
import time
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
import os
import MACA as maca
import scrublet as scr
from matplotlib import rcParams
import seaborn as sb
import matplotlib.pyplot as plt

package_dir = os.path.dirname(__file__)

def train_model_classifer(adata_training, epochs=20, max_depth=10, eta=0.15, predict=False):
    """ Train models for classifying cells into males and females through sex_classifier_universal function"""
    start_time = time.time()
    print('Initializing')
    warnings.simplefilter("ignore", UserWarning)
    warnings.simplefilter("ignore", RuntimeWarning)
    warnings.simplefilter("ignore", FutureWarning)
    
    print('Preparing Training Anndata')
    #Add Sex_Class to training adata if it does not exist
    adata_training_copy = adata_training.copy()
    adata_training_copy.obs.columns = adata_training_copy.obs.columns.str.capitalize()
    sex_classes = {'F':0, 'M':1}
    if 'Sex' in adata_training_copy.obs_keys():
        adata_training_copy.obs['Sex_Class'] = adata_training_copy.obs.Sex.map(sex_classes).astype('category')
    else:
        print('Please provide training Anndata with SEX column')
    
    print('Training Models')
    
    #Load Parameters:
    train = xgb.DMatrix(adata_training_copy.X, label=adata_training_copy.obs.Sex_Class)
    param_softmax = {'max_depth':max_depth, 'eta':eta, 'objective':'multi:softmax','num_class':2}
    param_softprob = {'max_depth':max_depth, 'eta':eta, 'objective':'multi:softprob','num_class':2}
    #Train Models
    model_softmax = xgb.train(param_softmax, train, epochs)
    model_softprob = xgb.train(param_softprob, train, epochs)
    
    #Make gene list
    #Select ensemble IDs if they are present instead of gene names
    adata_training_copy.var = adata_training_copy.var.astype(str)
    global genes_training
    for i,j in zip(adata_training_copy.var.iloc[0,:], range(0,(len(adata_training_copy.var.iloc[0,:])))):
        if 'ENS' in i:
            genes_training = adata_training_copy.var.iloc[:,j].tolist()
            break
        else:
            genes_training = adata_training_copy.var.index.tolist()
    print('Training Complete')
    
    if predict==True:
        #Make the test matrix on training adata
        test = xgb.DMatrix(adata_training_copy.X, label=adata_training_copy.obs.Sex_Class)
        #Add the predictions to the copy of the training adata
        predictions_softmax = model_softmax.predict(test)
        predictions_softprob = model_softprob.predict(test)
        #Add columns to obs of training adata
        adata_training_copy.obs['Predictions'] = predictions_softmax
        adata_training_copy.obs['Predictions_Probability_Female'] = predictions_softprob.T[0]
        adata_training_copy.obs['Predictions_Probability_Male'] = predictions_softprob.T[1]
        adata_training_copy.obs['Class_Prediction'] = np.where((adata_training_copy.obs['Predictions'] == 0.0), 
                                                  adata_training_copy.obs.Predictions_Probability_Female, 
                                                  adata_training_copy.obs.Predictions_Probability_Male)
        adata_training_copy.obs.Sex_Class = adata_training_copy.obs.Sex_Class.astype(float)
        adata_training_copy.obs.Predictions = adata_training_copy.obs.Predictions.astype(float)
        #Split the copy of training adata into males and females
        females = adata_training_copy[adata_training_copy.obs.Sex == 'F']
        males = adata_training_copy[adata_training_copy.obs.Sex == 'M']
        values = [0.75, 0.80, 0.85, 0.90, 0.95]
        dict_names = ['a', 'b', 'c', 'd', 'e']
        adata_list = [females, males]
        df_plot = ['df_females_plot', 'df_males_plot']
        for adata, df_name in zip(adata_list, df_plot):
            for cutoff,name in zip(values, dict_names):
                conditions = [((adata.obs.Sex_Class == adata.obs.Predictions) & (adata.obs.Class_Prediction > cutoff)),
                  ((adata.obs.Sex_Class != adata.obs.Predictions) & (adata.obs.Class_Prediction > cutoff)), 
                  ((adata.obs.Sex_Class == adata.obs.Predictions) & (adata.obs.Class_Prediction < cutoff)),
                  ((adata.obs.Sex_Class != adata.obs.Predictions) & (adata.obs.Class_Prediction < cutoff))]
                choices = [ 'True Positive', 'False Positive', 'False Negative' , 'True Negative']
                adata.obs["Condition"] = np.select(conditions, choices, default=np.nan)
                globals()['cutoff' + '_' + name ] = adata.obs.groupby('Condition').size().to_dict()
            cutoff_values = ['Cutoff: 0.75','Cutoff: 0.80', 'Cutoff: 0.85', 'Cutoff: 0.90', 'Cutoff: 0.95']
            globals()[df_name] = pd.DataFrame([cutoff_a, cutoff_b, cutoff_c,cutoff_d, cutoff_e], index=cutoff_values)
#             globals()[df_name] = globals()[df_name][['True Positive', 'True Negative','False Positive', 'False Negative']]
#             globals()[df_name]['Total'] = globals()[df_name].sum(axis=1)
        
        #Plot with Group
        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(10,8))
        fig = plt.figure()
        fig.subplots_adjust(hspace=0.5, wspace=0.6)
        sb.set_style(style='whitegrid')
        sb.lineplot(data=(df_females_plot.iloc[:,:4]['True Positive'], df_females_plot.iloc[:,:4]['True Negative'], df_females_plot.iloc[:,:4]['False Positive'], 
                          df_females_plot.iloc[:,:4]['False Negative']), 
                   ax=ax1, dashes=False, markers=True, linewidth = 3).set(title="Ground Truth: Females")
        sb.lineplot(data=(df_males_plot.iloc[:,:4]['True Positive'], df_males_plot.iloc[:,:4]['True Negative'], df_males_plot.iloc[:,:4]['False Positive'], 
                          df_males_plot.iloc[:,:4]['False Negative']), 
                   ax=ax2, dashes=False, markers=True, linewidth = 3).set(title="Ground Truth: Females")
        plt.show()
        
        #Accuracy Score
        acc_score_males = accuracy_score(males.obs.Sex_Class, males.obs.Predictions)
        print('The accuracy_score for males for universally trained model is {}'.format(acc_score_males))
        acc_score_females = accuracy_score(females.obs.Sex_Class, females.obs.Predictions)
        print('The accuracy_score for females for universally trained model is {}'.format(acc_score_females))
        #Add the columns to the original adata
        sex_classes_2 = {0: 'F', 1: 'M'}
        adata_training.obs['Predictions'] = predictions_softmax
        adata_training.obs.Predictions = adata_training.obs.Predictions.map(sex_classes_2)
        adata_training.obs['Predictions_Probability_Female'] = predictions_softprob.T[0]
        adata_training.obs['Predictions_Probability_Male'] = predictions_softprob.T[1]
    else:
        pass
    print("--- %s mins ---" % int((time.time() - start_time)/60))
    return model_softmax, model_softprob
    
    
def sex_classifier_universal(adata_test, model_softmax, model_softprob, class_prob_cutoff=0.85):
    """ Classifies cells into males and females based on the gene expression"""
    start_time = time.time()
    print('Initializing')
    warnings.simplefilter("ignore", UserWarning)
    warnings.simplefilter("ignore", RuntimeWarning)
    warnings.simplefilter("ignore", FutureWarning)
    
    #Slice genes in the test dataset and add genes from training dataset if they are unique to the training dataset. 
    adata_test_copy = adata_test.copy()
    genes_slice = [i for i in adata_test_copy.var.index if i in genes_training]
    #Rearrange the test dataset to match the index of training.
    adata_test_copy = adata_test_copy[:,genes_slice]
    
    print('Preparing Test Adata')
    
    #Create Pseudo adata
    var_pseudo = pd.DataFrame({'Gene_Names': genes_training}).set_index(['Gene_Names'])
    obs_pseudo = pd.DataFrame({'Cols':[1]}).set_index('Cols')
    matrix_pseudo = np.zeros((1,(len(genes_training))))
    adata_pseudo_train = sc.AnnData(sparse.csr_matrix(matrix_pseudo), obs_pseudo, var_pseudo)
    
    #Merge the datasets 
    adata_test_copy.obs['Set'] = 'Test'
    adata_pseudo_train.obs['Set'] = 'Training'
    adata_list = [adata_pseudo_train, adata_test_copy]
    adata_merged = ad.concat(adata_list, join='outer')
    #Split the dataset
    adata_test_copy = adata_merged[adata_merged.obs.Set == 'Test']
    
    #Reindex test dataset
    adata_test_copy.var = adata_test_copy.var.set_index(adata_pseudo_train.var.index)
    
    print('Test Anndata Modified For The Model')
        
    #Make the test matrix
    sex_classes = {'F':0, 'M':1}
    if ('Sex' or 'sex' or 'SEX') in adata_test.obs_keys():
        adata_test_copy.obs['Sex_Class'] = adata_test.obs.Sex.map(sex_classes).astype('category')
        test = xgb.DMatrix(adata_test_copy.X, label=adata_test_copy.obs.Sex_Class)
    else:
        test = xgb.DMatrix(adata_test_copy.X)
    
    #Predictions
    predictions_softmax = model_softmax.predict(test)
    predictions_softprob = model_softprob.predict(test)
    
    print('Sex Prediction Complete')
    
    #Adding columns to the adata
    adata_test.obs['Predictions'] =  predictions_softmax
    sex_classes_2 = {0: 'F', 1: 'M'}
    adata_test.obs['Predictions'] = adata_test.obs.Predictions.map(sex_classes_2).astype('category')
    adata_test.obs['Predictions_Probability_Female'] = predictions_softprob.T[0]
    adata_test.obs['Predictions_Probability_Male'] = predictions_softprob.T[1]
    adata_test_copy.obs['Predictions'] = predictions_softmax
    adata_test_copy.obs['Class_Prediction'] = np.where((adata_test_copy.obs['Predictions'] == 0.0), 
                                                  adata_test.obs.Predictions_Probability_Female, 
                                                  adata_test.obs.Predictions_Probability_Male)
    print('Adding Columns to the Test Anndata')
    
    #If loop for adding the class prediction of the provided sex data 
    if ('Sex' or 'sex' or 'SEX') in adata_test.obs_keys():
        conditions = [((adata_test_copy.obs.Sex_Class == adata_test_copy.obs.Predictions) & (adata_test_copy.obs.Class_Prediction > class_prob_cutoff)),
          ((adata_test_copy.obs.Sex_Class != adata_test_copy.obs.Predictions) & (adata_test_copy.obs.Class_Prediction > class_prob_cutoff)), 
          ((adata_test_copy.obs.Sex_Class == adata_test_copy.obs.Predictions) & (adata_test_copy.obs.Class_Prediction < class_prob_cutoff)),
          ((adata_test_copy.obs.Sex_Class != adata_test_copy.obs.Predictions) & (adata_test_copy.obs.Class_Prediction < class_prob_cutoff))]
        choices = [ "True Positive", 'False Positive', 'False Negative' , 'True Negative']
        adata_test.obs['Condition'] = np.select(conditions, choices, default=np.nan)
        #Accuracy Score
        adata_test_copy_males = adata_test_copy[adata_test_copy.obs.Sex == 'M']
        acc_score_males = accuracy_score(adata_test_copy_males.obs.Sex_Class, adata_test_copy_males.obs.Predictions)
        print('The accuracy_score for males for universally trained model is {}'.format(acc_score_males))
        adata_test_copy_females = adata_test_copy[adata_test_copy.obs.Sex == 'F']
        acc_score_females = accuracy_score(adata_test_copy_females.obs.Sex_Class, adata_test_copy_females.obs.Predictions)
        print('The accuracy_score for females for universally trained model is {}'.format(acc_score_females))
    else:
        pass
    
    #Add ROC Curve and ROC-AUC Value
    auroc = roc_auc_score(adata_test_copy.obs.Sex_Class, adata_test_copy.obs.Predictions)
    fpr, tpr, thresholds = roc_curve(adata_test_copy.obs.Sex_Class, adata_test_copy.obs.Predictions)
    #Plot the ROC Curve
    plt.figure(figsize=(5,5))
    plt.plot(fpr, tpr, linestyle = '-', label = 'ROC_AUC_Curve %0.3f' % auroc)
    plt.legend(loc='lower right')
    plt.show()
    print('Prediction Completed')
    print("--- %s mins ---" % int((time.time() - start_time)/60))
    
    
def misclassified(adata, min_ncounts=1100, min_genes=300, min_mtfrac=0.04):
    """Returns dataframe for misclassified cells"""
    print('Initializing')
    start_time = time.time()    
    sc.settings.verbosity = 0
    adata_test = adata.copy()
    adata_test.obs.columns = adata_test.obs.columns.str.capitalize()
    if ('log1p') not in adata.uns.keys():
        print('Please provide Normalized and Log transformed Anndata')
        return None
    else:
        pass
    if 'Annotation' not in adata_test.obs.keys():
            #Annotate the cells
            azimuth_markers = pd.read_csv((os.path.join(package_dir, 'azimuth_markers_MACA.csv')), index_col=['Unnamed: 0'])
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
            ad_az = ad_az[:,b]
            ad_az, annotation_az = maca.singleMACA(ad=ad_az, cell_markers=cell_markers_az,res=[1, 1.5, 2.0],n_neis=[3,5,10])
            #Add annotation to the original adata
            adata.obs['Annotation']=np.array(annotation_az)
            adata_test.obs['Annotation']=np.array(annotation_az)
            print('Annotation Complete')
    else:
        pass
    if ('Sex' or 'sex') in adata.obs.keys():
        #Add parameters to test Adata
        adata_test.obs['n_counts'] = adata_test.X.sum(axis = 1)
        adata_test.obs['n_genes'] = (adata_test.X > 0).sum(axis = 1)
        mt_gene = np.flatnonzero([gene.startswith('MT-') for gene in adata_test.var_names])
        adata_test.obs['mt_frac']= np.sum(adata_test[:, mt_gene].X, axis =1).A1/adata_test.obs['n_counts']
        adata_test = adata_test[adata_test.obs['mt_frac'] < 0.2]
        adata_test.obs.Sex = adata_test.obs.Sex.astype(str)
        adata_test.obs.Predictions = adata_test.obs.Predictions.astype(str)
        adata_wrong_predictions = adata_test[adata_test.obs.Sex != adata_test.obs.Predictions]

        # Doublet count
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
        for cell in cells_annot:
            total = len(adata_test.obs[adata_test.obs.Annotation == cell])
            adata_cell = adata_test.obs[adata_test.obs.Annotation == cell]
            total_cells.append(len(adata_cell))
            mispred = len(adata_cell[adata_cell.Sex != adata_cell.Predictions])
            perc = (mispred/total)*100
            mis_pred_cells[cell] = perc
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
            adata_annot = adata_test.obs[adata_test.obs.Annotation == i]
            mispred = len(adata_annot[adata_annot.Sex != adata_annot.Predictions])
            num_mis.append(mispred)
        mis_pred_df['Number_Cells_Misclass'] = num_mis

        #Some cells could have low gene count + count + doublet and therefore need to be filtered as low quality before individually assigning to columns
        adata_low_quality = adata_wrong_predictions[(adata_wrong_predictions.obs.n_counts < min_ncounts) | (adata_wrong_predictions.obs.n_genes < min_genes) | 
                                                (adata_wrong_predictions.obs.doublet == True)]
        low_qual = []
        num_doublet = []
        num_low_count_and_genes = []
        num_mt_frac = []
        for cell in cells_annot:
            total = len(adata_low_quality.obs[adata_low_quality.obs.Annotation == cell])
            low_qual.append(total)
            n_doub = len(adata_low_quality[(adata_low_quality.obs.Annotation == cell) & (adata_low_quality.obs.doublet == True)].obs)
            num_doublet.append(n_doub)
            low_count_genes = len(adata_low_quality[(adata_low_quality.obs.Annotation == cell) & ((adata_low_quality.obs.n_counts< min_ncounts) | (adata_low_quality.obs.n_genes< min_genes))].obs)
            num_low_count_and_genes.append(low_count_genes)
            ncells_mt = len(adata_test[(adata_test.obs.Annotation == cell) & (adata_test.obs.mt_frac > min_mtfrac)])
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
        mis_pred_df.iloc[-1,0] = 'Total'
        chosen_cell_type = mis_pred_df_temp.iloc[(np.argmax(mis_pred_df_temp['Perc_Unexplained_Total_Pop'])), :].Annotation
        print('Low quality cells detected and dataframe created')
        print("--- %s mins ---" % int((time.time() - start_time)/60))
        return mis_pred_df   
    else:
        print('Please provide Anndata with Sex Label or use ambiguously_assigned function from the package')
        
def ambiguously_classified(adata,min_ncounts=1100, min_genes=300, min_mtfrac=0.04, class_prob_cutoff=0.85):
    """Returns dataframe for misclassified cells"""
    print('Initializing')
    start_time = time.time()    
    sc.settings.verbosity = 0
    adata_test = adata.copy()
    adata_test.obs.columns = adata_test.obs.columns.str.capitalize()
    if ('log1p') not in adata.uns.keys():
        print('Please provide Normalized and Log transformed Anndata')
        return None
    else:
        pass
    
    if 'Annotation' not in adata_test.obs.keys():
            #Annotate the cells
            azimuth_markers = pd.read_csv((os.path.join(package_dir, 'azimuth_markers_MACA.csv')), index_col=['Unnamed: 0'])
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
            ad_az = ad_az[:,b]
            ad_az, annotation_az = maca.singleMACA(ad=ad_az, cell_markers=cell_markers_az,res=[1, 1.5, 2.0],n_neis=[3,5,10])
            #Add annotation to the original adata
            adata.obs['Annotation']=np.array(annotation_az)
            adata_test.obs['Annotation']=np.array(annotation_az)
            print('Annotation Complete')
    else:
        pass
    
    adata_test.obs['Class_Pred'] = np.where((adata.obs['Predictions'] == 'F'), 
                                              adata.obs.Predictions_Probability_Female, 
                                              adata.obs.Predictions_Probability_Male)
    adata_test.obs['n_counts'] = adata_test.X.sum(axis = 1)
    adata_test.obs['n_genes'] = (adata_test.X > 0).sum(axis = 1)
    mt_gene = np.flatnonzero([gene.startswith('MT-') for gene in adata_test.var_names])
    adata_test.obs['mt_frac']= np.sum(adata_test[:, mt_gene].X, axis =1).A1/adata_test.obs['n_counts']
    adata_test = adata_test[adata_test.obs['mt_frac'] < 0.2]
    adata_test.obs.Predictions = adata_test.obs.Predictions.astype(str)
    adata_wrong_predictions = adata_test[adata_test.obs.Class_Pred < class_prob_cutoff]

    # Doublet count
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

    # Calculate cell numbers
    cells_annot = adata_test.obs.Annotation.unique().tolist()
    mis_pred_cells = {}
    total_cells = []
    for cell in cells_annot:
        total = len(adata_test.obs[adata_test.obs.Annotation == cell])
        adata_cell = adata_test.obs[adata_test.obs.Annotation == cell]
        total_cells.append(len(adata_cell))
        mispred = len(adata_cell[adata_cell.Class_Pred < class_prob_cutoff])
        perc = (mispred/total)*100
        mis_pred_cells[cell] = perc
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
        adata_annot = adata_test.obs[adata_test.obs.Annotation == i]
        mispred = len(adata_annot[adata_annot.Class_Pred < class_prob_cutoff])
        num_mis.append(mispred)
    mis_pred_df['Number_Cells_Misclass'] = num_mis
    #Some cells could have low gene count + count + doublet and therefore need to be filtered as low quality before individually assigning to columns
    adata_low_quality = adata_wrong_predictions[(adata_wrong_predictions.obs.n_counts < min_ncounts) | (adata_wrong_predictions.obs.n_genes < min_genes) | 
                                            (adata_wrong_predictions.obs.doublet == True)]
    low_qual = []
    num_doublet = []
    num_low_count_and_genes = []
    num_mt_frac = []
    for cell in cells_annot:
        total = len(adata_low_quality.obs[adata_low_quality.obs.Annotation == cell])
        low_qual.append(total)
        n_doub = len(adata_low_quality[(adata_low_quality.obs.Annotation == cell) & (adata_low_quality.obs.doublet == True)].obs)
        num_doublet.append(n_doub)
        low_count_genes = len(adata_low_quality[(adata_low_quality.obs.Annotation == cell) & ((adata_low_quality.obs.n_counts< min_ncounts) | (adata_low_quality.obs.n_genes< min_genes))].obs)
        num_low_count_and_genes.append(low_count_genes)
        ncells_mt = len(adata_test[(adata_test.obs.Annotation == cell) & (adata_test.obs.mt_frac > min_mtfrac)])
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
    mis_pred_df.iloc[-1,0] = 'Total'
    chosen_cell_type = mis_pred_df_temp.iloc[(np.argmax(mis_pred_df_temp['Perc_Unexplained_Total_Pop'])), :].Annotation
    print('Low quality cells detected and dataframe created')
    print("--- %s mins ---" % int((time.time() - start_time)/60))
    return mis_pred_df 
    

def plot_avg_gene_expression(adata):
    """Returns average gene expression of differentially expressed genes for correctly and incorrectly classified cells"""
    if ('log1p') in adata.uns.keys():
        adata_test = adata.copy()
        start_time = time.time()
        print('Computing')
        #Change Sex and Prediction to Strings
        adata_test.obs.Sex = adata_test.obs.Sex.astype(str)
        adata_test.obs.Predictions = adata_test.obs.Predictions.astype(str)
        #Split the data into males and females
        females = adata_test[adata_test.obs.Sex == 'F']
        males = adata_test[adata_test.obs.Sex == 'M']
        #Make the Group Column for calculating DE genes
        females.obs['Group'] = np.where((females.obs['Sex'] == females.obs['Predictions']), 'Correct Females' , 'Incorrect Females')
        males.obs['Group'] = np.where((males.obs['Sex'] == males.obs['Predictions']), 'Correct Males' , 'Incorrect Males')
        #Calculate DE genes
        sc.tl.rank_genes_groups(females, 'Group', method='t-test')
        sc.tl.rank_genes_groups(males, 'Group', method='t-test')
        print('Genes Identified')
        #Create list of those genes
        females_names = pd.DataFrame((females.uns['rank_genes_groups']['names'])).head(20)
        males_names = pd.DataFrame((males.uns['rank_genes_groups']['names'])).head(20)
        fem_correct = females_names.iloc[:,0].tolist()
        fem_incorrect = females_names.iloc[:,1].tolist()
        male_correct = males_names.iloc[:,0].tolist()
        male_incorrect = males_names.iloc[:,1].tolist()
        #Create required lists for FOR loop
        list_all = [fem_correct, fem_incorrect, male_correct, male_incorrect]
        names = ['females_correct', 'females_incorrect', 'males_correct', 'males_incorrect']
        adata_list=[females, females, males, males]
        df_names_list = ['df_females_correct', 'df_females_incorrect', 'df_males_correct', 'df_males_incorrect']
        #Create the inital DF
        for ls,name,ad in zip(list_all, names, adata_list):
            temp_adata = ad[:,(ls[0])]
            temp_adata.obs[ls[0]] = temp_adata.X.sum(axis = 1)
            globals()['df' + '_' + name] = pd.DataFrame(temp_adata.obs.groupby('Annotation')[ls[0]].mean())
        #Create list of DFs
        df_list = []
        for i in names:
            df_list.append(globals()['df' + '_' + i])
        #Create final DF for CM, CF, IM, IF
        for genes_list, df, adata in zip(list_all, df_list, adata_list):
            for genes in genes_list[1:]:
                temp_adata = adata[:,genes]
                temp_adata.obs[genes] = temp_adata.X.sum(axis = 1)
                df[genes] = dict(temp_adata.obs.groupby('Annotation')[genes].mean())
            df.index.name = None
        print('Dataframes created')
        print('Plotting')
        #Plots
        fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(nrows=2, ncols=2,  figsize=(20,10))
        fig.subplots_adjust(hspace=0.5, wspace=0.001)
        sb.heatmap(df_females_correct, ax=ax1, cmap='GnBu', alpha=0.6, cbar=False).set(title="Correct Females" )
        fig.colorbar(ax1.collections[0], ax=ax1,location="left", use_gridspec=False, pad=0.5, shrink=0.5)
        sb.heatmap(df_females_incorrect, ax=ax2, cmap='OrRd', alpha=0.6,cbar=False).set(title="Incorrect Females")
        fig.colorbar(ax2.collections[0], ax=ax2,location="right", use_gridspec=False, pad=0.5, shrink=0.5)
        sb.heatmap(df_males_correct, ax=ax3, cmap='GnBu', alpha=0.6, cbar=False).set(title="Correct Males" )
        fig.colorbar(ax3.collections[0], ax=ax3,location="left", use_gridspec=False, pad=0.5, shrink=0.5)
        sb.heatmap(df_males_incorrect, ax=ax4, cmap='OrRd', alpha=0.6,cbar=False).set(title="Incorrect Males")
        fig.colorbar(ax4.collections[0], ax=ax4,location="right", use_gridspec=False, pad=0.5, shrink=0.5)
        ax2.yaxis.tick_right()
        ax4.yaxis.tick_right()
        ax2.yaxis.set_tick_params(rotation=0)
        ax4.yaxis.set_tick_params(rotation=0)
        plt.show()
        print("--- %s sec ---" % int((time.time() - start_time)))
        return fig
    else:
        print('Please provide Normalized and Log transformed Anndata')
        return None
    
    
        