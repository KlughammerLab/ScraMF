def misclassified(adata):
    print('Initializing') 
    sc.settings.verbosity = 0
    start_time = time.time()    
    #Annotate all datasets individually and then concat because they have batch effect in between them
    adata_test = adata.copy()
    #PCA analysis
    sc.pp.pca(adata_test)
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
    ad_az = ad_az[:,b]
    ad_az, annotation_az = maca.singleMACA(ad=ad_az, cell_markers=cell_markers_az,res=[1, 1.5, 2.0],n_neis=[3,5,10])
    print('Annotation Complete')
    
    #Add annotation to the original adata
    adata.obs['Annotation']=np.array(annotation_az)
    adata_test.obs['Annotation']=np.array(annotation_az)
    adata_test.obs['n_counts'] = adata_test.X.sum(axis = 1)
    adata_test.obs['n_genes'] = (adata_test.X > 0).sum(axis = 1)
    mt_gene = np.flatnonzero([gene.startswith('MT-') for gene in adata_test.var_names])
    adata_test.obs['mt_frac']= np.sum(adata_test[:, mt_gene].X, axis =1).A1/adata_test.obs['n_counts']
    adata_test = adata_test[adata_test.obs['mt_frac'] < 0.2]
    adata_test.obs.Sex = adata_test.obs.Sex.astype(str)
    adata_test.obs.Predictions = adata_test.obs.Predictions.astype(str)
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
    
    print('The most misclassified celltype is {}'.format(chosen_cell_type.upper()))
    print('Low quality cells detected and dataframe created')
    return mis_pred_df

    #Differentially expressed genes for females and males
    sc.tl.rank_genes_groups(females, 'Group', method='t-test')
    sc.tl.rank_genes_groups(males, 'Group', method='t-test')

def genes_cell_classified(adata)
    adata_test = adata.copy()
    females = adata_test[adata_test.obs.Sex == 'F']
    females.obs['Group'] = np.where((adata_test.obs['Sex'] == adata_test.obs['Predictions']), 'Correct Females' , 'Incorrect Females')
    males = adata_test[adata_test.obs.Sex == 'M']
    males.obs['Group'] = np.where((adata_test.obs['Sex'] == adata_test.obs['Predictions']), 'Correct Males' , 'Incorrect Males')
    #Plot the figures
    
    #Calculate dendogram and generate matrixplot for the genes expressed by the correct and incorrect males and females
    sc.tl.dendrogram(females, groupby='Annotation')
    sc.tl.dendrogram(males, groupby='Annotation')
    rcParams['figure.figsize']=(20,15)
    plt.figure()
    ax1 = plt.subplot(2,1,1)
    sc.pl.rank_genes_groups_matrixplot(females, n_genes=20 , key='rank_genes_groups', 
                               groupby='Annotation',cmap='BuPu', alpha=0.5, dendrogram=True, ax=ax1)
    ax2 = plt.subplot(2,1,2)
    sc.pl.rank_genes_groups_matrixplot(males, n_genes=20 , key='rank_genes_groups', 
                               groupby='Annotation',cmap='GnBu', alpha=0.8, dendrogram=True, ax=ax2)
    plots = ax1, ax2
    print('Dataframe and Plots Ready')
    print("--- %s mins ---" % int((time.time() - start_time)/60))
