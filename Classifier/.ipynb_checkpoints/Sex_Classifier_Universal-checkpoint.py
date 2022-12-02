def sex_classifier_universal(adata_training, adata_test):
    start_time = time.time()
    print('Initializing')
    warnings.simplefilter("ignore", UserWarning)
    warnings.simplefilter("ignore", RuntimeWarning)
    warnings.simplefilter("ignore", FutureWarning)
    
    print('Preparing Training Adata')
    #Add Sex_Class to training adata if it does not exist
    adata_training_copy = adata_training.copy()
    adata_training_copy.obs.columns = adata_training_copy.obs.columns.str.capitalize()
    sex_classes = {'F':0, 'M':1}
    if ('Sex' or 'sex') in adata_training_copy.obs_keys():
        adata_training_copy.obs['Sex_Class'] = adata_training_copy.obs.Sex.map(sex_classes).astype('category')
    else:
        print('Please provide training adata with SEX column')
    
    print('Training Models')
    
    #Load Parameters:
    train = xgb.DMatrix(adata_training_copy.X, label=adata_training_copy.obs.Sex_Class)
    param_softmax = {'max_depth':10, 'eta':0.15, 'objective':'multi:softmax','num_class':2}
    param_softprob = {'max_depth':10, 'eta':0.15, 'objective':'multi:softprob','num_class':2}
    epochs = 20
    #Train Models
    model_softmax = xgb.train(param_softmax, train, epochs)
    model_softprob = xgb.train(param_softprob, train, epochs)
    
    print('Training Complete')
    
    #Slice genes in the test dataset and add genes from training dataset if they are unique to the training dataset. 
    adata_test_copy = adata_test.copy()
    genes_training = [i for i in adata_test_copy.var.index if i in adata_training_copy.var.index]
    #Rearrange the test dataset to match the index of training.
    adata_test_copy = adata_test_copy[:,genes_training].copy()
    
    print('Preparing Test Adata')
    
    #Merge the datasets 
    adata_test_copy.obs['Set'] = 'Test'
    adata_training_copy.obs['Set'] = 'Training'
    adata_list = [adata_training_copy, adata_test_copy]
    adata_merged = ad.concat(adata_list, join='outer')
    #Split the dataset
    adata_test_copy = adata_merged[adata_merged.obs.Set == 'Test']
    
    #Reindex test dataset
    adata_test_copy.var = adata_test_copy.var.set_index(adata_training_copy.var.index)
    
    print('Test Adata Modified For The Model')
        
    #Make the test matrix
    if ('Sex' or 'sex') in adata_test.obs_keys():
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
    print('Adding Columns to the Test Adata')
    
    #If loop for adding the class prediction of the provided sex data 
    if ('Sex' or 'sex') in adata_test.obs_keys():
        conditions = [((adata_test_copy.obs.Sex_Class == adata_test_copy.obs.Predictions) & (adata_test_copy.obs.Class_Prediction > 0.85)),
          ((adata_test_copy.obs.Sex_Class != adata_test_copy.obs.Predictions) & (adata_test_copy.obs.Class_Prediction > 0.85)), 
          ((adata_test_copy.obs.Sex_Class == adata_test_copy.obs.Predictions) & (adata_test_copy.obs.Class_Prediction < 0.85)),
          ((adata_test_copy.obs.Sex_Class != adata_test_copy.obs.Predictions) & (adata_test_copy.obs.Class_Prediction < 0.85))]
        choices = [ "True Positive", 'False Positive', 'False Negative' , 'True Negative']
        adata_test.obs['Condition'] = np.select(conditions, choices, default=np.nan)
        #Accuracy Score
        from sklearn.metrics import accuracy_score
        adata_test_copy_males = adata_test_copy[adata_test_copy.obs.Sex == 'M']
        acc_score_males = accuracy_score(adata_test_copy_males.obs.Sex_Class, adata_test_copy_males.obs.Predictions)
        print('The accuracy_score for males for universally trained model is {}'.format(acc_score_males))
        adata_test_copy_females = adata_test_copy[adata_test_copy.obs.Sex == 'F']
        acc_score_females = accuracy_score(adata_test_copy_females.obs.Sex_Class, adata_test_copy_females.obs.Predictions)
        print('The accuracy_score for females for universally trained model is {}'.format(acc_score_females))
    else:
        pass
    print('Prediction Completed')
    print("--- %s mins ---" % int((time.time() - start_time)/60))
    return model_softmax, model_softprob