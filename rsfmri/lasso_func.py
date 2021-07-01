###################### ACT-R + PYTHON TEMPLATE #######################
#   Author: Cher Yang
#   Date: 3.20.2021
# This script provides lasso analysis functions
#
# Bugs: 
#
# TODO: 5/27: change upsampling to downsampling
# 
# Requirement: 
#
#
#
###################### ####################### #######################


import pandas as pd
import numpy as np
import json
import os, glob
import itertools
import time
import random
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_curve, auc, roc_auc_score, accuracy_score
from sklearn.metrics import plot_confusion_matrix, confusion_matrix
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import lasso_path, enet_path
from sklearn.svm import l1_min_c
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import KFold, RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import validation_curve
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.utils import resample
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.model_selection import HalvingRandomSearchCV
from nilearn import plotting
from nilearn import datasets


############### LOAD DATA ###############
def load_subj(model_dat, CORR_DIR='./connectivity_matrix', TASK_DIR='/REST1/', SES_DIR='/ses-01/', corr_fname='mr_pcorr.txt', znorm=True, warn=True):
    """ this function load correlation matrix for each subj """
    subj_dict = {}
    HCPIDs = model_dat['HCPID'].to_list()
    for HCPID in HCPIDs:
        sub_dir = 'sub-'+HCPID.split('_')[0]
        sub_fpath = CORR_DIR+TASK_DIR+sub_dir+SES_DIR+corr_fname
        try:
            sub_df = pd.read_csv(sub_fpath, header=0).round(10)

            # Fisher transformation
            # NOTE: zscore function default axis=0, need to specify as None to make whole matrix zscore
            # To eliminate rounding errors, need to round 10 digits
            if znorm:
                sub_df = pd.DataFrame(stats.zscore(sub_df, axis=None), columns=sub_df.columns, index=sub_df.index).round(10)
            subj_dict[HCPID] = sub_df
        except:
            if warn: print("WARNING: rsfMRI data missing", HCPID)

    # convert to wide format
    subj_long = pd.DataFrame()
    for HCPID, df in subj_dict.items():
        subj_long[HCPID] = matrix2vector(df)
    subj_wide = pd.DataFrame(subj_long).T

    # rename columns
    roi_list = list(range(1, 265))
    roi_list = list(itertools.product(roi_list, roi_list))
    subj_wide.columns = [str(list(t)) for t in roi_list]
    subj_wide = subj_wide.reset_index().rename(columns={'index': 'HCPID'})

    # include dv: categorical variable either 1/0 (1=model1, 0=model2)
    subj_wide = pd.merge(model_dat[['HCPID', 'best_model1']], subj_wide, on="HCPID")
    return subj_wide

def scale_X(subj_dat, features, DV):
    # standardize X
    std_scaler = preprocessing.StandardScaler()
    subj_wide = pd.DataFrame(std_scaler.fit_transform(subj_dat[features]), columns=subj_dat.columns)


############### REFORMAT MATRIX ###############
def matrix2vector(adj_matrix):
    """
    convert a NxN adjacency matrix to a Nx1 vector
    :param adj_matrix: NxN adjacency matrix
    :return: adjacency vector Nx1
    """
    vec = []
    for i, row in adj_matrix.iteritems():
        vec.extend(row.to_list())
    return np.array(vec)

def vector2matrix(adj_vector):
    """
    convert a N adjacency vector to a MxM vector
    :param adj_matrix: N adjacency vector
    :return: adjacency matrix MxM
    """
    size = int(np.sqrt(len(adj_vector)))
    assert size==264
    mat = np.zeros((size, size))
    count = 0
    for i in range(size):
        for j in range(size):
            mat[i][j] = adj_vector[count]
            count += 1
    return np.matrix(mat)

def vectorize_mat_name(row_name_list, col_name_list):
    """
    convert NxN adjacency matrix name into a Nx1 column name
    :param row_name_list: a list of 264 connectivity id
    :param col_name_list: a list of 264 connectivity id
    :return: a Nx1 vector name
    """
    vec = []
    for r in row_name_list:
        for c in col_name_list:
            vec.append(json.dumps([r, c]))
    return np.array(vec)


############### COI ###############
def censor(vec_df, COI):
    """
    select a subset of connectivity vector
    :param vec_df: a Nx1 vector dataframe
    :param COI: connectivity of interests
    :return: a subset of connectivity vector
    """
    vec_df['censor'] = vec_df['connID'].apply(lambda x: json.loads(x)[0] < json.loads(x)[1])
    vec_df = vec_df.merge(COI, on=['netName'])
    return vec_df[vec_df['censor'] == True]

def get_vector_df(power2011, NOI):
    """
    generate a connectivity vector df
    :return: connectivity vector df after censoring
    """
    vector_df = get_ROI_df(power2011)

    COI = pd.DataFrame({'netName': vectorize_mat_name(NOI, NOI)})

    censored_df = censor(vector_df, COI)
    return censored_df

def get_ROI_df(power2011):
    # vectorize col
    col_vec = vectorize_mat_name(range(1, 265), range(1, 265))
    # vectorize network
    net_vec = vectorize_mat_name(power2011['Network'].to_list(), power2011['Network'].to_list())
    # vectorize network_name
    netname_vec = vectorize_mat_name(power2011['NetworkName'].to_list(), power2011['NetworkName'].to_list())
    # concate vector df
    vector_df = pd.DataFrame({'connID': col_vec, 'netID': net_vec, 'netName': netname_vec})
    return vector_df

def get_subj_df(subj_wide, censor):
    """
    generate a subject df
    :return: a subject df
    """
    res = subj_wide[['HCPID', 'best_model1'] + censor['connID'].to_list()].dropna(how='any')
    return res

def reverse_connID(connID):
    roi1, roi2 = json.loads(connID)
    newconnID = [roi2, roi1]
    res = json.dumps(newconnID)
    return res

############### LOGISTIC MODEL ###############
def lasso_logistic_fit(train_data, test_data, features, DV, c=1e-2, verbose=True):
    logistic_model = LogisticRegression(penalty='l1', random_state=1, solver='saga', C=c)
    logistic_model.fit(train_data[features], train_data[DV])

    # calculate score
    train_score = logistic_model.score(train_data[features], train_data[DV])
    test_score = logistic_model.score(test_data[features], test_data[DV])

    # calculate accuracy
    test_accuracy = accuracy_score(test_data[DV], logistic_model.predict(test_data[features]))

    if verbose:
        print("L1 logistic model - Training score: ", round(train_score, 4))
        print("L1 logistic model - Testing score: ", round(test_score, 4))
        print("L1 logistic model - Testing accuracy: ", round(test_accuracy, 4))
    return logistic_model

def grid_search_lasso(X, y, lambda_values=None, num_cv=20, plot_path=False):
    start = time.time()

    # define model
    model = LogisticRegression(penalty='l1', solver='saga', fit_intercept=False, max_iter=10000, tol=0.01)

    # define parameter space
    space = dict()
    #space['solver'] = ['newton-cg', 'lbfgs', 'liblinear']
    #space['penalty'] = ['l1']
    if lambda_values == None: lambda_values=1.0/np.logspace(-3, 3, 100)
    space['C'] = 1.0/lambda_values


    # define search
    grid_search = GridSearchCV(estimator=model, param_grid=space, n_jobs=-1,
                               cv=num_cv, scoring='accuracy', error_score=0, return_train_score=True)

    # use LOOCV to evaluate model
    # scores = cross_val_score(model, train_data[features], train_data[DV], scoring='accuracy', cv=cv, n_jobs=-1)

    # execute search
    grid_result = grid_search.fit(X, y)

    # summarize result
    print('Best Score: %s' % grid_result.best_score_)
    print('Best Hyperparameters: %s' % grid_result.best_params_)
    print("Time usage: %0.3fs" % (time.time() - start))

    #if plot_path: plot_regularization_path(train_data, features, DV, lambda_values, 1.0/grid_result.best_params_['C'])

    return grid_result

def random_grid_search_lasso(X, y, lambda_values=None, num_cv=20, plot_path=False):
    start = time.time()

    # define model
    model = LogisticRegression(penalty='l1', solver='saga', fit_intercept=False, max_iter=10000, tol=0.1)

    # define evaluation
    # cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

    # define parameter space
    space = dict()
    #space['solver'] = ['newton-cg', 'lbfgs', 'liblinear']
    #space['penalty'] = ['l1']
    if lambda_values == None: lambda_values=1.0/np.logspace(-3, 3, 100)
    space['C'] = 1.0/lambda_values

    # define search
    grid_search = RandomizedSearchCV(model, space, n_iter=1000, scoring='accuracy', n_jobs=-1, cv=num_cv, random_state=1, return_train_score=True)
    
    # execute search
    grid_result = grid_search.fit(X, y)
    
    # summarize result
    print('Best Score: %s' % grid_result.best_score_)
    print('Best Hyperparameters: %s' % grid_result.best_params_)
    print("Time usage: %0.3fs" % (time.time() - start))

    #if plot_path: plot_regularization_path(train_data, features, DV, lambda_values, 1.0/grid_result.best_params_['C'])
    
    return grid_result


def balance_training_sample(subj_dat, DV, method='up'):
    num_class0 = subj_dat[DV].value_counts()[0]
    num_class1 = subj_dat[DV].value_counts()[1]

    if num_class0 > num_class1:
        subj_majority = subj_dat[subj_dat[DV]==0]
        subj_minority = subj_dat[subj_dat[DV]==1]
        majority_count = num_class0
        minority_count = num_class1
    else:
        subj_majority = subj_dat[subj_dat[DV]==1]
        subj_minority = subj_dat[subj_dat[DV]==0]
        majority_count = num_class1
        minority_count = num_class0
	
    if method=='up':
        # upsample minority class
        subj_minority_upsampled = resample(subj_minority, replace=True, n_samples=majority_count, random_state=1)
        
        # combine majorrity class with upsampled miniority class
        subj_upsampled = pd.concat([subj_majority, subj_minority_upsampled])
        subj_upsampled = subj_upsampled.reset_index()
        return subj_upsampled
    else:
        # downsample majority class
        subj_majority_downsampled = subj_majority.sample(n=minority_count, replace=False)
        
        # combine minority with downsampled majority class
        subj_downsampled = pd.concat([subj_minority, subj_majority_downsampled])
        subj_downsampled = subj_downsampled.reset_index()
        return subj_downsampled

def loocv_train_test_split_ith(subj_dat, i):
    assert i < len(subj_dat)

    cv = LeaveOneOut()
    
    index_list = list(cv.split(subj_dat))
    train_index, test_index = index_list[i][0], index_list[i][1]
    train_data = subj_dat.iloc[train_index]
    test_data = subj_dat.iloc[test_index]
    return train_data, test_data

def loocv_logistic_retrain(subj_censored, features, DV, best_c):
    start = time.time()
    cv = LeaveOneOut()
    num_cv = cv.get_n_splits(subj_censored)
    res = []

    for i in range(num_cv):
        # define train, test data
        train_data, test_data = loocv_train_test_split_ith(subj_censored, i)

        # define model
        best_lasso_model = LogisticRegression(penalty='l1', solver='saga', C=best_c, fit_intercept=False)

        # fit model 
        best_lasso_model.fit(train_data[features], train_data[DV])
        
        # predict prob
        yhat = best_lasso_model.predict(test_data[features])
        yprob = best_lasso_model.predict_proba(test_data[features])[:, 1]

        # save pred outcomes
        res.append([test_data['HCPID'].values[0], test_data[DV].values[0], yhat[0], yprob[0]])
    
    res = pd.DataFrame(res, columns = ['HCPID', 'ytrue', 'yhat', 'yprob'])

    print('Time Usage (s)', round((time.time() - start), 4))
    return res

def cv_logistic_retrain(subj_censored, features, DV, best_c, num_cv):
    start = time.time()

    # define model
    model = LogisticRegression(penalty='l1', solver='saga', C=best_c, fit_intercept=False)

    # define cross-validation
    # kfold = KFold(10, False, 2)

    # Evaluate the model using k-fold CV
    cross_val_scores = cross_val_score(model, subj_censored[features], subj_censored[DV], cv=num_cv, scoring='accuracy')

    all_ytrue = subj_censored[DV].values
    all_yhat = cross_val_predict(model, subj_censored[features], subj_censored[DV], cv=num_cv, method='predict')
    all_yprob = cross_val_predict(model, subj_censored[features], subj_censored[DV], cv=num_cv, method='predict_proba')

    # Get the model performance metrics
    print("Accuracy Mean: " + str(cross_val_scores.mean()))
    print("Time Usage", round(time.time() - start, 4))
    return all_ytrue, all_yhat, all_yprob, cross_val_scores



############### MODEL COMPARE ###############
def tune_hyperparam(model, X, y, param_grid, cv=20):
    #model = LogisticRegression(fit_intercept=False, penalty='l1', max_iter=10000, warm_start=True, solver='liblinear')
    #param_grid = {'C': 1/np.logspace(-3, 3, 100)}
    scoring = {'Accuracy': 'accuracy', 'AUC': 'roc_auc'}
    gs = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=cv, refit='AUC', return_train_score=True)
    gs.fit(X, y)
    print('='*20)
    print("best params: " + str(gs.best_estimator_))
    print("best params: " + str(gs.best_params_))
    print('best score:', gs.best_score_)
    print('='*20)
    return gs

def plot_hyperparam(results, save_path=False):
    #scoring = {'Accuracy': 'accuracy', 'AUC': 'roc_auc', 'Log_loss': 'neg_log_loss'}
    #param_grid = {'C': 1/np.logspace(-1, 3, 100)}
    scoring = {'Accuracy': 'accuracy', 'AUC': 'roc_auc'}
    param_name=results.params.values.tolist()[0].split("'")[1]
    param_grid={param_name:np.array([p[0] for p in results.filter(regex='param_').values.tolist()])}
    
    plt.figure(figsize=(10, 6))
    plt.title("GridSearchCV evaluating using multiple scorers simultaneously",fontsize=16)
    
    plt.xlabel("Hyper Parameter: {:}".format(param_name))
    plt.ylabel("Score")
    plt.grid()
    
    ax = plt.axes()
    ax.set_xlim(param_grid[param_name].min(), param_grid[param_name].max()) 
    #ax.set_ylim(0, 1.2)
    
    # Get the regular numpy array from the MaskedArray
    X_axis = np.array(param_grid[param_name], dtype=float)
    
    for scorer, color in zip(list(scoring.keys()), ['g', 'k', 'b']): 
        for sample, style in (('train', '--'), ('test', '-')):
            sample_score_mean = -results['mean_%s_%s' % (sample, scorer)] if scoring[scorer]=='neg_log_loss' else results['mean_%s_%s' % (sample, scorer)]
            sample_score_std = results['std_%s_%s' % (sample, scorer)]
            ax.fill_between(X_axis, sample_score_mean - sample_score_std,
                            sample_score_mean + sample_score_std,
                            alpha=0.1 if sample == 'test' else 0, color=color)
            ax.plot(X_axis, sample_score_mean, style, color=color,
                    alpha=1 if sample == 'test' else 0.7,
                    label="%s (%s)" % (scorer, sample))
        
        best_index = np.nonzero((results['rank_test_%s' % scorer] == 1).values)[0][0]
        best_score = -results['mean_test_%s' % scorer][best_index] if scoring[scorer]=='neg_log_loss' else results['mean_test_%s' % scorer][best_index]
        
        # Plot a dotted vertical line at the best score for that scorer marked by x
        ax.plot([X_axis[best_index], ] * 2, [0, best_score],
                linestyle='-.', color=color, marker='x', markeredgewidth=3, ms=8)
        
        # Annotate the best score for that scorer
        ax.annotate("%0.2f" % best_score,
                    (X_axis[best_index], best_score + 0.005))
        
    plt.legend(loc="best")
    plt.grid('off')
    if save_path: plt.savefig(save_path)
    else: plt.show()
        
def crossvalscore(model, X, y):
    scores_accuracy = cross_val_score(model, X, y, cv=10, scoring='accuracy', n_jobs=-1)
    scores_balanced_accuracy = cross_val_score(model, X, y, cv=10, scoring='balanced_accuracy', n_jobs=-1)
    scores_f1 = cross_val_score(model, X, y, cv=10, scoring='f1', n_jobs=-1)
    scores_auc = cross_val_score(model, X, y, cv=10, scoring='roc_auc', n_jobs=-1)
    scores_log_loss = cross_val_score(model, X, y, cv=10, scoring='neg_log_loss', n_jobs=-1)

    rand_scores = pd.DataFrame({
        'cv':range(0,10),
        'accuracy score':scores_accuracy, 
        'balanced accuracy score':scores_balanced_accuracy, 
        'f1 score':scores_f1, 
        'roc_auc score':scores_auc,
        'log loss score':scores_log_loss
        })
    
    print('Accuracy :',rand_scores['accuracy score'].mean())
    print('Balanced accuracy :',rand_scores['balanced accuracy score'].mean())
    print('f1 :',rand_scores['f1 score'].mean())
    print('ROC_AUC :',rand_scores['roc_auc score'].mean())
    print('Log Loss :',rand_scores['log loss score'].mean())
    return rand_scores.sort_values(by='roc_auc score',ascending=False)

def feature_selection(model, X, y):
    rfe = RFE(model, 8)
    rfe = rfe.fit(X, y)

    rfecv = RFECV(estimator=model, step=1, cv=10, scoring='accuracy')
    rfecv.fit(X, y)

def evaluate_model(model, X, y, cv=20):
    scores_accuracy = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    scores_auc = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')

    print('K-fold cross-validation results:')
    print(model.__class__.__name__+" average accuracy is %2.3f" % scores_accuracy.mean())
    print(model.__class__.__name__+" average auc is %2.3f" % scores_auc.mean())

    rand_scores = pd.DataFrame({
        'cv':range(0,cv),
        'accuracy':scores_accuracy, 
        'roc_auc':scores_auc,
        })
    return rand_scores.sort_values(by='roc_auc',ascending=False)

def random_forest_tuning(X, y):
    rf_model = RandomForestClassifier(random_state=0)
    random_grid={'bootstrap': [True, False],
                'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
                'max_features': ['auto', 'sqrt'],
                'min_samples_leaf': [1, 2, 4],
                'min_samples_split': [2, 5, 10],
                'n_estimators': [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]}
    rf_random = RandomizedSearchCV(estimator = rf_model, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=3, random_state=42, n_jobs = -1)
    rf_random.fit(X, y)
    print('best parameters:', rf_random.best_params_)
    print('best model:', rf_random.best_estimator_)
    # param_grid={'bootstrap': True,
    #             'max_depth':[6, 8, 10, 12, 14, 16],
    #             'max_features': 'sqrt', 
    #             'min_samples_leaf': 2,
    #             'min_samples_split': 10,
    # }
    return rf_model

def decision_tree_tuning(X, y):
    dt_model = DecisionTreeClassifier()
    param_dist = {"max_depth": [3, None],
              "max_features": list(range(1, 10)),
              "min_samples_leaf": list(range(1, 10)),
              "criterion": ["gini", "entropy"]}
    dt_model.fit(X, y)
    return dt_model

def svm_tuning(X, y):
    svm_model = svm.SVC()
    random_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1,0.1,0.01,0.001],'kernel': ['rbf', 'poly', 'sigmoid']}
    svm_random = RandomizedSearchCV(estimator = svm_model, param_distributions = random_grid, refit=True, n_iter = 100, cv = 3, verbose=3, random_state=42, n_jobs = -1)
    svm_random.fit(X, y)
    print('best parameters:', svm_random.best_params_)
    print('best model:', svm_random.best_estimator_)
    return svm_model

def neural_network(X, y):
    nn_model = MLPClassifier()
    random_grid = {
    'hidden_layer_sizes': [(10,30,10),(20,)],
    'alpha': [0.0001, 0.05],
    'learning_rate': ['constant','adaptive']}
    nn_random = RandomizedSearchCV(estimator = nn_model, param_distributions = random_grid, refit=True, n_iter = 100, cv = 3, verbose=3, random_state=42, n_jobs = -1)
    nn_random.fit(X, y)
    print('best parameters:', nn_random.best_params_)
    print('best model:', nn_random.best_estimator_)
    return nn_model


############### VISUALIZE LOG MODEL ###############
def print_coefficients(coef, features):
    """
    This function takes in a model column and a features column.
    And prints the coefficient along with its feature name.
    """
    feats = list(zip(features, coef))
    print(*feats, sep="\n")

def plot_roc_curve(logistic_model, test_data, features, DV):
    y_score = logistic_model.predict_proba(test_data[features])[:, 1]
    false_positive_rate, true_positive_rate, threshold = roc_curve(test_data[DV], y_score)
    print('ROC Accuracy Score for Logistic Regression: ', roc_auc_score(test_data[DV], y_score))
    
    # plot
    plt.subplots(1, figsize=(5, 5))
    plt.title('ROC - Logistic regression')
    plt.plot(false_positive_rate, true_positive_rate)
    plt.plot([0, 1], ls="--")
    plt.plot([0, 0], [1, 0], c=".7"), plt.plot([1, 1], c=".7")
    plt.ylabel('Sensitivity: TPR')
    plt.xlabel('Specifity: PPR')
    plt.show()
    plt.close()

def plot_roc_curve_loo(pred_data, save_plot=False, cache_prefix='r1s1_'):
    # calculate fp, tp
    fp, tp, _ = roc_curve(pred_data['ytrue'].values, pred_data['yprob'].values)
    
    # plot
    plt.subplots(1, figsize=(15, 10))
    plt.title('ROC - Logistic regression')
    roc_auc = auc(fp, tp)
    plt.plot(fp, tp, lw=15, 
             label='ROC (AUC = %0.2f)' % (roc_auc))
    plt.plot([0, 1], ls="--")
    plt.plot([0, 0], [1, 0], c=".7"), plt.plot([1, 1], c=".7")
    plt.ylabel('Sensitivity: TPR')
    plt.xlabel('Specifity: FPR')
    plt.legend(loc="lower right", fontsize='large')
    
    # save
    if save_plot: plt.savefig('./bin/'+cache_prefix+'roc.png')
    plt.show()
    plt.close()
    
def plot_confusion_matrix(logistic_model, test_data, features, DV):
    """
    Plots a confusion matrix using the values
       tp - True Positive
       fp - False Positive
       fn - False Negative
       tn - True Negative
    """
    y_pred = logistic_model.predict(test_data[features])
    tn, fp, fn, tp = confusion_matrix(test_data[DV], y_pred).ravel()
    data = np.matrix([[tp, fp], [fn, tn]])
    plt.subplots(1, figsize=(15, 10))
    sns.heatmap(data, annot=True, xticklabels=['Actual Pos', 'Actual Neg'], yticklabels=['Pred. Pos', 'Pred. Neg'])
    plt.show()

def plot_confusion_matrix_loo(pred_data, norm=None, save_plot=False, cache_prefix='r1s1_'):
    tn, fp, fn, tp = confusion_matrix(pred_data['ytrue'].values, pred_data['yhat'].values, normalize = norm).ravel()
    data = np.matrix([[tp, fp], [fn, tn]])
    
	# plot
    plt.subplots(1, figsize=(15, 18))
    plt.title('Confusion Matrix - Logistic Regression\n Accuracy Score: {:.4f}'.format(accuracy_score(pred_data['ytrue'].values, pred_data['yhat'].values)))
    sns.heatmap(data, annot=True, xticklabels=['Actual Pos', 'Actual Neg'], yticklabels=['Pred. Pos', 'Pred. Neg'])
    if save_plot: plt.savefig('./bin/'+cache_prefix+'confusion_matrix.png')                         
    plt.show()
    plt.close()

def save_regularization_path(X, y, best_lambda, lambda_values=None):
    start = time.time()
    model = LogisticRegression(penalty='l1', solver='saga', tol=1e-3, max_iter=int(1e2), warm_start=True)

    c_values = np.logspace(-2, 2, 100)
    if lambda_values.any() == None: lambda_values = 1.0/c_values
    coefs_ = []
    for c in c_values:
        model.set_params(C=c)
        model.fit(X, y)
        coefs_.append(model.coef_.ravel().copy())

    coefs_ = np.array(coefs_)
    print("Time usage: %0.3fs" % (time.time() - start))
    return coefs_

def plot_regularization_path(coefs_, best_lambda, lambda_values, save_plot=False, cache_prefix='r1s1_'):
	# plot
	sns.color_palette('Set2')
	fig, ax = plt.subplots(figsize=(20, 12))
	ax.plot(lambda_values, coefs_, marker='o', linewidth=2)
	ax.axvline(best_lambda, linestyle='--', color='k')
	ax.text(x=0.8, y=0.7, s='best lambda\n{:.2e}'.format(best_lambda), color='k', fontsize=40,
			transform=ax.transAxes,
			horizontalalignment='center', verticalalignment='center',
			bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
	ax.set_xscale('log')
	plt.xscale("log")
	plt.xlabel(r"$\lambda$")
	plt.ylabel('Coefficients')
	plt.title('Logistic Regression: Cross-Validation Path')
	plt.axis('tight')
	if save_plot: plt.savefig('./bin/'+cache_prefix+'coefs.png')
	plt.show()
	plt.close()

def save_regularization_score(X, y, best_lambda=None, lambda_values=None, num_cv=10):
    start = time.time()
    
    if lambda_values == None: 
        param_range = np.logspace(-2, 2, 100)
        lambda_values = 1.0/param_range
    else:
        param_range = 1.0/lambda_values
    
    model = LogisticRegression(penalty='l1', solver='saga', fit_intercept=False)
    train_scores, test_scores = validation_curve(model, X, y, param_name='C', error_score='raise', cv=num_cv,
                                                 param_range=param_range, scoring="roc_auc", n_jobs=1)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    best_lambda_new = lambda_values[np.argmax(test_scores_mean)]
    
    if best_lambda == None: 
        best_lambda = best_lambda_new
    
    score_df = pd.DataFrame({'train_scores_mean':train_scores_mean, 'train_scores_std':train_scores_std,
                             'test_scores_mean':test_scores_mean, 'test_scores_std':test_scores_std,
                             'lambda_values':lambda_values,})
    score_df['best_lambda'] = best_lambda
    score_df['best_lambda_new'] = best_lambda_new
    print("Time usage: %0.3fs" % (time.time() - start))
    return score_df

def plot_regularization_score(score_df, best_lambda, save_plot=False, cache_prefix='r1s1_'):
    lambda_values = score_df['lambda_values']
    train_scores_mean = score_df['train_scores_mean']
    train_scores_std = score_df['train_scores_std']
    test_scores_mean = score_df['test_scores_mean']
    test_scores_std = score_df['test_scores_std']
    
    # plot
    fig, ax = plt.subplots(figsize=(20, 12))
    plt.title("Logistic Regression: Cross-Validation Score")
    plt.xlabel(r"$\lambda$")
    plt.ylabel("Score")
    plt.ylim(0.0, 1.1)
    lw = 2
    
    ax.semilogx(lambda_values, train_scores_mean, label="Training score",
        color="darkorange", lw=lw)
    ax.fill_between(lambda_values, train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std, alpha=0.2, color="darkorange", lw=lw)
    ax.semilogx(lambda_values, test_scores_mean, label="Cross-validation score",
        color="navy", lw=lw)
    ax.fill_between(lambda_values, test_scores_mean - test_scores_std, 
        test_scores_mean + test_scores_std, alpha=0.2, color="navy", lw=lw)
    ax.axvline(best_lambda, linestyle='--', color='k')
    ax.text(x=best_lambda, y=0.7, s='best lambda\n{:.2e}'.format(best_lambda), color='k', fontsize=40,
            #transform=ax.transAxes, 
            horizontalalignment='center', verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    plt.legend(loc="best")
    plt.axis('tight') 
    if save_plot: plt.savefig('./bin/'+cache_prefix+'scores.png')
    plt.show()
    plt.close()

def plot_prediction(subj_wide, test_data, features, DV, best_lasso, save_plot):
	test_data[[DV + '_lasso_pred']] = best_lasso.predict_proba(test_data[features])[:, 1]
	test_data['threshold'] = subj_wide['best_model1'].mean()
	test_data = test_data.sort_values(by=['best_model1'])
	
	# plot
	plt.plot(test_data['HCPID'], test_data[DV], 'b^', label='observed')
	plt.plot(test_data['HCPID'], test_data[DV + '_lasso_pred'], 'rx', label='lasso_pred')
	plt.plot(test_data['HCPID'], test_data['threshold'], '-', label='thresh')
	
	plt.xlabel('subID')
	plt.ylabel(DV)
	plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.1)
	plt.show()
	plt.close()


def plot_prediction_loo(pred_data, threshold=0.5, drop_dup=False, save_plot=False, cache_prefix='r1s1_'):
    
    if drop_dup:
        x_ax = 'HCPID'
        x_labsize = 1
    else:
        x_ax = 'index'
        x_labsize = 20
    
    pred_data = pred_data.reset_index()
    pred_data['predcorrect'] = np.where(pred_data['ytrue'] == pred_data['yhat'], 'Logistic Model: Correct', 'Logistic Model: Incorrect')

	# plot
    fig, ax = plt.subplots(figsize=(15,12))

    sns.scatterplot(data=pred_data, x=x_ax, y="ytrue", marker="s", alpha=0.9, s=250)
    sns.scatterplot(data=pred_data, x=x_ax, y="yprob", s=250, hue='predcorrect', style='predcorrect', markers=['o', 'X'], palette="Set2")
    plt.axhline(y=threshold, color='black', linestyle='--')

    plt.xlabel('Subjects')
    plt.ylabel('Prediction(Probability)')
    plt.title('Logistic Regression: \nPrediction Performance (Leave-One-Out)')
    plt.tick_params(labelsize=x_labsize)
    plt.xticks(rotation=30)
    plt.legend(title="", bbox_to_anchor=(0.01, 0), borderaxespad=0, fontsize=20,
               loc='lower left',
               labels=['Threshold', 'ACT-R Model Identification', 'Logistic Model Prediction', 'Correct Prediction', 'Inorrect Prediction'])
    
    if save_plot: plt.savefig('./bin/'+cache_prefix+'prediction.png')
    plt.show()
    plt.close()

    
############### BRAIN CONNECTIONS ###############
def calc_beta_pr(pr_df, features, beta_df, subjID):
    # calculate beta * pr
    pr_df['beta'] = beta_df['beta']
    pr_df[features] = pr_df[features].apply(lambda x: x*pr_df['beta'])
    
    # create a beta_pr df
    bpr_df = pr_df[pr_df['HCPID']==subjID][features].T.reset_index()
    bpr_df.columns = ['connID', 'beta_pr']
    return bpr_df

def map_beta(censor, coeff_df, power2011):
    # create a left censor
    censor_left = censor.copy()
    censor_left['beta'] = coeff_df

    # make symmetric censor df
    censor_right = censor_left[['connID', 'beta']].copy()
    censor_right['connID'] = censor_right['connID'].apply(reverse_connID)
    censor_LR = censor_left.append(censor_right, ignore_index=True)
    censor_LR = censor_LR[['connID', 'beta']]

    # merge to main roi df
    roi_df = get_ROI_df(power2011)
    roi_df = roi_df.merge(censor_LR, how="left", on=['connID'])
    roi_df['beta'] = roi_df['beta'].replace(np.nan, 0.0)

    # reformat beta matrix
    adj_vector = roi_df['beta'].values
    adj_beta = pd.DataFrame(vector2matrix(adj_vector), dtype='float')

    # grab center coordinates for atlas labels
    coor_vector = np.array([(0, 0, 0) for i in range(264)])
    power = datasets.fetch_coords_power_2011()
    power_coords = np.vstack((power.rois['x'], power.rois['y'], power.rois['z'])).T

    return adj_beta, power_coords

def average_corr(subj_wide, DV):
    """
    This func averages subject corr matrix
    :param subj_wide: NxM dataframe, N=number of subjects; M=264*264
    :param DV:
    :return: averaged corr matrix for all subj
    """
    subj_zmean_vec =  subj_wide.drop(['HCPID', DV], axis=1).mean(axis=0)
    subj_zmean_mat = vector2matrix(subj_zmean_vec)
    subj_mean_mat = np.tanh(subj_zmean_mat)
    return subj_mean_mat

def concat_wbeta(power2011, NOI, beta_df, subj_mat, w_mat, subj_mean_mat):
    roi_df = get_vector_df(power2011, NOI)
    
    full_col = subj_mat.drop(['HCPID', 'best_model1'], axis=1).columns
    w_df = pd.DataFrame(matrix2vector(w_mat), columns=['weighted_corr'])
    w_df['connID'] = full_col
    
    subj_mean_df = pd.DataFrame(matrix2vector(subj_mean_mat), columns=['avg_corr'])
    subj_mean_df['connID'] = full_col
    
    res = roi_df.merge(beta_df, on = 'connID').merge(subj_mean_df, on = 'connID').merge(w_df, on = 'connID')
    return res

def plot_brain_connections(mat, power_coords, mat_name='beta_mat', thre='99.9%', save_plot=False, cache_prefix='r1s1_'):
    if mat_name=='beta_mat':
        tit = 'Beta'
    elif mat_name == 'wcorr_mat':
        tit = 'Weighted Connectivity'
    else:
        tit = 'Unknown'
    
    # plot
    sns.set_style('white')
    fig = plt.figure(figsize=(10,3))
    ax = fig.add_subplot(111)
    ax.axis("off")

    # plot connectome with 80% edge strength in the connectivity
    plotting.plot_connectome(mat, power_coords, figure=fig,
                             edge_threshold=thre,
                             #node_color=,
                             colorbar=True,
                             node_size=0, # size 264
                             #alpha=.8,
                             #title='Group analysis: ' + tit,
                             edge_kwargs = {'lw':8})
    if save_plot: plt.savefig('./bin/'+cache_prefix+mat_name+thre+'.png')
    plt.show()
    plt.close()

