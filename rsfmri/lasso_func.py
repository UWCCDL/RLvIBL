import pandas as pd
import numpy as np
import json
import os, glob
import itertools
import time
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
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import KFold, RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import validation_curve
from sklearn.feature_extraction import DictVectorizer
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
    model = LogisticRegression(penalty='l1', solver='saga', fit_intercept=False)

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
    model = LogisticRegression(penalty='l1', solver='saga', fit_intercept=False)

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

def halving_search_lasso(X, y, lambda_values=None, plot_path=False):
    start = time.time()

    # define model
    model = LogisticRegression(penalty='l1', solver='saga', fit_intercept=False)

    # define cross-validation method
    # cv = LeaveOneOut()

    # define parameter space
    space = dict()
    # space['solver'] = ['newton-cg', 'lbfgs', 'liblinear']
    # space['penalty'] = ['l1']
    if lambda_values == None: lambda_values=1.0/np.logspace(-3, 3, 100)
    space['C'] = 1.0 / lambda_values

    # define search
    grid_search = HalvingGridSearchCV(estimator=model, param_grid=space, factor=2, return_train_score=True)

    # execute search
    grid_result = grid_search.fit(X, y)

    # summarize result
    print('Best Score: %s' % grid_result.best_score_)
    print('Best Hyperparameters: %s' % grid_result.best_params_)
    print("Time usage: %0.3fs" % (time.time() - start))

    #if plot_path: plot_regularization_path(train_data, features, DV, lambda_values, 1.0 / grid_result.best_params_['C'])

    return grid_result

def balance_training_sample(subj_dat, DV):
    num_class0 = subj_dat[DV].value_counts()[0]
    num_class1 = subj_dat[DV].value_counts()[1]

    if num_class0 > num_class1:
        subj_majority = subj_dat[subj_dat[DV]==0]
        subj_minority = subj_dat[subj_dat[DV]==1]
        majority_count = num_class0
    else:
        subj_majority = subj_dat[subj_dat[DV]==1]
        subj_minority = subj_dat[subj_dat[DV]==0]
        majority_count = num_class1

    # upsample minority class
    subj_minority_upsampled = resample(subj_minority, replace=True, n_samples=majority_count, random_state=1)

    # combine majorrity class with upsampled miniority class
    subj_upsampled = pd.concat([subj_majority, subj_minority_upsampled])
    subj_upsampled = subj_upsampled.reset_index()
    return subj_upsampled

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

############### VISUALIE LOG MODEL ###############
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

    plt.subplots(1, figsize=(5, 5))
    plt.title('ROC - Logistic regression')
    plt.plot(false_positive_rate, true_positive_rate)
    plt.plot([0, 1], ls="--")
    plt.plot([0, 0], [1, 0], c=".7"), plt.plot([1, 1], c=".7")
    plt.ylabel('Sensitivity: TPR')
    plt.xlabel('Specifity: PPR')
    plt.show()

def plot_roc_curve_loo(pred_data, save=False):
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
    if save: plt.savefig('./bin/roc.png')
    plt.show()
    
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

def plot_confusion_matrix_loo(pred_data, save=False, norm=None):
    tn, fp, fn, tp = confusion_matrix(pred_data['ytrue'].values, pred_data['yhat'].values, normalize = norm).ravel()
    data = np.matrix([[tp, fp], [fn, tn]])
    
    plt.subplots(1, figsize=(15, 15))
    plt.title('Accuracy Score: {:.4f}'.format(accuracy_score(pred_data['ytrue'].values, pred_data['yhat'].values)))
    sns.heatmap(data, annot=True, xticklabels=['Actual Pos', 'Actual Neg'], yticklabels=['Pred. Pos', 'Pred. Neg'])
    if save: plt.savefig('./bin/confusion_matrix.png')                         
    plt.show()

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

def plot_regularization_path(coefs_, best_lambda, lambda_values):
    # define subplot
    sns.color_palette('Set2')
    fig, ax = plt.subplots(figsize=(20, 12))

    # the size of A4 paper
    #fig.set_size_inches(11.7, 8.27)

    # plot
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
    plt.show()
    
def plot_regularization_path_old(X, y, best_lambda, lambda_values=None):
    start = time.time()
    model = LogisticRegression(penalty='l1', solver='saga', tol=1e-3, max_iter=int(1e2), warm_start=True)

    c_values = np.logspace(-2, 2, 100)
    if lambda_values == None: lambda_values = 1.0/c_values
    coefs_ = []
    for c in c_values:
        model.set_params(C=c)
        model.fit(X, y)
        coefs_.append(model.coef_.ravel().copy())

    coefs_ = np.array(coefs_)

    # define subplot
    sns.color_palette('Set2')
    fig, ax = plt.subplots()

    # the size of A4 paper
    # fig.set_size_inches(11.7, 8.27)

    # plot
    ax.plot(lambda_values, coefs_, marker='.', linewidth=1)
    ax.axvline(best_lambda, linestyle='--', color='k')
    ax.text(x=0.8, y=0.7, s='best lambda\n{:.2e}'.format(best_lambda), color='k', fontsize=12,
             transform=ax.transAxes,
             horizontalalignment='center', verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    ax.set_xscale('log')
    plt.xlim(-10, 10)
    plt.xscale("log")
    # plt.ylim(-.3, .3)

    plt.xlabel(r"$\lambda$")
    plt.ylabel('Coefficients')
    plt.title('Logistic Regression: Cross-Validation Path')
    plt.axis('tight')
    plt.show()

    print("Time usage: %0.3fs" % (time.time() - start))
    return coefs_

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

def plot_regularization_score(score_df, gs_best_lambda):
    lambda_values = score_df['lambda_values']
    train_scores_mean = score_df['train_scores_mean']
    train_scores_std = score_df['train_scores_std']
    test_scores_mean = score_df['test_scores_mean']
    test_scores_std = score_df['test_scores_std']
    
    
    fig, ax = plt.subplots(figsize=(20, 12))
    plt.title("Logistic Regression: Cross-Validation Score")
    plt.xlabel(r"$\lambda$")
    plt.ylabel("Score")
    plt.ylim(0.0, 1.1)
    lw = 2

    ax.semilogx(lambda_values, train_scores_mean, label="Training score",
                 color="darkorange", lw=lw)
    ax.fill_between(lambda_values, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2,
                     color="darkorange", lw=lw)
    ax.semilogx(lambda_values, test_scores_mean, label="Cross-validation score",
                 color="navy", lw=lw)
    ax.fill_between(lambda_values, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.2,
                     color="navy", lw=lw)
    ax.axvline(gs_best_lambda, linestyle='--', color='k')
    ax.text(x=gs_best_lambda, y=0.7, s='best lambda\n{:.2e}'.format(gs_best_lambda), color='k', fontsize=40,
            #transform=ax.transAxes,
            horizontalalignment='center', verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    plt.legend(loc="best")
    plt.show()

def plot_regularization_score_old(X, y, gs_best_lambda, lambda_values=None, num_cv=10):
    start = time.time()
    if lambda_values == None: lambda_values = np.logspace(-2, 2, 100)
    param_range = 1.0/lambda_values
    
    model = LogisticRegression(penalty='l1', solver='saga', fit_intercept=False)
    train_scores, test_scores = validation_curve(model, X, y, param_name='C', error_score='raise', cv=num_cv,
                                                 param_range=param_range, scoring="roc_auc", n_jobs=1)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    best_lambda = lambda_values[np.argmax(test_scores_mean)]
    if gs_best_lambda == None: gs_best_lambda = best_lambda
    
    score_df = pd.DataFrame({'train_scores_mean':train_scores_mean, 'train_scores_std':train_scores_std,
                             'test_scores_mean':test_scores_mean, 'test_scores_std':test_scores_std,
                             'lambda_values':lambda_values,})
    score_df['best_lambda'] = best_lambda
    
    fig, ax = plt.subplots()
    plt.title("Logistic Regression: Cross-Validation Score")
    plt.xlabel(r"$\lambda$")
    plt.ylabel("Score")
    plt.ylim(0.0, 1.1)
    lw = 2

    ax.semilogx(lambda_values, train_scores_mean, label="Training score",
                 color="darkorange", lw=lw)
    ax.fill_between(lambda_values, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2,
                     color="darkorange", lw=lw)
    ax.semilogx(lambda_values, test_scores_mean, label="Cross-validation score",
                 color="navy", lw=lw)
    ax.fill_between(lambda_values, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.2,
                     color="navy", lw=lw)
    ax.axvline(gs_best_lambda, linestyle='--', color='k')
    ax.text(x=gs_best_lambda, y=0.7, s='best lambda\n{:.2e}'.format(gs_best_lambda), color='k', fontsize=12,
            #transform=ax.transAxes,
            horizontalalignment='center', verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    plt.legend(loc="best")
    plt.show()
    
    print("Time usage: %0.3fs" % (time.time() - start))
    return score_df

# def plot_regularization_score(grid_result):
#     """
#     This func plot the validation score changes as the function of lambda
#     :param grid_result: the grid search result
#     :return: a dataframe of grid search log
#     """
#     gs_df = pd.DataFrame(grid_result.cv_results_)
#     train_df = gs_df[['param_C', 'mean_train_score', 'std_train_score']]
#     test_df = gs_df[['param_C', 'mean_test_score', 'std_test_score']]
#     train_df = train_df.rename(columns={"mean_train_score": "score", "std_train_score": "sd"})
#     test_df = test_df.rename(columns={"mean_test_score": "score", "std_test_score": "sd"})
#
#     train_df['cv_split'] = 'train'
#     test_df['cv_split'] = 'test'
#     df = pd.concat([train_df, test_df], axis=0)
#     df['param_Lambda'] = 1.0 / df['param_C']
#
#     best_lambda = 1.0 / grid_result.best_params_['C']
#
#     # plot the lambda and score
#     sns.pointplot(data=df, x='param_Lambda', y='score', hue='cv_split', kind="point", dodge=True)
#     plt.axvline(np.log(best_lambda), linestyle='--', color='k')
#     #g.set_xticklabels(['{:.2e}'.format(x) for x in g.get_xticks()])
#     # plt.legend([],[], frameon=False)
#     plt.text(x=np.log(best_lambda)-10, y=0.8, s='best lambda\n{:.2e}'.format(best_lambda), color='k')
#
#     plt.ylim(0, 1.15)
#     plt.xscale("log")
#     #plt.xticks(rotation=45)
#     plt.gca().invert_xaxis()
#     plt.xlabel('Log(Lambda)')
#     plt.ylabel('Score')
#     plt.title('Logistic Regression: Cross-Validation Score')
#     plt.axis('tight')
#     plt.show()
#
#     return df

def plot_prediction(subj_wide, test_data, features, DV, best_lasso):
    test_data[[DV + '_lasso_pred']] = best_lasso.predict_proba(test_data[features])[:, 1]
    test_data['threshold'] = subj_wide['best_model1'].mean()
    test_data = test_data.sort_values(by=['best_model1'])

    plt.plot(test_data['HCPID'], test_data[DV], 'b^', label='observed')
    plt.plot(test_data['HCPID'], test_data[DV + '_lasso_pred'], 'rx', label='lasso_pred')
    plt.plot(test_data['HCPID'], test_data['threshold'], '-', label='thresh')

    plt.xlabel('subID')
    plt.ylabel(DV)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)


def plot_prediction_loo(pred_data, save=False, threshold=0.5, drop_dup=False):
    
    if drop_dup:
        x_ax = 'HCPID'
        x_labsize = 1
    else:
        x_ax = 'index'
        x_labsize = 20
    
    pred_data = pred_data.reset_index()
    pred_data['predcorrect'] = np.where(pred_data['ytrue'] == pred_data['yhat'], 'Logistic Model: Correct', 'Logistic Model: Incorrect')

    fig, ax = plt.subplots(figsize=(15,10))

    sns.scatterplot(data=pred_data, x=x_ax, y="ytrue", marker="s", alpha=0.9, s=250)
    sns.scatterplot(data=pred_data, x=x_ax, y="yprob", s=250, hue='predcorrect', style='predcorrect', markers=['o', 'X'], palette="Set2")
    plt.axhline(y=threshold, color='black', linestyle='--')

    plt.xlabel('Subjects')
    plt.ylabel('Prediction(Probability)')
    plt.tick_params(labelsize=x_labsize)
    plt.xticks(rotation=30)
    plt.legend(title="", bbox_to_anchor=(0.01, 0), borderaxespad=0, fontsize=20,
               loc='lower left',
               labels=['Threshold', 'ACT-R Model Identification', 'Logistic Model Prediction', 'Incorrect Prediction', 'Correct Prediction'])
    if save: plt.savefig('./bin/loo_pred.png')
    plt.show()

# def plot_prediction_loo(all_ytrue, all_yhat, all_yprob, threshold=0.5):
#
#     all_ytrue2 = [i[0] for i in all_ytrue]
#     all_yhat2 = [i[0] for i in all_yhat]
#     all_yprob2 = [i[0] for i in all_yprob]
#     pred_data2 = pd.DataFrame({'y_true':all_ytrue2, 'y_prob':all_yprob2, 'y_hat':all_yhat2}, dtype='float').rename_axis(columns='index').reset_index()
#     pred_data2['pred_corr'] = pred_data2['y_true'] == pred_data2['y_hat']
#
#     sns.scatterplot(data=pred_data2, x='index', y="y_true")
#     fig = sns.scatterplot(data=pred_data2, x='index', y="y_prob", hue = 'pred_corr', marker = 'x')
#     plt.axhline(y=threshold, color='black', linestyle='-.', label='threshold')
#
#     plt.xlabel('subj')
#     plt.ylabel('prediction')
#
#     plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
#
#     plt.show()



    
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

def plot_brain_connections(mat, power_coords, save=False, mat_name='beta_mat', thre='99.9%'):
    if mat_name=='beta_mat':
        tit = 'Beta'
    elif mat_name == 'wcorr_mat':
        tit = 'Weighted Connectivity'
    else:
        tit = 'Unknown'
    # create chart
    sns.set_style('white')
    fig = plt.figure(figsize=(15,5))
    ax = fig.add_subplot(111)
    ax.axis("off")

    # plot connectome with 80% edge strength in the connectivity
    plotting.plot_connectome(mat, power_coords, figure=fig,
                             edge_threshold=thre,
                             #node_color=,
                             colorbar=True,
                             node_size=0, # size 264
                             #alpha=.8,
                             title='Group analysis: ' + tit)
    if save: plt.savefig('./bin/'+mat_name+thre+'.png')
    plt.show()
