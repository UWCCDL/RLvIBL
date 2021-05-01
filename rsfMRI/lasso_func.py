import pandas as pd
import numpy as np
import json
import os, glob
import itertools
import time
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score
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
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import KFold, RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction import DictVectorizer
from sklearn.utils import resample


############### LOAD DATA ###############
def load_subj(CORR_DIR, model_dat):
    """ this function load correlation matrix for each subj """
    subj_dict = {}
    HCPIDs = model_dat['HCPID'].to_list()
    for HCPID in HCPIDs:
        sub_dir = 'sub-'+HCPID.split('_')[0]
        sub_fpath = CORR_DIR+sub_dir+'/ses-01/mr_corr_pearson.txt'
        try:
            sub_df = pd.read_csv(sub_fpath, header=0)
            subj_dict[HCPID] = sub_df
        except:
            print("WARNING: rsfMRI data missing", HCPID)

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

def grid_search_lasso(train_data, features, DV, plot_path=False):
    start = time.time()

    # define model
    model = LogisticRegression(penalty='l1', random_state=1, solver='saga', fit_intercept=False)

    # define cross-validation method
    cv = LeaveOneOut()

    # define parameter space
    space = dict()
    #space['solver'] = ['newton-cg', 'lbfgs', 'liblinear']
    #space['penalty'] = ['l1']
    space['C'] = np.logspace(-3, 3, 10)
    lambda_values = 1.0 / space['C']


    # define search
    grid_search = GridSearchCV(estimator=model, param_grid=space, n_jobs=-1, cv=cv, scoring='accuracy', error_score=0)

    # use LOOCV to evaluate model
    # scores = cross_val_score(model, train_data[features], train_data[DV], scoring='accuracy', cv=cv, n_jobs=-1)

    # execute search
    grid_result = grid_search.fit(train_data[features], train_data[DV])

    # summarize result
    print('Best Score: %s' % grid_result.best_score_)
    print('Best Hyperparameters: %s' % grid_result.best_params_)
    print("Time usage: %0.3fs" % (time.time() - start))

    if plot_path: plot_regularization_path(train_data, features, DV, lambda_values, 1.0/grid_result.best_params_['C'])

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

############### VISUALIE LOG MODEL ###############
def print_coefficients(coef, features):
    """
    This function takes in a model column and a features column.
    And prints the coefficient along with its feature name.
    """
    feats = list(zip(features, coef))
    print(*feats, sep="\n")

def plot_ROC_curve(logistic_model, test_data, features, DV):
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
    sns.heatmap(data, annot=True, xticklabels=['Actual Pos', 'Actual Neg'], yticklabels=['Pred. Pos', 'Pred. Neg'])
    plt.show()

def plot_regularization_path(train_data, features, DV, lambda_values, best_lambda):
    start = time.time()
    model = LogisticRegression(penalty='l1', solver='saga',
                             tol=1e-3, max_iter=int(1e2),
                             warm_start=True)
    coefs_ = []
    c_values = 1.0 / lambda_values
    for c in c_values:
        model.set_params(C=c)
        model.fit(train_data[features], train_data[DV])
        coefs_.append(model.coef_.ravel().copy())

    coefs_ = np.array(coefs_)
    plt.plot(np.log(lambda_values), coefs_, marker='o')
    plt.axvline(x=np.log(best_lambda), label='best', c='b')
    plt.xlim(-1, 100)
    plt.ylim(-.3, .3)

    plt.xlabel('Log(Lambda)')
    plt.ylabel('Coefficients')
    plt.title('Logistic Regression Path')
    plt.axis('tight')
    plt.show()
    print("Time usage: %0.3fs" % (time.time() - start))

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


