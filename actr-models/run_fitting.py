###################### ACT-R + PYTHON TEMPLATE #######################
#   Author: Cher Yang
#   Date: 4/14/2021
# This template provides a init python code for fitting an ACT-R model
#
# Bugs:
#
# TODO: 4.20 1) calculate PSwitch and RT
#
###################### ####################### #######################

import os
import pandas as pd
import numpy as np
import run_model as run
from scipy import optimize
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import StandardScaler
import pprint as p


subjs = pd.read_csv("../bin/gambling_clean_data.csv", index_col=0)
#################### LOAD DATA ####################
def load_subj(HCPID):
    return reformat_humam_data(subjs[subjs['HCPID'] == HCPID])

#################### REFORMAT DATA ####################
def reformat_model_data(mdat, znorm=False):
    """
    This function reformat model data into a single df with same column  name as human data
    :param mdat: model data: list(df) or df
    :param znorm: whether to z-score RT
    :return:
    """
    if isinstance(mdat, list):
        mdat = pd.concat(mdat)

    # mdat=mdat.replace("win", "Reward")
    # mdat=mdat.replace("lose", "Punishment")
    # mdat=mdat.replace("neutral", "Neutral")

    # mdat = mdat.replace("mostly_reward", "MostlyReward")
    # mdat = mdat.replace("mostly_punish", "MostlyPunishment")

    std_scaler = StandardScaler() # zscore normalization
    if znorm: mdat["RT"] = pd.DataFrame(std_scaler.fit_transform(mdat[["RT"]]), columns=["RT"])
    return mdat

def reformat_humam_data(hdat, znorm=False):
    """
    This func reformats human data with same column  name as model data
    :param hdat:
    :param znorm:
    :return:
    """
    hdat = hdat[["HCPID", "Trial", "TrialType", "BlockTypeCoded", "RunTrialNumber", "CurrentResponse", "PastResponse", "FutureResponse",
                 "ResponseSwitch", "PreviousFeedback", "RT"]]
    hdat = hdat.rename(columns={"RunTrialNumber":"BlockTrial", "CurrentResponse":"Response", "BlockTypeCoded":"BlockType"})
    std_scaler = StandardScaler()  # zscore normalization
    if znorm: hdat["RT"] = pd.DataFrame(std_scaler.fit_transform(hdat[["RT"]]), columns=["RT"])
    return hdat

#################### CALCULATE ####################
def add_previous_feedbck(dat):
    """
    This func adds a column "PreviousFeedback"
    :param dat:
    :return:
    """
    #model1.loc[model1['Response'].shift(1).eq('yes'), 'val'] = 'yes'
    dat["PreviousFeedback"] = dat[['TrialType']].shift(1)
    dat.loc[dat["BlockTrial"] == 0, "PreviousFeedback"] = np.nan
    return dat

def add_switch(dat):
    """
    This func adds a column "ResponseSwitch"
    :param dat:
    :return:
    """
    dat["PreviousResponse"] = dat[['Response']].shift(1)
    dat["ResponseSwitch"] = (dat["Response"] != dat["PreviousResponse"])
    dat.loc[dat["BlockTrial"] == 0, "ResponseSwitch"] = np.nan
    return dat

def cal_prob_switch(dat, group_vars=["BlockType", "TrialType"]):
    """
    This func calculates PSwitch by group variables: BlockType and PreviousFeedback
    :param dat:
    :param group_vars: ["BlockType", "PreviousFeedback"]
    :return: aggregated df
    """
    res = dat.dropna(axis=0, how='any', subset=group_vars+["ResponseSwitch", "RT"])
    res = res.groupby(group_vars).agg({"ResponseSwitch":"mean", "RT":"mean"}).reset_index()
    res = res.rename(columns={"ResponseSwitch": "PSwitch"})
    return res

def cal_rmse(magg, hagg, target_var="PSwitch", group_vars=["BlockType","TrialType"]):
    """
    This func calculates the RMSE of model aggregated df vs. human aggregated df by target_var ("PSwitch")
    :param mdat: model data
    :param hdat: human data
    :param target_var: Pswitch
    :return: RMSE
    """
    # make sure the condition matches order
    assert (magg[group_vars[0]].to_list()==hagg[group_vars[0]].to_list()) & \
           (magg[group_vars[1]].to_list()==hagg[group_vars[1]].to_list())

    return ((magg[target_var] - hagg[target_var]) ** 2).mean() ** .5

def count_null_response(dat):
    """
    This func count the number of null responses in either model_output or human_output
    :param dat: data
    :return: count of null response
    """
    assert isinstance(dat, pd.core.frame.DataFrame)
    response_count = dat[["Response"]].value_counts()
    try:
        null_count = response_count['']
    except:
        return 0
    return null_count

#################### Optimization ####################
def fit_simulation(model, param_array, HCPID, epoch = 50):
    """
    This func running simulation and save log file in local computer
    :param model: model name. :keyword "model1" :keyword "model2
    :param param_array: a list of parameters
    :param HCPID: subject ID
    :param epoch: number of simulation per parameter set
    :return: a list of  simulation output dataframe (length = epoch)
    """
    # assert isinstance(param_array, list) & isinstance(model, str) & isinstance(HCPID, str)

    # decide parameter set
    if model=="model1":
        param_set = {"ans": param_array[0], "bll": param_array[1], "lf": param_array[2]}
    elif model=="model2":
        param_set = {"egs": param_array[0], "alpha": param_array[1], "r": param_array[2]}
    else:
        param_set = None
        print("invalid model name")

    # run 50 times
    model_output = run.simulate(epoch=epoch, model=model, param_set=param_set, export=True, verbose=False,
                         file_suffix="_" + HCPID + "_log", HCPID=HCPID)
    return model_output

def model_target_func(param_array, HCPID, model):
    """
    This func serves as target function to find optimal parameter
    :param param_array: parameter set of three. If model1 is passed in, then three parameters are :ans, :bll, :lf.
    If model2 is passed in, then three paramters are :egs, :alpha, :r
    :param HCPID: subject ID number
    :param model: model name: "model1" or "model2"
    :return: target value for minimization
    """
    targ_value = None

    # run fitting simulation based on parameter_set, and HCPID
    model_output = fit_simulation(model, param_array, HCPID, epoch = 50)

    # calcualte rmse
    hdat = load_subj(HCPID) # load one subj given HCPID
    mdat = add_switch(add_previous_feedbck(reformat_model_data(model_output)))
    magg = cal_prob_switch(mdat, group_vars=["BlockType", "TrialType"])
    hagg = cal_prob_switch(hdat, group_vars=["BlockType", "TrialType"])
    rmse_value = cal_rmse(magg, hagg, target_var="PSwitch", group_vars=["BlockType","TrialType"])

    # add penalty term
    null_penalty = abs(int(count_null_response(mdat) - int(count_null_response(hdat))))

    targ_value = rmse_value + null_penalty
    print(">>", param_array, "rmse =", rmse_value, "targ_value =", targ_value)
    return targ_value

# def model2_target_func(param_array, HCPID):
#     assert isinstance(param_array, list) & isinstance(HCPID, str)
#
#     # decide parameter set
#     param_set = {"egs": param_array[0], "alpha": param_array[1], "r": param_array[2]}
#     epoch = 50
#     model = "model2"
#
#     # run 50 times
#     m_raw = run.simulate(epoch=epoch, model=model, param_set=param_set, export=True, verbose=False,
#                          file_suffix="_"+HCPID+"_log", HCPID=HCPID)
#     magg = cal_prob_switch(add_switch(add_previous_feedbck(reformat_model_data(m_raw))))
#     hagg = cal_prob_switch(load_subj(HCPID)) # load one subj given HCPID
#     rmse = cal_rmse(magg, hagg, target_var="PSwitch")
#     print(">>", param_set, "rmse =",rmse)
#     return rmse

def estimate_param(HCPID, model):
    """
    This func estimates the optimal parameter set for specific subj
    :param HCPID: Subj ID
    :param model: model name, either "model1" or "model2
    :return: optimization output
    """
    init = [.1, .1, .1]
    if model=="model1": bounds = [(0, 5), (0, 1), (0, 5)]  #:ans       :bll        :lf
    elif model=="model2": bounds = [(0, 5), (0, 1), (0, None)]  #:egs       :alpha        :r
    else: print("wrong model name")
    minmum = optimize.minimize(model_target_func, init, args=(HCPID, model), method='Powell', tol=1e-3, bounds=bounds,
                                 options={"maxiter": 200, "ftol": 1e-4, "xtol": 1e-3, "disp": True,
                                          "return_all": True})
    return minmum


def grid_seach_estimate_param(HCPID, model):
    param_set_list = []
    if model=="model1":
        param_set_list = [{"ans": 0.05, "bll": 0.5, "lf": 0.5}, {"ans": 0.1, "bll": 0.5, "lf": 0.5}, {"ans": 0.5, "bll": 0.5, "lf": 0.5}]     
    elif model=="model2":
        param_set_list = [{"egs": 0.05, "alpha": 0.2, "r": 1}, {"egs": 0.1, "alpha": 0.2, "r": 1}, {"egs": 0.5, "alpha": 0.2, "r": 1}] 

    for param_set in param_set_list:
        model_output = run.simulate(epoch=100, model=model, param_set=param_set, 
            export=True, verbose=False, file_suffix="_" + HCPID + "_gs", HCPID=HCPID)

# def estimate_param_model2(HCPID):
#     init = [.1, .1, .1]  #:egs       :alpha        :r
#     bounds = [(0, 5), (0, 1), (0, 100)]
#     minmum = optimize.minimize(model2_target_func, init, args=(HCPID), method='Powell', tol=1e-5, bounds=bounds,
#                                options={"maxiter": 200, "ftol": 0.0001, "xtol": 0.0001, "disp": True,
#                                         "return_all": True})
#     return minmum

#################### NOT USE ####################
# def calPSwitch(dat, human):
#     res = None
#     if human:
#         res = dat[["HCPID", "PreviousFeedback", "BlockTypeCoded", "ResponseSwitch"]].groupby(
#         ["HCPID", "PreviousFeedback", "BlockTypeCoded"]).agg(["mean"]).reset_index()
#     else:
#         res = dat[["BlockType", "TrialType", "PreviousFeedback", "ResponseSwitch"]].\
#             groupby(["BlockType", "PreviousFeedback"]).agg(["mean"]).reset_index()
#
# def calRT(dat, human):
#     res  = None
#     if human:
#         res = dat[["HCPID", "PreviousFeedback", "BlockTypeCoded", "RT"]].groupby(
#         ["HCPID", "PreviousFeedback", "BlockTypeCoded"]).agg(["mean"]).reset_index()
#     else:
#         res = dat[["RT", "BlockType", "TrialType"]].groupby(["BlockType", "TrialType"]).mean().reset_index()
#     return res
#
# def plotPSwitch(dat):
#     sns.set_theme()
#     g = sns.FacetGrid(dat, col="BlockType", hue="PreviousFeedback")
#     g.map(sns.pointplot, "PreviousFeedback", "PSwitch", order=["Reward", "Punishment", "Neutral"])





#################### TEST ####################
def test_unit1():
    """Some hits:
    1) if set seed, RMSE becomes a constant, even increasing epoch, RMSE still a constant
    2) if set differnet seeds, RMSE will change greatly
    3) if not set seed, RMSE changes very greatly, not sure will converge

    """
    # load subj data
    subjs = pd.read_csv("../bin/gambling_clean_data.csv", index_col=0)
    s100307 = reformat_humam_data(subjs[subjs['HCPID']=='100307_fnca'])
    s100307agg = cal_prob_switch(s100307)

    # model_dat = pd.concat(simulate(epoch=10, model="model1", param_set=None, export=False, verbose=True))

    # simulate model1 10 time using HCPID=100307_fnca
    param_grid = list(ParameterGrid({'ans': [.1, .9], 'lf': [.1, .9], "bll": [.1, .9]}))
    epoch = 1
    for param_set in param_grid:
        m_raw = run.simulate(epoch=epoch, model="model1", param_set=param_set, export=False, verbose=False,
                             file_suffix="", HCPID="100307_fnca")
        mdat = reformat_model_data(m_raw)
        mdat = add_previous_feedbck(mdat)
        mdat = add_switch(mdat)
        magg = cal_prob_switch(mdat)
        rmse = cal_rmse(magg, s100307agg, target_var="PSwitch")
        p.pprint(param_set)
        print('RMSE =', rmse)
    return

def test_unit2():
    """This test is to see if model1 estimation works, it took an hour for one subj, 50 epoch, no :seed """
    init = [.1, .1, .1] #:ans       :bll        :lf
    # bounds = [optimize.Bounds(0, 5, keep_feasible=False), optimize.Bounds(0, 1, keep_feasible=False),
    #               optimize.Bounds(0, 5, keep_feasible=False)]
    bounds = [(0, 5), (0,1), (0, 5)]

    # minimum = optimize.fmin_powell(target_func, init, maxiter=200, full_output=True, retall=True)
    minimum2 = optimize.minimize(model_target_func, init, args=("102311_fcna", "model1"), method='Powell',
                                 bounds=bounds, options={"maxiter":200,"ftol":0.0001, "xtol":0.01, "disp":True, "return_all":True})

    # test {'ans': 0.6905457428133445, 'bll': 0.046858992021032525, 'lf': 1.3775874589422552} rmse = 0.234849677208153
    # Optimization terminated successfully.
    #          Current function value: 0.221294
    #          Iterations: 2
    #          Function evaluations: 144
    #  allvecs: [array([0.1, 0.1, 0.1]), array([2.63470964, 0.14589803, 0.27864045]), array([0.69059423, 0.04685987, 1.37759088])]
    #    direc: array([[ 0.        ,  0.        ,  1.        ],
    #        [ 0.        ,  1.        ,  0.        ],
    #        [-1.45436976, -0.02633545, -0.1025006 ]])
    #      fun: 0.22129438361175593
    #  message: 'Optimization terminated successfully.'
    #     nfev: 144
    #      nit: 2
    #   status: 0
    #  success: True
    #        x: array([0.69059423, 0.04685987, 1.37759088])

    return minimum2

def test_unit3():
    model_output = run.simulate(epoch=5, model="model1", param_set={"ans":1.1831138, "bll":0.5401689, "lf":3.09},
                                export=False, verbose=True,
                                file_suffix="", HCPID="102311_fnca")
    return model_output



