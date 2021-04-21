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
# import seaborn as sns
from scipy import optimize
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import StandardScaler
import pprint as p


subjs = pd.read_csv("../bin/gambling_clean_data.csv", index_col=0)
#################### LOAD DATA ####################
def load_subj(subjs, HCPID):
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

def cal_prob_switch(dat, group_vars=["BlockType", "PreviousFeedback"]):
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

def cal_rmse(magg, hagg, target_var="PSwitch"):
    """
    This func calculates the RMSE of model aggregated df vs. human aggregated df by target_var ("PSwitch")
    :param mdat: model data
    :param hdat: human data
    :param target_var: Pswitch
    :return: RMSE
    """
    # make sure the condition matches order
    assert (magg["BlockType"].to_list()==hagg["BlockType"].to_list()) & \
           (magg["PreviousFeedback"].to_list()==hagg["PreviousFeedback"].to_list())

    return ((magg[target_var] - hagg[target_var]) ** 2).mean() ** .5

def fit_simulation(param_array, HCPID, model, epoch = 50):
    assert isinstance(param_array, list) & isinstance(model, str) & isinstance(HCPID, str)


def model1_target_func(param_array, HCPID):
    # simulate model1 10 time using HCPID=100307_fnca
    # param_grid = list(ParameterGrid({'ans': [.1, .9], 'lf': [.1, .9], "bll": [.1, .9]}))
    assert isinstance(param_array, list) & isinstance(HCPID, str)

    # decide parameter set
    param_set = {"ans":param_array[0], "bll":param_array[1], "lf":param_array[2]}
    epoch = 50
    model = "model1"
    #param_set = {"egs": param_array[0], "alpha": param_array[1], "r": param_array[2]}

    # run 50 times
    m_raw = run.simulate(epoch=epoch, model=model, param_set=param_set, export=True, verbose=False,
                         file_suffix="_"+HCPID+"_log", HCPID=HCPID)
    magg = cal_prob_switch(add_switch(add_previous_feedbck(reformat_model_data(m_raw))))
    hagg = cal_prob_switch(load_subj(HCPID)) # load one subj given HCPID
    rmse = cal_rmse(magg, hagg, target_var="PSwitch")
    print(">>", param_set, "rmse =",rmse)
    return rmse

def model2_target_func(param_array, HCPID):
    assert isinstance(param_array, list) & isinstance(HCPID, str)

    # decide parameter set
    param_set = {"egs": param_array[0], "alpha": param_array[1], "r": param_array[2]}
    epoch = 50
    model = "model2"

    # run 50 times
    m_raw = run.simulate(epoch=epoch, model=model, param_set=param_set, export=True, verbose=False,
                         file_suffix="_"+HCPID+"_log", HCPID=HCPID)
    magg = cal_prob_switch(add_switch(add_previous_feedbck(reformat_model_data(m_raw))))
    hagg = cal_prob_switch(load_subj(HCPID)) # load one subj given HCPID
    rmse = cal_rmse(magg, hagg, target_var="PSwitch")
    print(">>", param_set, "rmse =",rmse)
    return rmse

def estimate_param_model1(HCPID):
    init = [.1, .1, .1]  #:ans       :bll        :lf
    bounds = [(0, 5), (0, 1), (0, 5)]
    minmum = optimize.minimize(model1_target_func, init, args=(HCPID),method='Powell', tol=1e-5, bounds=bounds,
                                 options={"maxiter": 200, "ftol": 0.0001, "xtol": 0.0001, "disp": True,
                                          "return_all": True})
    return minmum


def estimate_param_model2(HCPID):
    init = [.1, .1, .1]  #:egs       :alpha        :r
    bounds = [(0, 5), (0, 1), (0, 100)]
    minmum = optimize.minimize(model2_target_func, init, args=(HCPID), method='Powell', tol=1e-5, bounds=bounds,
                               options={"maxiter": 200, "ftol": 0.0001, "xtol": 0.0001, "disp": True,
                                        "return_all": True})
    return minmum

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
    minimum2 = optimize.minimize(model1_target_func(), init, args=("100307_fcna"), method='Powell', tol=1e-10, bounds=bounds,
                                 options={"maxiter":200,"ftol":0.0001, "xtol":0.0001, "disp":True, "return_all":True})

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



