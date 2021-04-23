###################### ACT-R + PYTHON TEMPLATE #######################
#   Author: Cher Yang
#   Date: 4/14/2021
# This template provides a init python code for building an ACT-R model
#
# Bugs: 4.19 loading model is not always working. could be due to actr.reset() and actr.reload()
#
# TODO: reload model every block or very run? will make difference?
# TOOD: add BlockTrial column
#
###################### ####################### #######################


import actr
import random
import time
import pandas as pd
import pprint as p
import os.path
from sklearn.model_selection import ParameterGrid


random.seed(0)

response = False
response_time = False
reward = 0



#################### LOAD MODEL CORE ####################
def load_model(model="model1", param_set=None):
    actr.load_act_r_model(os.path.abspath(model+"_core.lisp"))
    # load new pramsets
    if param_set: set_parameters(**param_set)
    reward = 0 # init value
    actr.load_act_r_model(os.path.abspath(model+"_body.lisp"))
    print("######### LOADED MODEL " +model+ " #########")
    print(">>", get_parameters(*get_parameters_name()), "<<")

def check_load(model="model1"):
    has_model = actr.current_model().lower() == model
    has_productions = actr.all_productions() != None
    return has_model & has_productions


#################### PARAMETER SET ####################

def get_parameters_name():
    if actr.current_model() == "MODEL1":
        param_names = ['ans', 'bll', 'lf']
    elif actr.current_model() == "MODEL2":
        param_names = ['alpha', 'egs', 'r']
    return param_names

def get_parameter(param_name):
    """
    get parameter from current model
    :param keys: string, the parameter name (e.g. ans, bll, r1, r2)
    :return:
    """
    assert param_name in ("ans", "bll", "lf", "egs", "alpha", "r")
    if param_name=="r": return reward
    else: return actr.get_parameter_value(":"+param_name)

def get_parameters(*kwargs):
    param_set = {}
    for param_name in kwargs:
        param_set[param_name] = get_parameter(param_name)
    return param_set

def set_parameters(**kwargs):
    """
    set parameter to current model
    :param kwargs: dict pair, indicating the parameter name and value (e.g. ans=0.1, r1=1, r2=-1)
    :return:
    """
    global reward
    for key, value in kwargs.items():
        if key == "r": reward = value
        else: actr.set_parameter_value(':' + key, value)

#################### TASK ####################

def task(trials):
    """
    This function present task and monitor response from model
    :param size: number of trials to present
    :param trials: the trial list
    :return:
    """

    # monitor the output-key
    actr.add_command("paired-response", respond_to_key_press,
                     "Paired associate task key press response monitor")
    actr.monitor_command("output-key","paired-response")

    result = do_experiment(trials)

    actr.remove_command_monitor("output-key","paired-response")
    actr.remove_command("paired-response")
    return result

def respond_to_key_press(model, key, test=False):
    """
    This function is set to monitor the output-key command, will be called whenever
    a key is pressed in the experiment window
    :param model: name of the model. if None, indicates a person interacting with
                the window
    :param key: string name of the key
    :return:
    """
    global response, response_time

    # record the latency of key-press (real time, not ACT-R time)
    response_time = actr.get_time(model)
    response = key
    if test: print("TEST: in respond_to_key_press: ", response, response_time)

def do_guess(prompt, window):
    """
    this function allows model to do first half of the experiment, guessing
    :param prompt:"?"
    :param window:
    :return: response "f" for less, or "j" for more
    """

    # display prompt
    actr.clear_exp_window(window)
    actr.add_text_to_exp_window(window, prompt, x=150, y=150)

    # wait for response
    global response
    response = ''

    start = actr.get_time()
    actr.run_full_time(5)
    time = response_time - start

    return response, time

def do_feedback(feedback, window):
    """
    This  function allows the model to encode feedback
    :param feedback: "win" or "lose"
    :param window:
    :return:
    """

    actr.clear_exp_window(window)
    actr.add_text_to_exp_window(window, feedback, x=150, y=150)

    actr.run_full_time(5)

    # implement reward
    if actr.current_model() == "MODEL2":
        if feedback == "Reward":
            actr.trigger_reward(reward)
        elif feedback == "Punishment":
            actr.trigger_reward(-1.0*reward)

def do_experiment(trials, test=False):
    """
    This function run the experiment, and return simulated model behavior
    :param size:
    :param trials:
    :param human:
    :return:
    """
    #TODO: need to comment if seperate to core and body script
    # actr.reset()

    result = []

    window = actr.open_exp_window("Gambling Experiment", visible=False)
    actr.install_device(window)

    for trial in trials:
        # time = 0
        prompt, feedback, block_type = trial

        # guess
        response, time = do_guess(prompt, window)

        # this  test is to see if model can learn feedback
        if test: feedback = test_unit5(response)

        # encode feedback
        do_feedback(feedback, window)
        result.append((feedback, block_type, response, time))

    return result

def create_block(num_trials=8, num_reward=6, num_punish=1, num_neutral=1, block_type="MostlyReward", shuffle=False):
    """
    This function create experiment stimuli by blocks
    :param num_trials: number of trials =8
    :param num_reward: number of reward trials =6 (Mostly Reward  Block)
    :param num_punish: number of reward trials =1 (Mostly Reward  Block)
    :param num_neutral: number of reward trials =1 (Mostly Reward  Block)
    :param block_type: Mostly Reward  Block or Mostly Punishment  Block
    :param shuffle: whether to randomly shuffle trials within blocks
    :return: a block of trials (8)
    """
    prob_list = ["?"] * num_trials
    feedback_list = ["Reward"] * num_reward + ["Punishment"] * num_punish + ["Neutral"] * num_neutral
    block_list = [block_type] * num_trials
    trials = list(zip(prob_list, feedback_list, block_list))
    if shuffle: random.shuffle(trials)
    return trials

def create_stimuli(num_run=1):
    trials = []
    MR1 = create_block(8, 6, 1, 1, "MostlyReward", True) + create_block(8, 4, 2, 2, "MostlyReward", True)
    MR2 = create_block(8, 6, 1, 1, "MostlyReward", True) + create_block(8, 4, 2, 2, "MostlyReward", True)
    MP1 = create_block(8, 1, 6, 1, "MostlyPunishment", True) + create_block(8, 2, 4, 2, "MostlyPunishment", True)
    MP2 = create_block(8, 1, 6, 1, "MostlyPunishment", True) + create_block(8, 2, 4, 2, "MostlyPunishment", True)
    r1_trials = MR1 + MP1 + MP2 + MR2
    r2_trials = MP1 + MR1 + MP2 + MR2

    if num_run == 1:
        trials = r1_trials
    elif num_run == 2:
        trials = r1_trials + r2_trials
    else:
        trials = None
    return trials

def load_stimuli(HCPID):
    """
    This function enables the model to simulate based on specific HCP subj  stimuli order being accessed
    :param HCPID:
    :return:
    """
    stim = pd.read_csv("../bin/gambling_trials/"+HCPID+".csv", usecols=["TrialType", "BlockType"])
    stim["Probe"] = "?"
    stim = stim[['Probe', 'TrialType', 'BlockType']]
    trials = [tuple(x) for x in stim.to_numpy()]
    return trials

def experiment(model="model1", param_set=None, reload=True, stim_order=None):
    """
    This function call create_block() and task() to run experiment
    :param num_run: default =1, but could be 2
    :return: a dataframe of model outputs, with "TrialType", "BlockType", "Response", "RT" as columns
    """
    #only one run
    if reload: load_model(model=model, param_set=param_set)

    # if provided stimuli order, use it
    # otherwise, create psudorandom stim order
    if stim_order==None: trials=create_stimuli()
    else: trials = stim_order

    model_result =  task(trials)

    model_result = pd.DataFrame(model_result, columns=["TrialType", "BlockType", "Response", "RT"])
    model_result["BlockTrial"] = list(range(0, 8)) * int(len(model_result)/8)
    model_result["Trial"] = model_result.index
    return model_result

def print_averaged_results(model_data):
    """
    This function print aggregated results group by trial type
    :param model_data: model output
    :return: aggregated results
    """
    print(model_data.groupby("TrialType").mean())
    print()
    print(model_data["TrialType"].value_counts(normalize=True))
    print()
    print(model_data.groupby("Response").mean())
    print()
    print(model_data["Response"].value_counts(normalize=True))

#################### SIMULATION ####################

def simulate(epoch, model, param_set=None, export=True, verbose=True, file_suffix="", HCPID=None):


    # whether to load stimuli order or create random stimuli
    trials=None
    if HCPID:
        trials = load_stimuli(HCPID)

    model_output = []
    for i in range(epoch):
        simulation_start = time.time()
        model_dat = experiment(model=model, param_set=param_set, reload=True, stim_order=trials) # reset

        model_dat["Epoch"] = i
        param_names = get_parameters_name()
        for param_name in param_names:
            model_dat[param_name]=get_parameter(param_name)

        model_output.append(model_dat)
        simulation_end = time.time()

        if export:
            fpath = './model_output/' + actr.current_model() + pd.to_datetime('now').strftime('%Y%m%d') + file_suffix +".csv"
            model_dat.to_csv(fpath, mode='a', header=not(os.path.exists(fpath)), index=False)
            print(">> exported")

        if verbose:
            p.pprint(model_dat.head())
            print(">> running time", round(simulation_end-simulation_start, 2))
    return model_output


#################### TEST ####################

def test_unit1():
    """
    This is a unit test for RL model RT. The goal is to test whether RL remained same regardless of conditions
    :return:
    """
    assert check_load()
    sometrials=create_block()
    sometrials.sort()
    p.pprint(task(sometrials))

def test_unit2():
    """ This unit test is to observe single/two trials"""
    assert check_load()
    trials=create_block()[0]
    p.pprint(task([trials]))

def test_unit3():
    """this test unit examines trace"""
    assert check_load()
    trial = ('?', 'Punishment', 'MostlyReward')
    prompt, feedback, block_type = trial

    window = actr.open_exp_window("Gambling Experiment", visible=False)
    actr.install_device(window)
    actr.clear_exp_window(window)
    actr.add_text_to_exp_window(window, prompt, x=150, y=150)
    actr.run_full_time(5)

    actr.clear_exp_window(window)
    actr.add_text_to_exp_window(window, feedback, x=150, y=150)
    actr.run_full_time(5)

def test_unit4():
    assert check_load()
    "This test unit is to see if :lf can scale RT"
    print_averaged_results(experiment())

def test_unit5(model_resp):
    "This test is to see if model1 can learn from feedback"
    if (model_resp=="f"):
        feedback = "Reward"
    else:
        feedback = "Punishment"
    print("testing feedback learning", feedback)
    print("delivered reward", reward)
    return feedback

def test_unit6():
    "This test is to set param and simulate 10 times"
    assert check_load("model1")
    param_grid = list(ParameterGrid({'ans': [.2, .7], 'lf': [.2, .7], 'bll': [.2, .7]}))
    epoch = 10
    for param_set in param_grid:
        simulate(epoch, model="model1", param_set=param_set)
    return

def test_unit7():
    "This test is to set param and simulate 10 times"
    assert check_load(model="model2")
    param_grid = list(ParameterGrid({'egs': [.2, .7], 'alpha': [.2, .7], "r" : [.1, 10]}))
    epoch = 10
    for param_set in param_grid:
        simulate(epoch, model="model2", param_set=param_set)
    return

def test_unit8():
    """This test is to see if load stimuli func works"""
    simulate(epoch=2, model="model1", param_set=None, export=False, verbose=True, file_suffix="test1", HCPID="100307_fnca")

def test_unit9():
    HCPID = "102311_fnca"
    stimuli = load_stimuli(HCPID)
    p.pprint(task(stimuli))

def test_unit10():
    "This is testing parameter"
    HCPID = "102311_fnca"
    stimuli = load_stimuli(HCPID)
    load_model("model1", param_set={"ans":1.1831138, "bll":.2, "lf":1.5})
    m1 = pd.DataFrame(task(stimuli))
    m1[[2]]



