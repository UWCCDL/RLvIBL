###################### ACT-R + PYTHON TEMPLATE #######################
#   Author: Cher Yang
#   Date: 4/14/2021
# This template provides a init python code for building an ACT-R model

import actr
import random
import pandas as pd
import pprint as p
import os.path

random.seed(0)

response = False
response_time = False

#################### LOAD MODEL ####################
model = "model1"
if model=="model1":
    actr.load_act_r_model(os.path.abspath("model1.lisp"))
elif model=="model2":
    actr.load_act_r_model(os.path.abspath("model2.lisp"))

#################### PARAMETER SET ####################
def set_parameters(**kwargs):
    """
    set parameter to current model
    :param kwargs: dict pair, indicating the parameter name and value (e.g. ans=0.1, r1=1, r2=-1)
    :return:
    """
    # actr.reset() # this step makes sure current model getting rid of chunks and productions
                 # then new parameter can be set
    for key, value in kwargs.items():
        # # set reward parameter
        # if key=='r1':
        #     actr.spp('encode-reward', ':reward', value)
        # elif key=='r2':
        #     actr.spp('encode-punishment', ':reward', value)
        # normal parameters
        actr.set_parameter_value(':' + key, value)

def get_parameter(param_name):
    """
    get parameter from current model
    :param keys: string, the parameter name (e.g. ans, bll, r1, r2)
    :return:
    """
    assert param_name in ("ans", "bll", "lf", "egs", "alpha")
    # if param_name in ("r1", "r2"):
    #     param_reward = [x[0] for x in actr.spp(':reward') if x != [None]]
    #     if param_name == "r1":
    #         prarm_value = max(param_reward)
    #     else:
    #         prarm_value = min(param_reward)
    # else:
    return actr.get_parameter_value(":"+param_name)

def get_parameters_name():
    if actr.current_model() == "MODEL1":
        param_names = ['ans', 'bll', 'lf']
    elif actr.current_model() == "MODEL2":
        param_names = ['alpha', 'egs', 'r1', 'r2']
    return param_names

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

    # display prompt
    actr.clear_exp_window(window)
    actr.add_text_to_exp_window(window, prompt, x=150, y=150)

    # wait for response
    global response
    response = ''

    start = actr.get_time(model)
    actr.run_full_time(5)
    time = response_time - start

    return response, time

def do_feedback(feedback, window):

    actr.clear_exp_window(window)
    actr.add_text_to_exp_window(window, feedback, x=150, y=150)

    actr.run_full_time(5)

    # implement reward
    if actr.current_model() == "MODEL2":
        if feedback == "win":
            actr.trigger_reward(100)
        elif feedback == "lose":
            actr.trigger_reward(-100)

def do_experiment(trials, test=False, feedback_test=True):
    """
    This function run the experiment, and return simulated model behavior
    :param size:
    :param trials:
    :param human:
    :return:
    """
    #TODO: need to comment if seperate to core and body script
    actr.reset()

    result = []

    window = actr.open_exp_window("Gambling Experiment", visible=False)
    actr.install_device(window)

    for trial in trials:
        # time = 0
        prompt, feedback, block_type = trial

        # guess
        response, time = do_guess(prompt, window)

        # this  test is to see if model can learn feedback
        if feedback_test:
            if response=="f":
                feedback="win"
            else:
                feedback="lose"

        # encode feedback
        do_feedback(feedback, window)
        result.append((feedback, block_type, response, time/1000.0))

    return result


def create_block(num_trials=8, num_reward=6, num_punish=1, num_neutral=1, block_type="mostly_reward", shuffle=False):
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
    feedback_list = ["win"] * num_reward + ["lose"] * num_punish + ["neutral"] * num_neutral
    block_list = [block_type] * num_trials
    trials = list(zip(prob_list, feedback_list, block_list))
    if shuffle: random.shuffle(trials)
    return trials

def experiment(num_run=1):
    """
    This function call create_block() and task() to run experiment
    :param num_run: default =1, but could be 2
    :return: a dataframe of model outputs, with "TrialType", "BlockType", "Response", "RT" as columns
    """
    #only one run
    trials = []
    for i in range(num_run):
        MR1 = create_block(8, 6, 1, 1, "mostly_reward", True) + create_block(8, 4, 2, 2, "mostly_reward", True)
        MR2 = create_block(8, 6, 1, 1, "mostly_reward", True) + create_block(8, 4, 2, 2, "mostly_reward", True)
        MP1 = create_block(8, 1, 6, 1, "mostly_punish", True) + create_block(8, 2, 4, 2, "mostly_punish", True)
        MP2 = create_block(8, 1, 6, 1, "mostly_punish", True) + create_block(8, 2, 4, 2, "mostly_punish", True)

        trials += MR1+MR2+MP1+MP2
    model_result = pd.DataFrame(task(trials), columns=["TrialType", "BlockType", "Response", "RT"])
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

def test_unit1():
    """
    This is a unit test for RL model RT. The goal is to test whether RL remained same regardless of conditions
    :return:
    """
    actr.load_act_r_model(os.path.abspath("model1.lisp"))
    sometrials=create_block()
    sometrials.sort()
    p.pprint(task(sometrials))

def test_unit2():
    """ This unit test is to observe single/two trials"""
    actr.load_act_r_model(os.path.abspath("model1.lisp"))
    trials=create_block()[0]
    p.pprint(task([trials]))

def test_unit3():
    """this test unit examines trace"""
    #actr.load_act_r_model(os.path.abspath("model1_core.lisp"))
    actr.reset()
    trial = create_block()[0]
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
    "This test unit is to see if :lf can scale RT"
    actr.load_act_r_model(os.path.abspath("model1.lisp"))
    print_averaged_results(experiment())

def test_unit5():
    "This test is to see if model1 can learn from feedback"
    actr.load_act_r_model(os.path.abspath("model2.lisp"))
    print_averaged_results(experiment())







