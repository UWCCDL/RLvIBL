###################### ACT-R + PYTHON TEMPLATE #######################
#   Author: Cher Yang
#   Date: 4/14/2021
# This template provides a init python code for building an ACT-R model

import actr
import random
import pandas as pd
import pprint as p
random.seed(0)


actr.load_act_r_model("/Users/cheryang/Documents/GitProject/RLvIBL/actr-models/model2.lisp")   # load the model
response = False
response_time = False

def task(trials,human=False):
    """
    This function present task and monitor response from model
    :param size: number of trials to present
    :param trials: the trial list
    :param human: whether run a person or a model
    :return:
    """

    # monitor the output-key
    actr.add_command("paired-response", respond_to_key_press,
                     "Paired associate task key press response monitor")
    actr.monitor_command("output-key","paired-response")

    result = do_experiment(trials,human)

    actr.remove_command_monitor("output-key","paired-response")
    actr.remove_command("paired-response")
    return result


def respond_to_key_press(model, key, test=True):
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


def do_experiment(trials, human=False):
    """
    This function run the experiment, and return simulated model behavior
    :param size:
    :param trials:
    :param human:
    :return:
    """
    actr.reset()

    result = []
    model = not (human)
    window = actr.open_exp_window("Gambling Experiment", visible=human)

    if model:
        actr.install_device(window)

    for trial in trials:
        # time = 0
        prompt, feedback, block_type = trial

        # display prompt
        actr.clear_exp_window(window)
        actr.add_text_to_exp_window(window, prompt, x=150, y=150)

        # wait for response
        global response
        response = ''
        start = actr.get_time(model)
        # print("TEST: (START)>>", "start:", start, ", response_time:", response_time)

        if model:
            actr.run_full_time(5)
        else:
            while (actr.get_time(False) - start) < 5000:
                actr.process_events()
        time = response_time - start

        # print("TEST: ", "feedback start, actr time", actr.get_time())

        # display feedback
        actr.clear_exp_window(window)
        actr.add_text_to_exp_window(window, feedback, x=150, y=150)
        # start = actr.get_time(model)

        if model:
            actr.run_full_time(5)
        else:
            while (actr.get_time(False) - start) < 5000:
                actr.process_events()

        # print("TEST: ", "trial ends, actr time", actr.get_time())

        # report data
        # calculate RT

        result.append((feedback, block_type, response, time/1000.0))
        print("TEST: (END)>>", "start:", start, ", response_time:", response_time, ", time:", time)
    return result


def create_block(num_trials=8, num_reward=6, num_punish=1, num_neutral=1, block_type="mostly_reward", shuffle=False):
    #R1 block
    prob_list = ["?"] * num_trials
    feedback_list = ["win"] * num_reward + ["lose"] * num_punish + ["neutral"] * num_neutral
    block_list = [block_type] * num_trials
    trials = list(zip(prob_list, feedback_list, block_list))
    if shuffle: random.shuffle(trials)
    return trials

def experiment(num_run=1):
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
    # model_data = pd.DataFrame(model_data, columns=["feedback", "block_type", "response", "RT"])
    print()
    print(model_data.groupby("TrialType").mean())
    print()
    print(model_data["TrialType"].value_counts(normalize=True))

def test_unit1():
    sometrials=create_block()
    sometrials.sort()
    p.pprint(task(sometrials))