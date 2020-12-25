import json
import os
import matplotlib.pyplot as plt
import copy
import numpy as np
from rlplant import RLPlant
from ddpgagent import DDPGAgent
import matplotlib
matplotlib.use('Agg')

DIC_AGENTS = {
    'DDPGAgent': DDPGAgent,
    }


def test(dic_path, cnt_round, dic_traffic_conf, dic_exp_conf, dic_agent_conf, store=True):
    _dic_agent_conf = copy.deepcopy(dic_agent_conf)
    model_round = "round_%d" % cnt_round
    # do not explore
    _dic_agent_conf['NOISE_PARAMS'] = 0
    _dic_agent_conf['MIN_NOISE_SCALE'] = 0
    _dic_agent_conf['MAX_NOISE_SCALE'] = 0
    _agent = DIC_AGENTS[dic_exp_conf['MODEL_NAME']](_dic_agent_conf, cnt_round + 1, dic_traffic_conf, dic_path)
    # _agent.load_network_weights_and_architecture(model_round) replaced by +1 for cnt_round
    path_to_test_round = os.path.join(dic_path["PATH_TO_WORK_DIRECTORY"], "test_round")
    path_to_log = os.path.join(path_to_test_round, model_round)
    if not os.path.exists(path_to_log):
        os.makedirs(path_to_log)
    path_to_pics = os.path.join(dic_path["PATH_TO_PLOTS"], "test_" + model_round)
    if not os.path.exists(path_to_pics):
        os.makedirs(path_to_pics)

    _rl_plant = RLPlant(dic_traffic_conf, dic_exp_conf)
    _state = _rl_plant.reset()
    _step_num = 0
    _samples = []
    while _step_num < int(dic_exp_conf['EXP_TIME'] / dic_exp_conf['STEP_LENGTH']):
        # print("round {} step {}".format(cnt_round, _step_num))
        step_start_time = _step_num * dic_exp_conf['STEP_LENGTH']
        _action = _agent.choose_action(_state)
        _next_state, _reward, _accumulation, _, _ = _rl_plant.step(_state, _action, step_start_time)
        _samples.append([_state, _action, _accumulation, _next_state, _reward, _step_num])
        _state = _next_state
        _step_num += 1
    # store samples in this round into file
    json.dump(_samples, open(os.path.join(path_to_log, "samples.json"), "w"))
    summary_plot(_samples, path_to_pics, dic_exp_conf, store=store, cnt_round=cnt_round)

    # plot q value prediction; y_pred and y, y: [sample_size, 1]
    path_to_folder = os.path.join(dic_path["PATH_TO_WORK_DIRECTORY"], "train_round", 'round_' + str(cnt_round))
    train_set = open(os.path.join(path_to_folder, "train_set.json"), 'rb')
    train_set = json.load(train_set)
    states, actions, critic_targets = train_set
    y_true = np.array(critic_targets)
    y_pred = _agent.critic.critic_network.predict([np.array(states), np.array(actions)])
    t = list(range(1, 1 + len(states), 10))
    plt.ioff()
    plt.style.use('seaborn-whitegrid')
    plt.figure()  # plot q value prediction
    plt.plot(t, y_true[::10, 0].tolist(), '-.', label='Q_true')
    plt.plot(t, y_pred[::10, 0].tolist(), '-.', label='Q_pred')
    plt.legend(loc='best')
    plt.xlabel('sample number')
    plt.ylabel('q value (-)')
    legend = plt.legend(frameon=1)
    frame = legend.get_frame()
    frame.set_facecolor('white')
    if store:
        plt.savefig(os.path.join(path_to_pics, 'Q-value prediction.png'), bbox_inches='tight')
    plt.close()

    plt.figure()  # plot controls used for updating
    plt.plot(t, np.array(actions)[::10, 0], '-', label=r'$u_{12}$')
    plt.plot(t, np.array(actions)[::10, 1], '-', label=r'$u_{21}$')
    plt.legend(loc='best')
    plt.xlabel('sample number')
    plt.ylabel('u (-)')
    legend = plt.legend(frameon=1)
    frame = legend.get_frame()
    frame.set_facecolor('white')
    if store:
        plt.savefig(os.path.join(path_to_pics, 'train-set-actions.png'), bbox_inches='tight')
    plt.close()


def summary_plot(samples, save_path, dic_exp_conf, store=True, cnt_round=None):
    step_length = dic_exp_conf['STEP_LENGTH']
    # plot the figure at each round and calculate total reward, store plots into file
    u12 = []
    u21 = []
    n11 = []
    n12 = []
    n21 = []
    n22 = []
    n1 = []
    n2 = []
    rewards = []
    for i in range(len(samples)):
        u_12, u_21 = samples[i][1]['u_12'], samples[i][1]['u_21']
        u12.extend([u_12] * step_length)
        u21.extend([u_21] * step_length)

        n11.append(samples[i][0]['accumulations_ratio'][0] * 34000)
        n12.append(samples[i][0]['accumulations_ratio'][1] * 34000)
        n21.append(samples[i][0]['accumulations_ratio'][2] * 17000)
        n22.append(samples[i][0]['accumulations_ratio'][3] * 17000)
        n1.append(n11[-1] + n12[-1])
        n2.append(n21[-1] + n22[-1])
        rewards.extend([samples[i][4]] * step_length)
    total_time = len(samples) * step_length
    t = list(range(1, 1+total_time, 1))

    plt.ioff()
    plt.style.use('seaborn-whitegrid')
    plt.rcParams.update({'font.size': 16, "font.family": "Times new Roman", 'font.weight': 'bold'})
    exp_duration = [x * 60 for x in list(range(0, int(total_time / step_length), 1))]
    plt.figure()  # plot results
    plt.plot(exp_duration, n11, '-', label='n_11')
    plt.plot(exp_duration, n12, '-', label='n_12')
    plt.plot(exp_duration, n21, '-', label='n_21')
    plt.plot(exp_duration, n22, '-', label='n_22')
    plt.plot(exp_duration, n1, '-', label='n_1')
    plt.plot(exp_duration, n2, '-', label='n_2')
    plt.legend(loc='best')
    plt.xlabel('Time (sec)')
    plt.ylabel('Accumulation (veh)')
    legend = plt.legend(frameon=1)
    frame = legend.get_frame()
    frame.set_facecolor('white')
    if store:
        plt.savefig(os.path.join(save_path, 'accumulation.png'), bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(8.88, 6.66), dpi=100)  # plot control
    plt.plot(t, u12, '-', label='u_12')
    plt.plot(t, u21, '-', label='u_21')
    plt.legend(loc='best')
    plt.xlabel('Time (s)')
    plt.ylabel('(-)')
    legend = plt.legend(frameon=1)
    frame = legend.get_frame()
    frame.set_facecolor('white')
    plt.title("Round {}".format(cnt_round))
    if store:
        plt.savefig(os.path.join(save_path, 'control.png'), dpi=100)
    plt.close()

    plt.figure()  # plot rewards
    plt.plot(t, rewards, ':', label='Reward')
    plt.legend(loc='best')
    plt.xlabel('Time (s)')
    plt.ylabel('Rewards (-)')
    legend = plt.legend(frameon=1)
    frame = legend.get_frame()
    frame.set_facecolor('white')
    if store:
        plt.savefig(os.path.join(save_path, 'rewards.png'), bbox_inches='tight')
    plt.close()
    # plt.show()
