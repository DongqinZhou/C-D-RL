import copy
import os
import json
from rlplant import RLPlant
import matplotlib.pyplot as plt
from ddpgagent import DDPGAgent

DIC_AGENTS = {
    'DDPGAgent': DDPGAgent,
    }


class Generator:
    def __init__(self, cnt_round, cnt_gen, dic_exp_conf, dic_agent_conf, dic_traffic_conf, dic_path):
        self.cnt_round = cnt_round
        self.cnt_gen = cnt_gen
        self.dic_exp_conf = dic_exp_conf
        self.dic_agent_conf = copy.deepcopy(dic_agent_conf)
        self.dic_traffic_conf = dic_traffic_conf
        self.dic_path = dic_path
        self.agent = DIC_AGENTS[self.dic_exp_conf['MODEL_NAME']](
            self.dic_agent_conf,
            self.cnt_round,
            self.dic_traffic_conf,
            self.dic_path
        )
        self.samples = []
        self.rl_plant = RLPlant(self.dic_traffic_conf, self.dic_exp_conf)
        self.path_to_samples = os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"], "train_round",
                                            "round_" + str(self.cnt_round), "generator_" + str(self.cnt_gen))
        if not os.path.exists(self.path_to_samples):
            os.makedirs(self.path_to_samples)

    def generate(self):
        state = self.rl_plant.reset()
        step_num = 0
        while step_num < int(self.dic_exp_conf['EXP_TIME'] / self.dic_exp_conf['STEP_LENGTH']):
            # print("round {} generation {} step {}".format(self.cnt_round, self.cnt_gen, step_num))
            step_start_time = step_num * self.dic_exp_conf['STEP_LENGTH']
            action = self.agent.choose_action(state)
            next_state, reward, _, _, _ = self.rl_plant.step(state, action, step_start_time)
            self.samples.append([state, action, next_state, reward, step_num])
            state = next_state
            step_num += 1
        # store samples in this generation into file
        json.dump(self.samples, open(os.path.join(self.path_to_samples, "samples.json"), "w"))
        self.exp_plot()

    def exp_plot(self):
        step_length = self.dic_exp_conf['STEP_LENGTH']
        samples = self.samples
        u12 = []
        u21 = []

        for i in range(len(samples)):
            u_12, u_21 = samples[i][1]['u_12'], samples[i][1]['u_21']
            u12.extend([u_12] * step_length)
            u21.extend([u_21] * step_length)

        total_time = len(samples) * step_length
        t = list(range(1, 1 + total_time, 1))

        plt.ioff()
        plt.style.use('seaborn-whitegrid')
        plt.figure()  # plot results
        plt.plot(t, u12, '-', label=r'$u_{12}$')
        plt.plot(t, u21, '-', label=r'$u_{21}$')
        plt.legend(loc='best')
        plt.xlabel('Time (s)')
        plt.ylabel('(-)')
        legend = plt.legend(frameon=1)
        frame = legend.get_frame()
        frame.set_facecolor('white')
        plt.savefig(os.path.join(self.path_to_samples, 'control.png'), bbox_inches='tight')
        plt.close()
