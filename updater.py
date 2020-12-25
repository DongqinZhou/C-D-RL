import os
import json
import copy
from ddpgagent import DDPGAgent

DIC_AGENTS = {
    'DDPGAgent': DDPGAgent,
    }


class Updater:
    def __init__(self, cnt_round, dic_agent_conf, dic_exp_conf, dic_traffic_conf, dic_path):
        self.cnt_round = cnt_round
        self.dic_path = dic_path
        self.dic_exp_conf = dic_exp_conf
        self.dic_traffic_conf = dic_traffic_conf
        self.dic_agent_conf = copy.deepcopy(dic_agent_conf)
        self.agent = DIC_AGENTS[self.dic_exp_conf['MODEL_NAME']](
            self.dic_agent_conf,
            self.cnt_round,
            self.dic_traffic_conf,
            self.dic_path
        )
        self.round_sample = []
        self.all_samples = []
        self.path_to_folder = os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"], "train_round",
                                           'round_' + str(self.cnt_round))
        self.all_samples_path = os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"], "train_round")
        if not os.path.exists(self.all_samples_path):
            os.makedirs(self.all_samples_path)

    def load_round_sample(self):
        for folder in os.listdir(self.path_to_folder):
            if 'generator' not in folder:
                continue
            folder_sample = open(os.path.join(self.path_to_folder, folder, 'samples.json'), 'rb')
            folder_sample = json.load(folder_sample)
            self.round_sample.extend(folder_sample)

    def load_all_sample(self):
        if self.cnt_round == 0:
            self.all_samples = self.round_sample
        else:
            self.all_samples = open(os.path.join(self.all_samples_path, "all_samples" + ".json"), "rb")
            self.all_samples = json.load(self.all_samples)
            self.all_samples.extend(self.round_sample)

        if self.cnt_round > (  # this ensures we only have most-recent samples
                self.dic_agent_conf['MAX_MEMORY_LEN'] /
                (self.dic_exp_conf['NUM_GENERATORS'] * self.dic_exp_conf['STEP_LENGTH']) - 1
        ):
            self.all_samples = self.all_samples[-self.dic_agent_conf['MAX_MEMORY_LEN']:]

        json.dump(self.all_samples, open(os.path.join(self.all_samples_path, "all_samples.json"), "w"))

    def load_sample(self):
        self.load_round_sample()
        self.load_all_sample()

    def update_network(self):
        self.agent.prepare_X_Y(self.all_samples)
        json.dump([self.agent.states.tolist(), self.agent.actions.tolist(), self.agent.critic_targets.tolist()],
                  open(os.path.join(self.path_to_folder, "train_set.json"), "w"))
        self.agent.train_network()
        self.agent.save_network_weights_and_architecture("round_{}".format(self.cnt_round))
