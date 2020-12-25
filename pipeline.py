import os
import json
from shutil import copy
from generator import Generator
from updater import Updater
import model_test
import datetime as dt
import numpy as np
import random
import matplotlib.pyplot as plt
from multiprocessing import Process
import matplotlib
matplotlib.use('Agg')


def MFD(n, alpha=0):  # f(n) + error
    error = random.uniform(-alpha, alpha) * n
    if n < 14000:
        return (2.28e-8 * n ** 3 - 8.62e-4 * n ** 2 + 9.58 * n + error) / 3600
    elif n < 34000:
        return (27731 - 1.38655 * (n - 14000) + error) / 3600
    else:
        return 0


def inner_MFD(n, alpha=0):
    return 0.5 * MFD(2 * n, alpha)


class Pipeline:
    def __init__(self, dic_traffic_conf, dic_exp_conf, dic_agent_conf, dic_path):
        self.dic_exp_conf = dic_exp_conf
        self.dic_agent_conf = dic_agent_conf
        self.dic_traffic_conf = dic_traffic_conf
        self.dic_path = dic_path
        self._path_check()

    def _path_check(self):
        # check path
        if os.path.exists(self.dic_path["PATH_TO_WORK_DIRECTORY"]):
            print(self.dic_path["PATH_TO_WORK_DIRECTORY"])
        else:
            os.makedirs(self.dic_path["PATH_TO_WORK_DIRECTORY"])

        if os.path.exists(self.dic_path["PATH_TO_MODEL"]):
            print(self.dic_path["PATH_TO_MODEL"])
        else:
            os.makedirs(self.dic_path["PATH_TO_MODEL"])

        if os.path.exists(self.dic_path["PATH_TO_RESULTS"]):
            print(self.dic_path["PATH_TO_RESULTS"])
        else:
            os.makedirs(self.dic_path["PATH_TO_RESULTS"])

        self.all_samples_path = os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"], "train_round")
        if not os.path.exists(self.all_samples_path):
            os.makedirs(self.all_samples_path)

        json.dump(self.dic_agent_conf, open(os.path.join(self.dic_path['PATH_TO_MODEL'], "agent.conf"), "w"),
                  indent=4)
        json.dump(self.dic_traffic_conf, open(os.path.join(self.dic_path['PATH_TO_MODEL'], "traffic.conf"), "w"),
                  indent=4)
        json.dump(self.dic_exp_conf, open(os.path.join(self.dic_path['PATH_TO_MODEL'], "exp.conf"), "w"),
                  indent=4)
        json.dump(self.dic_path, open(os.path.join(self.dic_path['PATH_TO_MODEL'], "path.conf"), "w"),
                  indent=4)

    def print_throughput(self, cnt_round):
        sample_file = open(
            os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"], "test_round", 'round_' + str(cnt_round),
                         "samples" + ".json"), "rb")
        sample_set = json.load(sample_file)
        rl_completion = []

        for i in range(len(sample_set)):
            _m11 = sample_set[i][2][0] / sample_set[i][2][4] * MFD(sample_set[i][2][4]) * \
                   self.dic_exp_conf['STEP_LENGTH']
            _m22 = sample_set[i][2][3] / sample_set[i][2][5] * inner_MFD(sample_set[i][2][5]) * \
                   self.dic_exp_conf['STEP_LENGTH']
            rl_completion.append(_m11 + _m22)
        rl_completion = np.cumsum(rl_completion)

        completion = rl_completion[-1]
        print("completion is: {} veh".format(completion))

        return completion

    def plot_performance(self, completion, completion_avg, cnt_round):
        round_name = "round_" + str(cnt_round)
        path_to_pics = os.path.join(self.dic_path["PATH_TO_PLOTS"], "test_" + round_name)
        x = list(range(1, 1 + len(completion), 1))
        plt.style.use('seaborn-whitegrid')
        plt.ioff()

        plt.figure()  # plot results
        plt.plot(x, completion, '-', label='completion')
        plt.plot(x, completion_avg, '-', label='completion avg')
        plt.legend(loc='best')
        plt.xlabel('Round (-)')
        plt.ylabel('Completion (veh)')
        plt.title("Throughput")
        legend = plt.legend(frameon=1)
        frame = legend.get_frame()
        frame.set_facecolor('white')
        plt.savefig(os.path.join(path_to_pics, 'completion.png'), bbox_inches='tight')
        plt.close()

        # critic rmse
        rmse_list = open(os.path.join(self.dic_path["PATH_TO_MODEL"], "critic_rmse.json"), 'rb')
        rmse_list = json.load(rmse_list)
        rmse_avg = []
        for ind in range(len(rmse_list)):
            if ind < 19:
                rmse_avg.append(np.mean(rmse_list[:(ind + 1)]))
            else:
                rmse_avg.append(np.mean(rmse_list[(ind - 19):(ind + 1)]))

        plt.figure()  # plot results
        plt.plot(x, rmse_list, '-', label='rmse')
        plt.plot(x, rmse_avg, '-', label='rmse_avg')
        plt.legend(loc='best')
        plt.xlabel('Round (-)')
        plt.ylabel('RMSE')
        plt.title("Critic RMSE")
        legend = plt.legend(frameon=1)
        frame = legend.get_frame()
        frame.set_facecolor('white')
        plt.savefig(os.path.join(path_to_pics, 'critic_rmse.png'), bbox_inches='tight')
        plt.close()

        # actor rmse
        rmse_list = open(os.path.join(self.dic_path["PATH_TO_MODEL"], "actor_rmse.json"), 'rb')
        rmse_list = json.load(rmse_list)
        rmse_avg = []
        for ind in range(len(rmse_list)):
            if ind < 19:
                rmse_avg.append(np.mean(rmse_list[:(ind + 1)]))
            else:
                rmse_avg.append(np.mean(rmse_list[(ind - 19):(ind + 1)]))

        plt.figure()  # plot results
        plt.plot(x, rmse_list, '-', label='rmse')
        plt.plot(x, rmse_avg, '-', label='rmse_avg')
        plt.legend(loc='best')
        plt.xlabel('Round (-)')
        plt.ylabel('RMSE')
        plt.title("Actor RMSE")
        legend = plt.legend(frameon=1)
        frame = legend.get_frame()
        frame.set_facecolor('white')
        plt.savefig(os.path.join(path_to_pics, 'actor_rmse.png'), bbox_inches='tight')
        plt.close()

    def early_stopping(self, completion_avg, cnt_round):
        if cnt_round < 50:
            return 0
        cgavg_under_exam = np.array(completion_avg[-20:])
        mean_cgavg = np.mean(cgavg_under_exam)
        std_cgavg = np.std(cgavg_under_exam)

        if std_cgavg / mean_cgavg < self.dic_exp_conf['CV_THRESHOLD']:
            return 1
        else:
            return 0

    def write_results(self, completion, cnt_round):
        json.dump(completion, open(os.path.join(self.all_samples_path,
                                                "round_{}_completion.json".format(cnt_round)), "w"))

    def copy_files_to_result_folder(self, cnt_round):
        files_under_model_folder = [
            "path.conf", "traffic.conf", "agent.conf", "exp.conf", 'actor.json', 'critic.json', "actor_rmse.json",
            "critic_rmse.json", "actor_round_{}.h5".format(cnt_round), "critic_round_{}.h5".format(cnt_round)
        ]
        for file in files_under_model_folder:
            copy(
                os.path.join(self.dic_path['PATH_TO_MODEL'], file),
                os.path.join(self.dic_path['PATH_TO_RESULTS'], file)
            )
        copy(os.path.join(self.all_samples_path, "round_{}_completion.json".format(cnt_round)),
             os.path.join(self.dic_path['PATH_TO_RESULTS'], "round_{}_completion.json".format(cnt_round)))
        path_to_test_round = os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"], "test_round")
        copy(
            os.path.join(path_to_test_round, 'round_{}'.format(cnt_round), 'samples.json'),
            os.path.join(self.dic_path['PATH_TO_RESULTS'], 'round_{}_test_samples.json'.format(cnt_round))
        )
        path_to_pics = os.path.join(self.dic_path["PATH_TO_PLOTS"], "test_round_{}".format(cnt_round))
        pics = os.listdir(path_to_pics)
        for pic in pics:
            copy(
                os.path.join(path_to_pics, pic),
                os.path.join(self.dic_path['PATH_TO_RESULTS'], pic)
            )

    @staticmethod
    def generator_wrapper(cnt_round, cnt_gen, dic_exp_conf, dic_agent_conf, dic_traffic_conf, dic_path):
        generator = Generator(cnt_round=cnt_round,
                              cnt_gen=cnt_gen,
                              dic_exp_conf=dic_exp_conf,
                              dic_agent_conf=dic_agent_conf,
                              dic_traffic_conf=dic_traffic_conf,
                              dic_path=dic_path
                              )
        generator.generate()

    @staticmethod
    def updater_wrapper(cnt_round, dic_agent_conf, dic_exp_conf, dic_traffic_conf, dic_path):
        updater = Updater(
            cnt_round=cnt_round,
            dic_agent_conf=dic_agent_conf,
            dic_exp_conf=dic_exp_conf,
            dic_traffic_conf=dic_traffic_conf,
            dic_path=dic_path
        )
        updater.load_sample()
        updater.update_network()

    def run(self):
        all_completion = []
        all_completion_avg = []

        for cnt_round in range(self.dic_exp_conf['NUM_ROUNDS']):
            if (cnt_round + 1) % 50 == 0:
                self.write_results(all_completion, cnt_round)

            print("round %d starts" % cnt_round)
            t0 = dt.datetime.now()
            process_list = []

            print("========================== generator ==========================")
            for cnt_gen in range(self.dic_exp_conf['NUM_GENERATORS']):
                p = Process(target=self.generator_wrapper,
                            args=(cnt_round,
                                  cnt_gen,
                                  self.dic_exp_conf,
                                  self.dic_agent_conf,
                                  self.dic_traffic_conf,
                                  self.dic_path))
                print("generation {} starts".format(cnt_gen))
                p.start()
                process_list.append(p)

            print('join generator')
            for proc in process_list:
                proc.join()
            print("end join")

            t1 = dt.datetime.now()
            print("Round {} Generator takes: {} seconds".format(cnt_round, (t1 - t0).seconds))

            print("========================== updater ==========================")
            p = Process(target=self.updater_wrapper,
                        args=(cnt_round,
                              self.dic_agent_conf,
                              self.dic_exp_conf,
                              self.dic_traffic_conf,
                              self.dic_path))
            p.start()
            print("updater to join")
            p.join()
            print("updater finish join")

            t2 = dt.datetime.now()
            print("Round {} Updater takes: {} seconds".format(cnt_round, (t2 - t1).seconds))

            print("========================== model test and summary ==========================")
            test_round = True
            p = Process(target=model_test.test,
                        args=(self.dic_path,
                              cnt_round,
                              self.dic_traffic_conf,
                              self.dic_exp_conf,
                              self.dic_agent_conf,
                              test_round))
            p.start()
            p.join()
            t3 = dt.datetime.now()
            print("Round {} model test takes: {} seconds".format(cnt_round, (t3 - t2).seconds))

            print("========================== print throughput ==========================")
            completion = self.print_throughput(cnt_round)

            all_completion.append(completion)
            all_completion_avg.append(np.mean(all_completion[-20:]))
            if cnt_round >= 19:
                self.plot_performance(all_completion, all_completion_avg, cnt_round)

            print("==================== early stopping ====================")
            if self.dic_exp_conf["EARLY_STOPPING"]:
                flag = self.early_stopping(all_completion_avg, cnt_round)
                if flag == 1:
                    self.dic_agent_conf["LRA"] = 1e-4
                    self.dic_agent_conf["MIN_LRA"] = 1e-4
                    self.dic_agent_conf["FINAL_MIN_LRA"] = 1e-4

        self.write_results(all_completion, cnt_round)
        self.copy_files_to_result_folder(cnt_round)
