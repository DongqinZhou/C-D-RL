import copy
import random

U_ACCUMULATION_1 = 34000
U_ACCUMULATION_2 = 17000
L_ACCUMULATION = 0
MAX_COMPLETION = 33168 / 3600


class RLPlant:

    def __init__(self, dic_traffic_conf=None, dic_exp_conf=None):
        if dic_traffic_conf is None or dic_exp_conf is None:
            print("must have traffic configuration and experiment configuration")
            return
        self.dic_traffic_conf = dic_traffic_conf
        self.dic_exp_conf = dic_exp_conf
        self.sigma = dic_exp_conf['SIGMA']
        self.step_length = dic_exp_conf['STEP_LENGTH']
        self.alpha = dic_exp_conf['ALPHA']

    @staticmethod
    def q_11(sigma, t, scaler=1):
        randomness = random.gauss(0, sigma)
        if t <= 300:
            return max(0.25 * (1 + randomness), 0) * scaler
        elif t <= 1300:
            return max((0.00065 * t + 0.055) * (1 + randomness), 0) * scaler
        elif t <= 2200:
            return max(0.9 * (1 + randomness), 0) * scaler
        elif t <= 3200:
            return max((-0.00065 * t + 2.33) * (1 + randomness), 0) * scaler
        else:
            return max(0.25 * (1 + randomness), 0) * scaler

    @staticmethod
    def q_12(sigma, t, scaler=1):
        randomness = random.gauss(0, sigma)
        if t <= 200:
            return max((0.015 * t + 0.25) * (1 + randomness), 0) * scaler
        elif t <= 3000:
            return max(3.25 * (1 + randomness), 0) * scaler
        elif t <= 3600:
            return max((18.25 - t / 200) * (1 + randomness), 0) * scaler
        else:
            return max(0.25 * (1 + randomness), 0) * scaler

    @staticmethod
    def q_21(sigma, t, scaler=1):
        randomness = random.gauss(0, sigma)
        if t <= 300:
            return max(0.25 * (1 + randomness), 0) * scaler
        elif t <= 1800:
            return max((t / 1500 + 0.05) * (1 + randomness), 0) * scaler
        elif t <= 3200:
            return max(1.25 * (1 + randomness), 0) * scaler
        elif t <= 3600:
            return max((9.25 - t / 400) * (1 + randomness), 0) * scaler
        else:
            return max(0.25 * (1 + randomness), 0) * scaler

    @staticmethod
    def q_22(sigma, t, scaler=1):
        randomness = random.gauss(0, sigma)
        if t <= 100:
            return max(0.25 * (1 + randomness), 0) * scaler
        elif t <= 900:
            return max((1.25 / 800 * t + 0.09375) * (1 + randomness), 0) * scaler
        elif t <= 2700:
            return max(1.5 * (1 + randomness), 0) * scaler
        elif t <= 3500:
            return max((-1.25 / 800 * t + 5.71875) * (1 + randomness), 0) * scaler
        else:
            return max(0.25 * (1 + randomness), 0) * scaler

    def get_reward(self, results):

        accumulations = results[0]
        completion = results[1]
        if self.dic_exp_conf["IF_CLIP"]:
            reward = (completion[0] + completion[3]) / (
                    MAX_COMPLETION * self.step_length * 2 * self.dic_exp_conf["NORMAL_FACTOR"])
        else:
            reward = (completion[0] + completion[3]) / (MAX_COMPLETION * self.step_length * 2)
        if accumulations[4] > U_ACCUMULATION_1 or accumulations[5] > U_ACCUMULATION_2 \
                or min(accumulations) < L_ACCUMULATION:
            reward -= 2
        return reward

    def get_avg_demand(self, next_step_start_time=0):
        sigma = 0
        max_demand = self.dic_traffic_conf['MAX_DEMAND']
        demand_scaler = self.dic_traffic_conf['DEMAND_SCALER']
        t = next_step_start_time + self.step_length / 2
        avg_q11 = self.q_11(sigma, t, demand_scaler) / max_demand[0]
        avg_q12 = self.q_12(sigma, t, demand_scaler) / max_demand[1]
        avg_q21 = self.q_21(sigma, t, demand_scaler) / max_demand[2]
        avg_q22 = self.q_22(sigma, t, demand_scaler) / max_demand[3]
        return [avg_q11, avg_q12, avg_q21, avg_q22]

    @staticmethod
    def print_action(actions):
        for key, value in actions.items():
            print("{} = {}".format(key, value))

    @staticmethod
    def MFD(n, alpha=0):  # f(n) + error
        error = random.uniform(-alpha, alpha) * n
        if n < 14000:
            return (2.28e-8 * n ** 3 - 8.62e-4 * n ** 2 + 9.58 * n + error) / 3600
        elif n < 34000:
            return (27731 - 1.38655 * (n - 14000) + error) / 3600
        else:
            return 0

    def inner_MFD(self, n, alpha=0):
        return self.MFD(2 * n, alpha) * 0.5

    def calculate(self, n0, params, step_start_time):
        n_11, n_12, n_21, n_22 = n0
        sigma, alpha, u_12, u_21 = params

        M11 = n_11 / (n_11 + n_12) * self.MFD(n_11 + n_12, alpha)
        M12 = n_12 / (n_11 + n_12) * self.MFD(n_11 + n_12, alpha)
        M21 = n_21 / (n_21 + n_22) * self.inner_MFD(n_21 + n_22, alpha)
        M22 = n_22 / (n_21 + n_22) * self.inner_MFD(n_21 + n_22, alpha)

        demand_scalar = self.dic_traffic_conf['DEMAND_SCALER']
        q11 = self.q_11(sigma, step_start_time + self.step_length / 2, scaler=demand_scalar)
        q12 = self.q_12(sigma, step_start_time + self.step_length / 2, scaler=demand_scalar)
        q21 = self.q_21(sigma, step_start_time + self.step_length / 2, scaler=demand_scalar)
        q22 = self.q_22(sigma, step_start_time + self.step_length / 2, scaler=demand_scalar)
        n_11 += (q11 + u_21 * M21 - M11) * self.step_length
        n_12 += (q12 - u_12 * M12) * self.step_length
        n_21 += (q21 - u_21 * M21) * self.step_length
        n_22 += (q22 + u_12 * M12 - M22) * self.step_length

        # here the completion for reward is at the end of a time step
        M11 = n_11 / (n_11 + n_12) * self.MFD(n_11 + n_12, alpha)
        M12 = n_12 / (n_11 + n_12) * self.MFD(n_11 + n_12, alpha)
        M21 = n_21 / (n_21 + n_22) * self.inner_MFD(n_21 + n_22, alpha)
        M22 = n_22 / (n_21 + n_22) * self.inner_MFD(n_21 + n_22, alpha)

        next_state = [n_11, n_12, n_21, n_22, n_11 + n_12, n_21 + n_22]
        completion = [M11 * self.step_length, M12 * self.step_length, M21 * self.step_length, M22 * self.step_length]

        return [next_state, completion, [q11, q12, q21, q22]]

    def DAE(self, state, actions, step_start_time):

        accumulations = [U_ACCUMULATION_1 * x for x in state['accumulations_ratio'][0:2]] + \
                        [U_ACCUMULATION_2 * x for x in state['accumulations_ratio'][2:]]
        u_12, u_21 = actions['u_12'], actions['u_21']
        try:
            params = [self.sigma, self.alpha, round(u_12, 3), round(u_21, 3)]
            results = self.calculate(accumulations, params, step_start_time)
            return results, [round(u_12, 2), round(u_21, 2)]
        except:
            print("accumulations are: ", accumulations)
            print("step start time is: ", step_start_time)
            self.print_action(actions)
            raise NotImplementedError

    def reset(self):
        state = {}
        for feature_name in self.dic_traffic_conf['LIST_STATE_FEATURES']:
            if feature_name == 'avg_demand':
                state[feature_name] = self.get_avg_demand()
                continue
            state[feature_name] = self.dic_traffic_conf['INITIAL_' + feature_name.upper()]

        return state

    def step(self, state, actions, step_start_time):

        _state = copy.deepcopy(state)
        results, _ = self.DAE(_state, actions, step_start_time)
        end_accumulation = results[0][0:4]

        _state['accumulations_ratio'] = [x / U_ACCUMULATION_1 for x in end_accumulation[0:2]] + \
                                        [x / U_ACCUMULATION_2 for x in end_accumulation[2:]]
        _state['avg_demand'] = self.get_avg_demand(step_start_time + self.step_length)

        reward = self.get_reward(results)
        return _state, reward, results[0], results[1], results[2]  # accumulation, completion, demand


if __name__ == "__main__":
    plant = RLPlant()
    # plant.DAE()
    # plant.plot()
