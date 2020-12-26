from tensorflow.keras.models import Model, model_from_json
from tensorflow.keras.layers import Dense, Input, concatenate, Lambda
from tensorflow.keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.callbacks import TensorBoard
import os
import random
import numpy as np
import copy
import json
import tensorflow as tf


tf.keras.backend.set_floatx('float64')
SEED = 1234
random.seed(SEED)
tf.random.set_seed(SEED)


class Actor:

    def __init__(self, action_dim, lr, input_dim, target_freq=5, cnt_round=None, dic_path=None):

        self.dic_path = dic_path
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.lr = lr
        self.optimizer = Adam(learning_rate=self.lr)

        if cnt_round == 0:
            self.actor_network = self.build_network()
            self.actor_network_bar = self.build_network_from_copy(self.actor_network)
        else:
            self.load_network_weights_and_architecture(
                "round_{}".format(cnt_round - 1),
                "round_{}".format((cnt_round - 1) // target_freq * target_freq),
            )

    def build_network(self):
        _input = Input((self.input_dim,), name='input')
        dense1 = Dense(64, activation='relu', name='dense1', kernel_initializer='random_normal')(_input)
        dense2 = Dense(64, activation='relu', name='dense2', kernel_initializer='random_normal')(dense1)
        u21 = Dense(1, activation='tanh', kernel_initializer='random_normal')(dense2)
        u12 = Dense(1, activation='tanh', kernel_initializer='random_normal')(dense2)
        u21 = Lambda(lambda x: x * 0.4 + 0.5)(u21)
        u12 = Lambda(lambda x: x * 0.4 + 0.5)(u12)
        out = concatenate([u12, u21])
        model = Model(inputs=_input, outputs=out)
        return model

    def build_network_from_copy(self, network_copy):
        network_structure = network_copy.to_json()
        network_weights = network_copy.get_weights()
        network = model_from_json(network_structure)
        network.set_weights(network_weights)
        return network

    def target_choose_action(self, states):
        return self.actor_network_bar.predict(states)

    def save_network_weights_and_architecture(self, file_name):
        # only need to save weights, json structure will not change
        if 'round_0' in file_name:
            model_json = self.actor_network.to_json()
            with open(os.path.join(self.dic_path["PATH_TO_MODEL"], "actor.json"), "w") as json_file:
                json_file.write(model_json)
        self.actor_network.save_weights(os.path.join(self.dic_path["PATH_TO_MODEL"], "actor_{}.h5".format(file_name)))

    def load_network_weights_and_architecture(self, file_name, bar_file_name):
        json_file = open(os.path.join(self.dic_path["PATH_TO_MODEL"], "actor.json"), 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.actor_network = model_from_json(loaded_model_json)
        self.actor_network = self.build_network()
        self.actor_network.load_weights(os.path.join(self.dic_path["PATH_TO_MODEL"], "actor_{}.h5".format(file_name)))

        self.actor_network_bar = model_from_json(loaded_model_json)
        self.actor_network_bar = self.build_network()
        self.actor_network_bar.load_weights(os.path.join(self.dic_path["PATH_TO_MODEL"],
                                                         "actor_{}.h5".format(bar_file_name)))


class Critic:

    def __init__(self, action_dim, lr, input_dim, target_freq=5, cnt_round=None, dic_path=None):
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.lr = lr
        self.dic_path = dic_path
        # Build models and target models
        if cnt_round == 0:
            self.critic_network = self.build_network()
            self.critic_network_bar = self.build_network_from_copy(self.critic_network)
        else:
            self.load_network_weights_and_architecture(
                "round_{}".format(cnt_round - 1),
                "round_{}".format((cnt_round - 1) // target_freq * target_freq)
            )

    def build_network(self):
        state = Input((self.input_dim,))
        action = Input((self.action_dim,))
        state_action = concatenate([state, action])
        s1 = Dense(64, activation='relu', kernel_initializer='random_normal')(state_action)
        s2 = Dense(64, activation='relu', kernel_initializer='random_normal')(s1)
        out = Dense(1, kernel_initializer='random_normal')(s2)
        model = Model([state, action], out)
        model.compile(optimizer=Adam(lr=self.lr, epsilon=1e-8), loss='mse')
        return model

    @staticmethod
    def build_network_from_copy(network_copy):
        network_structure = network_copy.to_json()
        network_weights = network_copy.get_weights()
        network = model_from_json(network_structure)
        network.set_weights(network_weights)
        return network

    def target_predict(self, inp):
        return self.critic_network_bar.predict(inp)

    def save_network_weights_and_architecture(self, file_name):
        if 'round_0' in file_name:
            model_json = self.critic_network.to_json()
            with open(os.path.join(self.dic_path["PATH_TO_MODEL"], "critic.json"), "w") as json_file:
                json_file.write(model_json)

        self.critic_network.save_weights(os.path.join(self.dic_path["PATH_TO_MODEL"], "critic_{}.h5".format(file_name)))

    def load_network_weights_and_architecture(self, file_name, bar_file_name):
        json_file = open(os.path.join(self.dic_path["PATH_TO_MODEL"], "critic.json"), 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.critic_network = model_from_json(loaded_model_json)
        self.critic_network.load_weights(os.path.join(self.dic_path["PATH_TO_MODEL"], "critic_{}.h5".format(file_name)))
        self.critic_network.compile(optimizer=Adam(lr=self.lr, epsilon=1e-8), loss='mse')

        self.critic_network_bar = model_from_json(loaded_model_json)
        self.critic_network_bar.load_weights(os.path.join(self.dic_path["PATH_TO_MODEL"],
                                                          "critic_{}.h5".format(bar_file_name)))


class DDPGAgent:
    def __init__(self, _dic_agent_conf=None, cnt_round=None, dic_traffic_conf=None, dic_path=None):
        dic_agent_conf = copy.deepcopy(_dic_agent_conf)
        self.dic_path = dic_path
        self.dic_agent_conf = dic_agent_conf
        self.cnt_round = cnt_round
        self.action_dim = dic_agent_conf['ACTION_DIM']
        if cnt_round > self.dic_agent_conf['BEGIN_LINEAR_ROUND']:
            LRA = max(
                self.dic_agent_conf['MIN_LRA'] - self.dic_agent_conf['LRA_LINEAR_DECAY'] *
                (cnt_round - self.dic_agent_conf['BEGIN_LINEAR_ROUND']), self.dic_agent_conf['FINAL_MIN_LRA']
            )
        else:
            LRA = dic_agent_conf['LRA'] * dic_agent_conf['LRA_DECAY'] ** cnt_round
            LRA = max(LRA, dic_agent_conf['MIN_LRA'])
        LRC = dic_agent_conf['LRC'] * dic_agent_conf['LRC_DECAY'] ** cnt_round
        LRC = max(LRC, dic_agent_conf['MIN_LRC'])
        self.input_dim = 0
        feature_names = dic_traffic_conf['LIST_STATE_FEATURES']
        for feature_name in feature_names:
            self.input_dim += dic_traffic_conf["FEATURE_DIM"]["D_" + feature_name.upper()][0]
        noise_params = dic_agent_conf['NOISE_PARAMS'] - dic_agent_conf['NOISE_DECAY'] * cnt_round
        self.noise_params = float(
            np.clip(noise_params, dic_agent_conf['MIN_NOISE_SCALE'], dic_agent_conf['MAX_NOISE_SCALE']))
        target_freq = self.dic_agent_conf['UPDATE_Q_BAR_FREQ']

        self.actor = Actor(self.action_dim, LRA, self.input_dim, target_freq, cnt_round, dic_path)
        self.critic = Critic(self.action_dim, LRC, self.input_dim, target_freq, cnt_round, dic_path)

    @staticmethod
    def state_to_input(state):
        _input = []
        for key, value in state.items():
            _input.extend(value)
        _input = np.array(_input).reshape(1, -1)
        return _input

    def choose_action(self, state):
        predict_action = self.actor.actor_network.predict(self.state_to_input(state))[0]
        
        if self.noise_params == 0:
            actions = predict_action
        else:
            noise = np.array([random.gauss(0, self.noise_params), random.gauss(0, self.noise_params)])
            actions = predict_action + noise

        actions = np.clip(actions, self.dic_agent_conf["MIN_ACTION"], self.dic_agent_conf['MAX_ACTION'])
        action = {'u_12': round(actions[0], 3), 'u_21': round(actions[1], 3)}
        return action

    def prepare_X_Y(self, samples):
        ind_end = len(samples)
        ind_sta = max(0, ind_end - self.dic_agent_conf["MAX_MEMORY_LEN"])
        samples_after_forget = samples[ind_sta: ind_end]
        sample_size = min(self.dic_agent_conf["SAMPLE_SIZE"], len(samples_after_forget))
        sample_slice = random.sample(samples_after_forget, sample_size)

        states = np.zeros((len(sample_slice), self.input_dim))
        next_states = np.zeros((len(sample_slice), self.input_dim))
        actions = np.zeros((len(sample_slice), self.action_dim))
        rewards = np.zeros((len(sample_slice), 1))

        for i in range(len(sample_slice)):
            state, action, next_state, reward, _ = sample_slice[i]
            states[i] = self.state_to_input(state)
            actions[i] = np.fromiter(action.values(), dtype=float)
            next_states[i] = self.state_to_input(next_state)
            rewards[i] = reward

        target_actions = self.actor.target_choose_action(next_states)
        target_q = self.critic.target_predict([next_states, target_actions])
        critic_targets = rewards + self.dic_agent_conf['GAMMA'] * target_q

        self.states = states
        self.actions = actions
        self.critic_targets = critic_targets

    def train_network(self):

        actor_before_weights = self.actor.actor_network.get_weights()
        critic_before_weights = self.critic.critic_network.get_weights()

        # update critic
        epochs = self.dic_agent_conf["EPOCHS"]
        batch_size = min(self.dic_agent_conf["BATCH_SIZE"], len(self.states))
        if self.dic_agent_conf['IF_CALLBACK']:
            es = EarlyStopping(monitor='val_loss', patience=self.dic_agent_conf['PATIENCE'], verbose=0, mode='min')
            tb = TensorBoard(log_dir='./tensorboard')
            self.critic.critic_network.fit([self.states, self.actions], self.critic_targets, batch_size=batch_size,
                                           epochs=epochs, shuffle=False, verbose=2,
                                           validation_split=0.2, callbacks=[es, tb])
        else:
            self.critic.critic_network.fit([self.states, self.actions], self.critic_targets, batch_size=batch_size,
                                           epochs=epochs, verbose=2, shuffle=False)
        # update actor
        for _epoch in range(self.dic_agent_conf['ACTOR_EPOCHS']):
            with tf.GradientTape() as tape:  # chain rule
                actions = self.actor.actor_network(self.states)

                q = self.critic.critic_network([self.states, actions])
                loss = - tf.reduce_mean(q)  # minimize -q <=> maximize q; loss for actor is -q
            dq_dtheta = tape.gradient(loss, self.actor.actor_network.trainable_variables)
            if self.dic_agent_conf['GRADIENT_CLIP']:
                max_abs = np.max([np.max(np.abs(a)) for a in dq_dtheta])
                dq_dtheta_clip = dq_dtheta / max_abs
                self.actor.optimizer.apply_gradients(zip(dq_dtheta_clip, self.actor.actor_network.trainable_variables))
            else:
                self.actor.optimizer.apply_gradients(zip(dq_dtheta, self.actor.actor_network.trainable_variables))

        actor_after_weights = self.actor.actor_network.get_weights()
        critic_after_weights = self.critic.critic_network.get_weights()

        actor_before = []
        actor_after = []
        for ind in range(len(actor_before_weights)):
            actor_before.extend(list(actor_before_weights[ind].flatten()))
            actor_after.extend(list(actor_after_weights[ind].flatten()))
        rmse = np.sqrt(((np.array(actor_after) - np.array(actor_before)) ** 2).mean(axis=None)).astype(float)
        if self.cnt_round == 0:
            json.dump([rmse], open(os.path.join(self.dic_path["PATH_TO_MODEL"], "actor_rmse.json"), "w"))
        else:
            rmse_list = open(os.path.join(self.dic_path["PATH_TO_MODEL"], "actor_rmse.json"), 'rb')
            rmse_list = json.load(rmse_list)
            rmse_list.append(rmse)
            json.dump(rmse_list, open(os.path.join(self.dic_path["PATH_TO_MODEL"], "actor_rmse.json"), "w"))

        critic_before = []
        critic_after = []
        for ind in range(len(critic_before_weights)):
            critic_before.extend(list(critic_before_weights[ind].flatten()))
            critic_after.extend(list(critic_after_weights[ind].flatten()))
        rmse = np.sqrt(((np.array(critic_after) - np.array(critic_before)) ** 2).mean(axis=None)).astype(float)
        if self.cnt_round == 0:
            json.dump([rmse], open(os.path.join(self.dic_path["PATH_TO_MODEL"], "critic_rmse.json"), "w"))
        else:
            rmse_list = open(os.path.join(self.dic_path["PATH_TO_MODEL"], "critic_rmse.json"), 'rb')
            rmse_list = json.load(rmse_list)
            rmse_list.append(rmse)
            json.dump(rmse_list, open(os.path.join(self.dic_path["PATH_TO_MODEL"], "critic_rmse.json"), "w"))

    def save_network_weights_and_architecture(self, file_name):
        self.actor.save_network_weights_and_architecture(file_name)
        self.critic.save_network_weights_and_architecture(file_name)
