# C-D-RL

Code for the C-RL agent in the paper "Model-Free Perimeter Metering Control for Two-Region Urban Networks Using Deep Reinforcement Learning". Inspired by [this](https://github.com/wingsweihua/colight) work

## Usage

* Change `ROOT_PATH` in `run_rl_exp`; then start an experiment by `python run_rl_exp.py`
* Recommend running on Linux machines
* Compatible with GPU acceleration

## Structure

* `run_rl_exp.py`
    * set arguments for experiments and traffic environment
    * set hyper-parameters for RL agent
    * start an experiment

* `pipeline.py`
    * define a streamline for RL agents to interact with environment (`generator`), learn (`updater`), test its performance (`model_test`), and calculate the throughput in the system; finally store relevant files into disk for future use
    * implemente Ape-X architecture with `multiprocessing`
    
* `generator.py`
    * create an RL agent to interact with environment
    * visualize experience
    * store experiences into disk

* `updater.py`
    * load experiences from disk
    * update the actor and critic
    * store updated networks into disk

* `model_test.py`
    * evaluate policy periodically by testing it without exploration
    * visualize its performance after learning is carried out

* `rlplant.py`
    * implemented MFDs plant expressed by dynamics equations
    * receive action from agent
    * return reward back to agent

* `ddpgagent.py`
    * C-RL agent that adopts the DDPG learning algorithm with experience replay and target network
