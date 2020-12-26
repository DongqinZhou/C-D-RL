import argparse
import os
import time
from pipeline import Pipeline
import datetime as dt
from multiprocessing import Process

t0 = dt.datetime.now()
ROOT_PATH = r'/path/to/file/storage'


def pipeline_wrapper(dic_traffic_conf, dic_exp_conf, dic_agent_conf, dic_path):
    ppl = Pipeline(
        dic_traffic_conf=dic_traffic_conf,
        dic_exp_conf=dic_exp_conf,
        dic_agent_conf=dic_agent_conf,
        dic_path=dic_path
    )
    ppl.run()


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--step_length", type=int, default=60)
    parser.add_argument("--exp_time", type=int, default=3600)
    parser.add_argument("--alpha", type=int, default=0)
    parser.add_argument("--sigma", type=int, default=0)
    parser.add_argument("--model", type=str, default='DDPGAgent')
    parser.add_argument("--num_round", type=int, default=250)
    parser.add_argument("--num_gen", type=int, default=32)
    parser.add_argument("--memo", type=str, default='0_0')
    parser.add_argument("--demand_scaler", type=int, default=1.0)

    return parser.parse_args()


def main(args=None):
    assert float(args.memo.split('_')[0]) == args.sigma and float(args.memo.split('_')[1]) == args.alpha

    dic_exp_conf = {
        "MODEL_NAME": args.model,
        "EXP_TIME": args.exp_time,
        "NUM_ROUNDS": args.num_round,
        "NUM_GENERATORS": args.num_gen,
        "STEP_LENGTH": args.step_length,
        "ALPHA": args.alpha,
        "SIGMA": args.sigma,
        "EARLY_STOPPING": True,
        "CV_THRESHOLD": 0.0005,
        "NORMAL_FACTOR": 5,
        "IF_CLIP": False,
    }

    dic_traffic_conf = {
        "LIST_STATE_FEATURES": [
            "avg_demand",
            "accumulations_ratio",
        ],

        "FEATURE_DIM": dict(
            D_ACCUMULATIONS=(4,),
            D_AVG_DEMAND=(4,),
            D_ACCUMULATIONS_RATIO=(4,),
        ),

        "INITIAL_ACCUMULATIONS": [3000, 3000, 2500, 2500],
        "INITIAL_ACCUMULATIONS_RATIO": [3000 / 34000, 3000 / 34000, 2500 / 17000, 2500 / 17000],
        "MAX_DEMAND": [aa * args.demand_scaler for aa in [0.9, 3.25, 1.25, 1.5]],
        "DEMAND_SCALER": args.demand_scaler,
    }

    dic_agent_conf = {
        # buffer parameters
        "SAMPLE_SIZE": 1000,
        "MAX_MEMORY_LEN": 10000,

        # learning parameters
        "EPOCHS": 128,
        "BATCH_SIZE": 256,
        "LRC": 0.001,
        "LRC_DECAY": 0.98,
        "MIN_LRC": 0.0001,

        "ACTOR_EPOCHS": 2,  # this determines the number of gradient ascent steps for actor networks
        "LRA": 0.0025,
        "LRA_DECAY": 0.93,
        "MIN_LRA": 0.0004,
        "BEGIN_LINEAR_ROUND": 199,  # add linear decay for LRA
        "LRA_LINEAR_DECAY": 8e-6,
        "FINAL_MIN_LRA": 0.00005,
        "GRADIENT_CLIP": True,
        "UPDATE_Q_BAR_FREQ": 5,  # hard update for target networks
        "PATIENCE": 20,
        "IF_CALLBACK": True,

        # exploration parameters
        "NOISE_PARAMS": 0.3,  # Gaussian noise scale
        "NOISE_DECAY": 0.001,
        "MIN_NOISE_SCALE": 0.05,
        "MAX_NOISE_SCALE": 0.4,
        # other fixed parameters
        "LOSS_FUNCTION": "mean_squared_error",
        "GAMMA": 0.95,
        "MIN_ACTION": 0.1,
        "MAX_ACTION": 0.9,
        "ACTION_DIM": 2,
    }

    dic_path = {
        "PATH_TO_WORK_DIRECTORY": os.path.join(ROOT_PATH, "records", args.memo,
                                                time.strftime('%m_%d_%H_%M_%S', time.localtime(time.time()))),
        "PATH_TO_MODEL": os.path.join(ROOT_PATH, "model", args.memo,
                                        time.strftime('%m_%d_%H_%M_%S', time.localtime(time.time()))),
        "PATH_TO_PLOTS": os.path.join(ROOT_PATH, 'plots', args.memo,
                                        time.strftime('%m_%d_%H_%M_%S', time.localtime(time.time()))),
        "PATH_TO_TEST": os.path.join(ROOT_PATH, 'test', args.memo,
                                        time.strftime('%m_%d_%H_%M_%S', time.localtime(time.time()))),
        "PATH_TO_RESULTS": os.path.join(ROOT_PATH, 'results', args.memo,
                                        time.strftime('%m_%d_%H_%M_%S', time.localtime(time.time())))
    }

    p = Process(
        target=pipeline_wrapper,
        args=(
            dic_traffic_conf,
            dic_exp_conf,
            dic_agent_conf,
            dic_path
        )
    )
    p.start()
    print("pipeline to join")
    p.join()
    print("pipeline finish join")
    

if __name__ == "__main__":
    _args = parse_args()
    main(_args)
    tf = dt.datetime.now()
    print("time elapsed is: {} seconds".format((tf - t0).seconds))
