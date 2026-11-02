from common.rewards import improved_reward
from experiments.train_common import run_training

if __name__=="__main__":
    run_training(improved_reward,"improved",double_dqn=False,
                 epsilon_start=80,epsilon_decay=0.5,epsilon_min=5)
