from common.rewards import standard_reward
from experiments.train_common import run_training

if __name__=="__main__":
    run_training(standard_reward,"standard",double_dqn=False,
                 epsilon_start=80,epsilon_decay=2.0,epsilon_min=2)
