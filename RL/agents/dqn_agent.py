import random
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from common.config import MAX_MEMORY, BATCH_SIZE, LEARNING_RATE, GAMMA


class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(15, 128), nn.ReLU(), nn.Linear(128, 3))

    def forward(self, x):
        return self.net(x)


class DQNAgent:
    def __init__(
        self,
        double_dqn=False,
        epsilon_start=80,
        epsilon_decay=0.5,
        epsilon_min=5,
        loss="huber",
    ):
        if loss not in ("mse", "huber"):
            raise ValueError("loss must be 'mse' or 'huber'.")

        self.n_games = 0
        self.epsilon = 0
        self.double_dqn = double_dqn
        self.epsilon_start = epsilon_start
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.loss = loss
        self.update_target_every = 10
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = DQN()
        self.target_model = DQN()
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        # "mse" is Experiment 1 (original, no clipping).
        # "huber" is Experiment 2 (Huber loss, gradient clipped at 1.0).
        self.criterion = (
            nn.SmoothL1Loss() if loss == "huber" else nn.MSELoss()
        )
        self.clip_grad = loss == "huber"

    def get_action(self, state):
        self.epsilon = max(
            self.epsilon_start - self.n_games * self.epsilon_decay, self.epsilon_min
        )
        if random.randint(0, 200) < self.epsilon:
            return random.randrange(3)
        with torch.no_grad():
            return int(
                torch.argmax(
                    self.model(torch.tensor(state, dtype=torch.float32))
                ).item()
            )

    def remember(self, s, a, r, ns, done):
        self.memory.append((s, a, r, ns, done))

    def train_long(self):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.tensor(np.asarray(states), dtype=torch.float32)
        next_states = torch.tensor(np.asarray(next_states), dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.bool)
        batch_size = states.shape[0]
        pred = self.model(states)
        with torch.no_grad():
            target_next = self.target_model(next_states)
            if self.double_dqn:
                best_actions = self.model(next_states).argmax(dim=1)
                next_q = target_next.gather(
                    1, best_actions.unsqueeze(1)
                ).squeeze(1)
            else:
                next_q = target_next.max(dim=1).values
            # Terminal states get no bootstrapped value.
            target_q = rewards + GAMMA * next_q * (~dones).float()
        target = pred.detach().clone()
        target[torch.arange(batch_size), actions] = target_q
        self.optimizer.zero_grad()
        loss = self.criterion(pred, target)
        loss.backward()
        if self.clip_grad:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        return float(loss.item())

    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())
