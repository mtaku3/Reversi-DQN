import numpy as np
import creversi.gym_reversi
from creversi import *

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_tensor_type(torch.cuda.FloatTensor)

def get_state(board):
    features = np.empty((1, 2, 8, 8), dtype=np.float32)
    board.piece_planes(features[0])
    state = torch.from_numpy(features[:1]).to(device)
    return state

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(2, 128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(128)
        self.conv7 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm2d(128)
        self.conv8 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn8 = nn.BatchNorm2d(128)
        self.conv9 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn9 = nn.BatchNorm2d(128)
        self.conv10 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn10 = nn.BatchNorm2d(128)
        self.fcl1 = nn.Linear(128 * 64, 128)
        self.fcl2 = nn.Linear(128, 65)

    def predict(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = F.relu(self.bn7(self.conv7(x)))
        x = F.relu(self.bn8(self.conv8(x)))
        x = F.relu(self.bn9(self.conv9(x)))
        x = F.relu(self.bn10(self.conv10(x)))
        x = F.relu(self.fcl1(x.view(1, 128 * 64)))
        x = self.fcl2(x)
        return x.tanh()

class DQNAgent:
    def __init__(self):
        self.lr = 0.001
        self.gamma = 0.99
        self.exploration_proba = 0.9
        self.exploration_proba_decay = 0.05
        self.batch_size = 256
        
        self.memory_buffer = list()
        self.max_memory_buffer = 131072

        self.model = Net().to(device)

        self.loss_fn = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr = self.lr)

        self.model.train()

    def compute_action(self, bd):
        legalMoves = list(bd.legal_moves)
        if np.random.uniform(0, 1) < self.exploration_proba:
            return np.random.choice(legalMoves)
        q_values = self.model.predict(get_state(bd))[0]

        action = legalMoves[0]
        max = -1.0e6
        for i in legalMoves:
            if max < q_values[i]:
                max = q_values[i]
                action = i

        return action
    
    def update_exploration_probability(self):
        self.exploration_proba = self.exploration_proba * np.exp(-self.exploration_proba_decay)

    def store_episode(self,current_state, action, reward, next_state, done):
        self.memory_buffer.append({
            "current_state":current_state,
            "action":action,
            "reward":reward,
            "next_state":next_state,
            "done" :done
        })
        if self.max_memory_buffer < len(self.memory_buffer):
            self.memory_buffer.pop(0)

    def train(self):
        np.random.shuffle(self.memory_buffer)
        batch_sample = self.memory_buffer[0:self.batch_size]

        for experience in batch_sample:
            q_current_state = self.model.predict(get_state(experience["current_state"]))[0]
            q_target = q_current_state.clone()
            
            _q_target = experience["reward"]
            if not experience["done"]:
                _q_target = _q_target + self.gamma * torch.max(self.model.predict(get_state(experience["next_state"]))[0])
            q_target[experience["action"]] = _q_target

            loss = self.loss_fn(q_current_state, q_target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()