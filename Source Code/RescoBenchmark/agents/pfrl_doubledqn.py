from tokenize import Double
from typing import Any, Sequence
import numpy as np

import torch
import torch.nn as nn

import pfrl
from pfrl import explorers, replay_buffers
from pfrl.explorer import Explorer
from agents.setup_doubledqn import DoubleDQN
from pfrl.q_functions import DiscreteActionValueHead
from pfrl.utils.contexts import evaluating

from agents.agent import IndependentAgent, Agent


class IDoubleDQN(IndependentAgent):
    def __init__(self, config, obs_act, map_name, thread_number):
        super().__init__(config, obs_act, map_name, thread_number)
        for key in obs_act:
            obs_space = obs_act[key][0]
            act_space = obs_act[key][1]

            def conv2d_size_out(size, kernel_size=2, stride=1):
                return (size - (kernel_size - 1) - 1) // stride + 1

            h = conv2d_size_out(obs_space[1])
            w = conv2d_size_out(obs_space[2])

            model = nn.Sequential(
                nn.Conv2d(obs_space[0], 64, kernel_size=(2, 2)),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(h * w * 64, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, act_space),
                DiscreteActionValueHead()
            )

            self.agents[key] = DoubleDQNAgent(config, act_space, model)


class DoubleDQNAgent(Agent):
    def __init__(self, config, act_space, model, num_agents=0):
        super().__init__()

        self.model = model
        self.optimizer = torch.optim.Adam(self.model.parameters())
        replay_buffer = replay_buffers.ReplayBuffer(10000)

        explorer = explorers.LinearDecayEpsilonGreedy(
            config['EPS_START'],
            config['EPS_END'],
            config['steps'],
            lambda: np.random.randint(act_space),
        )

        self.agent = DoubleDQN(self.model, self.optimizer, replay_buffer, config['GAMMA'], explorer,
                            gpu=self.device.index,
                            minibatch_size=config['BATCH_SIZE'], replay_start_size=config['BATCH_SIZE'],
                            phi=lambda x: np.asarray(x, dtype=np.float32),
                            target_update_interval=config['TARGET_UPDATE'])
        

    def act(self, observation, valid_acts=None, reverse_valid=None):
        return self.agent.act(observation)

    def observe(self, observation, reward, done, info):
        self.agent.observe(observation, reward, done, False)

    def save(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path+'.pt')
    
    def load(self, path):
        print("Loading...")
        checkpoint = torch.load(path + '.pt')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(self.model.eval())
        print("Load Done! ")


