import environments.envs as envs 
import policies.cem as cem
import argparse
import torch
import torch.nn.functional as F
import math
import utils
import numpy as np
from collections import deque

class Trainer:
    def __init__(self, env, params):
        self.env = envs.make(env)

        self.iterations = params["iterations"]
        self.gamma = params["gamma"]
        self.seed = params["seed"]
        self.pop_size = params["pop_size"]
        self.elite_frac = params["elite_frac"]
        self.sigma = params["sigma"]
        self.render = params["render"]
        self.log_interval = params["log_interval"]
        self.save = params["save"]

        action_bound = self.env.action_bound[1]
        state_dim = env.observation_space
        action_dim = env.action_space
        hidden_dim = params["hidden_dim"]
        cuda = params["cuda"]

        self.agent = cem.CEM(state_dim, hidden_dim, action_dim, action_bound, GPU=cuda)
    
        if cuda:
            self.Tensor = torch.cuda.FloatTensor
            self.agent = self.agent.cuda()
        else:
            self.Tensor = torch.Tensor

    def train(self):
        def evaluate(weights):
            self.agent.set_weights(weights)
            episode_return = 0.0
            state = self.env.reset()
            for t in range(self.env.H):
                state = torch.from_numpy(state).float().to(device)
                action = self.agent(state)
                state, reward, done, _ = self.env.step(action)
                episode_return += reward*math.pow(self.gamma, t)
                if done:
                    break
            return episode_return
        
        n_elite=int(self.pop_size*self.elite_frac)

        scores_deque = deque(maxlen=100)
        best_weight = self.sigma*np.random.randn(self.agent.get_weights_dim())

        for i_iteration in range(self.iterations):
            weights_pop = [best_weight+(self.sigma*np.random.randn(self.agent.get_weights_dim())) for i in range(self.pop_size)]
            rewards = np.array([evaluate(weights) for weights in weights_pop])
            elite_idxs = rewards.argsort()[-n_elite:]
            elite_weights = [weights_pop[i] for i in elite_idxs]
            best_weight = np.array(elite_weights).mean(axis=0)
            reward = evaluate(best_weight)
            scores_deque.append(reward)
        
            if i_iteration % self.log_interval == 0:
                print('Episode {}\tAverage Score: {:.2f}'.format(i_iteration, np.mean(scores_deque)))
            if np.mean(scores_deque)>=90.0:
                print('\nEnvironment solved in {:d} iterations!\tAverage Score: {:.2f}'.format(i_iteration-100, np.mean(scores_deque)))
                break