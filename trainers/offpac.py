import environments.envs as envs 
import policies.ind.offpac as offpac
import argparse
import torch
import torch.nn.functional as F
import utils
import csv
import os
from torch.autograd import Variable


class Trainer:
    def __init__(self, env_name, params):
        self.env = envs.make(env_name)
        self.params = params

        self.iterations = params["iterations"]
        self.seed = params["seed"]
        self.render = params["render"]
        self.log_interval = params["log_interval"]
        self.save = params["save"]

        self.action_bound = self.env.action_bound[1]
        state_dim = self.env.observation_space
        action_dim = self.env.action_space
        hidden_dim = params["hidden_dim"]
        cuda = params["cuda"]
        network_settings = params["network_settings"]
        pi = utils.Actor(state_dim, hidden_dim, action_dim)
        beta = utils.Actor(state_dim, hidden_dim, action_dim)
        critic = utils.Critic(state_dim, hidden_dim, 1)
        self.agent = offpac.OFFPAC(pi, beta, critic, network_settings, GPU=cuda)
        self.optim = torch.optim.Adam(self.agent.parameters())

        if cuda:
            self.Tensor = torch.cuda.FloatTensor
            self.agent = self.agent.cuda()
        else:
            self.Tensor = torch.Tensor
        
        if self.render:
            self.env.init_rendering()
        
        self.best = None

        # initialize experiment logging
        self.logging = params["logging"]
        if self.logging:
            self.directory = os.getcwd()
            filename = self.directory + "/data/offpac.csv"
            with open(filename, "w") as csvfile:
                self.writer = csv.writer(csvfile)
                self.writer.writerow(["episode", "reward"])
                self.train()
        else:
            self.train()

    def train(self):
        interval_avg = []
        avg = 0
        for ep in range(1, self.iterations+1):
            running_reward = 0
            s_ = []
            a_ = []
            ns_ = []
            r_ = []
            lp_ = []
            state = self.Tensor(self.env.reset())
            if ep % self.log_interval == 0 and self.render:
                self.env.render()
            for _ in range(self.env.H):          
                #action, log_prob = self.agent.select_action(state)
                action, log_prob = Variable(self.Tensor([[1., 1., 1., 1.]])), self.Tensor([1e-5])
                next_state, reward, done, _ = self.env.step(action[0].cpu().numpy()*self.action_bound)
                running_reward += reward
                
                if ep % self.log_interval == 0 and self.render:
                    self.env.render()

                next_state = self.Tensor(next_state)
                s_.append(state[0])
                a_.append(action[0])
                ns_.append(next_state[0])
                r_.append(reward)
                lp_.append(log_prob[0])
                if done:
                    break
                state = next_state
            
            if (self.best is None or running_reward > self.best) and self.save:
                self.best = running_reward
                utils.save(self.agent, self.directory + "/saved_policies/offpac.pth.tar")
            
            trajectory = {"states": s_,
                        "actions": a_,
                        "next_states": ns_,
                        "rewards": r_,
                        "log_probs": lp_}
            self.agent.update(self.optim, trajectory)
            interval_avg.append(running_reward)
            avg = (avg*(ep-1)+running_reward)/ep   
            if ep % self.log_interval == 0:
                interval = float(sum(interval_avg))/float(len(interval_avg))
                print('Episode {}\t Interval average: {:.2f}\t Average reward: {:.2f}'.format(ep, interval, avg))
                interval_avg = []
                if self.logging:
                    self.writer.writerow([ep, avg])
                self.test_policy()
    
    def test_policy(self):
        running_reward = 0
        state = self.Tensor(self.env.reset())
        if self.render:
            self.env.render()
        for _ in range(self.env.H):
            action, _ = self.agent.select_action(state)
            state, reward, done, _ = self.env.step(action[0].cpu().numpy()*self.action_bound)
            state = self.Tensor(state)
            running_reward += reward
            if self.render:
                self.env.render()
            if done:
                break
        print("Test reward: {:.2f}".format(running_reward))
