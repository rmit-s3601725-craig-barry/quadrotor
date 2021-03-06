import environments.envs as envs 
import policies.ind.sw_scv as sw
import argparse
import torch
import torch.nn.functional as F
import utils
import csv
import os


class Trainer:
    def __init__(self, env_name, params):
        self.env = envs.make(env_name)
        self.params = params
        self.action_bound = self.env.action_bound[1]
        
        self.iterations = params["iterations"]
        self.mem_len = params["mem_len"]
        self.seed = params["seed"]
        self.render = params["render"]
        self.log_interval = params["log_interval"]
        self.warmup = params["warmup"]
        self.batch_size = params["batch_size"]
        self.save = params["save"]

        hidden_dim = params["hidden_dim"]
        state_dim = self.env.observation_space
        action_dim = self.env.action_space
        cuda = params["cuda"]
        network_settings = params["network_settings"]

        actor = utils.Actor(state_dim, hidden_dim, action_dim)
        target_actor = utils.Actor(state_dim, hidden_dim, action_dim)
        critic = utils.Critic(state_dim+action_dim, hidden_dim, 1)
        target_critic = utils.Critic(state_dim+action_dim, hidden_dim, 1)
        self.memory = utils.ReplayMemory(1000000)
        self.agent = sw.Sleepwalk(actor, 
                                critic,
                                target_actor, 
                                target_critic,
                                network_settings,
                                GPU=cuda)

        self.noise = utils.OUNoise(action_dim)
        self.noise.set_seed(self.seed)
        self.memory = utils.ReplayMemory(self.mem_len)

        self.pol_opt = torch.optim.Adam(actor.parameters())
        self.crit_opt = torch.optim.Adam(critic.parameters())

        if cuda:
            self.Tensor = torch.cuda.FloatTensor
        else:
            self.Tensor = torch.Tensor
        
        if self.render:
            self.env.init_rendering()
        
        self.best = None

        # initialize experiment logging
        self.logging = params["logging"]
        if self.logging:
            self.directory = os.getcwd()
            filename = self.directory + "/data/qprop.csv"
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
            state = self.Tensor(self.env.reset())
            states = []
            actions = []
            rewards = []
            log_probs = []
            if ep % self.log_interval == 0 and self.render:
                self.env.render()
            for t in range(self.env.H):     
                action, log_prob = self.agent.select_action(state)
                next_state, reward, done, _ = self.env.step(action.data[0].cpu().numpy()*self.action_bound)
                running_reward += reward

                if ep % self.log_interval == 0 and self.render:
                    self.env.render()     
                        
                next_state = self.Tensor(next_state)
                reward = self.Tensor([reward])
                self.memory.push(state[0], action[0], next_state[0], reward)
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                log_probs.append(log_prob)
                if ep >= self.warmup:
                    for i in range(3):               
                        transitions = self.memory.sample(self.batch_size)
                        batch = utils.Transition(*zip(*transitions))
                        self.agent.online_update(self.crit_opt, batch)
                if done:
                    break
                state = next_state
            
            if (self.best is None or running_reward > self.best) and self.save:
                self.best = running_reward
                utils.save(self.agent, self.directory + "/saved_policies/qprop.pth.tar")

            trajectory = {"states": states,
                        "actions": actions,
                        "rewards": rewards,
                        "log_probs": log_probs}
            self.agent.offline_update(self.pol_opt, trajectory)
            interval_avg.append(running_reward)
            avg = (avg*(ep-1)+running_reward)/ep   
            if ep % self.log_interval == 0:
                interval = float(sum(interval_avg))/float(len(interval_avg))
                print('Episode {}\t Interval average: {:.2f}\t Average reward: {:.2f}'.format(ep, interval, avg))
                interval_avg = []
                if self.logging:
                    self.writer.writerow([ep, avg]) 