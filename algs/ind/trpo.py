import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import scipy.optimize
from math import pi, log
import numpy as np
from collections import namedtuple
import gym
import utils
import numpy as np
import csv
import os

"""
    Port of Trust Region Policy Optimization by John Schulman.
"""

class Actor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Actor, self).__init__()
        self.affine1 = nn.Linear(input_dim, hidden_dim)
        self.action_mean = nn.Linear(hidden_dim, output_dim)
        self.action_log_std = nn.Parameter(torch.zeros(1, output_dim))

        self.action_mean.weight.data.mul_(0.1)
        self.action_mean.bias.data.mul_(0.0)

        self.saved_actions = []
        self.rewards = []
        self.final_value = 0

    def forward(self, x):
        x = F.tanh(self.affine1(x.float()))
        action_mean = self.action_mean(x)
        action_log_std = self.action_log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)
        return action_mean, action_log_std, action_std

class Critic(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Critic, self).__init__()
        self.affine1 = nn.Linear(input_dim, hidden_dim)
        self.value_head = nn.Linear(hidden_dim, output_dim)
        self.value_head.weight.data.mul_(0.1)
        self.value_head.bias.data.mul_(0.0)

    def forward(self, x):
        x = F.tanh(self.affine1(x))
        state_values = self.value_head(x)
        return state_values

class TRPO(nn.Module):
    def __init__(self, actor, critic, params, GPU=False):
        super(TRPO, self).__init__()
        self.actor = actor
        self.critic = critic
        
        self.gamma = params["gamma"]
        self.tau = params["tau"]
        self.l2_reg = params["l2_reg"]
        self.max_kl = params["max_kl"]
        self.damping = params["damping"]

        self.GPU = GPU

        if GPU:
            self.Tensor = torch.cuda.FloatTensor
            self.actor = self.actor.cuda()
            self.critic = self.critic.cuda()
        else:
            self.Tensor = torch.Tensor

    def select_action(self, state):
        state = torch.from_numpy(state).unsqueeze(0)
        action_mean, _, action_std = self.actor(Variable(state))
        action = torch.normal(action_mean, action_std)
        return F.tanh(action)

    def update(self, batch):
        
        # Original code uses the same LBFGS to optimize the value loss
        def get_value_loss(flat_params):
            set_flat_params_to(self.critic, torch.Tensor(flat_params))
            for param in self.critic.parameters():
                if param.grad is not None:
                    param.grad.data.fill_(0)
            values_ = self.critic(Variable(states))
            value_loss = (values_-targets).pow(2).mean()
            
            # weight decay
            for param in self.critic.parameters():
                value_loss += param.pow(2).sum()*self.l2_reg
            value_loss.backward()
            return (value_loss.data.double().numpy(), get_flat_grad_from(self.critic).data.double().numpy())

        def get_loss(volatile=False):
            action_means, action_log_stds, action_stds = self.actor(Variable(states, volatile=volatile))
            log_prob = normal_log_density(Variable(actions), action_means, action_log_stds, action_stds)
            action_loss = -Variable(advantages)*torch.exp(log_prob-Variable(fixed_log_prob))
            return action_loss.mean()

        def get_kl():
            mean1, log_std1, std1 = self.actor(Variable(states))
            mean0 = Variable(mean1.data)
            log_std0 = Variable(log_std1.data)
            std0 = Variable(std1.data)
            kl = log_std1-log_std0+(std0.pow(2)+(mean0-mean1).pow(2))/(2.*std1.pow(2))-0.5
            return kl.sum(1, keepdim=True)

        rewards = torch.Tensor(batch.reward)
        masks = torch.Tensor(batch.mask)
        actions = torch.Tensor(np.concatenate(batch.action, 0))
        states = torch.Tensor(batch.state)
        values = self.critic(Variable(states))
        
        returns = torch.Tensor(actions.size(0),1)
        deltas = torch.Tensor(actions.size(0),1)
        advantages = torch.Tensor(actions.size(0),1)
        prev_return = 0
        prev_value = 0
        prev_advantage = 0

        # calculate advantages
        for i in reversed(range(rewards.size(0))):
            returns[i] = rewards[i]+self.gamma*prev_return*masks[i]
            deltas[i] = rewards[i]+self.gamma*prev_value*masks[i]-values.data[i]
            advantages[i] = deltas[i]+self.gamma*self.tau*prev_advantage*masks[i]
            prev_return = returns[i, 0]
            prev_value = values.data[i, 0]
            prev_advantage = advantages[i, 0]
        
        targets = Variable(returns)
        flat_params, _, opt_info = scipy.optimize.fmin_l_bfgs_b(get_value_loss, get_flat_params_from(self.critic).double().numpy(), maxiter=25)
        set_flat_params_to(self.critic, torch.Tensor(flat_params))
        advantages = (advantages-advantages.mean())/advantages.std()
        action_means, action_log_stds, action_stds = self.actor(Variable(states))
        fixed_log_prob = normal_log_density(Variable(actions), action_means, action_log_stds, action_stds).data.clone()
        trpo_step(self.actor, get_loss, get_kl, self.max_kl, self.damping)

Transition = namedtuple('Transition', ('state', 'action', 'mask', 'next_state', 'reward'))
class Memory(object):
    def __init__(self):
        self.memory = []

    def push(self, *args):
        """Saves a transition."""
        self.memory.append(Transition(*args))

    def sample(self):
        return Transition(*zip(*self.memory))

    def __len__(self):
        return len(self.memory)

def conjugate_gradients(Avp, b, nsteps, residual_tol=1e-10):
    x = torch.zeros(b.size())
    r = b.clone()
    p = b.clone()
    rdotr = torch.dot(r, r)
    for i in range(nsteps):
        _Avp = Avp(p)
        alpha = rdotr/torch.dot(p, _Avp)
        x += alpha*p
        r -= alpha*_Avp
        new_rdotr = torch.dot(r, r)
        betta = new_rdotr/rdotr
        p = r + betta*p
        rdotr = new_rdotr
        if rdotr < residual_tol:
            break
    return x

def linesearch(model, f, x, fullstep, expected_improve_rate, max_backtracks=10, accept_ratio=.1):
    fval = f(True).data
    print("fval before", fval[0])
    for (_n_backtracks, stepfrac) in enumerate(.5**np.arange(max_backtracks)):
        xnew = x+stepfrac*fullstep
        set_flat_params_to(model, xnew)
        newfval = f(True).data
        actual_improve = fval-newfval
        expected_improve = expected_improve_rate*stepfrac
        ratio = actual_improve / expected_improve
        print("a/e/r", actual_improve[0], expected_improve[0], ratio[0])
        if ratio[0] > accept_ratio and actual_improve[0] > 0:
            print("fval after", newfval[0])
            return True, xnew
    return False, x

def trpo_step(model, get_loss, get_kl, max_kl, damping):
    def Fvp(v):
        kl = get_kl()
        kl = kl.mean()
        grads = torch.autograd.grad(kl, model.parameters(), create_graph=True)
        flat_grad_kl = torch.cat([grad.view(-1) for grad in grads])
        kl_v = (flat_grad_kl*Variable(v)).sum()
        grads = torch.autograd.grad(kl_v, model.parameters())
        flat_grad_grad_kl = torch.cat([grad.contiguous().view(-1) for grad in grads]).data
        return flat_grad_grad_kl+v*damping

    loss = get_loss()
    grads = torch.autograd.grad(loss, model.parameters())
    loss_grad = torch.cat([grad.view(-1) for grad in grads]).data
    stepdir = conjugate_gradients(Fvp,-loss_grad,10)
    shs = 0.5 * (stepdir*Fvp(stepdir)).sum(0, keepdim=True)
    lm = torch.sqrt(shs/max_kl)
    fullstep = stepdir/lm[0]
    neggdotstepdir = (-loss_grad*stepdir).sum(0, keepdim=True)
    print(("lagrange multiplier:", lm[0], "grad_norm:", loss_grad.norm()))
    prev_params = get_flat_params_from(model)
    success, new_params = linesearch(model, get_loss, prev_params, fullstep, neggdotstepdir/lm[0])
    set_flat_params_to(model, new_params)
    return loss

def normal_entropy(std):
    var = std.pow(2)
    entropy = 0.5+0.5*torch.log(2*var*pi)
    return entropy.sum(1, keepdim=True)

def normal_log_density(x, mean, log_std, std):
    var = std.pow(2)
    log_density = -(x-mean).pow(2)/(2*var)-0.5*log(2*pi)-log_std
    return log_density.sum(1, keepdim=True)

def get_flat_params_from(model):
    params = []
    for param in model.parameters():
        params.append(param.data.view(-1))
    flat_params = torch.cat(params)
    return flat_params

def set_flat_params_to(model, flat_params):
    prev_ind = 0
    for param in model.parameters():
        flat_size = int(np.prod(list(param.size())))
        param.data.copy_(flat_params[prev_ind:prev_ind+flat_size].view(param.size()))
        prev_ind += flat_size

def get_flat_grad_from(net, grad_grad=False):
    grads = []
    for param in net.parameters():
        if grad_grad:
            grads.append(param.grad.grad.view(-1))
        else:
            grads.append(param.grad.view(-1))
    flat_grad = torch.cat(grads)
    return flat_grad


class Trainer:
    def __init__(self, env_name, params):
        self.env = gym.make(env_name)
        self.trim = np.array(self.env.trim)
        self.params = params
        self.action_bound = self.env.action_bound[1]
        self.iterations = params["iterations"]
        self.seed = params["seed"]
        self.batch_size = params["batch_size"]
        self.render = params["render"]
        self.log_interval = params["log_interval"]
        self.save = params["save"]
        cuda = params["cuda"]
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]
        hidden_dim = params["hidden_dim"]
        self.pi = Actor(state_dim, hidden_dim, action_dim)
        self.critic = Critic(state_dim, hidden_dim, 1)
        self.agent = TRPO(self.pi, self.critic, params["network_settings"], GPU=cuda)
        self.running_state = ZFilter((state_dim,), clip=5)
        self.running_reward = ZFilter((1,), demean=False, clip=10)
        if cuda:
            self.Tensor = torch.cuda.FloatTensor
        else:
            self.Tensor = torch.Tensor
        self.best = None

        # initialize experiment logging
        self.logging = params["logging"]
        if self.logging:
            self.directory = os.getcwd()
            filename = self.directory + "/data/trpo.csv"
            with open(filename, "w") as csvfile:
                self.writer = csv.writer(csvfile)
                self.writer.writerow(["episode", "reward"])
                self.train()
        else:
            self.train()
    
    def train(self):
        for i_episode in range(1, self.iterations+1):
            memory = Memory()
            num_steps = 1
            reward_batch = 0
            num_episodes = 0
            while num_steps < self.batch_size+1:
                state = self.env.reset()
                state = self.running_state(state[0])
                reward_sum = 0
                if i_episode % self.log_interval == 0 and self.render:
                    self.env.render()
                for t in range(1, self.env.H):
                    action = self.agent.select_action(state)
                    action = action.data[0].numpy()
                    a = action
                    next_state, reward, done, _ = self.env.step(self.trim+a*15)
                    reward_sum += reward
                    
                    if i_episode % self.log_interval == 0 and self.render:
                        self.env.render()

                    next_state = self.running_state(next_state[0])
                    mask = 1
                    if done:
                        mask = 0
                    memory.push(state, np.array([action]), mask, next_state, reward)
                        
                    if done:
                        break
                    state = next_state
                num_steps += (t-1)
                num_episodes += 1
                reward_batch += reward_sum
            reward_batch /= num_episodes

            if (self.best is None or reward_batch > self.best) and self.save:
                self.best = reward_batch
                utils.save(self.agent, self.directory + "/saved_policies/trpo.pth.tar")

            batch = memory.sample()
            self.agent.update(batch)
            if i_episode % self.log_interval == 0:
                print('Episode {}\tLast reward: {}\tAverage reward {:.2f}'.format(i_episode, reward_sum, reward_batch))
                if self.logging:
                    self.writer.writerow([i_episode, reward_batch]) 

class RunningStat(object):
    def __init__(self, shape):
        self._n = 0
        self._M = np.zeros(shape)
        self._S = np.zeros(shape)

    def push(self, x):
        x = np.asarray(x)
        assert x.shape == self._M.shape
        self._n += 1
        if self._n == 1:
            self._M[...] = x
        else:
            oldM = self._M.copy()
            self._M[...] = oldM + (x - oldM) / self._n
            self._S[...] = self._S + (x - oldM) * (x - self._M)

    @property
    def n(self):
        return self._n

    @property
    def mean(self):
        return self._M

    @property
    def var(self):
        return self._S / (self._n - 1) if self._n > 1 else np.square(self._M)

    @property
    def std(self):
        return np.sqrt(self.var)

    @property
    def shape(self):
        return self._M.shape

class ZFilter:
    """
    y = (x-mean)/std
    using running estimates of mean,std
    """

    def __init__(self, shape, demean=True, destd=True, clip=10.0):
        self.demean = demean
        self.destd = destd
        self.clip = clip
        self.rs = RunningStat(shape)

    def __call__(self, x, update=True):
        if update: self.rs.push(x)
        if self.demean:
            x = x - self.rs.mean
        if self.destd:
            x = x / (self.rs.std + 1e-8)
        if self.clip:
            x = np.clip(x, -self.clip, self.clip)
        return x

    def output_shape(self, input_space):
        return input_space.shape
