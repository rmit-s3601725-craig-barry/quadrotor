import simulation.quadrotor as quad
import simulation.config as cfg
import models.one_step_velocity as model
import math
from math import pi
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as style
import torch
import torch.optim as optim
import torch.nn.functional as F
import utils

style.use("seaborn-deep")
GPU = True
def main():

    epochs = 100000
    state_dim = 12
    action_dim = 4
    hidden_dim = 64
    dyn = model.Transition(state_dim, action_dim, hidden_dim, GPU)
    if GPU:
        dyn = dyn.cuda()

    params = cfg.params
    iris = quad.Quadrotor(params)
    hover_rpm = iris.hov_rpm
    max_rpm = iris.max_rpm
    trim = np.array([hover_rpm, hover_rpm, hover_rpm, hover_rpm])
    dt = iris.dt

    print("HOVER RPM: ", trim)
    print("Terminal Velocity: ", iris.terminal_velocity)
    print("Terminal Rotation: ", iris.terminal_rotation)
    input("Press to continue")

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.set_title("Linear Velocity Loss")
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("Loss")
    fig1.subplots_adjust(hspace=0.3)
    fig1.subplots_adjust(wspace=0.3)
    fig1.show()
    
    av = []
    data = []
    iterations = []
    counter = 0

    running = True
    trajectory_len = 2
    optimizer = optim.LBFGS_C(dyn.parameters(),lr=1e-4)
    criterion = torch.nn.MSELoss(size_average=True)
    while running:
        
        # generate random state
        xyz_rand = np.random.uniform(low=-15, high=15, size=(3,1))
        zeta_rand = np.random.uniform(low=-2*pi,high=2*pi,size=(3,1))
        uvw_rand = np.random.uniform(low=-iris.terminal_velocity, high=iris.terminal_velocity, size=(3,1))
        pqr_rand = np.random.uniform(low=-iris.terminal_rotation, high=iris.terminal_rotation, size=(3,1))

        # set random state
        iris.set_state(xyz_rand, zeta_rand, uvw_rand, pqr_rand)
        xyz_ = xyz_rand.reshape((1,-1))
        zeta_ = zeta_rand.reshape((1,-1))
        uvw_ = uvw_rand.reshape((1,-1))
        pqr_ = pqr_rand.reshape((1,-1))
        
        xyz_nn = torch.from_numpy(xyz_).float()
        zeta_nn = torch.from_numpy(zeta_).float()
        uvw_nn = torch.from_numpy(uvw_).float()
        pqr_nn = torch.from_numpy(pqr_).float()
        if GPU:
                xyz_nn = xyz_nn.cuda()
                zeta_nn = zeta_nn.cuda()
                uvw_nn = uvw_nn.cuda()
                pqr_nn = pqr_nn.cuda()
        
        state_action_list = []
        next_state_list = []
        for i in range(trajectory_len):

            # generate random action, assume hover at 50% of max_rpm
            action = np.random.uniform(low=0, high=max_rpm, size=(4,))
            action_nn = torch.from_numpy(action.reshape(1,-1)).float()
            if GPU:
                action_nn = action_nn.cuda()
            
            xyz, zeta, uvw, pqr = iris.step(action)
            
            state = torch.cat([zeta_nn.sin(), zeta_nn.cos(), uvw_nn, pqr_nn],dim=1)
            state_action = torch.cat([state, action_nn],dim=1)
            state_action_list.append(state_action)
            xyz_nn, zeta_nn, uvw_nn, pqr_nn = dyn.transition(xyz_nn, state_action, dt)
            xyz_act, zeta_act, uvw_act, pqr_act = utils.numpy_to_pytorch(xyz, zeta, uvw, pqr)
            next_state = torch.cat([xyz_act, zeta_act, uvw_act, pqr_act],dim=1)
            next_state_list.append(next_state)
        
        xs = torch.stack(state_action_list)
        ys = torch.stack(next_state_list)
        dyn.update(optimizer, criterion, xs, ys)

        if len(av)>10:
            del av[0]
            av.append(loss.item())
        else:
            av.append(loss.item())

        average = float(sum(av))/float(len(av))
        
        if counter%100 == 0:
            data.append(average)
            iterations.append(counter/100.)
            ax1.clear()
            ax1.plot(iterations,data)
            ax1.set_title("Linear Velocity Loss")
            ax1.set_xlabel(r"Iterations $\times 10^{2}$")
            ax1.set_ylabel("Loss")
            fig1.canvas.draw()
        counter += 1

        print(loss.item())

        if counter > epochs:
            running = False
            print("Saving figures")
            fig1.savefig('multi_step_loss.pdf', bbox_inches='tight')
            print("Saving model")
            torch.save(dyn, "/home/seanny/quadrotor/models/multi_step.pth.tar")

        

if __name__ == "__main__":
    main()