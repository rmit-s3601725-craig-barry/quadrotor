import numpy as np
import simulation.quadrotor as quad
import simulation.animation as ani
import simulation.config as cfg
import matplotlib.pyplot as pl
import scipy.optimize as opt
import copy
from math import pi

class PID_Controller:
    def __init__(self, aircraft, pids):
        self.aircraft = aircraft
        self.p_xyz = pids["linear"]["p"]
        self.i_xyz = pids["linear"]["i"]
        self.d_xyz = pids["linear"]["d"]
        self.p_zeta = pids["angular"]["p"]
        self.i_zeta = pids["angular"]["i"]
        self.d_zeta = pids["angular"]["d"]
        self.last_error_xyz = 0.
        self.i_error_xyz = 0.
        self.last_error_zeta = 0.
        self.i_error_zeta = 0.
        
        self.kt = aircraft.kt
        self.kq = aircraft.kq
        self.mass = aircraft.mass
        self.J = aircraft.J
        self.g = aircraft.g
        self.dt = aircraft.dt
        self.hov_rpm = aircraft.hov_rpm
        self.max_rpm = aircraft.max_rpm
        self.n_motors = aircraft.n_motors
        print(self.hov_rpm)
        
    def compute_lin(self, state, target):
        error = target-state
        p_error = error
        self.i_error_xyz += (error + self.last_error_xyz)*self.dt
        i_error = self.i_error_xyz
        d_error = (error-self.last_error_xyz)/self.dt
        p_output = self.p_xyz*p_error
        i_output = self.i_xyz*i_error
        d_output = self.d_xyz*d_error
        self.last_error_xyz = error
        print(p_output+i_output+d_output)
        return p_output+i_output+d_output
    
    def compute_ang(self, state, target):
        error = target-state
        p_error = error
        self.i_error_zeta += (error + self.last_error_zeta)*self.dt
        i_error = self.i_error_zeta
        d_error = (error-self.last_error_zeta)/self.dt
        p_output = self.p_zeta*p_error
        i_output = self.i_zeta*i_error
        d_output = self.d_zeta*d_error
        self.last_error_zeta = error
        return p_output+i_output+d_output
    
    def action(self, state, target):
        xyz = state["xyz"]
        zeta = state["zeta"]
        target_xyz = target["xyz"]
        target_zeta = target["zeta"]
        u_s = self.compute_lin(xyz, target_xyz)
        roll = u_s[0,0]
        pitch = -u_s[1,0]
        yaw = 0.0
        throttle = self.n_motors*self.kt*self.hov_rpm**2+u_s[2,0]
        print(throttle)
        print(u_s)
        return np.array([[throttle],
                        [roll],
                        [pitch],
                        [yaw]])

def terminal(xyz, zeta, uvw, pqr):
    mask1 = zeta > pi/2.
    mask2 = zeta < -pi/2.
    mask3 = np.abs(xyz) > 6.
    term = np.sum(mask1+mask2+mask3)
    if term > 0: 
        return True
    else: 
        return False

def main():
    pl.close("all")
    pl.ion()
    fig = pl.figure(0)
    axis3d = fig.add_subplot(111, projection='3d')
    
    params = cfg.params
    iris = quad.Quadrotor(params)
    hover_rpm = iris.hov_rpm
    trim = np.array([hover_rpm, hover_rpm, hover_rpm, hover_rpm])
    vis = ani.Visualization(iris, 10)
    rpm = trim

    goal_zeta = np.array([[0.],
                        [0.],
                        [0.]])
    goal_xyz = np.array([[0.],
                        [0.],
                        [3.]])
    xyz_init = np.array([[0.],
                        [0.],
                        [1.5]])
    uvw_init = np.array([[0.],
                        [0.],
                        [0.]])
    pqr_init = np.array([[0.],
                        [0.],
                        [0.]])

    eps = np.random.rand(3,1)/10.
    zeta_init = goal_zeta+eps
    iris.set_state(xyz_init, zeta_init, uvw_init, pqr_init)
    xyz, zeta, uvw, pqr = iris.get_state()
    
    pids = {"linear":{"p": np.array([[0.1],
                                    [0.1],
                                    [0.1]]), 
                    "i": np.array([[0.1],
                                    [0.1],
                                    [0.1]]), 
                    "d": np.array([[0.1],
                                    [0.1],
                                    [0.1]])},
            "angular":{"p": np.array([[0.1],
                                    [0.1],
                                    [0.1]]), 
                    "i": np.array([[0.1],
                                    [0.1],
                                    [0.1]]), 
                    "d": np.array([[0.1],
                                    [0.1],
                                    [0.1]])}}
    targets = {"xyz": goal_xyz,
                "zeta": goal_zeta}
    controller = PID_Controller(iris, pids)

    counter = 0
    frames = 5
    running = True
    done = False
    t = 0
    
    while running:
        states = {"xyz": xyz,
                "zeta": zeta}
        if counter%frames == 0:
            pl.figure(0)
            axis3d.cla()
            vis.draw3d(axis3d)
            axis3d.set_xlim(-3, 3)
            axis3d.set_ylim(-3, 3)
            axis3d.set_zlim(0, 6)
            axis3d.set_xlabel('West/East [m]')
            axis3d.set_ylabel('South/North [m]')
            axis3d.set_zlabel('Down/Up [m]')
            axis3d.set_title("Time %.3f s" %t)
            pl.pause(0.001)
            pl.draw()
        actions = controller.action(states, targets)
        xyz, zeta, uvw, pqr = iris.step(actions, rpm_commands=False)
        done = terminal(xyz, zeta, uvw, pqr)
        t += iris.dt
        #counter += 1
        if done:
            print("Resetting vehicle to: {}, {}, {}, {}".format(xyz_init, zeta_init, uvw_init, pqr_init))
            iris.set_state(xyz_init, zeta_init, uvw_init, pqr_init)
            xyz, zeta, uvw, pqr = iris.get_state()
            t = 0
            counter = 0
            done = False

if __name__ == "__main__":
    main()