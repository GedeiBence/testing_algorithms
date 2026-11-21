# ==================== scara_sim.py ====================
from tkinter import *
import numpy as np
import time, random 
from math import sin, cos
import torch
import matplotlib.pyplot as plt



class Scara():
    def __init__(self):
        self.m1 = 1
        self.m2 = 1
        self.l1 = 1
        self.l2 = 1
        self.g = 9.81
        self.dt = 0.001
        self.control_steps = 20
        self.tau = self.dt * self.control_steps
        self.max_torque = 100
        self.stable_angle= np.deg2rad(3)
        self.j1_max_angle = np.deg2rad(180)
        self.j2_max_angle = np.deg2rad(180)
        self.max_time = 5
        self.reward_type = 'A'
        self.reset()

    
