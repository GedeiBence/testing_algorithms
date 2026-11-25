# ==================== scara_sim.py ====================
from tkinter import *
import numpy as np
import time
from math import sin, cos
import torch

class SCARA:
    def __init__(self):
        self.m1 = 0.4
        self.m2 = 0.3
        self.l1 = 4.0
        self.l2 = 3.0
        self.g = 9.81          # not used (horizontal)
        self.dt = 0.001
        self.control_steps = 20
        self.tau = self.dt * self.control_steps
        self.max_torque = 100.0
        self.stable_angle = np.deg2rad(3)
        self.j1_max_angle = np.deg2rad(180)
        self.j2_max_angle = np.deg2rad(180)
        self.max_time = 5.0
        self.reward_type = 'A'
        self.reset()

        # GUI
        self.root = None
        self.C = None
        self.arm_1 = None
        self.arm_2 = None
        self.origo_x = 350
        self.origo_y = 350
        self.r1 = 25 * self.l1
        self.r2 = 25 * self.l2
        self.stop_training = False

    def err(self):
        """Error from target (both arms pointing "up" = 0)"""
        return np.array([self.alpha, self.beta])

    def get_state(self):
        # Example: Cartesian error + velocities + joint angles
        x = self.l1 * sin(self.alpha) + self.l2 * sin(self.alpha + self.beta)
        y = self.l1 * cos(self.alpha) + self.l2 * cos(self.alpha + self.beta)
        # You need a target position (set it in reset)
        x_err = x - self.target_x
        y_err = y - self.target_y
        return np.array([x_err, y_err, self.alpha_dot, self.beta_dot, self.alpha, self.beta], dtype=np.float32)

    def step_sim(self, t1, t2):
        """One integration step of the dynamics (RK2 or simple Euler)"""
        # Mass matrix
        c = cos(self.beta)
        M11 = (1/3)*self.m1*self.l1**2 + self.m2*(self.l1**2 + self.l2**2/3 + self.l1*self.l2*c)
        M12 = self.m2*(self.l2**2/3 + 0.5*self.l1*self.l2*c)
        M22 = (1/3)*self.m2*self.l2**2

        # Coriolis / centrifugal
        s = sin(self.beta)
        h1 = -self.m2*self.l1*self.l2*s*(self.alpha_dot*self.beta_dot + 0.5*self.beta_dot**2)
        h2 = 0.5*self.m2*self.l1*self.l2*s*self.alpha_dot**2

        # Solve M * qdd = tau - h
        det = M11*M22 - M12*M12
        if abs(det) < 1e-8:
            det = 1e-8
        self.alpha_dot_dot = (M22*(t1 - h1) - M12*(t2 - h2)) / det
        self.beta_dot_dot  = (M11*(t2 - h2) - M12*(t1 - h1)) / det

        # Integrate
        self.alpha_dot += self.alpha_dot_dot * self.dt
        self.beta_dot  += self.beta_dot_dot  * self.dt
        self.alpha     += self.alpha_dot * self.dt
        self.beta      += self.beta_dot  * self.dt

    def step(self, action):
        """
        High-level action = Cartesian velocity [vx, vy]
        Low-level PD (Kp=90, Kd=78) tracks the resulting joint velocities
        """
        vx, vy = action[0], action[1]

        # Simple Jacobian (approximate)
        # α, β → Cartesian
        c1 = cos(self.alpha)
        s1 = sin(self.alpha)
        c12 = cos(self.alpha + self.beta)
        s12 = sin(self.alpha + self.beta)

        J = np.array([
            [self.l1 * c1 + self.l2 * c12, self.l2 * c12],
            [-self.l1 * s1 - self.l2 * s12, -self.l2 * s12]
        ])

        # Desired joint velocities from Cartesian velocity
        try:
            q_dot_des = np.linalg.solve(J, np.array([vx, vy]))
        except:
            q_dot_des = np.zeros(2)

        # Low-level PD on joint velocities
        kp = 90.0
        kd = 78.0
        t1 = kp * (q_dot_des[0] - self.alpha_dot) - kd * self.alpha_dot
        t2 = kp * (q_dot_des[1] - self.beta_dot) - kd * self.beta_dot

        # Clip and apply (rest of the method stays the same as before)
        t1 = np.clip(t1, -self.max_torque, self.max_torque)
        t2 = np.clip(t2, -self.max_torque, self.max_torque)
        self.current_torque_j1 = float(t1)
        self.current_torque_j2 = float(t2)

        total_reward = 0.0
        for _ in range(self.control_steps):
            self.step_sim(t1, t2)
            self.time += self.dt
            total_reward += self.calculate_reward(done=False)

            if abs(self.alpha) > self.j1_max_angle or abs(self.beta) > self.j2_max_angle:
                break

        done = (abs(self.alpha) > self.j1_max_angle or
                abs(self.beta) > self.j2_max_angle or
                self.time > self.max_time)

        return self.get_state(), total_reward, done, {}
    
    def calculate_reward(self, done=False):
        # Current TCP position
        x = self.l1 * sin(self.alpha) + self.l2 * sin(self.alpha + self.beta)
        y = self.l1 * cos(self.alpha) + self.l2 * cos(self.alpha + self.beta)

        # Distance to the current target
        dist = np.sqrt((x - self.target_x)**2 + (y - self.target_y)**2)

        reward = 0.0
        match self.reward_type:
            case 'A':
                # skálázott reward – ne legyen túl negatív
                reward = -0.5 * dist - 0.0001 * (self.current_torque_j1**2 + self.current_torque_j2**2)
                if dist < 0.15:          # közel van a célhoz
                    reward += 5.0
            case 'B':
                # Stronger shaping
                reward = -5.0 * dist - 0.01 * (self.alpha_dot**2 + self.beta_dot**2)
                if dist < 0.03:
                    reward += 20.0
                if done:
                    reward -= 5.0

            case 'C':
                # Dense reward for getting closer
                reward = -dist**2
                if dist < 0.05:
                    reward += 15.0

        return reward

    def _instant_reward(self):
        # Simple reward: penalize angles and velocities
        return -(self.alpha**2 + self.beta**2 + 0.1*self.alpha_dot**2 + 0.1*self.beta_dot**2)

    def reset(self):
        self.time = 0.0
        self.alpha = 0.0
        self.alpha_dot = 0.0
        self.alpha_dot_dot = 0.0
        self.beta = 0.0
        self.beta_dot = 0.0
        self.beta_dot_dot = 0.0
        self.current_torque_j1 = 0.0
        self.current_torque_j2 = 0.0
        r = np.random.uniform(2.0, 6.5)   
        theta = np.random.uniform(0, 2*np.pi)
        self.target_x = r * np.cos(theta)
        self.target_y = r * np.sin(theta)
        
        return self.get_state()

    def gui(self):
        self.root = Tk()
        self.root.title("SCARA")
        self.root.geometry("700x700")
        self.C = Canvas(self.root, bg="light gray", height=600, width=600)
        self.C.pack()
        self.arm_1 = self.C.create_line(0, 0, 0, 0, fill="orange", width=6)
        self.arm_2 = self.C.create_line(0, 0, 0, 0, fill="red", width=6)
        # axes
        self.C.create_line(50, self.origo_y, 650, self.origo_y, fill="black", width=1)
        self.C.create_line(self.origo_x, 50, self.origo_x, 650, fill="black", width=1)

    def update_gui(self):
        if self.C is None:
            return
        # joint 2 position
        x1 = self.origo_x + self.r1 * sin(self.alpha)
        y1 = self.origo_y - self.r1 * cos(self.alpha)
        # end effector
        x2 = x1 + self.r2 * sin(self.alpha + self.beta)
        y2 = y1 - self.r2 * cos(self.alpha + self.beta)

        self.C.coords(self.arm_1, self.origo_x, self.origo_y, x1, y1)
        self.C.coords(self.arm_2, x1, y1, x2, y2)
        self.root.update()