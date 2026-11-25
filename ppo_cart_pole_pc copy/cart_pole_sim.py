# ==================== cartpole_sim.py ====================
import numpy as np
from math import sin, cos
import torch
from tkinter import *
import time


class CartPole:
    def __init__(self):
        self.mc = 4.0
        self.mp = 0.15
        self.l = 0.7
        self.g = 9.81
        self.dt = 0.001
        self.control_steps = 20
        self.tau = self.dt * self.control_steps
        self.max_force = 100.0
        self.max_angle = np.deg2rad(180)
        self.stable_angle = np.deg2rad(25)
        self.max_time = 8
        self.max_x = 3.4
        self.max_fi_dot = 15
        self.reward_type = 'D'
        self.fi_desired = np.pi          # upright in your convention
        self.current_force = 0.0
        self.reset()

        # GUI
        self.root = None
        self.C = None
        self.cart = None
        self.pole = None
        self.stop_training = False

    def err(self):
        return np.arctan2(np.sin(self.fi - self.fi_desired), np.cos(self.fi - self.fi_desired))

    def get_state(self):
        x_norm = self.x / self.max_x
        v_norm = self.x_dot / 2.0
        e = self.err()
        sin_e = np.sin(e)
        cos_e = np.cos(e)
        w = self.fi_dot / 5.0
        return np.array([x_norm, v_norm, sin_e, cos_e, w], dtype=np.float32)

    def step_sim(self, force_x):
        denom = self.mc + self.mp * sin(self.fi)**2
        x_dot_dot = (force_x + self.mp * sin(self.fi) * (self.l * self.fi_dot**2 + self.g * cos(self.fi))) / denom
        self.x_dot += x_dot_dot * self.dt
        self.x += self.x_dot * self.dt

        fi_dot_dot = (-force_x * cos(self.fi) - self.mp * self.l * self.fi_dot**2 * cos(self.fi) * sin(self.fi)
                    - (self.mc + self.mp) * self.g * sin(self.fi)) / (self.l * denom)
        self.fi_dot += fi_dot_dot * self.dt
        self.fi += self.fi_dot * self.dt

    def calculate_reward(self, done=False):
        e = self.err()                          # angle error from upright
        x_err = self.x                          # cart position error
        match self.reward_type:
            case 'A':
                if done:
                    return -10.0
                elif abs(e) <= self.stable_angle and abs(x_err) < 0.5:
                    return (
                        2.0
                        - 1.0 * e**2
                        - 0.2 * self.fi_dot**2
                        - 1.5 * x_err**2             #was 0.5  # cart position penalty
                        - 0.1 * self.x_dot**2          # cart velocity penalty
                        - 0.01 * self.current_force**2
                    )
                else:
                    return -0.04

            case 'B':
               return (
                    np.cos(e)
                    - 0.1 * self.fi_dot**2 
                    - 1.0 * self.x**2          # was 0.1
                    - 0.5 * self.x_dot**2      # was 0.05
                    - 0.001 * self.current_force**2
                )

            case 'C':
                if done:
                    return -10.0
                elif abs(e) <= self.stable_angle and abs(x_err) < 0.7:
                    return (
                        10.0
                        - 1.0 * e**2
                        - 0.2 * self.fi_dot**2
                        - 1.5 * x_err**2
                        - 0.1 * self.x_dot**2
                        - 0.01 * self.current_force**2
                    )
                else:
                    return -0.03 -  0.005 * e**2

            case 'D':
                if done:
                    return -10.0

                # High reward near upright (±3°)
                if abs(e) <= np.deg2rad(3) and abs(x_err) < 0.5:
                    return (
                        5.0
                        - 2.0 * e**2
                        - 0.3 * self.fi_dot**2
                        - 2.0 * x_err**2
                        - 0.2 * self.x_dot**2
                        - 0.01 * self.current_force**2
                    )

                # Lower reward in the wider region (±25°)
                elif abs(e) <= np.deg2rad(25) and abs(x_err) < 0.8:
                    return (
                        1.5
                        - 0.5 * e**2
                        - 0.1 * self.fi_dot**2
                        - 0.8 * x_err**2
                        - 0.05 * self.x_dot**2
                        - 0.005 * self.current_force**2
                    )

                # Continuous guidance outside
                else:
                    return -0.03 - 0.01 * e**2


    def step(self, force):
        force = np.clip(force, -self.max_force, self.max_force)
        self.current_force = float(force)
        total_reward = 0.0
        hit_wall = False

        for _ in range(self.control_steps):
            self.step_sim(self.current_force)
            self.time += self.dt
            total_reward += self.calculate_reward()

            if abs(self.x) >= self.max_x:          # >= so it catches the clamp
                hit_wall = True
                self.x = np.clip(self.x, -self.max_x, self.max_x)
                self.x_dot = 0.0
                break
            if abs(self.err()) > self.max_angle or self.time > self.max_time:
                break

        done = abs(self.err()) > self.max_angle or hit_wall or self.time > self.max_time or self.fi_dot > self.max_fi_dot
        if hit_wall:
            total_reward += -10.0
        elif done:
            total_reward += self.calculate_reward(done=True)
        return self.get_state(), total_reward, done, {}

    def reset(self):
        # Ultimamento
        self.x = np.random.uniform(-0.5, 0.5)
        self.x_dot = np.random.uniform(-1.0, 1.0)
        self.fi = np.deg2rad(np.random.uniform(-180, 180))
        self.fi_dot = np.random.uniform(-2.0, 2.0)
        self.time = 0.0
        self.current_force = 0.0
        return self.get_state()
    
        # STAGE 4
        """
         self.x = np.random.uniform(-1.0, 1.0)
        self.x_dot = np.random.uniform(-1.0, 1.0)
        if np.random.rand() < 0.4:
            # közel függőleges (hogy ne felejtse el a balance-t)
            self.fi = np.pi + np.deg2rad(np.random.uniform(-40, 40))
        else:
            # teljes random
            self.fi = np.random.uniform(-np.pi, np.pi)
        self.fi_dot = np.random.uniform(-2.0, 2.0)
        self.time = 0.0
        self.current_force = 0.0
        return self.get_state()
        """
       

        # STAGE 3 (régi)
        """
        self.x = np.random.uniform(-0.3, 0.3)
        self.x_dot = np.random.uniform(-0.5, 0.5)
        self.fi = np.pi + np.deg2rad(np.random.uniform(-15, 15))
        self.fi_dot = np.random.uniform(-1.0, 1.0)
        self.time = 0.0
        self.current_force = 0.0
        return self.get_state()
        """
    

    def reset_test(self):
        # Ultimate
        self.x = np.random.uniform(-0.5, 0.5)
        self.x_dot = np.random.uniform(-5.0, 5.0)
        self.fi =np.pi- np.deg2rad(np.random.uniform(-90, 90))
        self.fi_dot = np.random.uniform(-2.0, 2.0)
        self.time = 0.0
        self.current_force = 0.0
        return self.get_state()
    
       # Stage 4
        """
        self.x = 0.0
        self.x_dot = 0.0
        self.fi = np.pi - np.deg2rad(40)# np.deg2rad(np.random.uniform(-80,80))
        self.fi_dot = 0.0
        self.time = 0.0
        self.current_force = 0.0
        return self.get_state()
        """
        
       
        
        
       

    def gui(self):
        self.root = Tk()
        self.root.title("Cart-Pole")
        self.root.geometry("1000x500")
        self.C = Canvas(self.root, bg="light gray", height=400, width=700)
        self.C.pack()
        self.cart = self.C.create_rectangle(0, 0, 20, 20, fill="blue")
        self.pole = self.C.create_line(0, 0, 0, 0, fill="green", width=3)
        self.x0 = 200
        self.y0 = 125
        self.px = 100
        self.r = 100 * self.l

    def update_gui(self):
        if self.C is None:
            return
        origo_x = self.x0 + self.x * self.px
        self.C.coords(self.cart, origo_x - 10, self.y0 - 10, origo_x + 10, self.y0 + 10)
        self.C.coords(self.pole,
                      origo_x, self.y0,
                      origo_x + self.r * sin(self.fi),
                      self.y0 + self.r * cos(self.fi))
        self.root.update()


def test_cartpole_gui(actor, episodes=5):
    actor.eval()
    env = CartPole()
    env.gui()
    print("\n" + "="*50)
    print("TESTING CART-POLE")
    print("="*50)

    for ep in range(episodes):
        state = env.reset_test()
        total_reward = 0.0
        last_log = 0.0
        print(f"\n--- Episode {ep+1} ---")
        print(f"{'Time':>6} | {'x':>6} | {'fi (deg)':>8} | {'fi_dot':>8} | {'Force':>8} | {'Reward':>8}")

        while True:
            state_t = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action, _ = actor.get_action(state_t, deterministic=True)
            force = action.item()

            state, reward, done, _ = env.step(force)
            total_reward += reward
            env.update_gui()
            time.sleep(0.01)

            if env.time - last_log >= 0.1 or done:
                fi_deg = np.rad2deg(env.fi)
                print(f"{env.time:6.2f} | {env.x:6.2f} | {fi_deg:8.2f} | {env.fi_dot:8.3f} | {force:8.2f} | {reward:8.2f}")
                last_log = env.time

            if done or env.stop_training:
                print(f"Ended | Time: {env.time:.2f}s | Reward: {total_reward:.1f}")
                break
        if env.stop_training:
            break
    actor.train()