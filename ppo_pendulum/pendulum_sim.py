# ==================== pendulum_sim.py ====================
from tkinter import *
import numpy as np
import time
from math import sin, cos
import torch
import matplotlib.pyplot as plt



class Pendulum():
    def __init__(self):
        self.m = 1.0
        self.l = 1.0
        self.g = 9.81
        self.dt = 0.001                    # simulation timestep
        self.control_steps = 20            # ZOH: hold torque for 20 sim steps
        self.tau = self.dt * self.control_steps  # control period = 0.02 s
        self.max_torque = 100.0
        self.max_angle = np.deg2rad(180)
        self.stable_angle = np.deg2rad(3)
        self.max_time = 2.0
        self.fi_desired = np.pi
        self.reward_type = 'C'
        self.current_torque = 0.0
        self.reset()

        # GUI
        self.fi_vector = []
        self.root = None
        self.C = None
        self.line = None
        self.origo_x = 250
        self.origo_y = 200
        self.r = 100 * self.l
        self.stop_training = False

    def err(self):
        return np.arctan2(np.sin(self.fi - self.fi_desired), np.cos(self.fi - self.fi_desired))

    def get_state(self):
        e = self.err() / self.max_angle
        w = self.fi_dot / 5.0
        return np.array([e, w], dtype=np.float32)

    def f(self, fi, fi_dot, torque):
        fi_dot_dot = torque / (self.m * self.l ** 2) - (self.g / self.l) * sin(fi)
        return fi_dot, fi_dot_dot

    def step_rk4(self, torque):
        dt = self.dt
        k1_fi, k1_w = self.f(self.fi, self.fi_dot, torque)
        k2_fi, k2_w = self.f(self.fi + 0.5 * dt * k1_fi, self.fi_dot + 0.5 * dt * k1_w, torque)
        k3_fi, k3_w = self.f(self.fi + 0.5 * dt * k2_fi, self.fi_dot + 0.5 * dt * k2_w, torque)
        k4_fi, k4_w = self.f(self.fi + dt * k3_fi, self.fi_dot + dt * k3_w, torque)
        self.fi += (dt / 6) * (k1_fi + 2 * k2_fi + 2 * k3_fi + k4_fi)
        self.fi_dot += (dt / 6) * (k1_w + 2 * k2_w + 2 * k3_w + k4_w)

    def calculate_reward(self, done=False):
        e = self.err()
        w = self.fi_dot
        reward = 0
        in_stable = False
        match self.reward_type:
            case 'A':
                e = self.err()
                w = self.fi_dot
                # Strong living reward
                reward = 2.0 - e**2 - 0.05 * w**2
                if done:
                    reward -= 10.0
            case 'B':
                e = self.err()
                w = self.fi_dot

                # Define "stable" region
                in_stable = abs(e) < 0.1 and abs(w) < 0.5

                if done:
                    reward = -10.0
                elif in_stable:
                    reward = 1.0
                else:
                    reward = 0.0
            case 'C':
                err = self.err()
                if abs(err)> self.max_angle:
                  done = True
                elif abs(err) <= self.stable_angle:
                    in_stable = True
                if done:
                    reward = -10
                elif in_stable:
                    reward = 1 - 0.5 * abs(err)**2 - 0.8 * abs(self.fi_dot)**4 - 0.001 * abs(self.current_torque)**2
                else:
                    reward = 0.0

        return reward
                    

    def step(self, torque):
        """One control step with ZOH."""
        torque = np.clip(torque, -self.max_torque, self.max_torque)
        self.current_torque = float(torque)
        total_reward = 0.0

        for _ in range(self.control_steps):
            self.step_rk4(self.current_torque)
            self.time += self.dt
            total_reward += self.calculate_reward()

            if abs(self.err()) > self.max_angle or self.time > self.max_time:
                break

        done = abs(self.err()) > self.max_angle or self.time > self.max_time
        if done:
            total_reward += self.calculate_reward(done=True)

        return self.get_state(), total_reward, done, {}

    def reset(self):
        self.fi = np.pi + np.deg2rad(30) #np.deg2rad(np.random.uniform(-1, 1))  # smaller noise
        self.fi_dot = 0.0
        self.time = 0.0
        self.score = 0.0
        self.current_torque = 0.0
        self.fi_vector = []
        return self.get_state()

    def reset_2(self):
            self.fi = np.pi  +np.deg2rad(np.random.uniform(-180, 180))  # smaller noise
            self.fi_dot = np.random.uniform(-15, 15)
            self.time = 0.0
            self.score = 0.0
            self.current_torque = 0.0
            self.fi_vector = []
            return self.get_state()

    def gui(self):
        self.root = Tk()
        self.root.title("Inverse Pendulum")
        self.root.geometry("500x500")
        self.C = Canvas(self.root, bg="light gray", height=400, width=500)
        self.C.pack()
        self.line = self.C.create_line(
            self.origo_x, self.origo_y,
            self.origo_x + self.r, self.origo_y,
            fill="orange", width=8
        )
        length = 200
        self.C.create_line(self.origo_x - length, self.origo_y, self.origo_x + length, self.origo_y, fill="black", width=1)
        self.C.create_line(self.origo_x, self.origo_y - length, self.origo_x, self.origo_y + length, fill="black", width=1)
        self.root.bind("<KeyPress-c>", lambda e: setattr(self, "stop_training", True))

    def update_gui(self):
        if self.C is None:
            return
        self.C.coords(
            self.line,
            self.origo_x, self.origo_y,
            self.origo_x + self.r * sin(self.fi),
            self.origo_y + self.r * cos(self.fi)
        )
        self.root.update()
        self.root.update_idletasks()


def test_pendulum_gui(actor, episodes=3):
    """Run trained policy with GUI (call after training)."""
    actor.eval()
    pend = Pendulum()
    pend.gui()

    print("\n" + "="*50)
    print("TESTING ON INVERSE PENDULUM (GUI)")
    print("="*50)

    for ep in range(episodes):
        state = pend.reset_2()
        total_reward = 0.0
        last_log_time = 0.0
        print(f"\n--- Episode {ep+1} ---")
        print(f"{'Time':>6} | {'fi (deg)':>8} | {'fi_dot':>8} | {'Torque':>8} | {'Reward':>8}")

        while True:
            state_t = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action, _ = actor.get_action(state_t, deterministic=True)
            torque = action.item()

            state, reward, done, _ = pend.step(torque)
            total_reward += reward
            pend.update_gui()
            time.sleep(0.02)

            # Log every 0.1 seconds
            if pend.time - last_log_time >= 0.1 or done:
                fi_deg = np.rad2deg(pend.fi)
                print(f"{pend.time:6.2f} | {fi_deg:8.2f} | {pend.fi_dot:8.3f} | {torque:8.2f} | {reward:8.2f}")
                last_log_time = pend.time

            if done or pend.stop_training:
                print(f"Episode ended | Time: {pend.time:.2f}s | Total Reward: {total_reward:.1f}")
                break

        if pend.stop_training:
            break

    print("GUI test finished.")
    actor.train()


def record_one_episode(actor, max_steps=200):
    """
    Run one episode with the trained policy and record fi vs time.
    Returns lists: times, fis (in degrees)
    """
    actor.eval()
    pend = Pendulum()
    state = pend.reset_2()          # or pend.reset() if you prefer fixed start

    times = []
    fis = []
    total_reward = 0.0

    for _ in range(max_steps):
        state_t = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action, _ = actor.get_action(state_t, deterministic=True)
        torque = action.item()

        state, reward, done, _ = pend.step(torque)
        total_reward += reward

        times.append(pend.time)
        fis.append(np.rad2deg(pend.fi))

        if done:
            break

    actor.train()
    print(f"Episode finished | Time: {pend.time:.2f}s | Reward: {total_reward:.1f}")
    return times, fis


def plot_fi_vs_time(times, fis, save_path="fi_vs_time.png"):
    """Plot and save angle vs time for one episode."""
    plt.figure(figsize=(10, 4))
    plt.plot(times, fis, label="fi (deg)", color="orange")
    plt.axhline(180, color="gray", linestyle="--", label="upright (180°)")
    plt.xlabel("Time (s)")
    plt.ylabel("Angle fi (degrees)")
    plt.title("Pendulum angle vs time (one episode)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Plot saved to {save_path}")