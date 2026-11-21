# ==================== pendulum_sim.py ====================
from tkinter import *
import numpy as np
import time, random 
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
        self.max_time = 4.0
        self.fi_desired = np.pi
        self.reward_type = 'C'
        self.current_torque = 0.0
        self.reset()

        # Control Barriers
        self.max_fi_dot = 20.0
        self.barrier_center = 0.0
        self.barrier_width  = np.deg2rad(40)

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

        # Barrier info (normalized)
        bc = self.wrap_angle(self.barrier_center) / np.pi
        bw = self.barrier_width / np.pi

        return np.array([e, w, bc, bw], dtype=np.float32)

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
        fi_wrapped = self.wrap_angle(self.fi)

        # Distance to barrier
        dist_to_barrier = abs(self.wrap_angle(fi_wrapped - self.barrier_center))
        half_width = self.barrier_width / 2

        match self.reward_type:
            case 'A':
                # 1. Goal reward (upright)
                if abs(e) <= np.deg2rad(5):
                    reward = 3.0 - 0.4 * e**2 - 0.3 * w**4 - 0.001 * self.current_torque**2
                else:
                    reward = -0.5 * e**2 - 0.05 * w**2

                # 2. Soft barrier penalty (before hitting it)
                if dist_to_barrier < half_width + np.deg2rad(15):
                    danger = half_width + np.deg2rad(15) - dist_to_barrier
                    reward -= 2.0 * danger

                # 3. Hard penalty if inside barrier or done
                if dist_to_barrier < half_width or done:
                    reward -= 20.0

                return reward

            case 'B':
                e = self.err()
                w = self.fi_dot
                fi_wrapped = self.wrap_angle(self.fi)
                dist = abs(self.wrap_angle(fi_wrapped - self.barrier_center))
                half = self.barrier_width / 2.0

                # Alap reward (cél)
                if abs(e) <= np.deg2rad(8):
                    reward = 5.0 - 1.0 * e**2 - 0.4 * w**2 - 0.002 * self.current_torque**2
                else:
                    reward = -0.3 * e**2 - 0.02 * w**2

                # Soft barrier (korai figyelmeztetés)
                if dist < half + np.deg2rad(20):
                    danger = (half + np.deg2rad(20) - dist) / np.deg2rad(20)
                    reward -= 8.0 * danger**2

                # Hard hit
                if dist < half or done:
                    reward -= 30.0

                return reward          # ← EZ HIÁNYZOTT
            case 'C':
                e = self.err()
                w = self.fi_dot
                fi_wrapped = self.wrap_angle(self.fi)
                dist = abs(self.wrap_angle(fi_wrapped - self.barrier_center))
                half = self.barrier_width / 2.0

                # Gyenge cél jutalom
                if abs(e) <= np.deg2rad(6):
                    reward = 1.5 - 0.5 * e**2 - 0.2 * w**2
                else:
                    reward = -0.003 * e**2

                # Soft barrier (korai figyelmeztetés)
                if dist < half + np.deg2rad(25):
                    danger = (half + np.deg2rad(25) - dist) / np.deg2rad(25)
                    reward -= 15.0 * danger**2

                # Nagyon erős hard büntetés
                if dist < half or done:
                    reward -= 150.0          # sokkal nagyobb

                return reward
            
            case _:
                # fallback
                return -e**2 - 0.1 * w**2
                    
    def wrap_angle(self, angle):
            """Wrap to [-π, π]"""
            return np.arctan2(np.sin(angle), np.cos(angle))

       
    def step(self, torque):
        """One control step with ZOH."""
        self.fi = self.wrap_angle(self.fi)
        torque = np.clip(torque, -self.max_torque, self.max_torque)
        self.current_torque = float(torque)
        total_reward = 0.0
        done = False

        for _ in range(self.control_steps):
            self.step_rk4(self.current_torque)
            self.time += self.dt

            err = self.err()
            fi_wrapped = self.wrap_angle(self.fi)

            # Prohibited zone relative to upright
            dist = abs(self.wrap_angle(fi_wrapped - self.barrier_center))
            in_prohibited = dist < (self.barrier_width / 2)

            if (abs(err) > self.max_angle or 
                abs(self.fi_dot) > self.max_fi_dot or 
                in_prohibited or
                self.time > self.max_time):
                done = True
                total_reward += -50.0   
                break

            total_reward += self.calculate_reward()

        return self.get_state(), total_reward, done, {}

    def reset(self):
        # Random barrier (make sure it doesn't cover the desired)
        for _ in range(20):
            self.barrier_center = np.random.uniform(-np.pi, np.pi)
            self.barrier_width  = np.deg2rad(np.random.uniform(25, 70))
            dist = abs(self.wrap_angle(self.barrier_center - self.fi_desired))
            if dist > self.barrier_width/2 + np.deg2rad(15):
                break

        self.fi = self.fi_desired + np.deg2rad(np.random.uniform(-60, 60))
        self.fi = self.wrap_angle(self.fi)
        self.fi_dot = np.random.uniform(-2, 2)
        self.time = 0.0
        self.current_torque = 0.0
        return self.get_state()

    def reset_2(self):
        # Barrier felül, de NEM a desired-en
        for _ in range(30):
            offset = np.deg2rad(np.random.uniform(25, 120))          # legalább 25°-ra az uprighttól
            side = np.random.choice([-1, 1])
            self.barrier_center = self.wrap_angle(np.pi + side * offset)
            self.barrier_width  = np.deg2rad(np.random.uniform(20, 40))
            # Ellenőrzés: ne fedje a desired-et
            dist = abs(self.wrap_angle(self.barrier_center - self.fi_desired))
            if dist > self.barrier_width / 2 + np.deg2rad(12):
                break

        # Start alul
        self.fi = 0#np.deg2rad(np.random.uniform(-120, 120))
        self.fi = self.wrap_angle(self.fi)
        self.fi_dot =0# np.random.uniform(-2, 2)

        self.time = 0.0
        self.current_torque = 0.0
        self.draw_barrier()
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
        # Barrier-t NEM itt rajzoljuk

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

    def draw_barrier(self):
        if self.C is None:
            return
        self.C.delete("barrier")

        half = self.barrier_width / 2.0
        center = self.wrap_angle(self.barrier_center)

        # Convert to degrees for easier handling in canvas (optional but clearer)
        # We draw in radians directly
        steps = 20
        points = [self.origo_x, self.origo_y]

        for i in range(steps + 1):
            # Go from -half to +half relative to center
            a = center - half + (2 * half) * i / steps
            a = self.wrap_angle(a)
            px = self.origo_x + self.r * 0.95 * np.sin(a)
            py = self.origo_y + self.r * 0.95 * np.cos(a)
            points.extend([px, py])

        # Only one polygon
        self.C.create_polygon(points, fill="red", stipple="gray25", outline="", tags="barrier")

        # Two boundary lines
        for sign in (-1, 1):
            a = self.wrap_angle(center + sign * half)
            x = self.origo_x + self.r * 1.15 * np.sin(a)
            y = self.origo_y + self.r * 1.15 * np.cos(a)
            self.C.create_line(self.origo_x, self.origo_y, x, y, fill="red", width=1, tags="barrier")


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