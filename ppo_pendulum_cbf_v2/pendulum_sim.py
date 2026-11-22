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
        self.dt = 0.001
        self.control_steps = 20
        self.tau = self.dt * self.control_steps
        self.max_torque = 100.0
        self.max_angle = np.deg2rad(180)
        self.stable_angle = np.deg2rad(3)
        self.max_time = 4.0
        self.fi_desired = np.pi
        self.reward_type = 'C'
        self.current_torque = 0.0

        # Control Barriers
        self.max_fi_dot = 20.0
        self.barrier_center = 0.0
        self.barrier_width  = np.deg2rad(40)

        # CBF-RL paraméterek
        self.alpha = 12.0
        self.cbf_lambda = 3.5
        self.cbf_sigma = 40.0

        # GUI
        self.fi_vector = []
        self.root = None
        self.C = None
        self.line = None
        self.origo_x = 250
        self.origo_y = 200
        self.r = 100 * self.l
        self.stop_training = False

        self.reset()

    def err(self):
        return np.arctan2(np.sin(self.fi - self.fi_desired), np.cos(self.fi - self.fi_desired))

    def get_state(self):
        e = self.err() / self.max_angle
        w = self.fi_dot / 5.0
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

    def wrap_angle(self, angle):
        return np.arctan2(np.sin(angle), np.cos(angle))

    def _signed_dist_to_barrier(self, fi):
        dist = abs(self.wrap_angle(fi - self.barrier_center))
        half = self.barrier_width / 2.0
        return dist - half

    def cbf_filter(self, proposed_torque):
        fi = self.fi
        w = self.fi_dot
        d = self._signed_dist_to_barrier(fi)

        to_center = self.wrap_angle(self.barrier_center - fi)
        sign = np.sign(to_center) if abs(to_center) > 1e-4 else 0.0
        v_approach = w * sign

        gamma1 = 0.30
        alpha = self.alpha

        h = d - gamma1 * v_approach
        a = -gamma1 * sign / (self.m * self.l ** 2)
        b_drift = -gamma1 * sign * (-(self.g / self.l) * np.sin(fi))
        b = -alpha * h - b_drift

        if d < 0.08 or (d < 0.18 and v_approach > 1.5):
            u_safe = -sign * self.max_torque * 0.9
            return np.clip(u_safe, -self.max_torque, self.max_torque)

        if abs(a) < 1e-6 or a * proposed_torque >= b:
            return proposed_torque

        correction = (b - a * proposed_torque) / (a ** 2 + 1e-8) * a
        u_safe = proposed_torque + correction
        return np.clip(u_safe, -self.max_torque, self.max_torque)

    def cbf_reward(self, proposed_torque, safe_torque):
        fi = self.fi
        w = self.fi_dot
        d = self._signed_dist_to_barrier(fi)

        to_center = self.wrap_angle(self.barrier_center - fi)
        sign = np.sign(to_center) if abs(to_center) > 1e-4 else 0.0
        v_approach = w * sign

        gamma1 = 0.30
        h = d - gamma1 * v_approach
        a = -gamma1 * sign / (self.m * self.l ** 2)
        b_drift = -gamma1 * sign * (-(self.g / self.l) * np.sin(fi))
        b = -self.alpha * h - b_drift

        violation = min(a * proposed_torque - b, 0.0)
        corr = proposed_torque - safe_torque
        correction_term = np.exp(-corr ** 2 / self.cbf_sigma ** 2) - 1.0

        proximity_penalty = 0.0
        if d < 0.25:
            proximity_penalty = -40.0 * (0.25 - d) ** 2

        return violation + 3.0 * correction_term + proximity_penalty

    def calculate_reward(self, done=False):
        e = self.err()
        w = self.fi_dot
        fi_wrapped = self.wrap_angle(self.fi)
        dist_to_barrier = abs(self.wrap_angle(fi_wrapped - self.barrier_center))
        half_width = self.barrier_width / 2

        if self.reward_type == 'A':
            if abs(e) <= np.deg2rad(5):
                reward = 3.0 - 0.4 * e**2 - 0.3 * w**4 - 0.001 * self.current_torque**2
            else:
                reward = -0.5 * e**2 - 0.05 * w**2
            if dist_to_barrier < half_width + np.deg2rad(15):
                danger = half_width + np.deg2rad(15) - dist_to_barrier
                reward -= 2.0 * danger
            if dist_to_barrier < half_width or done:
                reward -= 20.0
            return reward

        elif self.reward_type == 'B':
            if abs(e) <= np.deg2rad(8):
                reward = 5.0 - 1.0 * e**2 - 0.4 * w**2 - 0.002 * self.current_torque**2
            else:
                reward = -0.3 * e**2 - 0.02 * w**2
            if dist_to_barrier < half_width + np.deg2rad(20):
                danger = (half_width + np.deg2rad(20) - dist_to_barrier) / np.deg2rad(20)
                reward -= 8.0 * danger**2
            if dist_to_barrier < half_width or done:
                reward -= 30.0
            return reward

        elif self.reward_type == 'C':
            if abs(e) <= np.deg2rad(6):
                reward = 1.5 - 0.5 * e**2 - 0.2 * w**2
            else:
                reward = -0.003 * e**2
            if dist_to_barrier < half_width + np.deg2rad(25):
                danger = (half_width + np.deg2rad(25) - dist_to_barrier) / np.deg2rad(25)
                reward -= 15.0 * danger**2
            if dist_to_barrier < half_width or done:
                reward -= 150.0
            return reward

        else:
            return -e**2 - 0.1 * w**2

    def step(self, torque):
        self.fi = self.wrap_angle(self.fi)
        proposed_torque = np.clip(torque, -self.max_torque, self.max_torque)

        safe_torque = self.cbf_filter(proposed_torque)
        self.current_torque = float(safe_torque)

        r_cbf = self.cbf_reward(proposed_torque, safe_torque)

        total_reward = 0.0
        done = False

        for _ in range(self.control_steps):
            self.step_rk4(self.current_torque)
            self.time += self.dt

            err = self.err()
            fi_wrapped = self.wrap_angle(self.fi)
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

        total_reward += self.cbf_lambda * r_cbf
        return self.get_state(), total_reward, done, {}

    def reset(self):
        # Barrier generálása (ne fedje a desired-et)
        for _ in range(30):
            self.barrier_center = np.random.uniform(-np.pi, np.pi)
            self.barrier_width  = np.deg2rad(np.random.uniform(25, 55))
            dist = abs(self.wrap_angle(self.barrier_center - self.fi_desired))
            if dist > self.barrier_width / 2 + np.deg2rad(20):
                break

        # Spawn a barrier egyik széle közelében (kívülről)
        half = self.barrier_width / 2.0
        side = np.random.choice([-1.0, 1.0])
        offset = np.deg2rad(np.random.uniform(8, 25))   # 8–25 fokkal a barrieren kívül
        self.fi = self.wrap_angle(self.barrier_center + side * (half + offset))

        # Kis kezdősebesség a barrier irányába vagy semleges
        self.fi_dot = np.random.uniform(-1.5, 1.5)
        self.time = 0.0
        self.current_torque = 0.0
        return self.get_state()

    def reset_2(self):
        # Teszteléshez is hasonló logika
        for _ in range(30):
            offset = np.deg2rad(np.random.uniform(40, 110))
            side = np.random.choice([-1, 1])
            self.barrier_center = self.wrap_angle(np.pi + side * offset)
            self.barrier_width  = np.deg2rad(np.random.uniform(20, 40))
            dist = abs(self.wrap_angle(self.barrier_center - self.fi_desired))
            if dist > self.barrier_width / 2 + np.deg2rad(15):
                break

        half = self.barrier_width / 2.0
        spawn_side = np.random.choice([-1.0, 1.0])
        offset = np.deg2rad(np.random.uniform(10, 30))
        self.fi = self.wrap_angle(self.barrier_center + spawn_side * (half + offset))
        self.fi_dot = 0.0
        self.time = 0.0
        self.current_torque = 0.0
        if self.C is not None:
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
        steps = 20
        points = [self.origo_x, self.origo_y]
        for i in range(steps + 1):
            a = center - half + (2 * half) * i / steps
            a = self.wrap_angle(a)
            px = self.origo_x + self.r * 0.95 * np.sin(a)
            py = self.origo_y + self.r * 0.95 * np.cos(a)
            points.extend([px, py])
        self.C.create_polygon(points, fill="red", stipple="gray25", outline="", tags="barrier")
        for sign in (-1, 1):
            a = self.wrap_angle(center + sign * half)
            x = self.origo_x + self.r * 1.15 * np.sin(a)
            y = self.origo_y + self.r * 1.15 * np.cos(a)
            self.C.create_line(self.origo_x, self.origo_y, x, y, fill="red", width=1, tags="barrier")


def test_pendulum_gui(actor, episodes=3):
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
    actor.eval()
    pend = Pendulum()
    state = pend.reset_2()
    times, fis = [], []
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