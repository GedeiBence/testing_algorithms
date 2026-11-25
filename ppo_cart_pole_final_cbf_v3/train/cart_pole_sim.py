# ==================== cartpole_sim.py (cbf_v3) ====================
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
        self.stable_angle = np.deg2rad(3)
        self.max_time = 12.0
        self.max_x = 3.4
        self.max_fi_dot = 15.0
        self.reward_type = 'D'
        self.fi_desired = np.pi
        self.current_force = 0.0
        self.current_episode = 1

        # CBF
        self.use_cbf = False
        self.alpha = 8.0
        self.cbf_lambda = 0.0
        self.cbf_sigma = 40.0

        # Barrier
        self.barrier_center = 0.0
        self.barrier_half = 1.2
        self.barrier_vel = 0.0
        self.barrier_move = True

        # GUI
        self.root = None
        self.C = None
        self.cart = None
        self.pole = None
        self.stop_training = False

        self.reset()

    def err(self):
        return np.arctan2(np.sin(self.fi - self.fi_desired), np.cos(self.fi - self.fi_desired))

    def get_state(self):
        x_norm = self.x / self.max_x
        v_norm = self.x_dot / 2.0
        e = self.err()
        sin_e = np.sin(e)
        cos_e = np.cos(e)
        w = self.fi_dot / 5.0
        bc = self.barrier_center / self.max_x
        bh = self.barrier_half / self.max_x
        return np.array([x_norm, v_norm, sin_e, cos_e, w, bc, bh], dtype=np.float32)

    def step_sim(self, force_x):
        denom = self.mc + self.mp * sin(self.fi)**2
        x_dot_dot = (force_x + self.mp * sin(self.fi) * (self.l * self.fi_dot**2 + self.g * cos(self.fi))) / denom
        self.x_dot += x_dot_dot * self.dt
        self.x += self.x_dot * self.dt

        fi_dot_dot = (-force_x * cos(self.fi) - self.mp * self.l * self.fi_dot**2 * cos(self.fi) * sin(self.fi)
                    - (self.mc + self.mp) * self.g * sin(self.fi)) / (self.l * denom)
        self.fi_dot += fi_dot_dot * self.dt
        self.fi += self.fi_dot * self.dt

    def _signed_dist_to_barrier(self, x):
        return abs(x - self.barrier_center) - self.barrier_half

    def cbf_filter(self, proposed_force):
        x = self.x
        v = self.x_dot
        d = self._signed_dist_to_barrier(x)

        to_center = self.barrier_center - x
        sign = np.sign(to_center) if abs(to_center) > 1e-4 else 0.0
        v_approach = v * sign

        if d < 0.12 or (d < 0.28 and v_approach > 0.6):
            u_safe = -sign * self.max_force * 0.7
            return np.clip(u_safe, -self.max_force, self.max_force)

        gamma1 = 0.2
        alpha = self.alpha
        h = d - gamma1 * max(v_approach, 0.0)
        a = -gamma1 * sign / self.mc
        b = -alpha * h

        if abs(a) < 1e-6 or a * proposed_force >= b:
            return np.clip(proposed_force, -self.max_force, self.max_force)

        correction = (b - a * proposed_force) / (a ** 2 + 1e-8) * a
        u_safe = proposed_force + 0.6 * correction
        return np.clip(u_safe, -self.max_force, self.max_force)

    def cbf_reward(self, proposed_force, safe_force):
        d = self._signed_dist_to_barrier(self.x)
        to_center = self.barrier_center - self.x
        sign = np.sign(to_center) if abs(to_center) > 1e-4 else 0.0
        v_approach = self.x_dot * sign

        proximity_penalty = 0.0
        if d < 0.22 and v_approach > 0.4:
            proximity_penalty = -25.0 * (0.22 - d) ** 2

        corr = proposed_force - safe_force
        correction_term = np.exp(-corr ** 2 / self.cbf_sigma ** 2) - 1.0
        return 1.5 * correction_term + proximity_penalty

    def calculate_reward(self, done=False):
        e = self.err()
        x_err = self.x

        if self.reward_type == 'B':
            angle_reward = np.cos(e)
            vel_penalty = 0.06 * self.fi_dot**2
            pos_penalty = 0.4 * self.x**2
            vel_x_penalty = 0.2 * self.x_dot**2
            force_penalty = 0.0005 * self.current_force**2

            d = self._signed_dist_to_barrier(self.x)
            barrier_penalty = 0.0
            if d < 0.35:
                barrier_penalty = 8.0 * (0.35 - d)**2

            return (angle_reward 
                    - vel_penalty 
                    - pos_penalty 
                    - vel_x_penalty 
                    - force_penalty 
                    - barrier_penalty)

        elif self.reward_type == 'D':
            if done:
                return -10.0

            if abs(e) <= np.deg2rad(3) and abs(x_err) < 0.5:
                base = (
                    5.0 - 2.0 * e**2 - 0.3 * self.fi_dot**2
                    - 2.0 * x_err**2 - 0.2 * self.x_dot**2
                    - 0.01 * self.current_force**2
                )
            elif abs(e) <= np.deg2rad(25) and abs(x_err) < 0.8:
                base = (
                    1.5 - 0.5 * e**2 - 0.1 * self.fi_dot**2
                    - 0.8 * x_err**2 - 0.05 * self.x_dot**2
                    - 0.005 * self.current_force**2
                )
            else:
                base = -0.03 - 0.01 * e**2

            # Barrier közeli büntetés
            d = self._signed_dist_to_barrier(self.x)
            barrier_penalty = 0.0
            if d < 0.35:
                barrier_penalty = 6.0 * (0.35 - d)**2

            return base - barrier_penalty

        return 0.0

    def step(self, force):
        if self.barrier_move and self.use_cbf:
            direction = np.sign(self.x - self.barrier_center)
            if abs(direction) < 1e-4:
                direction = 1.0
            self.barrier_vel = direction * 0.08
            self.barrier_center += self.barrier_vel * self.tau
            self.barrier_center = np.clip(self.barrier_center, -self.max_x + 0.6, self.max_x - 0.6)

        proposed_force = np.clip(float(force), -self.max_force, self.max_force)

        if self.use_cbf:
            safe_force = self.cbf_filter(proposed_force)
            r_cbf = self.cbf_reward(proposed_force, safe_force)
        else:
            safe_force = proposed_force
            r_cbf = 0.0

        self.current_force = float(safe_force)
        total_reward = 0.0
        hit_wall = False

        for _ in range(self.control_steps):
            self.step_sim(self.current_force)
            self.time += self.dt
            total_reward += self.calculate_reward()

            if self.use_cbf:
                d = self._signed_dist_to_barrier(self.x)
                if d < 0.0:
                    hit_wall = True
                    break

            if abs(self.err()) > self.max_angle or self.time > self.max_time or abs(self.fi_dot) > self.max_fi_dot:
                break

        done = abs(self.err()) > self.max_angle or hit_wall or self.time > self.max_time or abs(self.fi_dot) > self.max_fi_dot

        if hit_wall:
            total_reward += -150.0
        elif done:
            total_reward += self.calculate_reward(done=True)

        total_reward += self.cbf_lambda * r_cbf
        return self.get_state(), total_reward, done, {}

    def _place_barrier(self):
        for _ in range(50):
            self.barrier_center = np.random.uniform(-2.0, 2.0)
            self.barrier_half = np.random.uniform(0.5, 1.1)

            left_edge = self.barrier_center - self.barrier_half
            right_edge = self.barrier_center + self.barrier_half
            left_space = left_edge - (-self.max_x)
            right_space = self.max_x - right_edge

            if left_space > 1.0 or right_space > 1.0:
                break

    def reset(self):
        self._place_barrier()

        if np.random.rand() < 0.25:
            left_edge = self.barrier_center - self.barrier_half
            right_edge = self.barrier_center + self.barrier_half
            left_space = left_edge - (-self.max_x)
            right_space = self.max_x - right_edge
            side = -1.0 if left_space > right_space else 1.0
            offset = np.random.uniform(0.2, 0.5)
            self.x = self.barrier_center + side * (self.barrier_half + offset)
            self.x = np.clip(self.x, -self.max_x + 0.2, self.max_x - 0.2)
        else:
            self.x = np.random.uniform(-0.3, 0.3)

        self.x_dot = np.random.uniform(-0.5, 0.5)
        self.fi = np.pi + np.deg2rad(np.random.uniform(-180, 180))
        self.fi_dot = np.random.uniform(-0.8, 0.8)
        self.time = 0.0
        self.current_force = 0.0
        self.barrier_vel = 0.0
        return self.get_state()

    def reset_test(self):
        self._place_barrier()

        left_edge = self.barrier_center - self.barrier_half
        right_edge = self.barrier_center + self.barrier_half
        left_space = left_edge - (-self.max_x)
        right_space = self.max_x - right_edge

        side = -1.0 if left_space > right_space else 1.0
        offset = np.random.uniform(0.25, 0.6)
        self.x = self.barrier_center + side * (self.barrier_half + offset)
        self.x = np.clip(self.x, -self.max_x + 0.2, self.max_x - 0.2)

        self.x_dot = np.random.uniform(-0.4, 0.4)
        self.fi = np.pi + np.deg2rad(np.random.uniform(-8, 8))
        self.fi_dot = np.random.uniform(-0.4, 0.4)
        self.time = 0.0
        self.current_force = 0.0
        self.barrier_vel = 0.0
        return self.get_state()

    def gui(self):
        self.root = Tk()
        self.root.title("Cart-Pole with CBF")
        self.root.geometry("1000x500")
        self.C = Canvas(self.root, bg="light gray", height=400, width=900)
        self.C.pack()
        self.cart = self.C.create_rectangle(0, 0, 20, 20, fill="blue")
        self.pole = self.C.create_line(0, 0, 0, 0, fill="green", width=3)
        self.x0 = 450
        self.y0 = 200
        self.px = 350 / self.max_x
        self.r = 80 * self.l
        self.info_text = self.C.create_text(20, 20, anchor="nw", font=("Arial", 14), fill="black")

    def update_gui(self):
        if self.C is None:
            return

        origo_x = self.x0 + self.x * self.px
        self.C.coords(self.cart, origo_x - 10, self.y0 - 10, origo_x + 10, self.y0 + 10)
        self.C.coords(self.pole,
                    origo_x, self.y0,
                    origo_x + self.r * sin(self.fi),
                    self.y0 + self.r * cos(self.fi))

        self.C.delete("barrier")
        left = self.x0 + (self.barrier_center - self.barrier_half) * self.px
        right = self.x0 + (self.barrier_center + self.barrier_half) * self.px
        top = self.y0 - 60
        bottom = self.y0 + 60
        self.C.create_rectangle(left, top, right, bottom,
                                fill="red", stipple="gray25", outline="darkred", width=2, tags="barrier")

        fi_deg = np.rad2deg(self.fi)
        text = f"Episode: {getattr(self, 'current_episode', 1)}\n"
        text += f"Time: {self.time:.1f}s\n"
        text += f"Angle: {fi_deg:.1f}°\n"
        text += f"x: {self.x:.2f} m\n"
        text += f"Force: {self.current_force:.1f} N\n"
        text += f"Barrier: {self.barrier_center:.1f} ± {self.barrier_half:.1f}\n"
        text += f"CBF: {'ON' if self.use_cbf else 'OFF'}"
        self.C.itemconfig(self.info_text, text=text)
        self.root.update()


def test_cartpole_gui(actor, episodes=5):
    actor.eval()
    env = CartPole()
    env.use_cbf = True
    env.cbf_lambda = 1.5
    env.gui()
    print("\n" + "="*50)
    print("TESTING CART-POLE WITH CBF")
    print("="*50)

    for ep in range(episodes):
        state = env.reset_test()
        total_reward = 0.0
        last_log = 0.0
        print(f"\n--- Episode {ep+1} ---")
        print(f"Barrier center: {env.barrier_center:.2f} | half: {env.barrier_half:.2f}")
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