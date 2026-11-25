# ==================== sim.py ====================
import numpy as np

class CartPole:
    def __init__(self, desired_x=0.0):
        self.mp = 1.0 # pole
        self.mc = 1.0 # cart
        self.l = 0.5
        self.g = 9.81
        self.dt = 0.001
        self.control_steps = 20
        self.max_force = 30.0
        self.max_time = 10.0
        self.max_x = 1.7
        self.max_angle = np.deg2rad(12)
        self.reward_type = 'A'
        self.desired_x = desired_x
        self.reset()

    def err(self):
        return np.arctan2(np.sin(self.fi - np.pi), np.cos(self.fi - np.pi))

    def get_state(self):
        e = self.err() / self.max_angle
        w = self.fi_dot / 2.0
        return np.array([self.x, self.x_dot, e, w], dtype=np.float32)

    def reset(self):
        """Used during training."""
        self.x = 0.0
        self.x_dot = 0.0
        self.fi = np.pi + np.deg2rad(7)
        self.fi_dot = 0.0
        self.time = 0.0
        self.current_force = 0.0
        return self.get_state()

    def reset_2(self):
        """Used during testing (wider initial conditions)."""
        self.x = 0.0
        self.x_dot = 0.0
        self.fi = np.pi + np.deg2rad(np.random.uniform(-5, 5))
        self.fi_dot = 0.0
        self.time = 0.0
        self.current_force = 0.0
        return self.get_state()

    
    def step(self, force):
        """Training step – terminates only on failure or time limit."""
        force = np.clip(force, -self.max_force, self.max_force)
        self.current_force = float(force)

        total_reward = 0.0

        for _ in range(self.control_steps):
            self._integrate(force)

            total_reward += self._instant_reward()

            fail = (
                abs(self.x) > self.max_x or
                abs(self.err()) > self.max_angle
            )

            if fail:
                break

        fail = (
            abs(self.x) > self.max_x or
            abs(self.err()) > self.max_angle
        )

        done = fail or self.time >= self.max_time

        if fail:
            total_reward -= 50.0

        return self.get_state(), total_reward, done, {}


    def step_test(self, force):
        """Testing step – no termination logic."""
        force = np.clip(force, -self.max_force, self.max_force)
        self.current_force = float(force)
        total_reward = 0.0

        for _ in range(self.control_steps):
            self._integrate(force)
            total_reward += self._instant_reward()

        return self.get_state(), total_reward

    def _integrate(self, force):
        cos_fi = np.cos(self.fi)
        sin_fi = np.sin(self.fi)
        temp = (force + self.mp * self.l * self.fi_dot**2 * sin_fi) / (self.mc + self.mp)
        fi_acc = (self.g * sin_fi - cos_fi * temp) / (self.l * (4.0/3.0 - self.mp * cos_fi**2 / (self.mc + self.mp)))
        x_acc = temp - self.mp * self.l * fi_acc * cos_fi / (self.mc + self.mp)

        self.fi_dot += fi_acc * self.dt
        self.fi += self.fi_dot * self.dt
        self.x_dot += x_acc * self.dt
        self.x += self.x_dot * self.dt
        self.time += self.dt

    
    def _instant_reward(self):
        pos_error = self.x - self.desired_x
        ang_error = self.err()

        match self.reward_type:
            case 'A':
                reward = (
                    -10.0 * pos_error**2
                    - 2.0 * self.x_dot**2
                    - 100.0 * ang_error**2
                    - 10.0 * self.fi_dot**2
                    - 0.01 * self.current_force**2
                )
            case 'B':
                reward = -1

        return reward


    def gui(self):
        pass

    def update_gui(self):
        pass