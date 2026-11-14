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
        self.max_force = 20.0
        self.max_time = 3.0
        self.max_x = 0.5
        self.max_angle = np.deg2rad(12)
        self.reward_type = 'G'
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
        self.fi = np.pi + np.deg2rad(np.random.uniform(-5, 5))
        self.fi_dot = 0.0
        self.time = 0.0
        self.current_force = 0.0
        return self.get_state()

    def reset_2(self):
        """Used during testing (wider initial conditions)."""
        self.x = 1.0
        self.x_dot = 0.0
        self.fi = np.pi + np.deg2rad(np.random.uniform(-5, 5))
        self.fi_dot = 0.0
        self.time = 0.0
        self.current_force = 0.0
        return self.get_state()

    def step(self, force):
        """Training step – terminates on success / fail / time."""
        force = np.clip(force, -self.max_force, self.max_force)
        self.current_force = float(force)
        total_reward = 0.0

        for _ in range(self.control_steps):
            self._integrate(force)
            total_reward += self._instant_reward()

            if abs(self.x) > self.max_x or abs(self.err()) > self.max_angle:
                break

        success = (abs(self.err()) < 0.05 and abs(self.fi_dot) < 0.1 and
                abs(self.x) < 0.1 and abs(self.x_dot) < 0.1)
        fail = abs(self.x) > self.max_x or abs(self.err()) > self.max_angle
        done = success or fail or self.time >= self.max_time

        if success:
            # Larger bonus when more stable
            pos_error = self.x - self.desired_x
            ang_error = self.err()
            stability = (
                1.0
                - 8.0 * pos_error**2
                - 2.0 * self.x_dot**2
                - 10.0 * ang_error**2
                - 0.8 * self.fi_dot**2
            )
            total_reward += 50.0 * max(0.0, stability)
        elif fail:
            total_reward -= 10.0

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
                    -pos_error**2
                    - 0.1 * self.x_dot**2
                    - 0.01 * self.current_force**2
                )
            case 'B':
                reward = (
                    -5.0 * pos_error**2
                    - 0.5 * self.x_dot**2
                    - 0.005 * self.current_force**2
                )
                reward -= 0.1
                if abs(pos_error) < 0.005 and abs(self.x_dot) < 0.01:
                    reward += 20.0
            case 'C':
                reward = -pos_error**2 - 0.5 * self.x_dot** - 0.01 * self.current_force**2
            case 'D':
                pos_error = self.x
                ang_error = self.err()

                reward = (
                    -4.0 * pos_error**2
                    - 1.0 * self.x_dot**2
                    - 0.5 * ang_error**2
                    - 0.01 * self.current_force**2
                )

                if abs(pos_error) > 0.3:
                    reward -= 4.0 * pos_error**2

                if abs(pos_error) < 0.05 and abs(self.x_dot) < 0.1 and abs(ang_error) < 0.05:
                    reward += 20.0

            case 'E':
                
                reward = (
                    -4.0 * pos_error**2
                    - 1.0 * self.x_dot**2
                    - 3.0 * ang_error**2
                    - 0.5 * self.fi_dot**2
                    - 0.005 * self.current_force**2
                )

                # Stronger penalty when moving away from center
                if pos_error * self.x_dot > 0:
                    reward -= 4.0 * self.x_dot**2

                # Extra penalty when far from center
                if abs(pos_error) > 0.2:
                    reward -= 8.0 * pos_error**2

                # Extra penalty when angle is large
                if abs(ang_error) > 0.1:
                    reward -= 6.0 * ang_error**2

                # Bonus only when both cart and pole are good
                if (abs(pos_error) < 0.05 and abs(self.x_dot) < 0.1 and
                    abs(ang_error) < 0.05 and abs(self.fi_dot) < 0.1):
                    reward += 25.0

            case 'F':
                ang_error = self.err()
                pos_error = self.x - self.desired_x

                reward = (
                    -5.0 * pos_error**2
                    - 3.0 * self.x_dot**2
                    - 5.0 * ang_error**2
                    - 1.0 * self.fi_dot**2
                    - 0.005 * self.current_force**2
                )

                # Extra penalty when moving away from center
                if pos_error * self.x_dot > 0:
                    reward -= 8.0 * self.x_dot**2

                # Stronger penalty when far from center
                if abs(pos_error) > 0.32:
                    reward -= 20.0 * pos_error**8

                # Extra penalty when angle is large
                if abs(ang_error) > 0.06:
                    reward -= 10.0 * ang_error**2

                # Instant bonus only when very stable
                if (abs(pos_error) < 0.04 and abs(self.x_dot) < 0.06 and
                    abs(ang_error) < 0.03 and abs(self.fi_dot) < 0.06):
                    reward += 30.0
            case 'G':
                # Fully Stabilized High-Precision Controller for PPO
                ang_error = self.err()
                pos_error = self.x - self.desired_x

                # 1. Stronger Base Damping: Prevent velocity and angular speed from spiraling
                reward = (
                    -6.0 * pos_error**2           # Position tracking weight
                    - 4.0 * self.x_dot**2         # Increased cart velocity damping (acts as brakes)
                    - 15.0 * ang_error**2         # High weight on pole angle
                    - 6.0 * self.fi_dot**2        # Increased angular velocity damping
                    - 0.005 * self.current_force**2 # Smooth action penalty
                )

                # 2. Symmetric Kinetic Energy Penalty (Penalizes high speed when far from center, regardless of direction)
                if abs(pos_error) > 0.05:
                    reward -= 2.0 * self.x_dot**2
                    reward -= 2.0 * self.fi_dot**2

                # 3. Soft Progressive Bounds for position and angle errors
                if abs(pos_error) > 0.15:
                    excess_pos = abs(pos_error) - 0.15
                    reward -= 8.0 * excess_pos**2

                if abs(ang_error) > 0.08:
                    excess_ang = abs(ang_error) - 0.08
                    reward -= 12.0 * excess_ang**2

                # 4. Rich Stabilization Bonus for precise hovering
                if (abs(pos_error) < 0.03 and abs(self.x_dot) < 0.05 and
                    abs(ang_error) < 0.03 and abs(self.fi_dot) < 0.05):
                    reward += 35.0

            case _:
                reward = 0.0

        return reward

    def gui(self):
        pass

    def update_gui(self):
        pass