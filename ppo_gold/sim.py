# ==================== sim.py ====================
import numpy as np

class SystemSim:
    def __init__(self, desired_x=1.0):
        self.max_force = 3.0          # [N]
        self.max_time = 3.0            # episode length [s]
        self.max_x = 2.0
        self.m = 1.0
        self.reward_type = 'B'
        self.dt = 0.001                # simulation timestep
        self.control_steps = 20        # ZOH: hold force for 25 sim steps
        self.tau = self.dt * self.control_steps   # control period = 0.025 s

        self.desired_x = desired_x
        self.reset()

    def reset(self):
        self.current_force = 0.0
        self.current_time = 0.0
        self.x = 0.0
        self.x_dot = 0.0
        self.x_dot_dot = 0.0
        return self.get_state()

    def get_state(self):
        """State seen by the policy: [x, x_dot]"""
        return np.array([self.x, self.x_dot], dtype=np.float32)

    def step(self, force):
        """
        One control step (ZOH).
        The force is held constant for `control_steps` simulation steps.
        Returns: next_state, reward, done, info
        """
        # Clip force to actuator limits
        force = np.clip(force, -self.max_force, self.max_force)
        self.current_force = float(force)

        total_reward = 0.0

        # Integrate physics at high rate while holding the force
        for _ in range(self.control_steps):
            self.x_dot_dot = self.current_force / self.m
            self.x_dot += self.x_dot_dot * self.dt
            self.x += self.x_dot * self.dt
            self.current_time += self.dt

            # Accumulate reward over the control interval
            total_reward += self._instant_reward()

            # Optional early termination
            if abs(self.x) > self.max_x or self.current_time >= self.max_time:
                break
        success = abs(self.x - self.desired_x) < 0.05 and abs(self.x_dot) < 0.1
        done = self.current_time >= 2.0 or abs(self.x) > self.max_x or success
        #done = self.current_time >= self.max_time or abs(self.x) > self.max_x
        next_state = self.get_state()

        return next_state, total_reward, done, {}

    def _instant_reward(self):
        match (self.reward_type):
            case 'A':
                """Reward at one simulation step (summed over the ZOH interval)"""
                pos_error = self.x - self.desired_x
                reward = (
                    -pos_error**2
                    - 0.1 * self.x_dot**2
                    - 0.01 * self.current_force**2
                )
            case 'B':
                pos_error = self.x - self.desired_x
                reward = (
                    - 5.0*pos_error**2
                    - 0.5 * self.x_dot**2
                    - 0.005 * self.current_force**2
                    
                )
                # Time penalty (encourages speed)
                reward -= 0.1

                # Bonus when close to target
                if abs(pos_error) < 0.005 and abs(self.x_dot) < 0.01:
                    reward += 20.0
            case 'C':
                pos_error = self.x - self.desired_x
                reward = -pos_error**2 - 0.5 * self.x_dot**2 - 0.01 * self.current_force**2
        return reward

    # Optional: useful for debugging / rendering
    def get_info(self):
        return {
            "x": self.x,
            "x_dot": self.x_dot,
            "force": self.current_force,
            "time": self.current_time,
        }