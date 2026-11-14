# ==================== agent.py (CPU, faster) ====================
import torch
import torch.nn.functional as F
import numpy as np
from models import ActorNetwork, CriticNetwork
from helper import plot, compute_moving_average
from cart_pole_sim import CartPole, test_cartpole_gui

# Hyperparameters
GAMMA = 0.99
LAMBDA = 0.95
LR = 3e-4
CLIP_EPS = 0.2
ENTROPY_COEF = 0.02
VALUE_COEF = 0.5
EPOCHS = 10
BATCH_SIZE = 256
ROLLOUT_STEPS = 2048

class Agent:
    def __init__(self, state_dim=4, action_dim=1):
        self.device = torch.device("cpu")
        print("Using device: cpu")

        self.actor = ActorNetwork(state_dim, action_dim, hidden_size=128)
        self.critic = CriticNetwork(state_dim, hidden_size=128)
        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=LR
        )

    def select_action(self, state):
        state_t = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action, log_prob = self.actor.get_action(state_t)
        return action.item(), log_prob.item()

    def compute_gae(self, rewards, values, dones, next_value):
        advantages = []
        gae = 0.0
        values = values + [next_value]
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + GAMMA * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + GAMMA * LAMBDA * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        returns = [adv + val for adv, val in zip(advantages, values[:-1])]
        return advantages, returns

    def save_policy(self):
        torch.save(self.actor.state_dict(), "cartpole_actor.pth")
        torch.save(self.critic.state_dict(), "cartpole_critic.pth")
        print("Models saved.")

    def update(self, states, actions, old_log_probs, advantages, returns):
        states = torch.FloatTensor(np.array(states))
        actions = torch.FloatTensor(np.array(actions)).unsqueeze(-1)
        old_log_probs = torch.FloatTensor(np.array(old_log_probs))
        advantages = torch.FloatTensor(np.array(advantages))
        returns = torch.FloatTensor(np.array(returns))

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        dataset_size = states.shape[0]
        for _ in range(EPOCHS):
            indices = torch.randperm(dataset_size)
            for start in range(0, dataset_size, BATCH_SIZE):
                end = min(start + BATCH_SIZE, dataset_size)
                mb_idx = indices[start:end]

                new_log_probs, entropy = self.actor.evaluate_actions(states[mb_idx], actions[mb_idx])
                ratio = torch.exp(new_log_probs - old_log_probs[mb_idx])
                surr1 = ratio * advantages[mb_idx]
                surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * advantages[mb_idx]
                actor_loss = -torch.min(surr1, surr2).mean()

                values = self.critic(states[mb_idx])
                critic_loss = F.mse_loss(values, returns[mb_idx])

                loss = actor_loss + VALUE_COEF * critic_loss - ENTROPY_COEF * entropy.mean()

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()), 0.5
                )
                self.optimizer.step()

    def train(self, total_timesteps=1_000_000):
        sim = CartPole()
        state = sim.reset()

        states, actions, log_probs = [], [], []
        rewards, dones, values = [], [], []
        scores = []

        timestep = 0
        episode_reward = 0.0
        episode_count = 0

        while timestep < total_timesteps:
            for _ in range(ROLLOUT_STEPS):
                action, log_prob = self.select_action(state)
                next_state, reward, done, _ = sim.step(action)

                states.append(state)
                actions.append(action)
                log_probs.append(log_prob)
                rewards.append(reward)
                dones.append(float(done))

                with torch.no_grad():
                    value = self.critic(torch.FloatTensor(state).unsqueeze(0)).item()
                values.append(value)

                state = next_state
                episode_reward += reward
                timestep += 1

                if done:
                    episode_count += 1
                    scores.append(episode_reward)

                    # Plot only every 50 episodes (biggest CPU speed-up)
                    if episode_count % 100 == 0:
                        print(f"Episode {episode_count:4d} | Reward: {episode_reward:8.2f} | Steps: {timestep}")
                        mean_scores = compute_moving_average(scores, window=20)
                        plot(scores, mean_scores, window=20)
                        if len(scores) >= 20:
                            print(f"Avg last 20: {np.mean(scores[-20:]):.2f}")

                    episode_reward = 0.0
                    state = sim.reset()

                if timestep >= total_timesteps:
                    break

            with torch.no_grad():
                next_value = self.critic(torch.FloatTensor(state).unsqueeze(0)).item() if not done else 0.0

            advantages, returns = self.compute_gae(rewards, values, dones, next_value)
            self.update(states, actions, log_probs, advantages, returns)

            states, actions, log_probs = [], [], []
            rewards, dones, values = [], [], []

        print("Training finished.")


if __name__ == "__main__":
    agent = Agent(state_dim=4, action_dim=1)
    agent.train(total_timesteps= 400_000)
    agent.save_policy()
    test_cartpole_gui(agent.actor, episodes= 20)