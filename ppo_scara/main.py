# ==================== main.py ====================
import torch
from train.agent import Agent
from train.scara_sim import SCARA

if __name__ == "__main__":
    agent = Agent(state_dim=6, action_dim=2)
    agent.train(total_timesteps=300_000)
    agent.save_policy()         