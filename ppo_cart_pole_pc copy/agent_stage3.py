# ==================== agent_stage3.py ====================
import torch
from agent import Agent
from cart_pole_sim import test_cartpole_gui

if __name__ == "__main__":
    agent = Agent(state_dim=5, action_dim=1)

    # Load stage 2 weights
    agent.actor.load_state_dict(torch.load("actor_stage2.pth", map_location="cpu"))
    agent.critic.load_state_dict(torch.load("critic_stage2.pth", map_location="cpu"))

    print("Loaded stage 2 weights. Starting stage 3 (centering)...")
    agent.train(total_timesteps=900_000)

    # Save stage 3
    torch.save(agent.actor.state_dict(), "actor_stage3.pth")
    torch.save(agent.critic.state_dict(), "critic_stage3.pth")
    print("Saved actor_stage3.pth and critic_stage3.pth")

    test_cartpole_gui(agent.actor, episodes=10)