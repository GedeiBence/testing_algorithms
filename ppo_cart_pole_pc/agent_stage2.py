# ==================== agent_stage2.py ====================
import torch
from agent import Agent
from cart_pole_sim import test_cartpole_gui

if __name__ == "__main__":
    agent = Agent(state_dim=5, action_dim=1)

    # Load stage 1 weights
    agent.actor.load_state_dict(torch.load("cartpole_actor.pth", map_location="cpu"))
    # agent.critic.load_state_dict(torch.load("cartpole_critic.pth", map_location="cpu"))  # if you saved it

    print("Loaded stage 1 weights. Starting stage 2...")
    agent.train(total_timesteps=1_000_000)

    # Save stage 2
    torch.save(agent.actor.state_dict(), "actor_stage2.pth")
    torch.save(agent.critic.state_dict(), "critic_stage2.pth")
    print("Saved actor_stage2.pth and critic_stage2.pth")

    test_cartpole_gui(agent.actor, episodes=10)