# ==================== agent_ultimatepy ====================
import torch
from agent import Agent
from cart_pole_sim import test_cartpole_gui

if __name__ == "__main__":
    agent = Agent(state_dim=5, action_dim=1)

    # Frist try
    agent.train(total_timesteps=2_300_000)

    # Save
    torch.save(agent.actor.state_dict(), "actor_ultimate2.pth")
    torch.save(agent.critic.state_dict(), "critic_ultimat2.pth")
    print("Saved actor_ultimate2.pth and critic_ultimate2.pth")

    test_cartpole_gui(agent.actor, episodes=10)