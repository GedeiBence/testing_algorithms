# ==================== continue_training.py ====================
import torch
from agent import Agent
from pendulum_sim import Pendulum

if __name__ == "__main__":
    agent = Agent()

    # Betöltés a korábbi modellekből
    try:
        agent.actor.load_state_dict(torch.load("pendulum_actor.pth", map_location="cpu"))
        agent.critic.load_state_dict(torch.load("pendulum_critic.pth", map_location="cpu"))
        print("Loaded existing models: pendulum_actor.pth, pendulum_critic.pth")
    except FileNotFoundError:
        print("No existing models found, starting from scratch.")

    # Továbbtanítás
    agent.train(total_timesteps=1_000_000)

    # Mentés új néven
    torch.save(agent.actor.state_dict(), "pendulum_actor_v2.pth")
    torch.save(agent.critic.state_dict(), "pendulum_critic_v2.pth")
    print("Saved continued models: pendulum_actor_v2.pth, pendulum_critic_v2.pth")