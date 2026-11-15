# ==================== train_stage4.py ====================
import torch
from cart_pole_sim import CartPole, test_cartpole_gui
from agent import Agent

if __name__ == "__main__":
    agent = Agent()

    agent.actor.load_state_dict(torch.load("actor_stage3.pth", map_location="cpu"))
    agent.critic.load_state_dict(torch.load("critic_stage3.pth", map_location="cpu"))

    print("Stage 4: continuing from stage 3 with wider reset")
    agent.train(total_timesteps=2_000_000)

    torch.save(agent.actor.state_dict(), "actor_stage4.pth")
    torch.save(agent.critic.state_dict(), "critic_stage4.pth")
    print("Saved actor_stage4.pth and critic_stage4.pth")

    test_cartpole_gui(agent.actor, episodes=10)