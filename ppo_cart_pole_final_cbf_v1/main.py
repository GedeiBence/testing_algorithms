# ==================== main.py ====================
import torch
from train.agent import Agent
from train.cart_pole_sim import test_cartpole_gui

if __name__ == "__main__":
    agent = Agent(state_dim=7, action_dim=1)

    # Load previous model (ultimate3)
    #agent.actor.load_state_dict(torch.load("actor_cbf_v1.pth"))
    #agent.critic.load_state_dict(torch.load("critic_cbf_v1.pth"))
    #print("Loaded actor_cbf_v1.pth and critic_cbf_v1.pth")
    # Continue training
    
    agent.train(total_timesteps=2_900_000)   

    # Save as ultimate4
    torch.save(agent.actor.state_dict(), "actor_cbf_v1.pth")
    torch.save(agent.critic.state_dict(), "critic_cbf_v1.pth")
    print("Saved actor_cbf_v1.pth and critic_cbf_v1.pth")

    test_cartpole_gui(agent.actor, episodes=10)