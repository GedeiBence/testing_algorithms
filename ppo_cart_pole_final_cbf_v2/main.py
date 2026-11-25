# ==================== main.py ====================
import torch
from train.agent import Agent
from train.cart_pole_sim import test_cartpole_gui

if __name__ == "__main__":
    agent = Agent(state_dim=7, action_dim=1)

    # Betöltés az 1. fázisból
    #agent.actor.load_state_dict(torch.load("actor_cbf_v3.pth")) eddig ez jó
    #agent.critic.load_state_dict(torch.load("critic_cbf_v3.pth"))
    agent.actor.load_state_dict(torch.load("actor_phase1.pth"))
    agent.critic.load_state_dict(torch.load("critic_phase1.pth"))   
    print("Loaded phase 1 models")

    # 2. fázis: CBF bekapcsolása
    print("Phase 2: Training with CBF")
    agent.train(total_timesteps=500_000, use_cbf=True, cbf_lambda=1.0)

    # Mentés
    torch.save(agent.actor.state_dict(), "actor_cbf_v4.pth")
    torch.save(agent.critic.state_dict(), "critic_cbf_v4.pth")
    print("Saved actor_cbf_v4.pth and critic_cbf_v4.pth")

    test_cartpole_gui(agent.actor, episodes=10)