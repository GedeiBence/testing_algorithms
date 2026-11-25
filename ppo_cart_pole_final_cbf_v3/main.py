# ==================== main.py ====================
import torch
from train.agent import Agent
from train.cart_pole_sim import test_cartpole_gui

if __name__ == "__main__":
    agent = Agent(state_dim=7, action_dim=1)

    # ========== 1. FÁZIS: CBF nélkül (stabilizálás) ==========
    #print("Phase 1: Stabilizing without CBF")
    #agent.train(total_timesteps=1_200_000, use_cbf=False, cbf_lambda=0)

    #torch.save(agent.actor.state_dict(), "actor_no_cbf_ulti.pth")
    #torch.save(agent.critic.state_dict(), "critic_no_cbf_ulti.pth")
    #print("Saved phase 1 models")

    # ========== 2. FÁZIS: CBF-fel a kezdettől ==========
    # agent.actor.load_state_dict(torch.load("actor_cbf_from_start.pth"))
    # agent.critic.load_state_dict(torch.load("critic_cbf_from_start.pth"))
    print("Training with CBF from the start")
    agent.train(total_timesteps=1_500_000, use_cbf=True, cbf_lambda=1.0)

    # Mentés
    torch.save(agent.actor.state_dict(), "actor_cbf_from_start_v4.pth")
    torch.save(agent.critic.state_dict(), "critic_cbf_from_start_v4.pth")
    print("Saved actor_cbf_from_start_v4.pth and critic_cbf_from_start_v4.pth")

    test_cartpole_gui(agent.actor, episodes=10)