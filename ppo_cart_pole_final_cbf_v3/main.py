# ==================== main.py ====================
import torch
from train.agent import Agent
from train.cart_pole_sim import test_cartpole_gui

if __name__ == "__main__":
    agent = Agent(state_dim=7, action_dim=1)

    # ========== 1. FÁZIS: CBF nélkül (stabilizálás) ==========
    print("Phase 1: Stabilizing without CBF")
    agent.train(total_timesteps=3_000_000, use_cbf=True, cbf_lambda=0.03)

    torch.save(agent.actor.state_dict(), "actor_no_cbf_ulti.pth")
    torch.save(agent.critic.state_dict(), "critic_no_cbf_ulti.pth")
    print("Saved phase 1 models")

    # ========== 2. FÁZIS: CBF bekapcsolása ==========
    #print("Phase 2: Adapting to CBF")
    #agent.train(total_timesteps=1_000_000, use_cbf=True, cbf_lambda=1.2)


     # ========== 3. FÁZIS: CBF bekapcsolása ==========
    #print("Phase 3: Adapting to CBF")
    #agent.train(total_timesteps=1_000_000, use_cbf=True, cbf_lambda=3.2)

     # ========== 4. FÁZIS: CBF bekapcsolása ==========
    #print("Phase 4: Adapting to CBF")
    #agent.train(total_timesteps=1_000_000, use_cbf=True, cbf_lambda=5.2)

    #torch.save(agent.actor.state_dict(), "actor_cbf_v3.pth")
    #torch.save(agent.critic.state_dict(), "critic_cbf_v3.pth")
    #print("Saved actor_cbf_v3.pth and critic_cbf_v3.pth")

    test_cartpole_gui(agent.actor, episodes=10)