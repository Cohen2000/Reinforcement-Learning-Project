import torch
import numpy as np
import gymnasium as gym
from torch.distributions import Categorical
from .rnd_agent import PPO, Memory
import hockey.hockey_env as h_env 

#Testing implementation for old training process

def test(env, episodes, env_name):
    print(f"Testing RNDAgent for {episodes} episodes.")
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    memory = Memory()

    agent = PPO(state_dim=state_dim, 
             action_dim=action_dim,
             n_latent_var=64,
             lr=0.00000,        
             betas=(0.9, 0.999),
             gamma=0.99,
             K_epochs=10,
             eps_clip=0.15,
             rnd_beta=0.1)

    model_path = f"models/rnd/rnd_{env_name}.pt"
    try:
        checkpoint = torch.load(model_path)
        agent.policy.load_state_dict(checkpoint['policy_state_dict'])
        agent.policy_old.load_state_dict(checkpoint['policy_state_dict'])
        agent.policy_old.eval()
        agent.policy.eval()
    except FileNotFoundError:
        print(f"Model file not found at {model_path}.")
        return

    total_rewards = []
    player2_weak = h_env.BasicOpponent(weak=True)
    player2_strong = h_env.BasicOpponent(weak=False)

    for episode in range(episodes):
        player2 = player2_weak if episode % 3 == 0 else player2_strong
        state, _ = env.reset()
        obs_agent2 = env.obs_agent_two()
        episode_reward = 0
        done = False
        print(f"Episode {episode+1}: Starting test...")
        while not done:
            if env_name == "Hockey-v0" or env_name == "Hockey-One-v0":
                env.render()
            action = agent.policy.act(state, memory)
            action_player_2 = player2.act(obs_agent2)
            discrete_action = env.discrete_to_continous_action(action)
            full_action = np.hstack([discrete_action, action_player_2])
            state, reward, done, _, _ = env.step(full_action)
            
            obs_agent2 = env.obs_agent_two()

            episode_reward += reward

        total_rewards.append(episode_reward)
        print(f"Episode {episode+1}: Total Reward = {episode_reward}")

    env.close()
    avg_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    print(f"\nTest Results over {episodes} episodes:")
    print(f"Average Reward: {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"Min Reward: {min(total_rewards):.2f}")
    print(f"Max Reward: {max(total_rewards):.2f}")