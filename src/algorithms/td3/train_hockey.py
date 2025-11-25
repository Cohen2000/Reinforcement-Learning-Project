import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import optparse
import pickle
import argparse
import hockey.hockey_env as h_env
import os

import utils.memory as mem

import td3_versions.td3_ou as td3_ou
import td3_versions.td3_pink as td3_pink
import td3_versions.td3_pink_noisy as td3_pink_noisy
import td3_versions.td3_noisy as td3_noisy

from neural.feedforward import Feedforward

agent_names = ["td3_ou", "td3_pink", "td3_pink_noisy", "td3_noisy"]
opp_modes = ["prefer_weak", "prefer_strong", "always_weak", "always_strong", "dumb_weak_strong", "dws+self-train"]

class RandomAgent():
    def act(self, obs):
        return np.random.uniform(-1,1,4)

def prepare_training(opts, mode = 0, td3_v = 0, existing_model = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("GPU not available, using CPU")

    ############## Hyperparameters ##############
    env = h_env.HockeyEnv()
    if mode == 1:
        env = h_env.HockeyEnv(mode=h_env.Mode.TRAIN_SHOOTING)
    elif mode == 2:
        env = h_env.HockeyEnv(mode=h_env.Mode.TRAIN_DEFENSE)
    
    render = False
    opts.log_interval = 50
    opts.save_interval = 2000
    opts.noise_decay = 1.0
    opts.training_start = 500
    #############################################

    print("Random Seed: {}".format(opts.seed))    

    if td3_v == 0:
        agent = td3_ou.TD3(env.observation_space, 
                        env.action_space, 
                        env.num_actions, 
                        env.max_timesteps,
                        eps = opts.eps, 
                        learning_rate_actor = opts.lr_a, 
                        learning_rate_critic = opts.lr_c)
    elif td3_v == 1:
        agent = td3_pink.TD3(env.observation_space, 
                        env.action_space, 
                        env.num_actions, 
                        env.max_timesteps,
                        eps = opts.eps, 
                        learning_rate_actor = opts.lr_a, 
                        learning_rate_critic = opts.lr_c)
    elif td3_v == 2:
        agent = td3_pink_noisy.TD3(env.observation_space, 
                        env.action_space, 
                        env.num_actions, 
                        env.max_timesteps,
                        eps = opts.eps, 
                        learning_rate_actor = opts.lr_a, 
                        learning_rate_critic = opts.lr_c)
    elif td3_v == 3:
        agent = td3_noisy.TD3(env.observation_space, 
                        env.action_space, 
                        env.num_actions, 
                        env.max_timesteps,
                        eps = opts.eps, 
                        learning_rate_actor = opts.lr_a, 
                        learning_rate_critic = opts.lr_c)
            

    if existing_model is not None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        agent.restore_state(torch.load(existing_model, map_location=torch.device(device)))

    return env, agent, opts

def save_model(agent, directory, model_name):
    os.makedirs(directory, exist_ok=True)
    model_path = os.path.join(directory, model_name)
    torch.save(agent.state(), model_path)
    print("Saved a model checkpoint under:", model_path)
    return model_path

def save_statistics(rewards, lengths, losses, win_draw_loose, episodes, total_steps, strong_weak_opponent, opts, directory, pkl_name):
    os.makedirs(directory, exist_ok=True)
    pkl_path = os.path.join(directory, pkl_name)
    with open(pkl_path, 'wb') as f:
        pickle.dump({"rewards" : rewards, 
                        "lengths": lengths, 
                        "losses": losses, 
                        "win_draw_loose": win_draw_loose,
                        "eps": opts.eps, 
                        "lr_a": opts.lr_a, 
                        "lr_c": opts.lr_c, 
                        "i_episode": episodes,
                        "total_steps": total_steps,
                        "strong_weak_opponent": strong_weak_opponent,
                        "max_episodes": opts.max_episodes}, f)
    print("Saved a statistics checkpoint under:", pkl_path)
    return pkl_path

def load_statistics(pkl_path):
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"No statistics checkpoint found at: {pkl_path}")
    
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    print("Loaded statistics checkpoint from:", pkl_path)
    return data

def train(opts, td3_v = 2, train_mode = 0, opp_mode = 0, existing_model = None, existing_statistics = None):
    env, agent, opts = prepare_training(opts, train_mode, td3_v, existing_model)
    ## Load opp
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    random_seed = opts.seed

    own_agent_opp = td3_pink_noisy.TD3(env.observation_space, 
                        env.action_space, 
                        env.num_actions, 
                        env.max_timesteps,
                        eps = opts.eps, 
                        learning_rate_actor = opts.lr_a, 
                        learning_rate_critic = opts.lr_c)
    # own_agent_opp.restore_state(torch.load(f'', map_location=torch.device(device)))

    # logging variables
    rewards = []
    lengths = []
    losses = []
    win_draw_loose = []
    strong_weak_opponent = []

    previous_model_names = []

    i_episode = 0
    total_steps = 0
    
    steps_until_new_save = 0
    if existing_statistics is not None:
        data = load_statistics(existing_statistics)
        rewards.extend(data['rewards'])
        lengths.extend(data['lengths'])
        losses = data['losses']
        win_draw_loose.extend(data['win_draw_loose'])
        strong_weak_opponent.extend(data['strong_weak_opponent'])
        i_episode = data['i_episode']
        total_steps = data['total_steps']
        steps_until_new_save = opts.save_interval


    if train_mode == 1:
        directory = "shooting"
        print("Starting training in shooting mode, Adversary is weak:", opp_mode)
    elif train_mode == 2:
        directory = "defense"
        print("Starting training in defense mode, Adversary is weak:", opp_mode)
    else:
        directory = "normal"
        print("Starting training in normal mode, Adversary is weak:", opp_mode)

    opponent_weak = h_env.BasicOpponent(weak=True)
    opponent_strong = h_env.BasicOpponent(weak=False)
    # training loop
    while i_episode < opts.max_episodes:
        ob, _info = env.reset()

        agent.reset()
        obs_agent2 = env.obs_agent_two()

        if opp_mode == 0:
            rng = np.random.uniform(low=0.0, high=1.0)
            if rng < (3.0/4.0):
                opponent_type = 0 
                agent_for_this_ep = opponent_weak
            else:
                opponent_type = 1
                agent_for_this_ep = opponent_strong
        elif opp_mode == 1:
            rng = np.random.uniform(low=0.0, high=1.0)
            if rng < (1.0/3.0):
                opponent_type = 0 
                agent_for_this_ep = opponent_weak
            else:
                opponent_type = 1
                agent_for_this_ep = opponent_strong
        elif opp_mode == 2:
            opponent_type = 0
            agent_for_this_ep = opponent_weak
        elif opp_mode == 3:
            opponent_type = 1
            agent_for_this_ep = opponent_strong
        elif opp_mode == 4:
            rng = np.random.uniform(low=0.0, high=1.0)
            if rng < (1.5/7.0):
                opponent_type = 2
                agent_for_this_ep = RandomAgent()
            elif rng > (1.5/7.0) and rng < (4.0/7.0):
                opponent_type = 0
                agent_for_this_ep = opponent_weak
            elif rng > (4.0/8.0):
                opponent_type = 1
                agent_for_this_ep = opponent_strong
        elif opp_mode == 5:
            rng = np.random.uniform(low=0.0, high=1.0)
            if rng < (1.5/12.0):
                opponent_type = 2
                agent_for_this_ep = RandomAgent()
            elif rng > (1.5/12.0) and rng < (5.0/12.0):
                opponent_type = 0
                agent_for_this_ep = opponent_weak
            elif rng > (5.0/12.0) and rng < (9.0/12.0):
                opponent_type = 1
                agent_for_this_ep = opponent_strong
            else:
                opponent_type = 3
                agent_for_this_ep = own_agent_opp


        done = False
        steps_per_ep = 0
        reward_per_ep = 0

        while not done:
            if opts.human == 1:
                env.render(mode="human")
            
            steps_per_ep += 1
            a = agent.act(ob)

            if train_mode == 0:
                a2 = agent_for_this_ep.act(obs_agent2)
            else:
                a2 = [0,0,0,0] 
            
            (ob_new, reward, done, trunc, _info) = env.step(np.hstack([a,a2]))
            agent.store_transition((ob, a, reward, ob_new, done))
            
            ob=ob_new
            obs_agent2 = env.obs_agent_two()

            if done or trunc: 
                win_draw_loose.append(_info['winner'])
                break

            total_steps += 1
            reward_per_ep += reward

        if total_steps > opts.training_start:
            critic_losses_per_ep, actor_losses_per_ep = agent.train()
            losses.append((critic_losses_per_ep, actor_losses_per_ep))
        
        rewards.append(reward_per_ep)
        lengths.append(steps_per_ep) 
        strong_weak_opponent.append(opponent_type)

        if i_episode % opts.log_interval == 0:
            win_draw_loose_filtered = np.array(win_draw_loose[-opts.log_interval:])
            num_wins = (win_draw_loose_filtered == 1).sum()
            
            win_draw_loose_without_draws = win_draw_loose_filtered[win_draw_loose_filtered != 0]
            num_wins_without_draws = (win_draw_loose_without_draws == 1).sum()
            
            avg_winrate = num_wins / opts.log_interval
            avg_winrate_without_draws = num_wins_without_draws / len(win_draw_loose_without_draws)

            avg_reward = np.mean(rewards[-opts.log_interval:])
            avg_length = int(np.mean(lengths[-opts.log_interval:]))
            print('Episode {} \t avg length: {} \t reward: {} \t Total episodes: {}'.format(steps_per_ep, avg_length, avg_reward, i_episode))
            print('Num wins {} \t of: {} \t Winrate: {}'.format(num_wins, opts.log_interval, avg_winrate))
            print('Num wins w/o draws {} \t of: {} \t Winrate w/o draws: {}'.format(num_wins_without_draws, len(win_draw_loose_without_draws), avg_winrate_without_draws))
    
        if i_episode % opts.save_interval == 0 and steps_until_new_save == 0:
            print("########## Saving a checkpoint... ##########")
            model_directory_name = save_model(agent, f'./models/{agent_names[opts.td3V]}/Hockey/intermediate/{directory}/Opp_mode-{opp_modes[opp_mode]}/prev-{opts.prev}/new/', f'TD3_Hockey_episode{i_episode}.pth')
            save_statistics(rewards, lengths, losses, win_draw_loose, i_episode, total_steps, strong_weak_opponent, opts, f'./results/{agent_names[opts.td3V]}/Hockey/intermediate/{directory}/Opp_mode-{opp_modes[opp_mode]}/prev-{opts.prev}/', f'TD3_Hockey-episode{i_episode}-stat.pkl')

        if i_episode % 50:
            agent._eps = agent._eps * opts.noise_decay

        i_episode += 1
        random_seed += 1
        steps_until_new_save = max(steps_until_new_save-1, 0)

    print("########## Saving the final model... ##########")
    model_directory_name = save_model(agent, f'./models/{agent_names[opts.td3V]}/Hockey/final/{directory}/Opp_mode-{opp_modes[opp_mode]}/prev-{opts.prev}/new/', f'TD3_Hockey_final_episode{i_episode}.pth')
    save_statistics(rewards, lengths, losses, win_draw_loose, i_episode, total_steps, strong_weak_opponent, opts, f'./results/{agent_names[opts.td3V]}/Hockey/final/{directory}/Opp_mode-{opp_modes[opp_mode]}/prev-{opts.prev}/', f'TD3_Hockey_final_episode{i_episode}-stat.pkl')
    return model_directory_name

if __name__ == '__main__':
    # Seed Everything
    optParser = optparse.OptionParser()
    optParser.add_option('-n', '--eps',action='store',  type='float',
                         dest='eps',default=0.1,
                         help='Policy noise (default %default)')
    optParser.add_option('-a', '--lr_a',action='store',  type='float',
                         dest='lr_a',default=1e-4,
                         help='learning rate for actor/policy (default %default)')
    optParser.add_option('-c', '--lr_c',action='store',  type='float',
                         dest='lr_c',default=1e-4,
                         help='learning rate for critic (default %default)')
    optParser.add_option('-m', '--maxepisodes',action='store',  type='int',
                         dest='max_episodes',default=30000,
                         help='number of episodes (default %default)')
    optParser.add_option('-s', '--seed',action='store',  type='int',
                         dest='seed',default=1,
                         help='random seed (default %default)')
    optParser.add_option('-t', '--human',action='store',  type='int',
                         dest='human',default=1,
                         help='render training process (default %default)')
    optParser.add_option('-z', '--td3',action='store',  type='int',
                         dest='td3V',default=2,
                         help='choose td3 version (default %default)')
    optParser.add_option('-o', '--opp',action='store',  type='int',
                         dest='opp',default=2,
                         help='choose opp mode (default %default)')
    optParser.add_option('-p', '--prev',action='store',  type='int',
                         dest='prev',default=1,
                         help='choose opp mode (default %default)')

    opts, args = optParser.parse_args()

    tmodel_dir_name = train(opts, td3_v=opts.td3V, train_mode=0, opp_mode=opts.opp)