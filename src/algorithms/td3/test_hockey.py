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

import td3_versions.td3_base as td3_base
import td3_versions.td3_ou as td3_ou
import td3_versions.td3_pink as td3_pink
import td3_versions.td3_pink_noisy as td3_pink_noisy
import td3_versions.td3_noisy as td3_noisy

from neural.feedforward import Feedforward

agent_names = ["td3_ou", "td3_pink", "td3_pink_noisy", "td3_noisy"]
opp_modes = ["prefer_weak", "prefer_strong", "always_weak", "always_strong", "dumb_weak_strong", "dws+self-train"]

def prepare_training(opts, mode = 0, td3_v = 0, existing_model = None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ############## Hyperparameters ##############
    env = h_env.HockeyEnv()
    if mode == 1:
        env = h_env.HockeyEnv(mode=h_env.Mode.TRAIN_SHOOTING)
    elif mode == 2:
        env = h_env.HockeyEnv(mode=h_env.Mode.TRAIN_DEFENSE)
    
    render = False
    opts.log_interval = 20
    opts.save_interval = 100
    opts.noise_decay = 1.0
    opts.training_start = 500
    ############################################# 

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
        agent.restore_state(torch.load(existing_model, map_location=torch.device(device)))

    return env, agent, opts


def save_statistics(rewards, lengths, losses, win_draw_loose, episodes, total_steps, strong_weak_opponent, opts, directory, pkl_name):
    os.makedirs(directory, exist_ok=True)
    pkl_path = os.path.join(directory, pkl_name)
    with open(pkl_path, 'wb') as f:
        pickle.dump({"rewards" : rewards, 
                        "lengths": lengths, 
                        "losses": losses, 
                        "win_draw_loose": win_draw_loose,
                        "i_episode": episodes,
                        "total_steps": total_steps,
                        "strong_weak_opponent": strong_weak_opponent,
                        "max_episodes": opts.max_episodes}, f)
    print("Saved a statistics checkpoint under:", pkl_path)
    return pkl_path

def test(opts, td3_v = 2, mode = 0, weak = False, opp_mode=1, existing_model = None, existing_statistics = None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    env, agent, opts = prepare_training(opts, mode, td3_v, existing_model)

    # logging variables
    rewards = []
    lengths = []
    losses = []
    win_draw_loose = []
    strong_weak_opponent = []

    i_episode = 0
    total_steps = 0

    if mode == 1:
        directory = "shooting"
        print("Starting training in shooting mode, Adversary is weak:", weak)
    elif mode == 2:
        directory = "defense"
        print("Starting training in defense mode, Adversary is weak:", weak)
    else:
        directory = "normal"
        print("Starting training in normal mode, Adversary is weak:", weak)

    opponent_weak = h_env.BasicOpponent(weak=True)
    opponent_strong = h_env.BasicOpponent(weak=False)
    agent.set_inference_mode(True)
    # training loop
    while i_episode < opts.max_episodes:
        ob, _info = env.reset()
        agent.reset()
        obs_agent2 = env.obs_agent_two()

        if weak:
            opponent_type = 0 
            agent_for_this_ep = opponent_weak
        else:
            opponent_type = 1
            agent_for_this_ep = opponent_strong

        done = False
        steps_per_ep = 0
        reward_per_ep = 0

        while not done:
            if opts.human == 1:
                env.render(mode="human")
            
            steps_per_ep += 1
            a = agent.act(ob)

            if mode == 0:
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
    
        i_episode += 1

    print("########## Saving the final model... ##########")
    statistics_dir = save_statistics(rewards, lengths, losses, win_draw_loose, i_episode, total_steps, strong_weak_opponent, opts, f'./test_results/{agent_names[td3_v]}/Hockey/final/{directory}/Opp_mode-{opp_modes[opp_mode]}/Weak-{weak}/', f'TD3_Hockey_final_episode{i_episode}-stat.pkl')
    return statistics_dir

if __name__  == "__main__":

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
                         dest='human',default=0,
                         help='render training process (default %default)')
    optParser.add_option('-z', '--td3',action='store',  type='int',
                         dest='td3V',default=0,
                         help='choose td3 version (default %default)')
    optParser.add_option('-o', '--opp',action='store',  type='int',
                         dest='opp',default=2,
                         help='choose opp mode (default %default)')
    optParser.add_option('-p', '--prev',action='store',  type='int',
                         dest='prev',default=1,
                         help='choose opp mode (default %default)')

    opts, args = optParser.parse_args()

    exisiting_model = 'src/algorithms/td3/models/base/TD3_Hockey.pth'
    tmodel_dir_name = test(opts, td3_v=opts.td3V, weak=False, existing_model=exisiting_model)