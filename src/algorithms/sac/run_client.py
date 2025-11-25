from __future__ import annotations
import argparse
import uuid
import torch
import numpy as np
import hockey.hockey_env as h_env
from comprl.client import Agent, launch_client
from src.algorithms.sac.agent import SACAgent  

class OwnAgent(Agent):

    def __init__(self, model_path: str) -> None:
        super().__init__()
        self.env = h_env.HockeyEnv()
        obs_size = self.env.observation_space.shape[0]

        self.config = {
            "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            "gamma": 0.95,
            "tau": 0.005,
            "batch_size": 128,
            "auto_entropy": True,
            "alpha": 0.2,
            "actor_lr": 1e-4,
            "critic_lr": 1e-3,
            "alpha_lr": 1e-4,
            "replay_capacity": 100000,
        }

        self.agent = SACAgent(obs_size, self.env.action_space, self.config)
        self.agent.load(model_path)
        print(f"Model loaded from {model_path}")

    def get_step(self, observation: list[float]) -> list[float]:
        return self.agent.act(observation, eval_mode=True).tolist()

    def on_start_game(self, game_id) -> None:
        print(f"Match started (ID: {uuid.UUID(int=int.from_bytes(game_id, byteorder='big'))})")


    def on_end_game(self, result: bool, stats: list[float]) -> None:
        print(f"Match finished: {'won' if result else 'lost'} | Score: {stats[0]} vs. {stats[1]}")

def initialize_agent(agent_args: list[str]) -> Agent:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to saved Model")
    args = parser.parse_args(agent_args)
    return OwnAgent(args.model_path)

def main() -> None:
    launch_client(initialize_agent)

if __name__ == "__main__":
    main()
