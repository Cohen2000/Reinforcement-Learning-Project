import argparse
from src.algorithms.sac.experiment_train import run_training

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', type=str, choices=["auto", "0", "0.2", "0.4"], required=True,
                        help="Alpha value (auto, 0, 0.2, or 0.4)")
    parser.add_argument('--gamma', type=float, choices=[0.90, 0.95, 0.99], required=True,
                        help="Gamma value")
    args = parser.parse_args()
    
    run_training(args.alpha, args.gamma)
