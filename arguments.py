import torch
import argparse
from warnings import warn
from datetime import datetime

DEFAULT_ARGS = {
    "architecture": "CNN-LSTM",
    "ticker": "SP500",
    "epochs": 1000,
    "batch_size": 64,
    "look_back": 64,
    "pred_horizon": 1,
    "hidden_width": 128,
    "wandb": False,
}

def parse_args():

    parser = argparse.ArgumentParser(description='Training script for stock market prediction')

    parser.add_argument('-p', '--project', type=str, default='stock-prediction-playground', help='Wandb project name')
    parser.add_argument('-n', '--name', type=str, default=None, help='Run name (default: "RUN--<TIMESTAMP>")')
    parser.add_argument('-d', '--device', type=str.lower, default='cpu', choices=['cpu', 'cuda'], help='Device to run the model on')
    parser.add_argument('-a', '--architecture', type=str, default=DEFAULT_ARGS['architecture'], choices=["LSTM", "CNN-LSTM"], help='Model architecture')
    parser.add_argument('-t', '--ticker', type=str, default=DEFAULT_ARGS['ticker'], help='Stock ticker name')
    parser.add_argument('-e', '--epochs', type=int, default=DEFAULT_ARGS['epochs'], help='Number of epochs')
    parser.add_argument('-b', '--batch_size', type=int, default=DEFAULT_ARGS['batch_size'], help='Batch size')
    parser.add_argument('-l', '--look-back', type=int, default=DEFAULT_ARGS['look_back'], help='Look-back length')
    parser.add_argument('-ph', '--pred-horizon', type=int, default=DEFAULT_ARGS['pred_horizon'], help='Prediction horizon size')
    parser.add_argument('-hw', '--hidden-width', type=int, default=DEFAULT_ARGS['hidden_width'], help='Hidden width')
    parser.add_argument('--eda', action='store_true', help='Perform exploratory data analysis before experiment starts')
    parser.add_argument('--wandb', action='store_true', help='Log to wandb')
    parser.add_argument('--ignore-timestamp', action='store_true', help='Ignore timestamp in run name')

    args = parser.parse_args()

    time_stamp_suffix = f"--{datetime.now().strftime('%Y-%m-%d_%H%M%S')}"
    if args.name is not None:
        args.name = args.name + ("" if args.ignore_timestamp else time_stamp_suffix)
    else:
        print(f"Run name not provided. Using default: RUN{time_stamp_suffix}")
        args.name = f"RUN{time_stamp_suffix}"

    if args.device == 'cuda':
        if not torch.cuda.is_available():
            warn("CUDA is not available. Using CPU instead.")
            args.device = torch.device('cpu')
        else:
            args.device = torch.device('cuda:0')
    else:
        args.device = torch.device('cpu')

    print(f"Run configuration:")
    print(f"  - Project: {args.project}")
    print(f"  - Run name: {args.name}")
    print(f"  - Device: {args.device.type}")
    print(f"  - Architecture: {args.architecture}")
    print(f"  - Stock ticker: {args.ticker}")
    print(f"  - Epochs: {args.epochs}")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Look-back: {args.look_back}")
    print(f"  - Prediction horizon: {args.pred_horizon}")
    print(f"  - Hidden width: {args.hidden_width}")

    return args