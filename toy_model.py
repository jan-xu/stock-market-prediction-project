from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from arguments import parse_args
from config import create_run_folder, wandb_config
from data.dataset import (TimeSeriesSliceDataset,
                          auto_time_series_cross_validation,
                          get_number_of_splits)
from data.transforms import FeatureNormalisation, get_relative_change
from models import CNNLSTMModel, LSTMModel
from toy import SyntheticStockData
from ui import UserPrompt, run_eda

np.random.seed(0)
torch.manual_seed(0)


def train_epoch(model, train_loader, optimizer, criterion, device):

    model.train()

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        pred = model(inputs, reset_hidden=True)
        loss = criterion(pred, targets)
        loss.backward()
        optimizer.step()

    return loss


def val_forward_pass(model, val_loader, criterion, device):

    model.eval()

    accuracies = []
    losses = []
    for inputs, targets in val_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        with torch.no_grad():
            pred = model(inputs, reset_hidden=True)
            loss = criterion(pred, targets)
            losses.append(loss.item())
            accuracies.append((torch.sum(pred.sign() == targets.sign()) / pred.numel()).item())

    return accuracies, losses


def main(args):

    # Set wandb configuration
    if args.wandb:
        wandb_config(args)

    epochs = args.epochs
    batch_size = args.batch_size
    look_back = args.look_back
    pred_horizon = args.pred_horizon
    hidden_width = args.hidden_width
    recurrent_pred_horizon = args.recurrent_pred_horizon

    #######################
    # DATA INITIALISATION #
    #######################

    syn_data_obj = (
        SyntheticStockData()
    )  # TODO: create database with data, both synthetic and real-life
    syn_data_obj()
    df = syn_data_obj.get_data_pandas()

    if args.eda:
        logged_plots = run_eda(
            df,
            label="SYN_DATA",
            value_col="Stock Price",
            add_ma=True,
            run_folder=run_folder,
        )
        prompt_answer = UserPrompt.prompt_continue()
        prompt_answer()
        if args.wandb:
            # TODO: See if we can simplify this
            FIGURE_LIST = [
                wandb.Html(str(html_plot))
                for html_plot in logged_plots
                if html_plot.endswith(".html")
            ]
            FIGURE_COLUMNS = [
                Path(html_plot).name
                for html_plot in logged_plots
                if html_plot.endswith(".html")
            ]
            FIGURE_TABLE = wandb.Table(columns=FIGURE_COLUMNS)
            FIGURE_TABLE.add_data(*FIGURE_LIST)
            wandb.log({"data_analysis": FIGURE_TABLE}, step=0)

    # Feature engineering
    df["Return"] = get_relative_change(df["Stock Price"])

    # Initialise data normalisation
    mean, std = df["Return"].mean(), df["Return"].std()
    normaliser = FeatureNormalisation(mean, std)

    # Specify features for the data and convert to NumPy array
    features = ["Return"]
    K = len(features)
    data_array = df[features].to_numpy()
    data_array = normaliser.forward(data_array)  # Apply normalisation

    # Time-series cross-validation on dataset
    n_splits = get_number_of_splits(data_array, look_back)
    epochs_per_split = epochs // n_splits
    ts_cv_data = auto_time_series_cross_validation(data_array, args)

    ########################
    # MODEL INITIALISATION #
    ########################

    device = args.device
    lr = 0.001

    if args.architecture == "LSTM":
        model = LSTMModel(input_size=K, hidden_layer_size=hidden_width, pred_horizon=pred_horizon, recurrent_pred_horizon=recurrent_pred_horizon).to(
            device
        )  # TODO: add static features as an option to model
    elif args.architecture == "CNN-LSTM":
        model = CNNLSTMModel(
            input_size=K, hidden_layer_size=hidden_width, conv_channels=32, pred_horizon=pred_horizon, recurrent_pred_horizon=recurrent_pred_horizon
        ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[epochs // 10 * 6, epochs // 10 * 8], gamma=0.1
    )

    ##################
    # TRAINING STAGE #
    ##################

    # Initialise time-series cross-validation iterator
    ts_cv_iter = iter(ts_cv_data)

    for epoch in tqdm(range(1, epochs + 1)):

        if (epoch - 1) % epochs_per_split == 0 and (
            epoch - 1
        ) < n_splits * epochs_per_split:
            train_data, val_data = next(ts_cv_iter)
            train_data = torch.tensor(train_data, dtype=torch.float32)
            val_data = torch.tensor(val_data, dtype=torch.float32)

            train_dataset = TimeSeriesSliceDataset(train_data, look_back, pred_horizon)
            val_dataset = TimeSeriesSliceDataset(val_data, look_back, pred_horizon)
            print(
                f"Initialising new training split with {len(train_dataset)} training samples and {len(val_dataset)} validation samples."
            )
            train_loader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True
            )

        loss = train_epoch(model, train_loader, optimizer, criterion, device)
        scheduler.step()

        if epoch % 50 == 0:
            if args.wandb:
                wandb.log(
                    {"loss": loss.item(), "lr": optimizer.param_groups[0]["lr"]},
                    step=epoch,
                )
            else:
                print(f"Epoch {epoch}, Loss: {loss.item()}")

        if epoch % 100 == 0:
            val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
            accuracies, losses = val_forward_pass(model, val_loader, criterion, device)
            if args.wandb:
                wandb.log(
                    {
                        "val_accuracy": np.mean(accuracies) * 100,
                        "val_loss": np.mean(losses),
                    },
                    step=epoch,
                )
            else:
                print(f"Validation accuracy: {np.mean(accuracies) * 100:.4f}%")
                print(
                    f"Accuracies: {''.join(['+' if a == 1 else '-' for a in accuracies])}"
                )
                print(f"Losses: {[round(l, 4) for l in losses]}")
                print(f"Mean loss: {np.mean(losses)}")


if __name__ == "__main__":

    args = parse_args()

    # Create run folder
    run_folder = create_run_folder(args.name)

    main(args)

    # TODO: create a logger class that keeps track of all scalar values to log to Wandb

    # Training loop
    # Log following (running) training variables:
    # - look_back
    # - pred_horizon
    # - learning_rate
    # - dataset_size
    # - batch_size

    # TODO: add logging of model weights and gradients
    # TODO: save state dict to disk
    # TODO: add logging of prediction curve
