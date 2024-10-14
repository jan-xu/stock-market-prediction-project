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
from data.dataset import (
    TimeSeriesSliceDataset,
    auto_time_series_cross_validation,
    get_number_of_splits,
)
from data.transforms import (
    FeatureNormalisation,
    apply_relative_change,
    get_relative_change,
)
from models import CNNLSTMModel, LSTMModel
from toy import SyntheticStockData
from ui import (
    LocalLogger,
    LoggerHandler,
    UserPrompt,
    WandbLogger,
    eda_plots,
    plot_pred_vs_gt,
)

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


def val_forward_pass(
    model,
    val_dataset,
    criterion,
    device,
    normaliser,
    val_stock_price,
    loggers,
    val_args,
):

    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    model.eval()

    losses = []
    accuracies = []
    predictions = []
    for inputs, targets in val_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        with torch.no_grad():
            pred = model(inputs, reset_hidden=True)
            loss = criterion(pred, targets)

            losses.append(loss.item())
            accuracies.append(
                (torch.sum(pred.sign() == targets.sign()) / pred.numel()).item()
            )
            predictions.append(pred)

    loggers.add_scalar("val/mean_accuracy_pct", np.mean(accuracies) * 100)
    loggers.add_scalar("val/mean_loss", np.mean(losses))
    loggers.add_scalar("val/validation_dataset_size", len(val_loader.dataset))

    gt_return_data = normaliser.inverse(val_loader.dataset.data)
    pred_return_data = normaliser.inverse(torch.cat(predictions, dim=0).cpu().numpy())

    next_day_return_fig, multi_day_return_fig = plot_pred_vs_gt(
        gt_return_data,
        pred_return_data,
        val_args.val_size,
        val_args.pred_horizon,
        var_type="Return",
    )

    loggers.add_plotly("val_plots/next_day_return_pred", next_day_return_fig)
    if multi_day_return_fig is not None:
        loggers.add_plotly("val_plots/multi_day_return_pred", multi_day_return_fig)

    gt_price_data = np.expand_dims(val_stock_price, -1)
    pred_price_data = np.expand_dims(
        apply_relative_change(
            pred_return_data[..., 0],
            gt_price_data[val_args.look_back - 1 : -val_args.pred_horizon],
        ),
        -1,
    )

    next_day_price_fig, multi_day_price_fig = plot_pred_vs_gt(
        gt_price_data,
        pred_price_data,
        val_args.val_size,
        val_args.pred_horizon,
        var_type="Stock Price",
    )

    loggers.add_plotly("val_plots/next_day_price_pred", next_day_price_fig)
    if multi_day_price_fig is not None:
        loggers.add_plotly("val_plots/multi_day_price_pred", multi_day_price_fig)

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

    loggers = LoggerHandler(
        [LocalLogger(run_folder)] + ([WandbLogger()] if args.wandb else [])
    )

    #######################
    # DATA INITIALISATION #
    #######################

    syn_data_obj = (
        SyntheticStockData()
    )  # TODO: create database with data, both synthetic and real-life
    syn_data_obj()
    df = syn_data_obj.get_data_pandas()

    if args.eda:
        logged_plots = eda_plots(
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
                Path(html_plot).stem
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
    n_splits = get_number_of_splits(data_array, look_back, val_size=args.val_size)
    epochs_per_split = epochs // n_splits
    ts_cv_data = auto_time_series_cross_validation(
        data_array, args, return_indices=True
    )

    ########################
    # MODEL INITIALISATION #
    ########################

    device = args.device
    lr = 0.001

    if args.architecture == "LSTM":
        model = LSTMModel(
            input_size=K,
            hidden_layer_size=hidden_width,
            pred_horizon=pred_horizon,
            recurrent_pred_horizon=recurrent_pred_horizon,
        ).to(
            device
        )  # TODO: add static features as an option to model
    elif args.architecture == "CNN-LSTM":
        model = CNNLSTMModel(
            input_size=K,
            hidden_layer_size=hidden_width,
            conv_channels=32,
            pred_horizon=pred_horizon,
            recurrent_pred_horizon=recurrent_pred_horizon,
        ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[epochs // 10 * 6, epochs // 10 * 8], gamma=0.1
    )

    ##################
    # TRAINING STAGE #
    ##################

    # Get logging frequencies
    train_log_freq = epochs // args.train_logs
    val_log_freq = epochs // args.val_logs

    # Initialise time-series cross-validation iterator
    ts_cv_iter = iter(ts_cv_data)

    with tqdm(total=epochs, desc="Training model...") as pbar:
        for epoch in range(1, epochs + 1):

            if (epoch - 1) % epochs_per_split == 0 and (
                epoch - 1
            ) < n_splits * epochs_per_split:
                (_, val_idx), (train_data, val_data) = next(ts_cv_iter)
                train_data = torch.tensor(train_data, dtype=torch.float32)
                train_dataset = TimeSeriesSliceDataset(
                    train_data, look_back, pred_horizon
                )
                train_loader = DataLoader(
                    train_dataset, batch_size=batch_size, shuffle=True
                )

            loss = train_epoch(model, train_loader, optimizer, criterion, device)
            scheduler.step()

            if epoch % train_log_freq == 0:
                loggers.add_scalar("train/epoch", epoch)
                loggers.add_scalar("train/loss", loss.item())
                loggers.add_scalar("train/lr", optimizer.param_groups[0]["lr"])
                loggers.add_scalar("train/batch_size", batch_size)
                loggers.add_scalar("train/look_back", look_back)
                loggers.add_scalar("train/pred_horizon", pred_horizon)
                loggers.add_scalar("train/train_dataset_size", len(train_dataset))
                loggers.push_scalars(step=epoch)

            if epoch % val_log_freq == 0:
                val_data = torch.tensor(val_data, dtype=torch.float32)
                val_dataset = TimeSeriesSliceDataset(val_data, look_back, pred_horizon)
                val_stock_price = df.loc[val_idx, "Stock Price"].to_numpy()

                accuracies, losses = val_forward_pass(
                    model,
                    val_dataset,
                    criterion,
                    device,
                    normaliser,
                    val_stock_price,
                    loggers,
                    val_args=args,
                )
                loggers.push_scalars(step=epoch)
                loggers.push_plotly(step=epoch)
                print(
                    f"\nAccuracies: {''.join(['+' if a > 0.5 else '-' for a in accuracies])}"
                )
                print(f"Losses: {[round(l, 4) for l in losses]}")

            pbar.update(1)


if __name__ == "__main__":

    args = parse_args()

    # Create run folder
    run_folder = create_run_folder(args.name)

    main(args)

    # TODO: add logging of model weights and gradients
    # TODO: save state dict to disk
