from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from arguments import parse_args
from config import create_run_folder, stock_mapping, wandb_config
from data.database import DATABASE_PATH, StockDatabase
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


def train_epoch(model, train_loader, optimizer, criterion, device, loggers):

    model.train()

    epoch_loss = 0
    epoch_baseline_loss = 0
    epoch_accuracy = 0

    for inputs, targets in train_loader:
        batch_size = inputs.shape[0]

        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        pred = model(inputs, reset_hidden=True)

        loss = criterion(pred, targets)
        loss.backward()
        optimizer.step()

        # Auxiliary / monitoring metrics
        baseline_loss = criterion(torch.zeros_like(pred), targets)
        accuracy = torch.sum(pred.sign() == targets.sign()) / pred.numel()

        epoch_loss += loss.item() * batch_size
        epoch_baseline_loss += baseline_loss.item() * batch_size
        epoch_accuracy += accuracy.item() * batch_size

    N = len(train_loader.dataset)
    loggers.add_scalar("train/loss", epoch_loss / N)
    loggers.add_scalar("train/baseline_loss", epoch_baseline_loss / N)
    loggers.add_scalar("train/accuracy_pct", epoch_accuracy / N * 100)
    loggers.add_scalar(
        "train/loss_gain", max(epoch_loss, 1e-8) / max(epoch_baseline_loss, 1e-8) - 1
    )


def val_forward_pass(
    model,
    val_data,
    criterion,
    device,
    normaliser,
    val_stock_price,
    loggers,
    val_args,
):
    # Initialise validation dataset and dataloader
    look_back = val_args.look_back
    pred_horizon = val_args.pred_horizon

    val_dataset = TimeSeriesSliceDataset(val_data, look_back, pred_horizon)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # Get ground truth data for plotting
    gt_return_data = normaliser.inverse(val_loader.dataset.data)
    gt_price_data = np.expand_dims(val_stock_price, -1)

    model.eval()

    return_predictions = []
    stock_price_predictions = []
    for i, (inputs, targets) in enumerate(val_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        with torch.no_grad():
            pred = model(inputs, reset_hidden=True)
            loss = criterion(pred, targets)
            baseline_loss = criterion(torch.zeros_like(pred), targets)
            accuracy = torch.sum(pred.sign() == targets.sign()) / pred.numel()

            loggers.add_scalar("val/loss", loss.item())
            loggers.add_scalar("val/baseline_loss", baseline_loss.item())
            loggers.add_scalar("val/accuracy_pct", accuracy.item() * 100)

            pred_return = normaliser.inverse(pred)
            return_predictions.append(pred_return)

            prev_stock_price = val_stock_price[val_args.look_back - 1 + i]
            real_stock_price = val_stock_price[
                val_args.look_back + i : val_args.look_back + i + val_args.pred_horizon
            ]
            real_stock_price = (
                torch.tensor(real_stock_price, device=device).unsqueeze(0).unsqueeze(-1)
            )

            pred_stock_price = apply_relative_change(
                pred_return[..., 0], prev_stock_price
            ).unsqueeze(-1)
            stock_price_predictions.append(pred_stock_price)

            rmse = torch.sqrt(criterion(pred_stock_price, real_stock_price))
            mae = torch.mean(torch.abs(pred_stock_price - real_stock_price))
            mape = torch.mean(
                torch.abs(pred_stock_price - real_stock_price) / real_stock_price
            )

            loggers.add_scalar("val/stock_price_rmse", rmse.item())
            loggers.add_scalar("val/stock_price_mae", mae.item())
            loggers.add_scalar("val/stock_price_mape", mape.item() * 100)

    loggers.log_apply_scalars(
        "val/loss_gain",
        lambda loss, baseline_loss: max(loss, 1e-8) / max(baseline_loss, 1e-8) - 1,
        "val/loss",
        "val/baseline_loss",
    )

    # Next-day predictions only
    if pred_horizon > 1:
        next_day_val_dataset = TimeSeriesSliceDataset(val_data, look_back, 1)
        next_day_val_loader = DataLoader(
            next_day_val_dataset, batch_size=1, shuffle=False
        )

        next_day_return_predictions = []
        next_day_stock_price_predictions = []
        for i, (inputs, targets) in enumerate(next_day_val_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            with torch.no_grad():
                pred = model(inputs, reset_hidden=True)
                pred = pred[:, :1]  # Select only next-day prediction
                loss = criterion(pred, targets)
                baseline_loss = criterion(torch.zeros_like(pred), targets)
                accuracy = torch.sum(pred.sign() == targets.sign()) / pred.numel()

                loggers.add_scalar("val_next_day/loss", loss.item())
                loggers.add_scalar("val_next_day/baseline_loss", baseline_loss.item())
                loggers.add_scalar("val_next_day/accuracy_pct", accuracy.item() * 100)

                pred_return = normaliser.inverse(pred)
                next_day_return_predictions.append(pred_return)

                prev_stock_price = val_stock_price[val_args.look_back - 1 + i]
                real_stock_price = val_stock_price[
                    val_args.look_back
                    + i : val_args.look_back
                    + i
                    + val_args.pred_horizon
                ]
                real_stock_price = (
                    torch.tensor(real_stock_price, device=device)
                    .unsqueeze(0)
                    .unsqueeze(-1)
                )

                pred_stock_price = apply_relative_change(
                    pred_return[..., 0], prev_stock_price
                ).unsqueeze(-1)
                next_day_stock_price_predictions.append(pred_stock_price)

                rmse = torch.sqrt(criterion(pred_stock_price, real_stock_price))
                mae = torch.mean(torch.abs(pred_stock_price - real_stock_price))
                mape = torch.mean(
                    torch.abs(pred_stock_price - real_stock_price) / real_stock_price
                )

                loggers.add_scalar("val_next_day/stock_price_rmse", rmse.item())
                loggers.add_scalar("val_next_day/stock_price_mae", mae.item())
                loggers.add_scalar("val_next_day/stock_price_mape", mape.item() * 100)

        loggers.log_apply_scalars(
            "val_next_day/loss_gain",
            lambda loss, baseline_loss: max(loss, 1e-8) / max(baseline_loss, 1e-8) - 1,
            "val_next_day/loss",
            "val_next_day/baseline_loss",
        )
    else:
        loggers.log_apply_scalars("val_next_day/loss", lambda x: x, "val/loss")
        loggers.log_apply_scalars(
            "val_next_day/baseline_loss", lambda x: x, "val/baseline_loss"
        )
        loggers.log_apply_scalars(
            "val_next_day/accuracy_pct", lambda x: x, "val/accuracy_pct"
        )
        loggers.log_apply_scalars(
            "val_next_day/stock_price_rmse", lambda x: x, "val/stock_price_rmse"
        )
        loggers.log_apply_scalars(
            "val_next_day/stock_price_mae", lambda x: x, "val/stock_price_mae"
        )
        loggers.log_apply_scalars(
            "val_next_day/stock_price_mape", lambda x: x, "val/stock_price_mape"
        )
        loggers.log_apply_scalars(
            "val_next_day/loss_gain", lambda x: x, "val/loss_gain"
        )
        next_day_return_predictions = return_predictions
        next_day_stock_price_predictions = stock_price_predictions

    loggers.add_scalar("val/validation_dataset_size", len(val_loader.dataset))

    pred_return_data = torch.cat(return_predictions, dim=0).cpu().numpy()
    pred_price_data = torch.cat(stock_price_predictions, dim=0).cpu().numpy()

    next_day_pred_return_data = (
        torch.cat(next_day_return_predictions, dim=0).cpu().numpy()
    )
    next_day_pred_price_data = (
        torch.cat(next_day_stock_price_predictions, dim=0).cpu().numpy()
    )

    next_day_return_fig, multi_day_return_fig = plot_pred_vs_gt(
        gt_return_data,
        pred_return_data,
        next_day_pred_return_data,
        val_args.val_size,
        val_args.pred_horizon,
        var_type="Return",
    )

    loggers.add_plotly("val_plots/next_day_return_pred", next_day_return_fig)
    if multi_day_return_fig is not None:
        loggers.add_plotly("val_plots/multi_day_return_pred", multi_day_return_fig)

    next_day_price_fig, multi_day_price_fig = plot_pred_vs_gt(
        gt_price_data,
        pred_price_data,
        next_day_pred_price_data,
        val_args.val_size,
        val_args.pred_horizon,
        var_type="Stock Price",
    )

    loggers.add_plotly("val_plots/next_day_price_pred", next_day_price_fig)
    if multi_day_price_fig is not None:
        loggers.add_plotly("val_plots/multi_day_price_pred", multi_day_price_fig)


def main(args):

    # Set wandb configuration
    if args.wandb:
        wandb_config(args)

    ticker = args.ticker
    epochs = args.epochs
    batch_size = args.batch_size
    look_back = args.look_back
    pred_horizon = args.pred_horizon
    hidden_width = args.hidden_width
    dropout = args.dropout
    recurrent_pred_horizon = args.recurrent_pred_horizon

    loggers = LoggerHandler(
        [LocalLogger(run_folder)] + ([WandbLogger()] if args.wandb else [])
    )

    #######################
    # DATA INITIALISATION #
    #######################

    database = StockDatabase(DATABASE_PATH)
    metadata = database.get_metadata(
        ticker_symbol=stock_mapping(ticker), printable=True
    )
    print(f"Metadata for ticker {ticker}:\n{metadata}")

    df = database.get_pandas_dataframe(ticker_symbol=stock_mapping(ticker))

    if args.eda:
        logged_plots = eda_plots(
            df,
            label=ticker,
            value_col="Adj Close",
            date_col="Date",
            volume_col="Volume",
            add_ma=True,
        )
        for plot_name, fig in logged_plots.items():
            loggers.add_plotly(f"eda_plot/{plot_name}", fig)
        loggers.push_plotly(step=0)
        print(
            f"Logged plots in {run_folder}/figures: \n-> {'\n-> '.join(logged_plots.keys())}\n"
        )
        prompt_answer = UserPrompt.prompt_continue()
        prompt_answer()

    # Feature engineering
    df["Return"] = get_relative_change(df["Adj Close"])

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
            dropout=dropout,
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
            dropout=dropout,
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

    if args.wandb:
        wandb.watch(model, log="all", log_freq=train_log_freq)

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

            train_epoch(model, train_loader, optimizer, criterion, device, loggers)
            scheduler.step()

            if epoch % train_log_freq == 0:
                loggers.add_scalar("train/epoch", epoch)
                loggers.add_scalar("train/lr", optimizer.param_groups[0]["lr"])
                loggers.add_scalar("train/batch_size", batch_size)
                loggers.add_scalar("train/look_back", look_back)
                loggers.add_scalar("train/pred_horizon", pred_horizon)
                loggers.add_scalar("train/train_dataset_size", len(train_dataset))
                loggers.push_scalars(step=epoch)

            if epoch % val_log_freq == 0:
                val_data = torch.tensor(val_data, dtype=torch.float32)
                val_stock_price = df.iloc[val_idx]["Adj Close"].to_numpy()

                val_forward_pass(
                    model,
                    val_data,
                    criterion,
                    device,
                    normaliser,
                    val_stock_price,
                    loggers,
                    val_args=args,
                )
                loggers.push_scalars(step=epoch)
                loggers.push_plotly(step=epoch)

            pbar.update(1)


if __name__ == "__main__":

    args = parse_args()

    # Create run folder
    run_folder = create_run_folder(args.name)

    main(args)

    # TODO: save state dict to disk
