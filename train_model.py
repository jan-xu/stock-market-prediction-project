from tqdm import tqdm
import wandb
import yfinance as yf
from datetime import datetime
import numpy as np
import pandas as pd

import plotly.graph_objects as go

import argparse
from config import create_run_folder, wandb_config, STOCK_MAPPING

from arguments import parse_args

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset


np.random.seed(0)


def train_test_split(data, test_size=0.05):
    split_idx = int(len(data) * (1 - test_size))
    train_data = data[:split_idx].copy()
    test_data = data[split_idx:].copy()
    return train_data, test_data


if __name__ == "__main__":

    args = parse_args()

    run_name = args.name

    # Set wandb configuration
    if args.wandb:
        wandb_config(args)
        FIGURE_TABLE = wandb.Table(columns=["generated_data", "generated_data_train_test", "generated_rel_diff_train_test", "generated_volume_train_test"])
        FIGURE_LIST = []

    epochs = args.epochs
    batch_size = args.batch_size
    look_back = args.look_back
    pred_horizon = args.pred_horizon
    hidden_width = args.hidden_width

    # Create run folder
    run_folder = create_run_folder(run_name)

    ### DATA GENERATION AND VISUALISATION

    # Generate and plot data

    today = datetime.today().strftime("%Y-%m-%d")
    stock_symbol = STOCK_MAPPING[args.dataset] if args.dataset in STOCK_MAPPING else args.dataset
    sp500_data = yf.download(stock_symbol, start="1983-01-01", end=today)

    time_series = sp500_data[['Close', 'Volume']].reset_index().rename({"Close": "value", "Volume": "volume"}, axis=1)
    time_series["time"] = range(len(time_series))
    time_series["tuesday"] = (time_series["Date"].dt.day_name() == "Tuesday").astype(int)
    time_series["wednesday"] = (time_series["Date"].dt.day_name() == "Wednesday").astype(int)
    time_series["thursday"] = (time_series["Date"].dt.day_name() == "Thursday").astype(int)
    time_series["friday"] = (time_series["Date"].dt.day_name() == "Friday").astype(int)
    time_series["month_cos_embed"] = np.cos(2 * np.pi * (time_series["Date"].dt.month - 1) / 12)
    time_series["month_sin_embed"] = np.sin(2 * np.pi * (time_series["Date"].dt.month - 1) / 12)
    time_series["day_cos_embed"] = np.cos(2 * np.pi * (time_series["Date"].dt.day - 1) / time_series["Date"].dt.days_in_month)
    time_series["day_sin_embed"] = np.sin(2 * np.pi * (time_series["Date"].dt.day - 1) / time_series["Date"].dt.days_in_month)

    # features = ["value", "tuesday", "wednesday", "thursday", "friday", "month_cos_embed", "month_sin_embed", "day_cos_embed", "day_sin_embed"]
    features = ["value", "volume"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time_series['time'], y=time_series['value'], mode='lines'))
    fig.update_layout(title='Simulated Time Series Data')
    fig.write_image(run_folder / "figures" / "png" / f"generated_data.png")
    fig.write_html(run_folder / "figures" / "html" / f"generated_data.html")
    if args.wandb:
        FIGURE_LIST.append(wandb.Html(str(run_folder / "figures" / "html" / f"generated_data.html")))

    # Split into train and test dataset
    train_data, test_data = train_test_split(time_series)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train_data['time'], y=train_data['value'], mode='lines', name='train', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=test_data['time'], y=test_data['value'], mode='lines', name='test', line=dict(color='red')))
    fig.update_layout(title='Simulated Time Series Data (train-test split)')
    fig.write_image(run_folder / "figures" / "png" / f"generated_data_train_test.png")
    fig.write_html(run_folder / "figures" / "html" / f"generated_data_train_test.html")
    if args.wandb:
        FIGURE_LIST.append(wandb.Html(str(run_folder / "figures" / "html" / f"generated_data_train_test.html")))

    train_data.to_csv("train.csv", index=False)
    test_data.to_csv("test.csv", index=False)

    # Compute relative time-series differences
    train_data['rel_diff'] = (train_data['value'].diff() / train_data['value'].shift()).fillna(0)
    test_data['rel_diff'] = (test_data['value'].diff() / test_data['value'].shift()).fillna(0)

    # Back to value: np.exp(np.log(test_data['rel_diff'] + 1).cumsum()) * test_data['value'].iloc[0]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train_data['time'], y=train_data['rel_diff'], mode='lines', name='train', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=test_data['time'], y=test_data['rel_diff'], mode='lines', name='test', line=dict(color='red')))
    fig.update_layout(title='Simulated Time Series Relative Diff (train-test split)')
    fig.write_image(run_folder / "figures" / "png" / f"generated_rel_diff_train_test.png")
    fig.write_html(run_folder / "figures" / "html" / f"generated_rel_diff_train_test.html")
    if args.wandb:
        FIGURE_LIST.append(wandb.Html(str(run_folder / "figures" / "html" / f"generated_rel_diff_train_test.html")))

    fig = go.Figure()
    fig.add_trace(go.Bar(x=train_data['time'], y=train_data['volume'], marker_color='blue', name='train'))
    fig.add_trace(go.Bar(x=test_data['time'], y=test_data['volume'], marker_color='red', name='test'))
    fig.update_layout(title='Volume (train-test split)')
    fig.write_image(run_folder / "figures" / "png" / f"generated_volume_train_test.png")
    fig.write_html(run_folder / "figures" / "html" / f"generated_volume_train_test.html")
    if args.wandb:
        FIGURE_LIST.append(wandb.Html(str(run_folder / "figures" / "html" / f"generated_volume_train_test.html")))

    if args.wandb:
        FIGURE_TABLE.add_data(*FIGURE_LIST)
        wandb.log({"train-test-data": FIGURE_TABLE}, step=0)

    ### MACHINE LEARNING MODELS

    # LSTM model

    def apply_lstm(train_data, test_data, look_back=look_back, pred_horizon=pred_horizon, epochs=epochs, lr=0.001):

        K = len(features) # number of features
        features[0] = "rel_diff"

        scaler = MinMaxScaler(feature_range=(-1, 1))
        if "volume" in features:
            volume_scaler = MinMaxScaler(feature_range=(0, 1))

        scaled_train_data = scaler.fit_transform(train_data['rel_diff'].values.reshape(-1, 1))
        if K > 1:
            if "volume" in features:
                scaled_volume = volume_scaler.fit_transform(train_data['volume'].values.reshape(-1, 1))
                scaled_train_data = np.concat([scaled_train_data, scaled_volume], axis=1)
            else:
                scaled_train_data = np.concat([scaled_train_data, train_data[features[1:]].values.reshape(-1, K-1)], axis=1)
        # scaled_train_data = scaler.fit_transform(train_data['rel_diff'].values.reshape(-1, 1))

        X_train, y_train = [], []

        for i in range(len(scaled_train_data) - look_back):
            X_train.append(scaled_train_data[i:i+look_back, :])
            if pred_horizon > 1:
                y_train.append(scaled_train_data[i+look_back:i+look_back+pred_horizon, 0])
            else:
                y_train.append(scaled_train_data[i+look_back, 0])

        X_train, y_train = np.array(X_train), np.array(y_train)
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], K))

        train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)) # scaled
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        device = torch.device("cuda:0")

        from models import LSTMModel, CNNLSTMModel

        if args.architecture == "LSTM":
            model = LSTMModel(input_size=K, hidden_layer_size=hidden_width).to(device)
        elif args.architecture == "CNN-LSTM":
            model = CNNLSTMModel(input_size=K, hidden_layer_size=hidden_width, conv_channels=32).to(device)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # Module for testing

        def _test_forward_pass(model):

            model.eval()

            # Forward pass all ground-truth data once
            # train_loader_for_test = DataLoader(train_dataset, batch_size=1, shuffle=False)
            # model.reset_hidden_cell(batch_size=1, verbose=True)

            # with torch.no_grad():
            #     count = 0
            #     for inputs, _ in train_loader_for_test:
            #         if count < len(train_loader_for_test) - look_back:
            #             pass
            #         else:
            #             inputs = inputs.to(device)
            #             _ = model(inputs, reset_hidden=False)


            # Prepare test data

            # Use ground truth

            last_train_data_values = train_data[features].values.reshape(-1, K)[-look_back:]
            test_data_values = test_data[features].values.reshape(-1, K)
            inputs = np.concat([last_train_data_values, test_data_values], axis=0)
            inputs[:, :1] = scaler.transform(inputs[:, :1])
            if "volume" in features:
                inputs[:, 1:] = volume_scaler.transform(inputs[:, 1:])

            X_test = []
            for i in range(len(inputs)-look_back):
                X_test.append(inputs[i:i+look_back, :])

            X_test = np.array(X_test)
            X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], K))

            predictions = []
            with torch.no_grad():
                X_test = torch.tensor(X_test, dtype=torch.float32, device=device)

                for i in range(len(X_test)):
                    predictions.append(model(X_test[i:i+1], reset_hidden=True).cpu().numpy())

            predictions = np.array(predictions)
            predictions = np.reshape(predictions, (predictions.shape[0], 1))

            predictions = scaler.inverse_transform(predictions)

            return predictions

        for epoch in tqdm(range(1, epochs+1)):
            for inputs, labels in train_loader:

                model.train()

                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                y_pred = model(inputs, reset_hidden=True)
                loss = criterion(y_pred, labels.view(-1, 1))
                loss.backward()
                optimizer.step()

            if epoch % 200 == 0:
                print(f'Epoch {epoch}, Loss: {loss.item()}')
                if args.wandb:
                    wandb.log({"train_loss": loss.item()}, step=epoch)

            if epoch % 200 == 0:
                predictions = _test_forward_pass(model)
                test_data['val_pred'] = np.exp(np.log(predictions + 1).cumsum()) * train_data[['value']].iloc[-1].item()
                acc = calc_acc(test_data["rel_diff"], test_data['val_pred'])
                rmse = calc_rmse(test_data['value'], test_data['val_pred'])
                mae = calc_mae(test_data['value'], test_data['val_pred'])
                print(f"Test accuracy: {acc * 100:.4f}%, RMSE: {rmse:.4f}, MAE: {mae:.4f}")
                if args.wandb:
                    wandb.log({"test_acc": acc, "test_rmse": rmse, "test_mae": mae}, step=epoch)
                    lr = optimizer.param_groups[0]['lr']
                    wandb.log({"lr": lr}, step=epoch)

            if epoch == int(epochs * 0.6):
                optimizer.param_groups[0]['lr'] = 0.0001

            if epoch == int(epochs * 0.8):
                optimizer.param_groups[0]['lr'] = 0.00001

        print(f'Epoch {epoch}, Loss: {loss.item()}')

        predictions = _test_forward_pass(model)
        # test_data = test_data.iloc[look_back:].copy()
        # test_data['lstm_predicted_value_gt'] = predictions
        test_data['lstm_predicted_value_gt'] = np.exp(np.log(predictions + 1).cumsum()) * train_data[['value']].iloc[-1].item()

        return test_data

    # Performance metrics
    def calc_acc(rel_diff, pred):
        return np.mean(np.sign(rel_diff) == np.sign(pred.diff().fillna(0)))

    def calc_rmse(value, pred):
        return np.sqrt(np.mean(value - pred)**2)

    def calc_mae(value, pred):
        return np.mean(np.abs(value - pred))

    lstm_test_data = apply_lstm(train_data, test_data, look_back=look_back)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train_data['time'], y=train_data['value'], mode='lines', name='Training data'))
    fig.add_trace(go.Scatter(x=test_data['time'], y=test_data['value'], mode='lines', name='Test data'))
    fig.add_trace(go.Scatter(x=test_data['time'], y=lstm_test_data['lstm_predicted_value_gt'], mode='lines', name='LSTM prediction (ground-truth)'))
    fig.update_layout(title='Time-series prediction')
    fig.write_image(run_folder / "figures" / "png" / f"time_series_prediction.png")
    fig.write_html(run_folder / "figures" / "html" / f"time_series_prediction.html")

    if args.wandb:
        RESULT_TABLE = wandb.Table(columns=["time_series_prediction"])
        RESULT_TABLE.add_data(wandb.Html(str(run_folder / "figures" / "html" / f"time_series_prediction.html")))
        wandb.log({"result_plots": RESULT_TABLE}, step=epochs)

    # Report accuracy, defined as the accuracy of predicting the sign of the relative difference

    print("Accuracy of predicting the sign of the relative difference")
    print(f"LSTM: {calc_acc(test_data["rel_diff"], lstm_test_data["lstm_predicted_value_gt"]) * 100:.4f}%")

    # Report RMSE
    print("RMSE")
    print(f"LSTM: {calc_rmse(lstm_test_data['value'], lstm_test_data['lstm_predicted_value_gt']):.4f}")

    # Report MAE
    print("MAE")
    print(f"LSTM: {calc_mae(lstm_test_data['value'], lstm_test_data['lstm_predicted_value_gt']):.4f}")
