from tqdm import tqdm
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

from toy import SyntheticStockData

np.random.seed(0)


def train_test_split(data, test_size=0.05):
    split_idx = int(len(data) * (1 - test_size))
    train_data = data[:split_idx].copy()
    test_data = data[split_idx:].copy()
    return train_data, test_data


if __name__ == "__main__":

    syn_data_obj = SyntheticStockData()
    syn_data_obj()
    df = syn_data_obj.get_data_pandas()





    # NUM_EPOCHS = 10000
    # BATCH_SIZE = 64

    # ### DATA GENERATION AND VISUALISATION

    # # Generate and plot data

    # today = datetime.today().strftime("%Y-%m-%d")
    # sp500_data = yf.download('^GSPC', start="1983-01-01", end=today)

    # time_series = sp500_data[['Close', 'Volume']].reset_index().rename({"Close": "value", "Volume": "volume"}, axis=1)
    # time_series["time"] = range(len(time_series))
    # time_series["tuesday"] = (time_series["Date"].dt.day_name() == "Tuesday").astype(int)
    # time_series["wednesday"] = (time_series["Date"].dt.day_name() == "Wednesday").astype(int)
    # time_series["thursday"] = (time_series["Date"].dt.day_name() == "Thursday").astype(int)
    # time_series["friday"] = (time_series["Date"].dt.day_name() == "Friday").astype(int)
    # time_series["month_cos_embed"] = np.cos(2 * np.pi * (time_series["Date"].dt.month - 1) / 12)
    # time_series["month_sin_embed"] = np.sin(2 * np.pi * (time_series["Date"].dt.month - 1) / 12)
    # time_series["day_cos_embed"] = np.cos(2 * np.pi * (time_series["Date"].dt.day - 1) / time_series["Date"].dt.days_in_month)
    # time_series["day_sin_embed"] = np.sin(2 * np.pi * (time_series["Date"].dt.day - 1) / time_series["Date"].dt.days_in_month)

    # # features = ["value", "tuesday", "wednesday", "thursday", "friday", "month_cos_embed", "month_sin_embed", "day_cos_embed", "day_sin_embed"]
    # features = ["value", "volume"]

    # fig, ax = plt.subplots()
    # ax.plot(time_series['time'], time_series['value'])
    # ax.set_title('Simulated Time Series Data')
    # fig.savefig("generated_data.png")

    # # Split into train and test dataset
    # train_data, test_data = train_test_split(time_series)

    # fig, ax = plt.subplots()
    # ax.plot(train_data['time'], train_data['value'], color="blue", label="train")
    # ax.plot(test_data['time'], test_data['value'], color="red", label="test")
    # ax.set_title('Simulated Time Series Data (train-test split)')
    # ax.legend()
    # fig.savefig("generated_data_train_test.png")

    # train_data.to_csv("train.csv", index=False)
    # test_data.to_csv("test.csv", index=False)

    # # Compute relative time-series differences
    # train_data['rel_diff'] = (train_data['value'].diff() / train_data['value'].shift()).fillna(0)
    # test_data['rel_diff'] = (test_data['value'].diff() / test_data['value'].shift()).fillna(0)

    # # Back to value: np.exp(np.log(test_data['rel_diff'] + 1).cumsum()) * test_data['value'].iloc[0]

    # fig, ax = plt.subplots()
    # ax.plot(train_data['time'], train_data['rel_diff'], color="blue", label="train")
    # ax.plot(test_data['time'], test_data['rel_diff'], color="red", label="test")
    # ax.set_title('Simulated Time Series Relative Diff (train-test split)')
    # ax.legend()
    # fig.savefig("generated_rel_diff_train_test.png")


    # fig, ax = plt.subplots()
    # ax.bar(train_data['time'], train_data['volume'], color="blue", label="train")
    # ax.bar(test_data['time'], test_data['volume'], color="red", label="test")
    # ax.set_title('Volume (train-test split)')
    # ax.legend()
    # fig.savefig("generated_volume_train_test.png")


    # ### MACHINE LEARNING MODELS

    # # LSTM model

    # def apply_lstm(train_data, test_data, look_back=10, epochs=NUM_EPOCHS, lr=0.001):

    #     K = len(features) # number of features
    #     features[0] = "rel_diff"

    #     # scaler = MinMaxScaler(feature_range=(0, 1))
    #     scaler = MinMaxScaler(feature_range=(-1, 1))
    #     if "volume" in features:
    #         volume_scaler = MinMaxScaler(feature_range=(0, 1))

    #     scaled_train_data = scaler.fit_transform(train_data['rel_diff'].values.reshape(-1, 1))
    #     if K > 1:
    #         if "volume" in features:
    #             scaled_volume = volume_scaler.fit_transform(train_data['volume'].values.reshape(-1, 1))
    #             scaled_train_data = np.concat([scaled_train_data, scaled_volume], axis=1)
    #         else:
    #             scaled_train_data = np.concat([scaled_train_data, train_data[features[1:]].values.reshape(-1, K-1)], axis=1)
    #     # scaled_train_data = scaler.fit_transform(train_data['rel_diff'].values.reshape(-1, 1))

    #     X_train, y_train = [], []
    #     # from IPython import embed; embed()
    #     for i in range(len(scaled_train_data) - look_back):
    #         X_train.append(scaled_train_data[i:i+look_back, :])
    #         y_train.append(scaled_train_data[i+look_back, 0])

    #     X_train, y_train = np.array(X_train), np.array(y_train)
    #     X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], K))

    #     train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    #     train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    #     device = torch.device("cuda:0")

    #     from model import LSTMModel, CNNLSTMModel
    #     # model = LSTMModel(input_size=K, hidden_layer_size=128).to(device)
    #     # print("Now training with a LSTM model")
    #     model = CNNLSTMModel(input_size=K, hidden_layer_size=128, conv_channels=32).to(device)
    #     print("Now training with a CNN-LSTM model")
    #     model.device = device
    #     criterion = nn.MSELoss()
    #     optimizer = optim.Adam(model.parameters(), lr=lr)

    #     # Module for testing

    #     def _test_forward_pass(model):

    #         model.eval()

    #         # Forward pass all ground-truth data once
    #         train_loader_for_test = DataLoader(train_dataset, batch_size=1, shuffle=False)
    #         model.reset_hidden_cell(batch_size=1, verbose=True)

    #         with torch.no_grad():
    #             count = 0
    #             for inputs, _ in train_loader_for_test:
    #                 if count < len(train_loader_for_test) - look_back:
    #                     pass
    #                 else:
    #                     inputs = inputs.to(device)
    #                     _ = model(inputs, reset_hidden=False)


    #         # Prepare test data

    #         # Use ground truth

    #         # last_train_data_values = train_data['value'].values.reshape(-1, 1)[-look_back:]
    #         last_train_data_values = train_data[features].values.reshape(-1, K)[-look_back:]
    #         # last_train_data_values = train_data['rel_diff'].values.reshape(-1, 1)[-look_back:]
    #         # test_data_values = test_data['value'].values.reshape(-1, 1)
    #         test_data_values = test_data[features].values.reshape(-1, K)
    #         # test_data_values = test_data['rel_diff'].values.reshape(-1, 1)
    #         inputs = np.concat([last_train_data_values, test_data_values], axis=0)
    #         inputs[:, :1] = scaler.transform(inputs[:, :1])
    #         if "volume" in features:
    #             inputs[:, 1:] = volume_scaler.transform(inputs[:, 1:])

    #         X_test = []
    #         for i in range(len(inputs)-look_back):
    #             X_test.append(inputs[i:i+look_back, :])

    #         X_test = np.array(X_test)
    #         X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], K))

    #         predictions = []
    #         with torch.no_grad():
    #             X_test = torch.tensor(X_test, dtype=torch.float32, device=device)

    #             for i in range(len(X_test)):
    #                 predictions.append(model(X_test[i:i+1], reset_hidden=False).cpu().numpy())

    #         predictions = np.array(predictions)
    #         predictions = np.reshape(predictions, (predictions.shape[0], 1))

    #         predictions = scaler.inverse_transform(predictions)

    #         return predictions

    #     for epoch in tqdm(range(epochs)):
    #         for inputs, labels in train_loader:

    #             model.train()

    #             inputs = inputs.to(device)
    #             labels = labels.to(device)

    #             optimizer.zero_grad()
    #             y_pred = model(inputs, reset_hidden=True)
    #             loss = criterion(y_pred, labels.view(-1, 1))
    #             loss.backward()
    #             optimizer.step()

    #         if epoch % 200 == 0:
    #             print(f'Epoch {epoch}, Loss: {loss.item()}')
    #             # wandb.log({"train_loss": loss.item()})

    #         if epoch % 200 == 0 and epoch > 0:
    #             predictions = _test_forward_pass(model)
    #             test_data['val_pred'] = np.exp(np.log(predictions + 1).cumsum()) * train_data[['value']].iloc[-1].item()
    #             acc = calc_acc(test_data["rel_diff"], test_data['val_pred'])
    #             rmse = calc_rmse(test_data['value'], test_data['val_pred'])
    #             mae = calc_mae(test_data['value'], test_data['val_pred'])
    #             # wandb.log({"test_acc": acc, "test_rmse": rmse, "test_mae": mae})
    #             print(f"Test accuracy: {acc * 100:.4f}%, RMSE: {rmse:.4f}, MAE: {mae:.4f}")

    #         if epoch == int(NUM_EPOCHS * 0.6):
    #             optimizer.param_groups[0]['lr'] = 0.0001

    #         if epoch == int(NUM_EPOCHS * 0.8):
    #             optimizer.param_groups[0]['lr'] = 0.00001

    #     print(f'Epoch {epoch}, Loss: {loss.item()}')

    #     predictions = _test_forward_pass(model)
    #     # test_data = test_data.iloc[look_back:].copy()
    #     # test_data['lstm_predicted_value_gt'] = predictions
    #     test_data['lstm_predicted_value_gt'] = np.exp(np.log(predictions + 1).cumsum()) * train_data[['value']].iloc[-1].item()

    #     return test_data

    # # Performance metrics
    # def calc_acc(rel_diff, pred):
    #     return np.mean(np.sign(rel_diff) == np.sign(pred.diff().fillna(0)))

    # def calc_rmse(value, pred):
    #     return np.sqrt(np.mean(value - pred)**2)

    # def calc_mae(value, pred):
    #     return np.mean(np.abs(value - pred))

    # lstm_test_data = apply_lstm(train_data, test_data, look_back=14)

    # fig, ax = plt.subplots()
    # ax.plot(train_data['time'][-2000:], train_data['value'][-2000:], label='Training data')
    # ax.plot(test_data['time'], test_data['value'], label='Test data')
    # ax.plot(test_data['time'], lstm_test_data['lstm_predicted_value_gt'], label='LSTM prediction (ground-truth)')
    # ax.set_title('Time-series prediction')
    # ax.legend()
    # fig.savefig("time_series_prediction.png")

    # # Report accuracy, defined as the accuracy of predicting the sign of the relative difference

    # print("Accuracy of predicting the sign of the relative difference")
    # print(f"LSTM: {calc_acc(test_data["rel_diff"], lstm_test_data["lstm_predicted_value_gt"]) * 100:.4f}%")

    # # Report RMSE
    # print("RMSE")
    # print(f"LSTM: {calc_rmse(lstm_test_data['value'], lstm_test_data['lstm_predicted_value_gt']):.4f}")

    # # Report MAE
    # print("MAE")
    # print(f"LSTM: {calc_mae(lstm_test_data['value'], lstm_test_data['lstm_predicted_value_gt']):.4f}")
