import torch
from models import LSTMModel, CNNLSTMModel

def _test_lstm_models(input_size, batch_size, time_steps, pred_horizon):

    # Test LSTM model
    model = LSTMModel(input_size=input_size)
    input_seq = torch.randn(batch_size, time_steps, input_size)
    output = model(input_seq)
    assert output.shape == (batch_size, 1, 1)

    # Test CNN-LSTM model
    model = CNNLSTMModel(input_size=input_size)
    input_seq = torch.randn(batch_size, time_steps, input_size)
    output = model(input_seq)
    assert output.shape == (batch_size, 1, 1)

    # Test LSTM with pred_horizon (non-recurrent)
    model = LSTMModel(input_size=input_size, pred_horizon=pred_horizon, recurrent_pred_horizon=False)
    input_seq = torch.randn(batch_size, time_steps, input_size)
    output = model(input_seq)
    assert output.shape == (batch_size, pred_horizon, 1)

    # Test CNN-LSTM with pred_horizon (non-recurrent)
    model = CNNLSTMModel(input_size=input_size, pred_horizon=pred_horizon, recurrent_pred_horizon=False)
    input_seq = torch.randn(batch_size, time_steps, input_size)
    output = model(input_seq)
    assert output.shape == (batch_size, pred_horizon, 1)

    # Test LSTM with pred_horizon (recurrent)
    model = LSTMModel(input_size=input_size, pred_horizon=pred_horizon, output_size=input_size, recurrent_pred_horizon=True)
    input_seq = torch.randn(batch_size, time_steps, input_size)
    output = model(input_seq)
    assert output.shape == (batch_size, pred_horizon, input_size)

    # Test CNN-LSTM with pred_horizon (recurrent)
    model = CNNLSTMModel(input_size=input_size, pred_horizon=pred_horizon, output_size=input_size, recurrent_pred_horizon=True)
    input_seq = torch.randn(batch_size, time_steps, input_size)
    output = model(input_seq)
    assert output.shape == (batch_size, pred_horizon, input_size)


def test_lstm_1():

    INPUT_SIZE = 1
    BATCH_SIZE = 1
    TIME_STEPS = 50
    PRED_HORIZON = 5

    _test_lstm_models(INPUT_SIZE, BATCH_SIZE, TIME_STEPS, PRED_HORIZON)

def test_lstm_2():

    INPUT_SIZE = 1
    BATCH_SIZE = 4
    TIME_STEPS = 50
    PRED_HORIZON = 5

    _test_lstm_models(INPUT_SIZE, BATCH_SIZE, TIME_STEPS, PRED_HORIZON)

def test_lstm_3():

    INPUT_SIZE = 5
    BATCH_SIZE = 4
    TIME_STEPS = 50
    PRED_HORIZON = 5

    _test_lstm_models(INPUT_SIZE, BATCH_SIZE, TIME_STEPS, PRED_HORIZON)

def test_lstm_forward_one():

    INPUT_SIZE = 5
    BATCH_SIZE = 4
    CNN_LSTM_CONV_KERNEL_SIZE = 5

    model = LSTMModel(input_size=INPUT_SIZE)
    input_seq = torch.randn(BATCH_SIZE, 1, INPUT_SIZE)
    model.reset_hidden_cell(BATCH_SIZE)
    output = model.forward_one_step(input_seq)
    assert output.shape == (BATCH_SIZE, 1, 1)

    model = CNNLSTMModel(input_size=INPUT_SIZE, conv_kernel_size=CNN_LSTM_CONV_KERNEL_SIZE)
    input_seq = torch.randn(BATCH_SIZE, CNN_LSTM_CONV_KERNEL_SIZE, INPUT_SIZE)
    model.reset_hidden_cell(BATCH_SIZE)
    output = model.forward_one_step(input_seq)
    assert output.shape == (BATCH_SIZE, 1, 1)

def test_lstm_pred_horizon_recurrency():

    INPUT_SIZE = 5
    BATCH_SIZE = 4
    CNN_LSTM_CONV_KERNEL_SIZE = 5
    PRED_HORIZON = 5
    TIME_STEPS = 50

    model = LSTMModel(input_size=INPUT_SIZE, output_size=INPUT_SIZE, pred_horizon=PRED_HORIZON, recurrent_pred_horizon=True)
    input_seq = torch.randn(BATCH_SIZE, TIME_STEPS, INPUT_SIZE)
    output = model(input_seq, reset_hidden=True)
    assert output.shape == (BATCH_SIZE, PRED_HORIZON, INPUT_SIZE)

    model = CNNLSTMModel(input_size=INPUT_SIZE, output_size=INPUT_SIZE, conv_kernel_size=CNN_LSTM_CONV_KERNEL_SIZE, pred_horizon=PRED_HORIZON, recurrent_pred_horizon=True)
    input_seq = torch.randn(BATCH_SIZE, TIME_STEPS, INPUT_SIZE)
    output = model(input_seq, reset_hidden=True)
    assert output.shape == (BATCH_SIZE, PRED_HORIZON, INPUT_SIZE)

def test_lstm_pred_horizon_single_pass():

    INPUT_SIZE = 5
    BATCH_SIZE = 4
    CNN_LSTM_CONV_KERNEL_SIZE = 5
    PRED_HORIZON = 5
    TIME_STEPS = 50

    model = LSTMModel(input_size=INPUT_SIZE, pred_horizon=PRED_HORIZON, recurrent_pred_horizon=False)
    input_seq = torch.randn(BATCH_SIZE, TIME_STEPS, INPUT_SIZE)
    output = model(input_seq, reset_hidden=True)
    assert output.shape == (BATCH_SIZE, PRED_HORIZON, 1)

    model = CNNLSTMModel(input_size=INPUT_SIZE, conv_kernel_size=CNN_LSTM_CONV_KERNEL_SIZE, pred_horizon=PRED_HORIZON, recurrent_pred_horizon=False)
    input_seq = torch.randn(BATCH_SIZE, TIME_STEPS, INPUT_SIZE)
    output = model(input_seq, reset_hidden=True)
    assert output.shape == (BATCH_SIZE, PRED_HORIZON, 1)