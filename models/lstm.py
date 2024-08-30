import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=128, output_size=1, pred_window=1):
        super(LSTMModel, self).__init__()
        self.input_size = input_size
        self.hidden_layer_size = hidden_layer_size
        self.pred_window = pred_window
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.device = next(self.parameters()).device
        self.hidden_cell = (torch.zeros(self.lstm.num_layers, 1, self.hidden_layer_size, device=self.device),
                            torch.zeros(self.lstm.num_layers, 1, self.hidden_layer_size, device=self.device))
        self._recurrence_count = 0

    @property
    def is_reset(self):
        return torch.all(self.hidden_cell[0] == 0) and torch.all(self.hidden_cell[1] == 0) and self.recurrence_count == 0

    @property
    def recurrence_count(self):
        return self._recurrence_count

    def _increment_recurrence_count(self, steps=1):
        self._recurrence_count += steps

    def reset_hidden_cell(self, batch_size, verbose=False):
        if verbose:
            print(f"Resetting hidden cell with batch size {batch_size}")
        self.current_batch_size = batch_size
        self.device = next(self.parameters()).device
        self.hidden_cell = (torch.zeros(self.lstm.num_layers, batch_size, self.hidden_layer_size, device=self.device),
                            torch.zeros(self.lstm.num_layers, batch_size, self.hidden_layer_size, device=self.device))
        self._recurrence_count = 0

    def forward_one_step(self, input):
        """
        Predict one timestep ahead
        """
        # input.shape = (batch_size, 1, num_features)
        assert len(input.shape) == 3 and input.shape[1] == 1, "Input should have shape (batch_size, 1, num_features)"
        lstm_out, self.hidden_cell = self.lstm(input, self.hidden_cell)
        self._increment_recurrence_count(steps=1)
        predictions = self.linear(lstm_out)
        return predictions

    def multi_step_forward(self, input, steps):
        """
        Predict multiple timesteps ahead and concatenate the results in a tensor
        """
        # input.shape = (batch_size, 1, num_features)
        multi_timestep_preds = [input]
        for _ in range(1, steps):
            input = self.forward_one_step(input)
            multi_timestep_preds.append(input)
        return torch.cat(multi_timestep_preds, dim=1)

    def forward(self, input_seq, reset_hidden=True):
        # input_seq.shape = (batch_size, seq_len, num_features)
        if reset_hidden:
            self.reset_hidden_cell(input_seq.shape[0])

        lstm_out, self.hidden_cell = self.lstm(input_seq, self.hidden_cell)
        self._increment_recurrence_count(steps=input_seq.shape[1])
        predictions = self.linear(lstm_out[:, -1]) # Hidden layer of last time step, same as self.hidden_cell[0][0]

        if self.pred_window > 1: # If we want to predict multiple timesteps
            multi_step_input = predictions.unsqueeze(1)
            predictions = self.multi_step_forward(multi_step_input, self.pred_window)

        return predictions

class CNNLSTMModel(LSTMModel):
    def __init__(self, input_size=1, hidden_layer_size=128, output_size=1, pred_window=1, conv_kernel_size=5, conv_channels=16):
        # Note: input_size will be times 2 due to conv features
        super(CNNLSTMModel, self).__init__(input_size=2*input_size, hidden_layer_size=hidden_layer_size, output_size=output_size, pred_window=pred_window)
        self.conv_kernel_size = conv_kernel_size
        self.conv_channels = conv_channels

        self.conv1 = nn.Conv1d(input_size, conv_channels, conv_kernel_size) # NOTE: no padding, input size will shrink by conv_kernel_size - 1
        self.conv_relu = nn.ReLU()
        self.down_proj = nn.Linear(conv_channels, input_size) # Project down to 1 channel

    def _check_sequence_length(self, input_seq):
        input_seq_len = input_seq.shape[1]
        assert input_seq_len >= self.conv_kernel_size, f"Input sequence length {input_seq_len} should be at least {self.conv_kernel_size}"

    def multi_step_forward(self, input, steps):
        """
        First, recompute the conv features for the last prediction, and then predict multiple timesteps ahead
        """
        # input.shape = (B, 1, P)
        multi_timestep_preds = [input]

        for _ in range(1, steps):
            # Recompute conv features for the last prediction
            input_for_conv = torch.cat([self.input_seq_cache, input], dim=1) # (B, K, P)

            conv_feat = self.conv_forward(input_for_conv) # (B, 1, P)

            input_aug = torch.cat([input, conv_feat], dim=2) # (B, 1, 2*P)
            pred = self.forward_one_step(input_aug) # (B, 1, P)
            multi_timestep_preds.append(pred)

            self.input_seq_cache = input_for_conv[:, 1:] # Update cache

            input = pred

        return torch.cat(multi_timestep_preds, dim=1)

    def conv_forward(self, input_seq):
        input_seq_for_conv = input_seq.permute(0, 2, 1).contiguous() # (B, N, P) -> (B, P, N)
        conv_out = self.conv1(input_seq_for_conv) # (B, P, N) -> (B, CH, N - K + 1)
        relu_out = self.conv_relu(conv_out) # (B, CH, N - K + 1) -> (B, CH, N - K + 1)
        relu_out = relu_out.permute(0, 2, 1).contiguous() # (B, CH, N - K + 1) -> (B, N - K + 1, CH)
        conv_feat = self.down_proj(relu_out) # (B, N - K + 1, CH) -> (B, N - K + 1, P)
        return conv_feat

    def forward(self, input_seq, reset_hidden=True):
        # input_seq.shape = (B, N, P)
        # Input Seq should be at least conv_kernel_size
        self._check_sequence_length(input_seq)

        # Get conv features
        conv_feat = self.conv_forward(input_seq)

        # Cache input features if we want to predict multiple timesteps
        if self.pred_window > 1:
            self.input_seq_cache = input_seq[:, -(self.conv_kernel_size - 1):].clone()

        input_seq_aug = torch.cat([input_seq[:, self.conv_kernel_size - 1:], conv_feat], dim=2)
        return super().forward(input_seq_aug, reset_hidden=reset_hidden)

if __name__ == "__main__":
    # Test LSTM model
    model = LSTMModel()
    input_seq = torch.randn(4, 50, 1)
    output = model(input_seq)
    print(f"{input_seq.shape} -> {output.shape}")

    # Test CNN-LSTM model
    model = CNNLSTMModel()
    input_seq = torch.randn(4, 50, 1)
    output = model(input_seq)
    print(f"{input_seq.shape} -> {output.shape}")

    # Test LSTM with pred_window
    model = LSTMModel(pred_window=5)
    input_seq = torch.randn(4, 50, 1)
    output = model(input_seq)
    print(f"{input_seq.shape} -> {output.shape}")

    # Test CNN-LSTM with pred_window
    model = CNNLSTMModel(pred_window=5)
    input_seq = torch.randn(4, 50, 1)
    output = model(input_seq)
    print(f"{input_seq.shape} -> {output.shape}")
