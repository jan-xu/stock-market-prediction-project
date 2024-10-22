import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(
        self,
        input_size=1,
        hidden_layer_size=128,
        output_size=1,
        pred_horizon=1,
        dropout=0.0,
        recurrent_pred_horizon=False,
    ):
        """LSTM model for stock price prediction.

        Args:
            input_size (int, optional): Number of input features. Defaults to 1.
            hidden_layer_size (int, optional): Number of hidden features. Defaults to 128.
            output_size (int, optional): Number of output features. Defaults to 1.
            pred_horizon (int, optional): Prediction horizon of LSTM output (how many future
                time steps to predict). Defaults to 1.
            recurrent_pred_horizon (bool, optional): If `pred_horizon > 1` and if this is
                True, the prediction horizon will be outputted recurrently using previous
                outputs as inputs. A necessary constraint of this is that
                `input_size == output_size`. Otherwise, the entire prediction horizon will
                be outputted at once. Defaults to False.

        -> Use case for setting pred_horizon > 1 and recurrent_pred_horizon = False:
            This is equivalent to predicting multiple time steps (T = pred_horizon) ahead of
            time in a single pass. The linear layer outputs precisely T * output_size
            features, which are reshaped to (T, output_size) in the forward pass before
            returning.
        -> Use case for setting pred_horizon > 1 and recurrent_pred_horizon = True:
            This enforces the model to predict multiple time steps (T = pred_horizon) ahead
            of time in a recurrent manner, by reutilising previous outputs as inputs in the
            next time step prediction. A necessary condition for this to work is that the
            input size is equal to output size, otherwise we can't establish recurrency. In
            the future, this functionality can be extended to admit auxiliary input variables
            for the recurrent time step prediction, if we know them ahead of time.
        """
        super(LSTMModel, self).__init__()
        self.input_size = input_size
        self.hidden_layer_size = hidden_layer_size
        self.output_size = output_size
        self.pred_horizon = pred_horizon
        self.dropout = dropout
        self.recurrent_pred_horizon = recurrent_pred_horizon

        self._assert_pred_horizon()

        self.lstm = nn.LSTM(
            input_size,
            hidden_layer_size,
            num_layers=2,
            dropout=dropout,
            batch_first=True,
        )
        self.linear = nn.Linear(
            hidden_layer_size,
            output_size * (pred_horizon if not recurrent_pred_horizon else 1),
        )
        self.device = next(self.parameters()).device
        self.hidden_cell = (
            torch.zeros(
                self.lstm.num_layers, 1, self.hidden_layer_size, device=self.device
            ),
            torch.zeros(
                self.lstm.num_layers, 1, self.hidden_layer_size, device=self.device
            ),
        )
        self._recurrence_count = 0

    def _assert_pred_horizon(self):
        if self.pred_horizon > 1 and self.recurrent_pred_horizon:
            assert (
                self.input_size == self.output_size
            ), f"For recurrent prediction horizon, `input_size = {self.input_size}` and `output_size = {self.output_size}` must be equal."

    @property
    def is_reset(self):
        return (
            torch.all(self.hidden_cell[0] == 0)
            and torch.all(self.hidden_cell[1] == 0)
            and self.recurrence_count == 0
        )

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
        self.hidden_cell = (
            torch.zeros(
                self.lstm.num_layers,
                batch_size,
                self.hidden_layer_size,
                device=self.device,
            ),
            torch.zeros(
                self.lstm.num_layers,
                batch_size,
                self.hidden_layer_size,
                device=self.device,
            ),
        )
        self._recurrence_count = 0

    def forward_one_step(self, input):
        """
        Predict one timestep ahead
        """
        # input.shape = (batch_size, 1, num_features)
        assert (
            len(input.shape) == 3 and input.shape[1] == 1
        ), "Input should have shape (batch_size, 1, num_features)"
        lstm_out, self.hidden_cell = self.lstm(input, self.hidden_cell)
        self._increment_recurrence_count(steps=1)
        predictions = self.linear(lstm_out)
        if not self.recurrent_pred_horizon:
            predictions = predictions.reshape(-1, self.pred_horizon, self.output_size)
        return predictions

    def recurrent_step_forward(self, input, steps):
        """
        Predict multiple timesteps ahead recurrently and concatenate the results in a tensor
        """
        # input.shape = (batch_size, 1, num_features)
        recurrent_timestep_preds = [input]
        for _ in range(1, steps):
            input = self.forward_one_step(input)
            recurrent_timestep_preds.append(input)
        return torch.cat(recurrent_timestep_preds, dim=1)

    def forward(self, input_seq, reset_hidden=True):
        # input_seq.shape = (batch_size, seq_len, num_features)
        if reset_hidden:
            self.reset_hidden_cell(input_seq.shape[0])

        lstm_out, self.hidden_cell = self.lstm(input_seq, self.hidden_cell)
        self._increment_recurrence_count(steps=input_seq.shape[1])
        lstm_out_last = lstm_out[
            :, -1:
        ]  # Hidden layer of last time step, same as self.hidden_cell[0][:1].transpose(0,1)
        predictions = self.linear(lstm_out_last)

        if self.pred_horizon > 1:  # If we want to predict multiple timesteps
            if self.recurrent_pred_horizon:
                predictions = self.recurrent_step_forward(
                    predictions, self.pred_horizon
                )
            else:
                predictions = predictions.reshape(
                    -1, self.pred_horizon, self.output_size
                )

        return predictions


class CNNLSTMModel(LSTMModel):
    """CNN-LSTM model for stock price prediction.

    This augments the inherited LSTM model by instantiating a 1D-Conv kernel of a specified
    size that is applied on the input sequence before concatenated with a truncated version
    of the input sequence (as the convolution reduces the input sequence length by
    conv_kernel_size - 1). Afterwards, the LSTM model is applied on this augmented input
    sequence as per usual.

    The benefit of this is that the convolution generalises autoregression, as the
    augmentation is dependent entirely on lagged time steps of a given backwards-looking
    time horizon (stated by conv_kernel_size). This enhances the predictability if the
    prediction model has a strong dependency on multiple previous time steps.

    Args (in addition to those of LSTMModel):
        conv_kernel_size (int, optional): Size of 1D-Conv kernel for feature pre-processing.
        Defaults to 5.
        conv_channels (int, optional): Number of convolution features. Defaults to 16.
    """

    def __init__(
        self,
        input_size=1,
        hidden_layer_size=128,
        output_size=1,
        pred_horizon=1,
        conv_kernel_size=5,
        conv_channels=16,
        dropout=0.0,
        recurrent_pred_horizon=False,
    ):
        # Note: input_size will be times 2 due to conv features
        super(CNNLSTMModel, self).__init__(
            input_size=2 * input_size,
            hidden_layer_size=hidden_layer_size,
            output_size=output_size,
            pred_horizon=pred_horizon,
            dropout=dropout,
            recurrent_pred_horizon=recurrent_pred_horizon,
        )
        self.conv_kernel_size = conv_kernel_size
        self.conv_channels = conv_channels

        self.conv1 = nn.Conv1d(
            input_size, conv_channels, conv_kernel_size
        )  # NOTE: no padding, input size will shrink by conv_kernel_size - 1
        self.conv_relu = nn.ReLU()
        self.down_proj = nn.Linear(
            conv_channels, input_size
        )  # Project down to 1 channel

    def _assert_pred_horizon(self):
        if self.pred_horizon > 1 and self.recurrent_pred_horizon:
            assert (
                self.input_size == 2 * self.output_size
            ), f"For recurrent prediction horizon, `input_size = {self.input_size}` and 2 * `output_size = {self.output_size}` must be equal."

    def _check_sequence_length(self, input_seq):
        input_seq_len = input_seq.shape[1]
        assert (
            input_seq_len >= self.conv_kernel_size
        ), f"Input sequence length {input_seq_len} should be at least {self.conv_kernel_size}"

    def forward_one_step(self, input):
        """
        Predict one timestep ahead, but recompute the conv features for the last prediction before and store in cache after
        """
        if self.recurrent_pred_horizon:
            input_for_conv = torch.cat(
                [self.input_seq_cache, input], dim=1
            )  # (B, K, P)
            conv_feat = self.conv_forward(input_for_conv)  # (B, 1, P)
            input_aug = torch.cat([input, conv_feat], dim=2)  # (B, 1, 2*P)
            pred = super().forward_one_step(input_aug)
            self.input_seq_cache = input_for_conv[:, 1:]  # Update cache
            return pred
        else:
            conv_feat = self.conv_forward(input)
            input_aug = torch.cat(
                [input[:, self.conv_kernel_size - 1 :], conv_feat], dim=2
            )
            return super().forward_one_step(input_aug)

    def conv_forward(self, input_seq):
        input_seq_for_conv = input_seq.permute(
            0, 2, 1
        ).contiguous()  # (B, N, P) -> (B, P, N)
        conv_out = self.conv1(input_seq_for_conv)  # (B, P, N) -> (B, CH, N - K + 1)
        relu_out = self.conv_relu(conv_out)  # (B, CH, N - K + 1) -> (B, CH, N - K + 1)
        relu_out = relu_out.permute(
            0, 2, 1
        ).contiguous()  # (B, CH, N - K + 1) -> (B, N - K + 1, CH)
        conv_feat = self.down_proj(relu_out)  # (B, N - K + 1, CH) -> (B, N - K + 1, P)
        return conv_feat

    def forward(self, input_seq, reset_hidden=True):
        # input_seq.shape = (B, N, P)
        # input_seq should be at least conv_kernel_size
        self._check_sequence_length(input_seq)

        # Get conv features
        conv_feat = self.conv_forward(input_seq)

        # Cache input features if we want to predict multiple timesteps
        if self.pred_horizon > 1 and self.recurrent_pred_horizon:
            self.input_seq_cache = input_seq[:, -(self.conv_kernel_size - 1) :].clone()

        input_seq_aug = torch.cat(
            [input_seq[:, self.conv_kernel_size - 1 :], conv_feat], dim=2
        )
        return super().forward(input_seq_aug, reset_hidden=reset_hidden)
