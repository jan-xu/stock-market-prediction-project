import argparse
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class SyntheticStockData:
    """
    A toy time series object that generates synthetic stock price data according to the following formula:

    P_t = P0 * exp(mu * t + sigma * W_t) * (1 + A * sin(2 * pi * f * t + phi))

    where W_t is a Wiener process (Brownian motion), representing the random component and is expressed as

    W_t = W_{t-1} + N(0, 1)

    Attributes:
    - P0 (float): Initial price of the toy. Default value is 100.
    - mu (float): Drift coefficient for the geometric Brownian motion. Default value is 0.001.
    - sigma (float): Volatility coefficient for the geometric Brownian motion. Default value is 0.02.
    - A (float): Amplitude of the sine wave. Default value is 0.05.
    - f (float): Frequency of the sine wave. Default value is 1/252 (assume 252 trading days in a year).
    - phi (float): Phase shift of the sine wave. Default value is 0.
    - T (int): Number of time steps. Default value is 2520 (10 years).

    Returns:
    - P_t (np.ndarray): Array of synthetic stock price values.
    """

    def __init__(
        self,
        P0: float = 100,
        mu: float = 0.001,
        sigma: float = 0.02,
        A: float = 0.05,
        f: float = 1 / 252,
        phi: float = 0,
        T: int = 2520,
    ) -> None:
        """
        Initializes the toy object with the given parameters.
        """
        self.P0 = P0  # Initial stock price
        self.mu = mu  # Drift (trend)
        self.sigma = sigma  # Volatility (randomness)
        self.A = A  # Amplitude of seasonal component
        self.f = f  # Frequency of seasonal component
        self.phi = phi  # Phase shift
        self.T = T  # Time horizon

        self.t = None  # Time array
        self.W_t = None
        self.P_t = None

    def __call__(self) -> np.array:
        """
        Generates the synthetic stock price data using the given parameters.
        """

        # Time array
        if self.t is None:
            self.t = np.arange(self.T)

        # Brownian motion (Wiener process)
        if self.W_t is None:
            self.W_t = np.random.normal(0, 1, self.T).cumsum()

        # Stock price formula
        self.P_t = (
            self.P0
            * np.exp(self.mu * self.t + self.sigma * self.W_t)
            * (1 + self.A * np.sin(2 * np.pi * self.f * self.t + self.phi))
        )

    def _check_is_called(self) -> bool:
        """
        Checks if the object has been called to generate synthetic stock price data.
        """
        if self.P_t is None:
            warn(
                "No synthetic stock price data generated yet. Please call the object first."
            )
            return False
        return True

    def get_data_numpy(self) -> np.array:
        """
        Returns the time array and synthetic stock price data as a NumPy array of size (T, 2).
        """
        if not self._check_is_called():
            return

        # Return the concatenated the time array and the stock price array
        return np.stack([self.t, self.P_t], axis=1)

    def get_data_pandas(self) -> pd.DataFrame:
        """
        Returns the time array and synthetic stock price data as a pandas DataFrame.
        """

        if not self._check_is_called():
            return

        # Create a DataFrame from the synthetic stock price data
        df = pd.DataFrame(self.get_data_numpy(), columns=["Time", "Stock Price"])
        df = df.astype({"Time": int})
        df = df.set_index("Time")
        return df

    def plot(self, save_fig_name=None) -> None:
        """
        Plots the synthetic stock price data.

        Args:
        - save_fig_name (str): Name of the file to save the plot (unless None). Default is None.
        """

        if not self._check_is_called():
            return

        fig = plt.figure(figsize=(12, 6))
        plt.plot(self.t, self.P_t, label="Stock Price")
        plt.suptitle("Synthetic Stock Price Curve", fontsize=16)
        plt.title(
            f"P0={self.P0}, mu={self.mu}, sigma={self.sigma}, A={self.A}, f={self.f:.6f}, phi={self.phi}, T={self.T}",
            fontsize=12,
        )
        plt.xlabel("Time (days)")
        plt.ylabel("Stock Price")
        if save_fig_name is not None:
            fig.savefig(save_fig_name)
        else:
            plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generate synthetic stock price data.")
    parser.add_argument("--P0", type=float, default=100, help="Initial stock price")
    parser.add_argument("--mu", type=float, default=0.001, help="Drift (trend)")
    parser.add_argument(
        "--sigma", type=float, default=0.02, help="Volatility (randomness)"
    )
    parser.add_argument(
        "--A", type=float, default=0.05, help="Amplitude of seasonal component"
    )
    parser.add_argument(
        "--f", type=float, default=1 / 252, help="Frequency of seasonal component"
    )
    parser.add_argument("--phi", type=float, default=0, help="Phase shift")
    parser.add_argument("--T", type=int, default=2520, help="Time horizon")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")

    args = parser.parse_args()

    P0 = args.P0
    mu = args.mu
    sigma = args.sigma
    A = args.A
    f = args.f
    phi = args.phi
    T = args.T
    np.random.seed(args.seed)

    # Initialize the toy object
    syn_data_obj = SyntheticStockData(
        P0=P0,
        mu=mu,
        sigma=sigma,
        A=A,
        f=f,
        phi=phi,
        T=T,
    )

    # Generate the synthetic stock price data
    syn_data_obj()

    # Plot the stock price curve
    syn_data_obj.plot(save_fig_name="synthetic_stock_price_curve.png")

    # Create a DataFrame from the synthetic stock price data
    df = syn_data_obj.get_data_pandas()

    # Save the DataFrame to a CSV file
    df.to_csv("synthetic_stock_price_data.csv")
