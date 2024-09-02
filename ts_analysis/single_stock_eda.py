from warnings import warn
from typing import Union, List, Tuple
from datetime import datetime
import numpy as np
import pandas as pd

from statsmodels.tsa.seasonal import STL
from scipy.stats import t
from scipy.stats import norm

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import PercentFormatter

import seaborn as sns
sns.set_style('whitegrid')
plt.style.use("fivethirtyeight")

np.random.seed(0)

class EDA:
    """
    Class for exploratory data analysis (EDA) of stock prices.
    """
    def __init__(self, data: pd.DataFrame, label: str, value_col: str = "Close", date_col: Union[str, None] = None, volume_col: Union[str, None] = None) -> None:
        """
        Initializes the EDA object with the given stock price data.
        """
        self.data = data
        self.label = label
        self.value_col = value_col
        self.date_col = date_col
        self.volume_col = volume_col
        self.return_col = "Return"

        if self.date_col is not None:
            self.data[self.date_col] = pd.to_datetime(self.data[self.date_col])

        # Compute daily return
        self.data[self.return_col] = self.data[self.value_col].pct_change()

    def get_summary(self) -> pd.DataFrame:
        """
        Returns a summary of the stock price data.
        """
        summary = self.data.select_dtypes(include=[np.number]).describe()
        return summary

    def get_index_range(self) -> Tuple[int, int]:
        """
        Returns the index range of the stock price data.
        """
        return int(self.data.index.min()), int(self.data.index.max())

    def get_date_range(self) -> Union[Tuple[str, str], Tuple[int, int]]:
        """
        Returns the date range of the stock price data.
        """
        if self.date_col is not None:
            start_date = self.data[self.date_col].min().strftime("%Y-%m-%d")
            end_date = self.data[self.date_col].max().strftime("%Y-%m-%d")
            return start_date, end_date
        else:
            warn("Date column not provided. Returning index range instead.")
            return self.get_index_range()

    def add_moving_average(self, window: Union[int, List[int]], column: pd.Series) -> None:
        """Add a moving average column of the stock price data.

        Args:
        - window (Union[int, List[int]]): moving average window size(s).
        - column (pd.Series): column to calculate moving average on.
        """
        if isinstance(window, int):
            window = [window]

        column_label = column.name

        for w in window:
            ma_label = f"{column_label}_{w}-day-MA"
            self.data[ma_label] = column.rolling(window=w).mean()

    def plot_price(self, plot_ma: bool = False, save_fig_name: Union[str, None] = None) -> None:
        """
        Plots the stock price data using Seaborn.

        Args:
        - plot_ma (bool): If True, plot all moving averages along with the daily returns. Default is False.
        - save_fig_name (str): Name of the file to save the plot (unless None). Default is None.
        """
        plt.figure(figsize=(12, 6))
        if self.date_col is not None:
            sns.lineplot(data=self.data, x=self.date_col, y=self.value_col, label="Close Price", color="b", lw=2)
            plt.xlabel("Date")
            locator = mdates.AutoDateLocator()
            plt.gca().xaxis.set_major_locator(locator)
            plt.gca().xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
            plt.gcf().autofmt_xdate()  # Rotate the x labels for better visibility
        else:
            sns.lineplot(data=self.data, x=self.data.index, y=self.value_col, label="Close Price", color="b", lw=2)
            plt.xlabel("Index")
        plt.title(f"Stock Price of {self.label}")
        plt.ylabel("Price")

        if plot_ma:
            ma_columns = [col for col in self.data.columns if col.startswith(self.value_col) and col.endswith("-day-MA")]
            x_col = self.date_col if self.date_col is not None else self.data.index
            for ma_col in ma_columns:
                sns.lineplot(data=self.data, x=x_col, y=ma_col, label=ma_col[len(self.value_col)+1:], lw=3, alpha=0.75)

        plt.legend()
        plt.tight_layout()
        if save_fig_name is not None:
            plt.savefig(save_fig_name)
            plt.close()
        else:
            plt.show()

    def plot_volume(self, save_fig_name: Union[str, None] = None) -> None:
        """
        Plots the stock volume data using Seaborn.

        Args:
        - save_fig_name (str): Name of the file to save the plot (unless None). Default is None.
        """
        if self.volume_col is None:
            warn("Volume column not provided. Please provide the volume column to plot the volume data.")
            return

        plt.figure(figsize=(12, 6))
        if self.date_col is not None:
            sns.barplot(data=self.data, x=self.date_col, y=self.volume_col, label="Volume", color="g")
            plt.xlabel("Date")
            locator = mdates.AutoDateLocator()
            plt.gca().xaxis.set_major_locator(locator)
            plt.gcf().autofmt_xdate()
        else:
            sns.barplot(data=self.data, x=self.data.index, y=self.volume_col, label="Volume", color="g")
            plt.xlabel("Index")
        plt.title(f"Stock Volume of {self.label}")
        plt.ylabel("Volume")
        plt.legend()
        plt.tight_layout()
        if save_fig_name is not None:
            plt.savefig(save_fig_name)
            plt.close()
        else:
            plt.show()

    def plot_return(self, plot_ma: bool = False, hist_bins: int = 50, save_fig_name: Union[str, None] = None) -> None:
        """
        Plots the daily stock price returns using Seaborn.

        Args:
        - plot_ma (bool): If True, plot all moving averages along with the daily returns. Default is False.
        - hist_bins (int): Number of bins to plot in histogram of daily returns. Default is 50.
        - save_fig_name (str): Name of the file to save the plot (unless None). Default is None.
        """
        fig, axs = plt.subplots(2, 1, figsize=(12, 12), sharex=False)

        # Plot top figure
        ax1 = axs[0]
        if self.date_col is not None:
            sns.lineplot(data=self.data, x=self.date_col, y=self.return_col, label="Daily Return of Close Price", color="b", lw=2, linestyle='--', marker='o', ax=ax1)
            ax1.set_xlabel("Date")
            locator = mdates.AutoDateLocator()
            ax1.xaxis.set_major_locator(locator)
            ax1.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
            ax1.tick_params(axis='x', rotation=45)
        else:
            sns.lineplot(data=self.data, x=self.data.index, y=self.return_col, label="Daily Return of Close Price", color="b", lw=2, linestyle='--', marker='o', ax=ax1)
            ax1.set_xlabel("Index")
        ax1.set_title(f"Daily Stock Price Returns of {self.label}")
        ax1.set_ylabel("Daily Return")
        ax1.yaxis.set_major_formatter(PercentFormatter(1))

        if plot_ma:
            ma_columns = [col for col in self.data.columns if col.startswith(self.return_col) and col.endswith("-day-MA")]
            x_col = self.date_col if self.date_col is not None else self.data.index
            for ma_col in ma_columns:
                sns.lineplot(data=self.data, x=x_col, y=ma_col, label=ma_col[len(self.return_col)+1:], lw=3, alpha=0.75, ax=ax1)

        ax1.legend()

        # Plot bottom figure
        ax2 = axs[1]
        sns.histplot(data=self.data, x=self.return_col, bins=hist_bins, ax=ax2, stat="density")

        # Fit t-student distribution
        t_params = t.fit(self.data[self.return_col].dropna())
        t_dist = t(*t_params)

        # Fit Gaussian distribution
        gaussian_params = norm.fit(self.data[self.return_col].dropna())
        gaussian_dist = norm(*gaussian_params)

        # Plot fitted distributions
        x = np.linspace(self.data[self.return_col].min(), self.data[self.return_col].max(), 100)
        ax2.plot(x, t_dist.pdf(x), label=f"Student's t (df={t_params[0]:.2f})", color="r", lw=2, alpha=0.75)
        ax2.plot(x, gaussian_dist.pdf(x), label=f"Gaussian (mean={gaussian_params[0]:.3f}, std={gaussian_params[1]:.3f})", color="m", lw=2, alpha=0.75)

        ax2.set_xlabel("Daily Return")
        ax2.set_ylabel("Frequency")
        ax2.xaxis.set_major_formatter(PercentFormatter(1))
        ax2.legend()

        plt.tight_layout()
        if save_fig_name is not None:
            plt.savefig(save_fig_name)
            plt.close()
        else:
            plt.show()


    def stl_decomposition(self, freq: int = 252, save_fig_name: Union[str, None] = None) -> None:
        """
        Decomposes the stock price data into trend, seasonal, and residual components using STL decomposition.

        Args:
        - freq (int): Frequency of the data. Default is 252 (for daily data), but choose 5 for weekly data and 21 for monthly data.
        - save_fig_name (str): Name of the file to save the plot (unless None). Default is None.
        """

        if self.date_col is None:
            warn("Date column not provided. Please provide the date column to decompose the stock price data.")
            return

        if self.date_col is not None:
            data_series = pd.Series(self.data[self.value_col].values, index=self.data[self.date_col], name=f"STL decomposition of {self.label} {self.value_col}")
        else:
            data_series = pd.Series(self.data[self.value_col].values, name=f"STL decomposition of {self.label} {self.value_col}")
        stl = STL(data_series, period=freq)
        res = stl.fit()
        res.plot()
        plt.gcf().autofmt_xdate()
        plt.tight_layout()
        if save_fig_name is not None:
            plt.savefig(save_fig_name)
            plt.close()
        else:
            plt.show()


if __name__ == "__main__":

    from pathlib import Path
    PROJECT_PATH = Path("/home/janx/repos/stock-market-prediction-project")

    import sys
    sys.path.append(str(PROJECT_PATH))

    # Import toy data
    from toy import SyntheticStockData

    # Generate synthetic stock price data
    syn_obj = SyntheticStockData()
    syn_obj()
    df = syn_obj.get_data_pandas()

    # Initialize EDA object
    eda = EDA(data=df, label="SYN_DATA", value_col="Stock Price")
    eda.add_moving_average(window=[7, 15, 30], column=eda.data[eda.value_col])
    eda.add_moving_average(window=[7, 15, 30], column=eda.data[eda.return_col])
    eda.plot_price(plot_ma=True)
    eda.plot_return(plot_ma=True)
    print("Summary of synthetic data:")
    print(eda.get_summary())
    print(f"Index range: {eda.get_index_range()}")

    real_data = pd.read_csv(PROJECT_PATH / "csv" / "AAPL_historical_data.csv")
    eda_real = EDA(data=real_data, label="AAPL", value_col="Adj Close", date_col="Date", volume_col="Volume")
    eda_real.add_moving_average(window=[7, 15, 30], column=eda_real.data[eda_real.value_col])
    eda_real.add_moving_average(window=[7, 15, 30], column=eda_real.data[eda_real.return_col])
    eda_real.plot_price(plot_ma=True)
    eda_real.plot_volume()
    eda_real.plot_return(plot_ma=True)
    eda_real.stl_decomposition()
    print("Summary of AAPL data:")
    print(eda_real.get_summary())
    print(f"Date range: {eda_real.get_date_range()}")
