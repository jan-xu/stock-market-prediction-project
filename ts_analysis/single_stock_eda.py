from warnings import warn
from typing import Union, List, Tuple
from datetime import datetime
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
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

        if self.date_col is not None:
            self.data[self.date_col] = pd.to_datetime(self.data[self.date_col])

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

    def plot_price(self, save_fig_name: Union[str, None] = None) -> None:
        """
        Plots the stock price data using Seaborn.

        Args:
        - save_fig_name (str): Name of the file to save the plot (unless None). Default is None.
        """
        plt.figure(figsize=(12, 6))
        if self.date_col is not None:
            sns.lineplot(data=self.data, x=self.date_col, y=self.value_col, label="Close Price", color="b", lw=1)
            plt.xlabel("Date")
            locator = mdates.AutoDateLocator()
            plt.gca().xaxis.set_major_locator(locator)
            plt.gca().xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
            plt.gcf().autofmt_xdate()  # Rotate the x labels for better visibility
        else:
            sns.lineplot(data=self.data, x=self.data.index, y=self.value_col, label="Close Price", color="b", lw=1)
            plt.xlabel("Index")
        plt.title(f"Stock Price of {self.label}")
        plt.ylabel("Price")
        plt.legend()
        if save_fig_name is not None:
            plt.savefig(save_fig_name)
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
            sns.lineplot(data=self.data, x=self.date_col, y=self.volume_col, label="Volume", color="g", lw=1)
            plt.xlabel("Date")
            locator = mdates.AutoDateLocator()
            plt.gca().xaxis.set_major_locator(locator)
            plt.gca().xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
            plt.gcf().autofmt_xdate()
        else:
            sns.lineplot(data=self.data, x=self.data.index, y=self.volume_col, label="Volume", color="g", lw=1)
            plt.xlabel("Index")
        plt.title(f"Stock Volume of {self.label}")
        plt.ylabel("Volume")
        plt.legend()
        if save_fig_name is not None:
            plt.savefig(save_fig_name)
        else:
            plt.show()

if __name__ == "__main__":

    # Import toy data
    from toy import SyntheticStockData

    # Generate synthetic stock price data
    syn_obj = SyntheticStockData()
    syn_obj()
    df = syn_obj.get_data_pandas()

    # Initialize EDA object
    eda = EDA(data=df, label="SYN_DATA", value_col="Stock Price")
    eda.plot_price(save_fig_name="SYN_DATA.png")
    print(eda.get_summary())
    print(f"Index range: {eda.get_index_range()}")

    real_data = pd.read_csv("csv/AAPL_historical_data.csv")
    eda_real = EDA(data=real_data, label="AAPL", value_col="Adj Close", date_col="Date", volume_col="Volume")
    eda_real.plot_price(save_fig_name="AAPL.png")
    eda_real.plot_volume(save_fig_name="AAPL_Vol.png")
    print(eda_real.get_summary())
    print(f"Date range: {eda_real.get_date_range()}")
    from IPython import embed; embed()