from typing import List, Tuple, Union
from warnings import catch_warnings, simplefilter, warn

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm, t
from statsmodels.tsa.seasonal import STL

np.random.seed(0)


class EDA:
    """
    Class for exploratory data analysis (EDA) of stock prices.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        label: str,
        value_col: str = "Adj Close",
        date_col: Union[str, None] = None,
        volume_col: Union[str, None] = None,
    ) -> None:
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
            if (
                self.date_col not in self.data.columns
                and self.data.index.name == self.date_col
            ):
                self.data = self.data.copy().reset_index()

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

    def add_moving_average(
        self, window: Union[int, List[int]], column: pd.Series
    ) -> None:
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

    def get_dist_params(self, data, dist_type="t"):
        """Fits a distribution to the given data and returns the parameters.
        Currently, only Student's t and Gaussian distributions are supported.

        Args:
            data (np.ndarray or pd.Series): Data to fit the distribution
            dist_type (str): Distribution type (choices: "t" or "gaussian"). Default is "t".
        """
        assert dist_type in ["t", "gaussian"], f"Invalid distribution type: {dist_type}"
        if dist_type == "t":
            return t.fit(data.dropna())
        elif dist_type == "gaussian":
            return norm.fit(data.dropna())

    def _save_figures(
        self, fig, save_fig_name: str, save_fmts: List[str] = ["html", "png"]
    ) -> None:
        # Ensure valid save formats
        valid_formats = ["html", "png"]
        for fmt in save_fmts:
            if fmt not in valid_formats:
                raise ValueError(
                    f"Invalid save format: {fmt}. Please choose from: {valid_formats}"
                )

        # Ensure figure is a Plotly object

        # Save figure in the specified formats
        with catch_warnings():
            simplefilter("ignore")
            if "png" in save_fmts:
                fig.write_image(save_fig_name + ".png")
            if "html" in save_fmts:
                fig.write_html(save_fig_name + ".html")

    def plot_price(
        self,
        plot_ma: bool = False,
        save_fig_name: Union[str, None] = None,
        save_fmts: List[str] = ["html", "png"],
    ) -> None:
        """
        Plots the stock price data using Plotly.

        Args:
        - plot_ma (bool): If True, plot all moving averages along with the daily returns. Default is False.
        - save_fig_name (str): Name of the file to save the plot (unless None). Default is None.
        """
        # Create the figure
        fig = go.Figure()

        # Determine the x-axis values (date or index)
        x_col = (
            self.data[self.date_col] if self.date_col is not None else self.data.index
        )

        # Add the line plot for the close price
        fig.add_trace(
            go.Scatter(
                x=x_col,
                y=self.data[self.value_col],
                mode="lines",
                name="Close Price",
                line=dict(color="blue", width=2),  # Line color and width
            )
        )

        # Set x-axis label based on the presence of date_col
        x_label = "Date" if self.date_col is not None else "Index"

        # Update layout: title, labels
        fig.update_layout(
            title_text=f"Stock Price of {self.label}",  # Set title
            xaxis_title=x_label,  # X-axis label
            yaxis_title="Price",  # Y-axis label
            width=1200,  # Set figure size (width)
            height=600,  # Set figure size (height)
            showlegend=True,  # Display the legend
        )

        # Automatically handle date formatting if 'date_col' is present
        if self.date_col:
            fig.update_xaxes(tickformat="%Y-%m-%d")  # Format x-axis for date

        # Add moving averages if available
        if plot_ma:
            ma_columns = [
                col
                for col in self.data.columns
                if col.startswith(self.value_col) and col.endswith("-day-MA")
            ]
            for ma_col in ma_columns:
                fig.add_trace(
                    go.Scatter(
                        x=x_col,
                        y=self.data[ma_col],
                        mode="lines",
                        name=ma_col[len(self.value_col) + 1 :],
                        line=dict(width=3),
                        opacity=0.9,  # Add transparency to the line
                    )
                )

        if save_fig_name is not None:
            self._save_figures(fig, save_fig_name, save_fmts=save_fmts)
        else:
            fig.show()

    def plot_volume(
        self,
        save_fig_name: Union[str, None] = None,
        save_fmts: List[str] = ["html", "png"],
    ) -> None:
        """
        Plots the stock volume data using Plotly.

        Args:
        - save_fig_name (str): Name of the file to save the plot (unless None). Default is None.
        """
        if self.volume_col is None:
            warn(
                "Volume column not provided. Please provide the volume column to plot the volume data."
            )
            return

        # Create the figure
        fig = go.Figure()

        # Check if 'date_col' is available
        x_col = (
            self.data[self.date_col] if self.date_col is not None else self.data.index
        )

        # Add bar plot for volume
        fig.add_trace(
            go.Bar(
                x=x_col,  # Date or index
                y=self.data[self.volume_col],  # Volume column
                name="Volume",
                marker=dict(color="green"),  # Set bar color to green
            )
        )

        # Set x-axis label based on the presence of date_col
        x_label = "Date" if self.date_col is not None else "Index"

        # Update layout: title, labels, and format
        fig.update_layout(
            title_text=f"Stock Volume of {self.label}",  # Set title
            xaxis_title=x_label,  # X-axis label
            yaxis_title="Volume",  # Y-axis label
            width=1200,  # Set figure size (width)
            height=600,  # Set figure size (height)
            showlegend=True,  # Display the legend
        )

        # Automatically handle date formatting if 'date_col' is present
        if self.date_col:
            fig.update_xaxes(tickformat="%Y-%m-%d")  # Format x-axis for date

        if save_fig_name is not None:
            self._save_figures(fig, save_fig_name, save_fmts=save_fmts)
        else:
            fig.show()

    def plot_return(
        self,
        plot_ma: bool = False,
        hist_bins: int = 50,
        save_fig_name: Union[str, None] = None,
        save_fmts: List[str] = ["png", "html"],
    ) -> None:
        """
        Plots the daily stock price returns using Plotly.

        Args:
        - plot_ma (bool): If True, plot all moving averages along with the daily returns. Default is False.
        - hist_bins (int): Number of bins to plot in histogram of daily returns. Default is 50.
        - save_fig_name (str): Name of the file to save the plot (unless None). Default is None.
        """

        # Top figure: Line plot of daily return and moving averages
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=False,
            subplot_titles=[
                f"Daily Stock Price Returns of {self.label}",
                "Distribution of Returns",
            ],
            vertical_spacing=0.1,
        )

        # Plot daily return
        x_col = (
            self.data[self.date_col] if self.date_col is not None else self.data.index
        )

        if self.date_col:
            hovertemplate = (
                lambda label: f"%{{x|%Y-%m-%d}}<br>Daily Return{label}: %{{y:.2%}}<extra></extra>"
            )  # Format x as date
        else:
            hovertemplate = (
                lambda label: f"Index: %{{x}}<br>Daily Return{label}: %{{y:.2%}}<extra></extra>"
            )  # Show index value

        fig.add_trace(
            go.Scatter(
                x=x_col,
                y=self.data[self.return_col],
                mode="lines+markers",
                name="Daily Return of Close Price",
                line=dict(color="blue", dash="dash"),
                marker=dict(symbol="circle", color="blue"),
                hovertemplate=hovertemplate(""),
            ),
            row=1,
            col=1,
        )

        # Plot moving averages if available
        if plot_ma:
            ma_columns = [
                col
                for col in self.data.columns
                if col.startswith(self.return_col) and col.endswith("-day-MA")
            ]
            for ma_col in ma_columns:
                fig.add_trace(
                    go.Scatter(
                        x=x_col,
                        y=self.data[ma_col],
                        mode="lines",
                        name=ma_col[len(self.return_col) + 1 :],
                        line=dict(width=3),
                        opacity=0.9,
                        hovertemplate=hovertemplate(
                            label=f" ({ma_col[len(self.return_col)+1:]})"
                        ),
                    ),
                    row=1,
                    col=1,
                )

        # Set x-axis labels
        if self.date_col:
            fig.update_xaxes(title_text="Date", row=1, col=1)
        else:
            fig.update_xaxes(title_text="Index", row=1, col=1)

        # Set y-axis format to percentage
        fig.update_yaxes(title_text="Daily Return", tickformat=".0%", row=1, col=1)

        # Bottom figure: Histogram with fitted distributions
        data_min = self.data[self.return_col].min()
        data_max = self.data[self.return_col].max()
        data_range = [
            min(-abs(data_min), -abs(data_max)),
            max(abs(data_min), abs(data_max)),
        ]
        hist = np.histogram(
            self.data[self.return_col].dropna(),
            bins=hist_bins,
            range=data_range,
            density=True,
        )
        x_hist = hist[1][:-1]
        y_hist = hist[0]

        # Plot histogram
        fig.add_trace(
            go.Bar(
                x=x_hist,
                y=y_hist,
                name="Histogram",
                marker=dict(color="blue"),
                showlegend=False,
                hovertemplate="Bin centre: %{x:.2%}<br>Density: %{y:.2f}<extra></extra>",
            ),
            row=2,
            col=1,
        )

        # Fit Student's t distribution
        t_params = self.get_dist_params(self.data[self.return_col], dist_type="t")
        t_dist = t(*t_params)
        x_fit = np.linspace(
            self.data[self.return_col].min(), self.data[self.return_col].max(), 100
        )

        # Plot t-distribution
        fig.add_trace(
            go.Scatter(
                x=x_fit,
                y=t_dist.pdf(x_fit),
                mode="lines",
                name=f"Student's t (df={t_params[0]:.6f})",
                line=dict(color="red", width=4, dash="dot"),
                opacity=0.9,
            ),
            row=2,
            col=1,
        )

        # Fit Gaussian distribution
        gaussian_params = self.get_dist_params(
            self.data[self.return_col], dist_type="gaussian"
        )
        gaussian_dist = norm(*gaussian_params)

        # Plot Gaussian distribution
        fig.add_trace(
            go.Scatter(
                x=x_fit,
                y=gaussian_dist.pdf(x_fit),
                mode="lines",
                name=f"Gaussian (mean={gaussian_params[0]:.6f}, std={gaussian_params[1]:.6f})",
                line=dict(color="magenta", width=4, dash="dot"),
                opacity=0.9,
            ),
            row=2,
            col=1,
        )

        # Update x-axis format for percentage and y-axis format for frequency
        fig.update_xaxes(title_text="Daily Return", tickformat=".0%", row=2, col=1)
        fig.update_yaxes(title_text="Frequency", row=2, col=1)

        # Set layout
        fig.update_layout(
            height=800,
            width=1200,
            title_text=f"Daily Stock Price Returns of {self.label}",
            showlegend=True,
        )

        if save_fig_name is not None:
            self._save_figures(fig, save_fig_name, save_fmts=save_fmts)
        else:
            fig.show()

    def stl_decomposition(
        self,
        freq: int = 252,
        save_fig_name: Union[str, None] = None,
        save_fmts: List[str] = ["html", "png"],
    ) -> None:
        """
        Decomposes the stock price data into trend, seasonal, and residual components using STL decomposition.

        Args:
        - freq (int): Frequency of the data. Default is 252 (for daily data), but choose 5 for weekly data and 21 for monthly data.
        - save_fig_name (str): Name of the file to save the plot (unless None). Default is None.
        """

        if self.date_col is None:
            warn(
                "Date column not provided. Please provide the date column to decompose the stock price data."
            )
            return

        data_series = pd.Series(
            self.data[self.value_col].values,
            index=self.data[self.date_col],
            name=f"STL decomposition of {self.label} {self.value_col}",
        )

        stl = STL(data_series, period=freq)
        res = stl.fit()

        # Create the figure for STL decomposition components
        fig = go.Figure()

        # Add observed data trace
        fig.add_trace(
            go.Scatter(
                x=data_series.index,
                y=data_series.values,
                mode="lines",
                name="Observed",
                line=dict(color="blue"),
            )
        )

        # Add trend component trace
        fig.add_trace(
            go.Scatter(
                x=data_series.index,
                y=res.trend,
                mode="lines",
                name="Trend",
                line=dict(color="red"),
            )
        )

        # Add seasonal component trace
        fig.add_trace(
            go.Scatter(
                x=data_series.index,
                y=res.seasonal,
                mode="lines",
                name="Seasonal",
                line=dict(color="green"),
            )
        )

        # Add residual component trace
        fig.add_trace(
            go.Scatter(
                x=data_series.index,
                y=res.resid,
                mode="lines",
                name="Residual",
                line=dict(color="orange"),
            )
        )

        # Update layout
        fig.update_layout(
            title=f"STL Decomposition of {self.label} {self.value_col}",
            xaxis_title="Date" if self.date_col is not None else "Index",
            yaxis_title="Value",
            width=1200,  # Set width
            height=800,  # Set height
            legend_title="Components",
            showlegend=True,
        )

        # Format x-axis dates
        if self.date_col:
            fig.update_xaxes(tickformat="%Y-%m-%d")  # Format x-axis for date

        if save_fig_name is not None:
            self._save_figures(fig, save_fig_name, save_fmts=save_fmts)
        else:
            fig.show()


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
    print("Summary of synthetic data:\n")
    print(eda.get_summary(), "\n")
    print(f"Index range: {eda.get_index_range()}\n")

    # Import real data
    label = "MSFT"
    real_data = pd.read_csv(PROJECT_PATH / "data" / "stock_data" / f"{label}.csv")
    eda_real = EDA(
        data=real_data,
        label=label,
        value_col="Adj Close",
        date_col="Date",
        volume_col="Volume",
    )
    eda_real.add_moving_average(
        window=[7, 15, 30], column=eda_real.data[eda_real.value_col]
    )
    eda_real.add_moving_average(
        window=[7, 15, 30], column=eda_real.data[eda_real.return_col]
    )
    eda_real.plot_price(plot_ma=True)
    eda_real.plot_volume()
    eda_real.plot_return(plot_ma=True)
    eda_real.stl_decomposition()
    print(f"Summary of {label} data:\n")
    print(eda_real.get_summary(), "\n")
    print(f"Date range: {eda_real.get_date_range()}\n")
