from itertools import cycle
from pathlib import Path

import plotly.express as px
import plotly.graph_objects as go

from ts_analysis import EDA

COLORS = px.colors.qualitative.Plotly


def eda_plots(
    df,
    label,
    value_col,
    date_col=None,
    volume_col=None,
    add_ma=False,
    save_folder=None,
    run_folder=None,
):
    """
    Return plots from exploratory data analysis on the given time-series data.

    Parameters
    ----------
    df : pd.DataFrame
        The time-series data to be analyzed.
    label : str
        The label (stock ticker symbol) of the time-series data.
    value_col : str
        The name of the column containing the value of the time-series data (usually "Adj Close").
    date_col : str, optional
        The name of the column containing the date of the time-series data (usually "Date").
    volume_col : str, optional
        The name of the column containing the volume of the time-series data (usually "Volume").
    add_ma : bool, optional
        Whether to add moving averages to the plots. Default: False.
    save_folder : str, optional
        The folder to save plots in. If both `save_folder` and `run_folder` are None,
        the plots will only be shown, not saved. Default: None.
    run_folder : str, optional
        The experiment folder (with defined folder structure) to save plots in. If both
        `save_folder` and `run_folder` are None, the plots will only be shown, not saved.
        Default: None.
    """

    if save_folder is not None and run_folder is not None:
        raise ValueError(
            "Only one of `save_folder` and `run_folder` should be provided."
        )
    else:
        if save_folder is not None:
            Path(save_folder).mkdir(parents=True, exist_ok=True)
        elif run_folder is not None:
            Path(f"{run_folder}/figures/png").mkdir(parents=True, exist_ok=True)
            Path(f"{run_folder}/figures/html").mkdir(parents=True, exist_ok=True)

    eda = EDA(
        data=df,
        label=label,
        value_col=value_col,
        date_col=date_col,
        volume_col=volume_col,
    )
    if add_ma:
        eda.add_moving_average(window=[7, 15, 30], column=eda.data[eda.value_col])
        eda.add_moving_average(window=[7, 15, 30], column=eda.data[eda.return_col])

    logged_plots = []

    def _save_fig_name(fig_type, fmt, add_suffix=False):
        if save_folder is not None:
            return f"{save_folder}/{eda.label}_{fig_type}" + (
                f".{fmt}" if add_suffix else ""
            )
        elif run_folder is not None:
            return f"{run_folder}/figures/{fmt}/{eda.label}_{fig_type}" + (
                f".{fmt}" if add_suffix else ""
            )
        return None

    eda.plot_price(
        plot_ma=add_ma,
        save_fig_name=_save_fig_name("price_plot", "png"),
        save_fmts=["png"],
    )
    eda.plot_price(
        plot_ma=add_ma,
        save_fig_name=_save_fig_name("price_plot", "html"),
        save_fmts=["html"],
    )
    eda.plot_return(
        plot_ma=add_ma,
        save_fig_name=_save_fig_name("return_plot", "png"),
        save_fmts=["png"],
    )
    eda.plot_return(
        plot_ma=add_ma,
        save_fig_name=_save_fig_name("return_plot", "html"),
        save_fmts=["html"],
    )

    if save_folder is not None or run_folder is not None:
        logged_plots.extend(
            [
                _save_fig_name("price_plot", "png", add_suffix=True),
                _save_fig_name("price_plot", "html", add_suffix=True),
                _save_fig_name("return_plot", "png", add_suffix=True),
                _save_fig_name("return_plot", "html", add_suffix=True),
            ]
        )

    if eda.volume_col is not None:
        eda.plot_volume(
            save_fig_name=_save_fig_name("volume_plot", "png"), save_fmts=["png"]
        )
        eda.plot_volume(
            save_fig_name=_save_fig_name("volume_plot", "html"), save_fmts=["html"]
        )
        if save_folder is not None or run_folder is not None:
            logged_plots.extend(
                [
                    _save_fig_name("volume_plot", "png", add_suffix=True),
                    _save_fig_name("volume_plot", "html", add_suffix=True),
                ]
            )

    if eda.date_col is not None:
        eda.stl_decomposition(
            save_fig_name=_save_fig_name("stl_decomposition", "png"), save_fmts=["png"]
        )
        eda.stl_decomposition(
            save_fig_name=_save_fig_name("stl_decomposition", "html"),
            save_fmts=["html"],
        )
        if save_folder is not None or run_folder is not None:
            logged_plots.extend(
                [
                    _save_fig_name("stl_decomposition", "png", add_suffix=True),
                    _save_fig_name("stl_decomposition", "html", add_suffix=True),
                ]
            )

    if save_folder is not None or run_folder is not None:
        print(f"Logged plots: \n-> {'\n-> '.join(logged_plots)}\n")

    print("Summary of synthetic data:")
    print(eda.get_summary(), "\n")

    if eda.date_col is None:
        print(f"Index range: {eda.get_index_range()}\n")
    else:
        print(f"Date range: {eda.get_date_range()}\n")

    return logged_plots


def plot_pred_vs_gt(gt_data, pred_data, val_size, pred_horizon, var_type="Return"):
    """
    Plot the return predictions against the ground truth.

    Parameters
    ----------
    gt_data : np.ndarray or torch.Tensor
        The ground truth values of the time-series data.
    pred_data : np.ndarray or torch.Tensor
        The predicted values of the time-series data.
    val_size : int
        The size of the validation set.
    pred_horizon : int
        Prediction horizon size.
    var_type : str, optional
        The type of variable to plot. Default: "Return".
    """
    color_cycle = cycle(COLORS)

    assert var_type in [
        "Return",
        "Stock Price",
    ], "Variable type must be 'Return' or 'Stock Price'."

    len_gt = gt_data.shape[0]
    len_pred = pred_data.shape[0]
    len_history = len_gt - val_size

    gt_indices = list(range(-len_history, val_size))
    next_day_pred_indices = list(range(0, len_pred))

    next_day_fig = go.Figure()

    next_day_fig.add_trace(
        go.Scatter(
            x=gt_indices,
            y=gt_data[:, 0],
            mode="lines",
            name="Ground truth",
            line=dict(color="black", width=2),
        )
    )

    next_day_fig.add_trace(
        go.Scatter(
            x=next_day_pred_indices,
            y=pred_data[:, 0, 0],
            mode="lines",
            name="Pred (next-day)",
            line=dict(color="blue", width=2),
        )
    )

    next_day_fig.update_layout(
        title_text=f"Next-day prediction",
        xaxis_title="Time index (negative: history)",
        yaxis_title=var_type,
        width=1200,
        height=600,
        showlegend=True,
    )

    if pred_horizon > 1:
        multi_day_fig = go.Figure()

        multi_day_fig.add_trace(
            go.Scatter(
                x=gt_indices[-(val_size + 1) :],
                y=gt_data[-(val_size + 1) :, 0],
                mode="lines",
                name="Ground truth",
                line=dict(color="black", width=4),
            )
        )

        multi_day_fig.add_trace(
            go.Scatter(
                x=next_day_pred_indices,
                y=pred_data[:, 0, 0],
                mode="lines+markers",
                name="Pred (next-day)",
                opacity=0.8,
                line=dict(color="blue", width=4),
                marker=dict(color="blue", size=12, symbol="star"),
            )
        )

        for i in range(len_pred):
            color = next(color_cycle)

            multi_day_indices = list(range(i, pred_horizon + i))
            multi_day_fig.add_trace(
                go.Scatter(
                    x=multi_day_indices,
                    y=pred_data[i, :, 0],
                    mode="lines",
                    name=f"Multi-day pred (start: index = {i})",
                    line=dict(width=2, color=color),
                    legendgroup=i,
                )
            )
            # Additional line connected to the start of the existing line
            multi_day_fig.add_trace(
                go.Scatter(
                    x=[i - 1, i],
                    y=[gt_data[-(val_size + 1) + i, 0], pred_data[i, 0, 0]],
                    mode="lines",
                    opacity=0.5,
                    line=dict(color=color, width=2, dash="dash"),
                    showlegend=False,
                    legendgroup=i,
                )
            )

        multi_day_fig.update_layout(
            title_text=f"Multi-day prediction analysis",
            xaxis_title="Time index (negative: history)",
            yaxis_title=var_type,
            width=1200,
            height=600,
            showlegend=True,
        )
    else:
        multi_day_fig = None

    return next_day_fig, multi_day_fig
