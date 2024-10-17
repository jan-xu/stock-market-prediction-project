from typing import Union

import numpy as np
import pandas as pd
import torch

# TODO: unify typings for NumPy, Pandas and Torch data types


def get_relative_change(
    data: Union[pd.DataFrame, pd.Series, np.ndarray]
) -> Union[pd.DataFrame, pd.Series]:
    """
    Calculate the relative change of a pandas DataFrame / Series or NumPy array.
    """
    if isinstance(data, np.ndarray):
        return np.diff(data, axis=0, prepend=data[:1]) / np.vstack(
            [np.ones(data[:1].shape), data[:-1]]
        )
    elif isinstance(data, pd.Series) or isinstance(data, pd.DataFrame):
        return data.pct_change().fillna(0.0)
    else:
        raise ValueError("Data must be a pandas DataFrame / Series or NumPy array.")


def apply_relative_change(
    relative_change: Union[np.ndarray, torch.Tensor], start_value
):
    """
    Apply the relative change to a starting value. Assume the relative change dimension is the last axis.
    """
    if isinstance(relative_change, torch.Tensor):
        return (relative_change + 1).log().cumsum(dim=-1).exp() * start_value
    elif isinstance(relative_change, np.ndarray):
        return np.exp(np.log(relative_change + 1).cumsum(axis=-1)) * start_value
