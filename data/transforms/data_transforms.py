import numpy as np
import pandas as pd

from typing import Union

def get_relative_change(data: Union[pd.DataFrame, pd.Series, np.ndarray]) -> Union[pd.DataFrame, pd.Series]:
    """
    Calculate the relative change of a pandas DataFrame / Series or NumPy array.
    """
    if isinstance(data, np.ndarray):
        return np.diff(data, axis=0, prepend=data[:1]) / np.vstack([np.ones(data[:1].shape), data[:-1]])
    elif isinstance(data, pd.Series) or isinstance(data, pd.DataFrame):
        return data.pct_change().fillna(0.)
    else:
        raise ValueError("Data must be a pandas DataFrame / Series or NumPy array.")

if __name__ == "__main__":

    data = np.array([[1., 2., 3., 4., 5.], [2., 3., 4., 5., 6.]]).T
    df = pd.DataFrame(data, columns=["value1", "value2"])
    expected_output = np.array([[0., 1., 0.5, 0.33333333, 0.25], [0., 0.5, 0.33333333, 0.25, 0.2]]).T

    assert np.allclose(get_relative_change(data), expected_output)
    assert np.allclose(get_relative_change(df).to_numpy(), expected_output)