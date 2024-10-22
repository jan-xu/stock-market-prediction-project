import numpy as np
from sklearn.model_selection import TimeSeriesSplit


def get_number_of_splits(data, look_back, val_size):
    """
    Get the number of splits for time-series cross-validation based on the look-back and validation set size.
    """
    return int(len(data) / (look_back + val_size)) - 1


def auto_time_series_cross_validation(data, args, return_indices=False):
    """Automatic configuration of time-series cross-validation based on experiment configuration
    (conveyed through args).

    Specifically, the number of splits is selected based on the look_back and val_size parameters.
    The gap parameter is the negative of look_back (in order to include the input parameters in the
    validation set, which intersects with the last samples in the training set), and the test_size
    parameter is equal to the look_back parameter plus the specified validation set size.

    Parameters
    ----------
    data : np.ndarray
        The time-series data to be split, with shape (n_samples, n_features).
    args : argparse.Namespace
        The experiment configuration.
    return_indices : bool, optional
        Whether to return the indices of the split data along with the data itself. If True,
        the function yields ((train_indices, test_indices), (train_data, test_data)).
    """
    assert isinstance(data, np.ndarray), "Data must be a NumPy array."

    n_splits = get_number_of_splits(data, args.look_back, val_size=args.val_size)
    gap = -args.look_back
    test_size = args.look_back + args.val_size
    return time_series_cross_validation(
        data, n_splits, gap=gap, test_size=test_size, return_indices=return_indices
    )


def time_series_cross_validation(
    data, n_splits, gap=0, test_size=None, return_indices=False
):
    ts_cv = TimeSeriesSplit(n_splits=n_splits, gap=gap, test_size=test_size)
    for train_indices, test_indices in ts_cv.split(data):
        if return_indices:
            yield (train_indices, test_indices), (
                data[train_indices],
                data[test_indices],
            )
        else:
            yield data[train_indices], data[test_indices]
