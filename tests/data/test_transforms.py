import numpy as np
import pandas as pd

from data.transforms import (
    FeatureStandardiser,
    apply_relative_change,
    get_relative_change,
)


def test_get_relative_change():
    data = np.array([[1.0, 2.0, 3.0, 4.0, 5.0], [2.0, 3.0, 4.0, 5.0, 6.0]]).T
    df = pd.DataFrame(data, columns=["value1", "value2"])
    expected_output = np.array(
        [[0.0, 1.0, 0.5, 0.33333333, 0.25], [0.0, 0.5, 0.33333333, 0.25, 0.2]]
    ).T

    assert np.allclose(get_relative_change(data), expected_output)
    assert np.allclose(get_relative_change(df).to_numpy(), expected_output)


def test_apply_relative_change():
    data = np.array(
        [[0.0, 1.0, 0.5, 0.33333333, 0.25], [0.0, 0.5, 0.33333333, 0.25, 0.2]]
    )
    start_value = np.array([[1.0], [2.0]])
    expected_output = np.array([[1.0, 2.0, 3.0, 4.0, 5.0], [2.0, 3.0, 4.0, 5.0, 6.0]])

    assert np.allclose(apply_relative_change(data, start_value), expected_output)


def test_feature_standardiser():
    # 1-D data

    f_norm = FeatureStandardiser(mean=0.5, std=0.25)

    data = np.array([0.5, 0.75, 0.25, 1.0])
    data_normed = np.array([0.0, 1.0, -1.0, 2.0])
    f_normed = f_norm.forward(data)
    f_inv = f_norm.inverse(f_normed)

    assert np.allclose(data_normed, f_normed)
    assert np.allclose(data, f_inv)

    # 1-D data, but parameters as a np.array

    f_norm = FeatureStandardiser(mean=np.array(0.5), std=np.array(0.25))

    data = np.array([0.5, 0.75, 0.25, 1.0])
    data_normed = np.array([0.0, 1.0, -1.0, 2.0])
    f_normed = f_norm.forward(data)
    f_inv = f_norm.inverse(f_normed)

    assert np.allclose(data_normed, f_normed)
    assert np.allclose(data, f_inv)

    # 1-D data, but parameters as a list

    f_norm = FeatureStandardiser(mean=[0.5], std=[0.25])

    data = np.array([0.5, 0.75, 0.25, 1.0])
    data_normed = np.array([0.0, 1.0, -1.0, 2.0])
    f_normed = f_norm.forward(data)
    f_inv = f_norm.inverse(f_normed)

    assert np.allclose(data_normed, f_normed)
    assert np.allclose(data, f_inv)

    # 2-D data

    f_norm = FeatureStandardiser(mean=[0.5, 0.25], std=[0.25, 0.1])

    data = np.array([[0.5, 0.25], [0.75, 0.35], [0.25, 0.15], [1.0, 0.45]])
    data_normed = np.array([[0.0, 0.0], [1.0, 1.0], [-1.0, -1.0], [2.0, 2.0]])
    f_normed = f_norm.forward(data)
    f_inv = f_norm.inverse(f_normed)

    assert np.allclose(data_normed, f_normed)
    assert np.allclose(data, f_inv)
