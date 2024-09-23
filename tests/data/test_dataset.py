import torch
from torch.utils.data import DataLoader
from data.dataset import TimeSeriesSliceDataset

torch.manual_seed(0)

def test_time_series_slice_dataset():

    # Initialise example parameters and data
    features = 3
    dataset_size = 24
    train_length = 5
    target_length = 2
    batch_size = 4
    data = torch.randn([dataset_size, features])

    # Create the dataset
    dataset = TimeSeriesSliceDataset(data, train_length=train_length, target_length=target_length)

    # Example: iterate over the dataset using a DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    assert len(dataset) == len(data) - train_length - target_length + 1

    visited_indices = []

    for x, y in dataloader:
        assert x[0].shape == (train_length, features)
        assert y[0].shape == (target_length, features)

        xy = torch.cat([x, y], dim=1)

        for b in range(x.shape[0]):
            xy_batch = xy[b]

            match = False
            for i in range(len(dataset)):
                if i in visited_indices:
                    continue
                elif torch.all(xy_batch == data[i:i + train_length + target_length]):
                    visited_indices.append(i)
                    match = True
                    break

            assert match

    assert set(visited_indices) == set(range(len(dataset)))
