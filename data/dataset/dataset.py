from torch.utils.data import Dataset

class TimeSeriesSliceDataset(Dataset):
    def __init__(self, data, train_length, target_length):
        """
        Args:
            data (torch.Tensor): The input data to sample slices from.
            train_length (int): The length of the training slice.
            target_length (int): The length of the target slice.
        """
        self.data = data
        self.train_length = train_length
        self.target_length = target_length

    def __len__(self):
        # The number of samples is the number of possible starting indices
        return self.data.shape[0] - self.train_length - self.target_length + 1

    def __getitem__(self, idx):
        # Get the slice starting at index `idx` of length `self.length`
        return self.data[idx:idx + self.train_length], self.data[idx + self.train_length:idx + self.train_length + self.target_length]
