from numbers import Number
from typing import List


class FeatureStandardiser:

    def __init__(self, mean, std):
        if not isinstance(mean, Number):
            if not isinstance(mean, List) and mean.shape == ():
                mean = mean.item()
                std = std.item()
                self._dim = 1
            else:
                assert len(mean) == len(std), "Mean and std must have the same length."
                self._dim = len(mean)
                if self.dim == 1:
                    mean = mean[0]
                    std = std[0]
        else:
            self._dim = 1

        self.mean = mean
        self.std = std

    @property
    def dim(self):
        return self._dim

    def __call__(self, x):
        print(
            f"Use .forward() or .inverse() instead of calling {self.__class__.__name__} directly."
        )
        return x

    def forward(self, x):
        return (x - self.mean) / self.std

    def inverse(self, x):
        return x * self.std + self.mean

    def __str__(self):
        if self.dim == 1:
            return f"FeatureStandardiser(mean={self.mean:.6f}, std={self.std:.6f})"
        else:
            printable_mean = [
                round(m.item(), 3) if hasattr(m, "item") else round(m, 3)
                for m in self.mean
            ]
            printable_std = [
                round(s.item(), 3) if hasattr(s, "item") else round(s, 3)
                for s in self.std
            ]
            return f"FeatureStandardiser(mean={printable_mean}, std={printable_std})"

    def __repr__(self):
        return str(self)
