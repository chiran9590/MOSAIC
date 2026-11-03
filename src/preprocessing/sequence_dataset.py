
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class LOBSequenceDataset(Dataset):
    def __init__(
        self,
        filename,
        sequence_length=100,
        stride=10,
        start_index=None,
        end_index=None,
    ):
        self.data_path = Path("data/processed") / filename

        data = np.load(self.data_path)

        X = data["X"].astype(np.float32)
        y = data["y"].astype(np.int64)

        self.sequence_length = sequence_length
        self.stride = stride

        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y - 1)

        if start_index is None:
            start_index = 0

        if end_index is None:
            end_index = len(X)

        self.indices = list(
            range(
                max(start_index, sequence_length - 1),
                end_index,
                stride,
            )
        )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        end = self.indices[index]
        start = end - self.sequence_length + 1

        sequence = self.X[start:end + 1]
        target = self.y[end]

        return sequence, target
