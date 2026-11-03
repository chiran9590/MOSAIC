
from torch.utils.data import DataLoader

from src.preprocessing.sequence_dataset import LOBSequenceDataset


dataset = LOBSequenceDataset(
    filename="train_cf_1.npz",
    sequence_length=100,
    stride=10,
)

loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True,
)

X, y = next(iter(loader))

print("Dataset size:", len(dataset))
print("Batch input shape:", X.shape)
print("Batch target shape:", y.shape)
print("Target classes:", y.unique().tolist())
