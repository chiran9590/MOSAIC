
import numpy as np


def extract_sequence_features(sequences):
    """
    Convert a batch of LOB sequences into fixed-size feature vectors.

    Input shape:
        (samples, sequence_length, features)

    Output shape:
        (samples, features * 4)

    Features:
        1. Last observation
        2. Mean across the sequence
        3. Standard deviation across the sequence
        4. Linear trend across the sequence
    """
    sequences = np.asarray(sequences, dtype=np.float32)

    if sequences.ndim != 3:
        raise ValueError(
            "Expected input shape (samples, sequence_length, features)"
        )

    last = sequences[:, -1, :]
    mean = sequences.mean(axis=1)
    std = sequences.std(axis=1)

    time = np.arange(sequences.shape[1], dtype=np.float32)
    time = time - time.mean()

    trend = np.einsum(
        "t,ntf->nf",
        time,
        sequences,
    ) / np.sum(time ** 2)

    features = np.concatenate(
        [last, mean, std, trend],
        axis=1,
    )

    return features
