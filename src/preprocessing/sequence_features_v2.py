
import numpy as np


def extract_sequence_features_v2(sequences):
    """
    Convert LOB sequences into an expanded feature representation.

    Input shape:
        (samples, sequence_length, features)

    Output shape:
        (samples, features * 10)
    """

    sequences = np.asarray(sequences, dtype=np.float32)

    if sequences.ndim != 3:
        raise ValueError(
            "Expected shape (samples, sequence_length, features)"
        )

    if sequences.shape[1] < 4:
        raise ValueError("Sequence length must be at least 4.")

    if not np.isfinite(sequences).all():
        raise ValueError("Sequences contain NaN or infinite values.")

    sequence_length = sequences.shape[1]

    # 1. Latest observation
    last = sequences[:, -1, :]

    # 2. Mean
    mean = sequences.mean(axis=1)

    # 3. Standard deviation
    std = sequences.std(axis=1)

    # 4. Linear trend
    time = np.arange(sequence_length, dtype=np.float32)
    time = time - time.mean()

    trend = np.einsum(
        "t,ntf->nf",
        time,
        sequences
    ) / np.sum(time ** 2)

    # 5. Minimum
    minimum = sequences.min(axis=1)

    # 6. Maximum
    maximum = sequences.max(axis=1)

    # 7. Median
    median = np.median(sequences, axis=1)

    # 8. Interquartile range
    q75 = np.percentile(sequences, 75, axis=1)
    q25 = np.percentile(sequences, 25, axis=1)
    iqr = q75 - q25

    # 9. Recent mean
    recent_length = max(1, sequence_length // 4)
    recent = sequences[:, -recent_length:, :]
    recent_mean = recent.mean(axis=1)

    # 10. Recent standard deviation
    recent_std = recent.std(axis=1)

    return np.concatenate(
        [
            last,
            mean,
            std,
            trend,
            minimum,
            maximum,
            median,
            iqr,
            recent_mean,
            recent_std,
        ],
        axis=1,
    )
