import numpy as np
import pandas as pd

from engineering import add_cyclical_time_encoding


def test_add_cyclical_time_encoding_hour_wraparound() -> None:
    """Hour 23 and hour 0 should have similar sine/cosine encodings."""
    timestamps = [
        pd.Timestamp("2024-01-01 00:00:00", tz="UTC"),
        pd.Timestamp("2024-01-01 23:00:00", tz="UTC"),
    ]
    df = pd.DataFrame({"timestamp": timestamps})

    engineered = add_cyclical_time_encoding(df, timestamp_col="timestamp")

    for col in ("sin_hour", "cos_hour"):
        assert col in engineered.columns

    v0 = engineered.loc[0, ["sin_hour", "cos_hour"]].to_numpy(dtype=float)
    v23 = engineered.loc[1, ["sin_hour", "cos_hour"]].to_numpy(dtype=float)

    distance = float(np.linalg.norm(v0 - v23))

    # Because of the circular encoding, the vectors for 0:00 and 23:00
    # should be close in Euclidean distance.
    assert distance < 0.5

