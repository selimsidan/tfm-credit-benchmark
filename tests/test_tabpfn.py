from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tfmbench.models.tabpfn35 import predict_with_chunk_retries


class FakeChunkModel:
    classes_ = np.array([0, 1])

    def __init__(self, maximum: int) -> None:
        self.maximum = maximum
        self.calls: list[int] = []

    def predict_proba(self, frame):
        self.calls.append(len(frame))
        if len(frame) > self.maximum:
            raise RuntimeError("CUDA out of memory")
        return np.tile([0.7, 0.3], (len(frame), 1))


def test_chunk_retry_halves_only_query_size():
    frame = pd.DataFrame({"x": np.arange(1_000)})
    model = FakeChunkModel(128)
    probability, chunk, retries = predict_with_chunk_retries(
        model, frame, initial_chunk_rows=512, minimum_chunk_rows=128
    )
    assert len(probability) == len(frame)
    assert np.all(probability == 0.3)
    assert chunk == 128
    assert retries == 2
    assert model.calls[:3] == [512, 256, 128]


def test_chunk_retry_raises_after_minimum_without_subsampling():
    frame = pd.DataFrame({"x": np.arange(500)})
    with pytest.raises(RuntimeError, match="not subsampled"):
        predict_with_chunk_retries(
            FakeChunkModel(64), frame, initial_chunk_rows=512, minimum_chunk_rows=128
        )
