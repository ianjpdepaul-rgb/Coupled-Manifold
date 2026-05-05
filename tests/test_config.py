"""Characterization tests for graceful.config."""

from graceful.config import (
    _cosine_lr, _trace_valid, load_keys,
    MODEL, RANK, LR, DATA_DIR, MAX_CTX,
)


class TestCosLR:
    def test_step_zero_is_max(self):
        lr = _cosine_lr(0, 100, lr_max=1e-4)
        assert abs(lr - 1e-4) < 1e-8

    def test_step_end_is_min(self):
        lr = _cosine_lr(100, 100, lr_max=1e-4, lr_min=1e-7)
        assert abs(lr - 1e-7) < 1e-8

    def test_midpoint_between(self):
        lr = _cosine_lr(50, 100, lr_max=1e-4, lr_min=1e-7)
        assert 1e-7 < lr < 1e-4

    def test_total_one_returns_max(self):
        assert _cosine_lr(0, 1) == LR


class TestTraceValid:
    def test_none_invalid(self):
        assert _trace_valid(None) is False

    def test_nan_invalid(self):
        assert _trace_valid(float("nan")) is False

    def test_inf_invalid(self):
        assert _trace_valid(float("inf")) is False

    def test_zero_valid(self):
        assert _trace_valid(0.0) is True

    def test_negative_valid(self):
        assert _trace_valid(-150.0) is True

    def test_int_valid(self):
        assert _trace_valid(42) is True


class TestConstants:
    def test_model_is_gemma(self):
        assert "gemma" in MODEL.lower()

    def test_rank_positive(self):
        assert RANK > 0

    def test_data_dir(self):
        assert DATA_DIR == "./manifold_data"
