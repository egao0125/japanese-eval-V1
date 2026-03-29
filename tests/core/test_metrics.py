"""Tests for Character Error Rate (CER) computation."""

from jaeval.core.metrics import compute_cer, CERResult


class TestComputeCER:
    def test_identical(self):
        result = compute_cer("こんにちは", "こんにちは")
        assert result.cer == 0.0
        assert result.substitutions == 0

    def test_completely_different(self):
        result = compute_cer("あ", "い")
        assert result.cer == 1.0
        assert result.substitutions == 1

    def test_insertion(self):
        result = compute_cer("あい", "あいう")
        assert result.insertions == 1

    def test_deletion(self):
        result = compute_cer("あいう", "あい")
        assert result.deletions == 1

    def test_empty_reference(self):
        result = compute_cer("", "あ")
        assert result.cer == 1.0

    def test_empty_both(self):
        result = compute_cer("", "")
        assert result.cer == 0.0

    def test_normalization_applied(self):
        # Full-width and half-width should be equivalent
        result = compute_cer("１２３", "123")
        assert result.cer == 0.0

    def test_returns_dataclass(self):
        result = compute_cer("テスト", "テスト")
        assert isinstance(result, CERResult)
        assert hasattr(result, "ref_normalized")
        assert hasattr(result, "hyp_normalized")
