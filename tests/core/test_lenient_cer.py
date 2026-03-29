"""Tests for lenient CER computation."""
from jaeval.core.metrics import compute_lenient_cer


class TestLenientCER:
    def test_katakana_hiragana_equivalent(self):
        # Katakana and hiragana should match after lenient normalization
        result = compute_lenient_cer("わたし", "ワタシ")
        assert result.cer == 0.0

    def test_mixed_scripts(self):
        result = compute_lenient_cer("テスト", "てすと")
        assert result.cer == 0.0

    def test_standard_cer_still_works(self):
        # Non-kana differences should still count as errors
        result = compute_lenient_cer("あいう", "かきく")
        assert result.cer > 0

    def test_identical_strings(self):
        result = compute_lenient_cer("こんにちは", "こんにちは")
        assert result.cer == 0.0

    def test_empty_ref_and_hyp(self):
        result = compute_lenient_cer("", "")
        assert result.cer == 0.0

    def test_empty_ref_nonempty_hyp(self):
        result = compute_lenient_cer("", "テスト")
        assert result.cer == 1.0

    def test_partial_katakana_match(self):
        # "アイウ" -> "あいう" via lenient normalization, compared with "あいう"
        result = compute_lenient_cer("あいう", "アイウ")
        assert result.cer == 0.0

    def test_mixed_kana_and_other_chars(self):
        # Numbers and punctuation should still pass through
        result = compute_lenient_cer("テスト123", "てすと123")
        assert result.cer == 0.0

    def test_returns_dataclass_fields(self):
        result = compute_lenient_cer("テスト", "てすと")
        assert result.ref_len > 0
        assert result.ref_normalized is not None
        assert result.hyp_normalized is not None
