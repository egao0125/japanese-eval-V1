"""Tests for Japanese text normalization."""

from jaeval.core.normalize import normalize_japanese


class TestNormalizeJapanese:
    def test_strip_whitespace(self):
        assert normalize_japanese("こんにちは 世界") == normalize_japanese("こんにちは世界")

    def test_fullwidth_digits(self):
        assert normalize_japanese("１２３") == "123"

    def test_fullwidth_alpha(self):
        assert normalize_japanese("ＡＢＣ") == "ABC"
        assert normalize_japanese("ａｂｃ") == "abc"

    def test_kanji_numbers_simple(self):
        result = normalize_japanese("三百二十一")
        assert result == "321"

    def test_kanji_numbers_large(self):
        result = normalize_japanese("一万五千")
        assert result == "15000"

    def test_punctuation(self):
        result = normalize_japanese("はい。いいえ、")
        assert "." in result
        assert "," in result

    def test_zero_variants(self):
        result = normalize_japanese("ゼロ")
        assert result == "0"

    def test_mixed_text(self):
        # Should not break on mixed Japanese+ASCII
        result = normalize_japanese("テスト123")
        assert "123" in result

    def test_empty_string(self):
        assert normalize_japanese("") == ""
