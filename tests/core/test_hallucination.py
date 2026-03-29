"""Tests for hallucinated kanji detection."""

from jaeval.core.hallucination import detect_hallucinated_kanji, hallucination_count


class TestHallucinationDetection:
    def test_no_hallucination(self):
        result = detect_hallucinated_kanji("会社に電話", "会社に電話")
        assert result == []

    def test_hallucinated_kanji(self):
        # 猫 is not in reference and not in default business whitelist
        result = detect_hallucinated_kanji("テスト", "猫テスト")
        assert "猫" in result

    def test_whitelist_allowed(self):
        # 会 is in the default business whitelist, so even if not in ref, it's allowed
        result = detect_hallucinated_kanji("テスト", "会テスト")
        assert result == []

    def test_custom_whitelist(self):
        result = detect_hallucinated_kanji("テスト", "猫テスト", whitelist={"猫"})
        assert result == []

    def test_count(self):
        c = hallucination_count("テスト", "猫犬テスト")
        assert c == 2

    def test_non_cjk_ignored(self):
        # Latin characters should not be flagged
        result = detect_hallucinated_kanji("テスト", "ABCテスト")
        assert result == []
