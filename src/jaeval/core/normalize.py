"""Japanese text normalization for CER comparison.

Normalizes Japanese text so that semantically identical transcriptions
(e.g. "二千二十六年" vs "2026年", full-width "１２３" vs "123") are treated
as equal during character error rate computation.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Kanji number tables
# ---------------------------------------------------------------------------

_KANJI_DIGIT: dict[str, int] = {
    "〇": 0, "零": 0, "一": 1, "二": 2, "三": 3, "四": 4,
    "五": 5, "六": 6, "七": 7, "八": 8, "九": 9,
}

_MULTIPLIERS: dict[str, int] = {"十": 10, "百": 100, "千": 1000}

_LARGE_UNITS: dict[str, int] = {"万": 10000, "億": 100000000, "兆": 1000000000000}

_NUM_CHARS: set[str] = set(_KANJI_DIGIT) | set(_MULTIPLIERS) | set(_LARGE_UNITS)

_KANJI_CHARS_ESCAPED = re.escape("".join(_NUM_CHARS))

_KANJI_NUMBER_PATTERN = re.compile(
    r"(?:[" + _KANJI_CHARS_ESCAPED + r"0-9])+"
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_kanji_number(s: str) -> str:
    """Convert a kanji/mixed number string to its digit representation.

    Handles pure kanji (二千二十六), pure digit sequences of kanji digits
    (〇四二), and mixed Arabic+kanji (12万3千400).
    """
    has_kanji = any(c in _NUM_CHARS for c in s)
    if not has_kanji:
        return s  # Pure Arabic digits, return as-is

    # Handle pure kanji digit sequences like 〇四二 (used in order IDs)
    if all(c in _KANJI_DIGIT for c in s):
        return "".join(str(_KANJI_DIGIT[c]) for c in s)

    # Parse structured numbers (pure kanji or mixed Arabic+kanji).
    total = 0
    current_section = 0  # accumulator for current large-unit section
    current = 0  # accumulator for current positional value
    i = 0

    while i < len(s):
        ch = s[i]

        if ch.isdigit():
            # Read full Arabic number
            num_str = ""
            while i < len(s) and s[i].isdigit():
                num_str += s[i]
                i += 1
            current = int(num_str)
            continue  # don't increment i again
        elif ch in _KANJI_DIGIT:
            current = _KANJI_DIGIT[ch]
        elif ch in _MULTIPLIERS:
            mult = _MULTIPLIERS[ch]
            if current == 0:
                current = 1  # 十 alone means 10, not 0
            current_section += current * mult
            current = 0
        elif ch in _LARGE_UNITS:
            unit = _LARGE_UNITS[ch]
            current_section += current
            total += (current_section if current_section > 0 else 1) * unit
            current_section = 0
            current = 0

        i += 1

    # Add remaining
    total += current_section + current
    return str(total)


def _should_convert(match_str: str, after: str) -> bool:
    """Decide whether to convert a kanji number match.

    Always converts if 2+ chars, or if a single kanji digit is followed
    by a Japanese counter/unit (月日時分秒年円個回件枚本台).
    """
    if any(c in _NUM_CHARS for c in match_str):
        if len(match_str) >= 2:
            return True
        # Single kanji number char: convert if followed by a counter
        if after and after[0] in "月日時分秒年円個回件枚本台階番号":
            return True
        # Single multiplier (十=10) or large unit
        if match_str in _MULTIPLIERS or match_str in _LARGE_UNITS:
            return True
    return False


def _normalize_japanese_numbers(text: str) -> str:
    """Normalize Japanese number representations to digit form.

    Converts kanji number words to Arabic digits so that semantically
    identical transcriptions (e.g. "二千二十六年" vs "2026年") are treated
    as equal during CER computation.

    Handles:
    - Basic kanji digits: 一二三四五六七八九零/〇
    - Positional multipliers: 十百千
    - Large units: 万億兆
    - Compound numbers: 十二万三千四百 → 123400
    - Katakana digits: ゼロ → 0
    """
    # Step 1: Replace katakana number words
    text = text.replace("ゼロ", "0")

    # Step 2: Parse and convert kanji number sequences
    def _replace(m: re.Match) -> str:
        s = m.group()
        after = text[m.end():]
        if _should_convert(s, after):
            return _parse_kanji_number(s)
        return s

    result = _KANJI_NUMBER_PATTERN.sub(_replace, text)
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def normalize_japanese(text: str) -> str:
    """Normalize Japanese text for CER comparison.

    Steps:
    1. Strip all whitespace (half-width and full-width)
    2. Full-width digits/letters to half-width
    3. Kanji numbers to Arabic digits
    4. Punctuation variants to ASCII equivalents
    """
    # 1. Strip whitespace
    text = text.replace(" ", "").replace("\u3000", "")

    # 2. Full-width to half-width
    result = []
    for ch in text:
        cp = ord(ch)
        if 0xFF10 <= cp <= 0xFF19:
            result.append(chr(cp - 0xFF10 + ord("0")))
        elif 0xFF21 <= cp <= 0xFF3A:
            result.append(chr(cp - 0xFF21 + ord("A")))
        elif 0xFF41 <= cp <= 0xFF5A:
            result.append(chr(cp - 0xFF41 + ord("a")))
        else:
            result.append(ch)
    text = "".join(result)

    # 3. Kanji numbers to Arabic
    text = _normalize_japanese_numbers(text)

    # 4. Punctuation normalization
    text = text.replace("。", ".").replace("、", ",")

    return text
