"""Hallucinated kanji detection for Japanese STT evaluation.

Hallucination detection flags inserted characters that are CJK Unified
Ideographs (U+4E00-U+9FFF) not present in the reference and not in a
whitelist of common business kanji.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Default business kanji whitelist -- characters frequently used in Japanese
# business phone calls. Inserted kanji NOT in this set are flagged as
# hallucinations.
# ---------------------------------------------------------------------------

# fmt: off
DEFAULT_BUSINESS_KANJI: set[str] = set(
    "会社員様方電話番号名前住所日時間月火水木金土曜年度週"
    "予約確認注文商品金額税込円万千百十"
    "担当者部長課長係主任代表取締役"
    "株式有限合同事業本支店営業販売開発企画"
    "東京都大阪府北海道神奈川埼玉千葉愛知福岡"
    "区市町村丁目番地階号室"
    "申訳承知了解検討連絡対応処理手続"
    "送受届届出届先届出届人届出届出届出"
    "見積請求納品発注契約書類資料"
    "御案内説明紹介提案報告相談依頼"
    "変更追加削除修正更新登録解除"
    "上下左右前後内外中"
    "一二三四五六七八九零"
    "新旧大小高低多少長短早遅"
    "入出来行帰着送届届"
    "言話聞読書見思考知分"
    "使用利便不可能必要重"
    "田中山本井藤原田村松木林森川"
    "渋谷品川目黒港新宿世田"
    "銀行口座振込普通当座預貯蓄"
    "平成令和昭和"
    "先次今回目的件点"
    "打合伺参拝"
    "頂戴致存候"
    "恐縮失礼"
    "早速改至急折返"
    "宜敷"
    "頁"
    "自動機能設定画面"
    "情報管利"
    "始終了完成"
)
# fmt: on


def _is_cjk_ideograph(ch: str) -> bool:
    """True if character is in CJK Unified Ideographs block (U+4E00-U+9FFF)."""
    return 0x4E00 <= ord(ch) <= 0x9FFF


def detect_hallucinated_kanji(
    ref: str, hyp: str, *, whitelist: set[str] | None = None
) -> list[str]:
    """Find kanji in hypothesis that are not in reference and not in whitelist.

    These are likely hallucinated characters -- rare kanji the STT model
    invented that have no basis in the spoken audio.

    Args:
        ref: Reference (ground truth) transcript.
        hyp: Hypothesis (STT output) transcript.
        whitelist: Optional custom whitelist. Defaults to DEFAULT_BUSINESS_KANJI.

    Returns:
        List of hallucinated kanji characters found in the hypothesis.
    """
    wl = whitelist if whitelist is not None else DEFAULT_BUSINESS_KANJI
    ref_chars = set(ref)
    return [ch for ch in hyp if _is_cjk_ideograph(ch) and ch not in ref_chars and ch not in wl]


def hallucination_count(ref: str, hyp: str, *, whitelist: set[str] | None = None) -> int:
    """Count hallucinated kanji in hypothesis.

    Convenience wrapper around detect_hallucinated_kanji.
    """
    return len(detect_hallucinated_kanji(ref, hyp, whitelist=whitelist))
