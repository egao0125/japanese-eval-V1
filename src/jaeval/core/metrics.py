"""Character Error Rate (CER) computation for Japanese text.

CER is character-level (not word-level) because Japanese has no word
boundaries. CER = (S + I + D) / len(reference).
"""

from __future__ import annotations

from dataclasses import dataclass

from .normalize import normalize_japanese


@dataclass
class CERResult:
    """Result of a CER computation."""

    cer: float
    substitutions: int
    insertions: int
    deletions: int
    ref_len: int
    ref_normalized: str
    hyp_normalized: str


def _edit_distance_ops(ref: str, hyp: str) -> tuple[int, int, int]:
    """Compute character-level edit distance and return (S, I, D).

    Uses the standard DP algorithm. Returns substitutions, insertions,
    and deletions to transform ``ref`` into ``hyp``.
    """
    n, m = len(ref), len(hyp)

    # dp[i][j] = (cost, substitutions, insertions, deletions)
    dp: list[list[tuple[int, int, int, int]]] = [
        [(0, 0, 0, 0) for _ in range(m + 1)] for _ in range(n + 1)
    ]

    for i in range(1, n + 1):
        dp[i][0] = (i, 0, 0, i)  # deletions
    for j in range(1, m + 1):
        dp[0][j] = (j, 0, j, 0)  # insertions

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                # substitution
                sub = dp[i - 1][j - 1]
                sub_cost = (sub[0] + 1, sub[1] + 1, sub[2], sub[3])

                # insertion (extra char in hyp)
                ins = dp[i][j - 1]
                ins_cost = (ins[0] + 1, ins[1], ins[2] + 1, ins[3])

                # deletion (missing char from ref)
                delete = dp[i - 1][j]
                del_cost = (delete[0] + 1, delete[1], delete[2], delete[3] + 1)

                dp[i][j] = min(sub_cost, ins_cost, del_cost, key=lambda x: x[0])

    _, s, i, d = dp[n][m]
    return s, i, d


def compute_cer(ref: str, hyp: str) -> CERResult:
    """Compute Character Error Rate between reference and hypothesis.

    Both strings are normalized (whitespace stripped, full-width to half-width,
    kanji numbers to Arabic) before comparison.

    Returns a CERResult dataclass with the CER value and detailed edit
    operation counts.
    """
    ref_clean = normalize_japanese(ref)
    hyp_clean = normalize_japanese(hyp)

    if not ref_clean:
        return CERResult(
            cer=0.0 if not hyp_clean else 1.0,
            substitutions=0,
            insertions=len(hyp_clean),
            deletions=0,
            ref_len=0,
            ref_normalized=ref_clean,
            hyp_normalized=hyp_clean,
        )

    s, i, d = _edit_distance_ops(ref_clean, hyp_clean)
    cer = (s + i + d) / len(ref_clean)

    return CERResult(
        cer=round(cer, 4),
        substitutions=s,
        insertions=i,
        deletions=d,
        ref_len=len(ref_clean),
        ref_normalized=ref_clean,
        hyp_normalized=hyp_clean,
    )
