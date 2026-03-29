"""Example: Post-call auto-evaluation using jaeval.

Shows how voice-fullduplex (or any voice AI system) can integrate jaeval
to automatically evaluate call quality after each call ends.

Usage:
    # Tier 1 only (fast, no API call)
    python examples/post_call_eval.py

    # Tier 1 + Tier 2 LLM judge (requires ANTHROPIC_API_KEY)
    python examples/post_call_eval.py --with-judge
"""

import json
import sys
from pathlib import Path

from jaeval import evaluate_call


def main():
    with_judge = "--with-judge" in sys.argv

    # In production, this data comes from your voice pipeline's call logs.
    # Each turn needs at minimum: user_text, bot_text
    # Optional: confidence, has_uncertainty, duration_s, latency_ms
    turns = [
        {
            "user_text": "もしもし、Recoについて聞きたいんですけど",
            "bot_text": "はい、StepAIのRecoについてですね。お電話ありがとうございます。どのような点についてお知りになりたいですか？",
            "confidence": -0.05,
            "latency_ms": 850,
            "duration_s": 2.1,
        },
        {
            "user_text": "料金プランを教えてください",
            "bot_text": "Recoの料金プランをご案内します。スタータープランは月額5,000円、プロプランは月額15,000円です。",
            "confidence": -0.03,
            "latency_ms": 1100,
            "duration_s": 2.8,
        },
        {
            "user_text": "デモを見たいんですが",
            "bot_text": "デモのご予約を承りました。担当者から折り返しご連絡いたします。お名前とご連絡先をお伺いしてもよろしいですか？",
            "confidence": -0.02,
            "latency_ms": 950,
            "duration_s": 1.9,
        },
        {
            "user_text": "田中です。03-1234-5678です",
            "bot_text": "田中様、お電話番号03-1234-5678ですね。ありがとうございます。担当者よりご連絡いたします。",
            "confidence": -0.08,
            "latency_ms": 1200,
            "duration_s": 3.2,
        },
    ]

    print("Evaluating call...")
    result = evaluate_call(
        call_sid="EXAMPLE_001",
        turns=turns,
        duration_sec=65.0,
        run_judge=with_judge,
        judge_model="claude-sonnet-4-20250514",
    )

    # Tier 1 results (always available)
    print(f"\n--- Tier 1: Scorecard ---")
    print(f"  Grade:            {result.grade}")
    print(f"  Task Completion:  {result.task_completion}")
    print(f"  STT Errors:       {result.stt_error_count}")
    print(f"  Hallucinations:   {result.hallucination_count}")
    print(f"  Banned Words:     {result.banned_words_used}")
    print(f"  Avg Latency:      {result.avg_latency_sec:.3f}s")

    # Tier 2 results (only if judge was run)
    if result.weighted_score is not None:
        print(f"\n--- Tier 2: LLM Judge ---")
        print(f"  Weighted Score:   {result.weighted_score}/5")
        print(f"  Production Ready: {result.production_ready}")
        if result.biggest_issue:
            print(f"  Biggest Issue:    {result.biggest_issue}")
        for dim, entry in result.dimension_scores.items():
            print(f"    {dim}: {entry['score']}/5")
    elif with_judge:
        print(f"\n  Judge skipped: {result.biggest_issue or 'unknown reason'}")

    # Save full result to JSON
    output = Path("results/example_eval.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    print(f"\nFull result saved to {output}")


if __name__ == "__main__":
    main()
