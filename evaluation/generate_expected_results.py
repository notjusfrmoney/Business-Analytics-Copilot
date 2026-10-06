from __future__ import annotations

from evaluation.evaluator import generate_expected_results


if __name__ == "__main__":
    results = generate_expected_results()
    print(f"Generated trusted expected results for {len(results)} answerable questions.")
