"""Create a blank human-review worksheet from the benchmark's held-out sample."""
import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODELS = ["multilingual-e5-base", "BGE-M3", "LaBSE"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-questions", type=int, default=250)
    args = parser.parse_args()
    data = ROOT / "data" / "evaluation"
    qpath = data / "test_questions.csv"
    dpath = data / "model_comparison_details.csv"
    if not qpath.exists() or not dpath.exists():
        raise SystemExit("Run benchmark_ensemble.py first to create the held-out sample and model details.")

    questions = pd.read_csv(qpath, keep_default_na=False)
    if args.max_questions and len(questions) > args.max_questions:
        questions = questions.sample(args.max_questions, random_state=24).reset_index(drop=True)
    else:
        questions = questions.reset_index(drop=True)
    details = pd.read_csv(dpath, keep_default_na=False)
    details["question_id"] = details["question_id"].astype(str)
    by_key = {(str(row.question_id), row.model): row for row in details.itertuples(index=False)}

    rows = []
    manual_fields = ["manual_selected_scheme", "manual_selected_field", "source_claims_supported",
                     "source_is_official", "language_faithful", "conflict_handled", "review_outcome",
                     "reviewer", "notes"]
    for i, q in questions.iterrows():
        row = {"question_id": i, "scheme_id": q.scheme_id, "scheme_name": q.scheme_name,
               "field": q.field, "language": q.language, "question": q.question,
               "expected_source_url": q.gold_source_url, "question_origin": q.question_origin}
        for model in MODELS:
            detail = by_key.get((str(i), model))
            prefix = model.lower().replace("-", "_")
            for source, target in [("top1_scheme_id", "top1_scheme_id"), ("top1_field", "top1_field"),
                                   ("rank_of_gold", "rank_of_gold"), ("top1_approved_evidence", "top1_approved_evidence")]:
                row[f"{prefix}_{target}"] = getattr(detail, source, "") if detail is not None else ""
        row.update({name: "" for name in manual_fields})
        rows.append(row)

    out = data / "manual_review_template.csv"
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
    print(f"Wrote {len(rows)} blank human-review rows to {out}")


if __name__ == "__main__":
    main()
