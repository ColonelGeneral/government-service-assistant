"""Compute top-1 scheme/field classification metrics for the benchmark sample."""
from pathlib import Path

import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

ROOT = Path(__file__).resolve().parents[1]
MODELS = ["multilingual-e5-base", "BGE-M3", "LaBSE"]


def main():
    evaluation = ROOT / "data" / "evaluation"
    questions = pd.read_csv(evaluation / "test_questions.csv", keep_default_na=False)
    questions = questions.sample(250, random_state=24).reset_index(drop=True)
    details = pd.read_csv(evaluation / "model_comparison_details.csv", keep_default_na=False)
    retrieval = pd.read_csv(evaluation / "model_comparison.csv", keep_default_na=False)

    rows = []
    for model in MODELS:
        result = details[details["model"].eq(model)].copy()
        result["question_id"] = pd.to_numeric(result["question_id"])
        result = result.sort_values("question_id")
        if len(result) != len(questions):
            raise SystemExit(f"Expected {len(questions)} benchmark rows for {model}; found {len(result)}.")

        truth = (questions["scheme_id"].astype(str) + "|" + questions["field"].astype(str)).tolist()
        predicted = (result["top1_scheme_id"].astype(str) + "|" + result["top1_field"].astype(str)).tolist()
        labels = sorted(set(truth) | set(predicted))
        precision, recall, f1, _ = precision_recall_fscore_support(
            truth, predicted, labels=labels, average="macro", zero_division=0
        )
        retrieval_row = retrieval[retrieval["setup"].eq(model)].iloc[0]
        rows.append({
            "model": model,
            "queries": len(questions),
            "top1_scheme_field_accuracy": sum(a == b for a, b in zip(truth, predicted)) / len(truth),
            "macro_precision": precision,
            "macro_recall": recall,
            "macro_f1": f1,
            "retrieval_recall_at_1": float(retrieval_row["recall_at_1"]),
            "retrieval_recall_at_5": float(retrieval_row["recall_at_5"]),
        })

    output = evaluation / "classification_metrics.csv"
    pd.DataFrame(rows).to_csv(output, index=False, encoding="utf-8-sig")
    print(f"Wrote top-1 classification metrics for {len(questions)} questions to {output}")


if __name__ == "__main__":
    main()
