# Model selection

Status: preliminary benchmark complete; final selection pending human review.

The app runs all three models on every valid query. The final passage is selected by reciprocal-rank fusion among reviewed evidence, with a two-model support requirement. The single strongest model should only be named after reviewing `model_comparison.csv` and completing `docs/REVIEW_RUBRIC.md`.

## Preliminary benchmark (2026-11-22)

`model_comparison.csv` reports 250 sampled, scheme-grouped, template-generated test questions:

| Setup | Recall@1 | Recall@5 | MRR |
|---|---:|---:|---:|
| multilingual-e5-base | 0.348 | 0.728 | 0.5119 |
| BGE-M3 | 0.548 | 0.832 | 0.6704 |
| LaBSE | 0.228 | 0.444 | 0.3309 |
| Three-model RRF, same approved passage in at least two top-eight lists | 0.472 accuracy | — | — |

The ensemble had 1.000 coverage on this sample. These are preliminary retrieval results only: the questions are synthetic, the benchmark searches the full corpus while the app narrows candidates by its scheme and intent resolver, and no manual source/language review has been completed. Do not treat these figures as answer accuracy or a final model-selection result. The paired review worksheet is `manual_review_template.csv`.
