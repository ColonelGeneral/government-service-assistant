# Three-model workflow

1. **Collect official English/Hindi pages.** `scripts/prepare_project.py` creates a manifest from known English myScheme page URLs plus blank Hindi URL rows. Add only confirmed official Hindi URLs. The collector is rate-limited and restricted to government domains.
2. **Extract facts and provenance.** Run `python scripts/extract_facts_for_review.py`. It produces heading-based candidates with URL, language, and fetch date. These are proposals and must be checked against the page.
3. **Align language facts.** A reviewer places confirmed rows in `data/official_facts.csv`, keyed by `scheme_id + field + language`, with official URL, checked date, validity dates, reviewer, and review status.
4. **Check data quality.** Run `python scripts/prepare_project.py`. It normalizes both datasets, builds a quality report for missing required fields, absent English/Hindi pairs, unverified QA rows, missing/stale checks, expiry, and numeric/date conflicts in reviewed pairs.
5. **Build knowledge base and model indexes.** The normalized records are stored in `data/processed/records.jsonl`. Run `python scripts/build_indexes.py` to download and encode passages sequentially for all three embedding models.
6. **Create evaluation questions.** Run `python scripts/make_eval_questions.py`. English, Devanagari Hindi, and romanized code-mixed questions are split by scheme into train/dev/test so one scheme cannot leak across sets. These are template-generated; add human-written test questions before drawing final claims.
7. **Run all three for each query.** The app sends the same query through E5-base, BGE-M3, and LaBSE, displays each model's top passages, and fuses reviewed-source ranks with reciprocal-rank fusion. It requires two-model support for the selected source fact.
8. **Check evidence and answer.** The app uses the selected source passage verbatim (extractive answer), displays URL and checked date, blocks a reviewed EN/HI numeric/date conflict, and abstains on ambiguous scheme names or missing reviewed Hindi evidence. It does not fabricate or silently translate facts.
9. **Evaluate and choose.** Run `python scripts/benchmark_ensemble.py --max-questions 250`; compare per-model Recall@1/5, MRR, and ensemble accuracy. Use `docs/REVIEW_RUBRIC.md` for manual evidence, conflict, and language review. Record model comparisons and limitations.

## Data/accuracy limits

The bundled myScheme CSVs are English and lack independent checked-at timestamps. Yojana Sahayak contains useful bilingual Q&A examples but can contain incomplete answers; it is indexed as retrieval context and **not** accepted as official answer evidence without human review. Until verified Hindi official fact pairs are added, a Hindi request may receive a clear “Hindi source missing” notice and an English source passage. No software can guarantee every policy answer is correct from unverified/incomplete source data; this workflow makes that gap explicit and prevents the app from guessing across schemes.

The three named models are embedding retrievers, not answer-generating LLMs. Answers are extractive and source-grounded by design.

## Current collection blocker

The static collector was tried against 10 English myScheme URLs on 2026-11-22. Those pages returned a JavaScript shell and an error dialog instead of scheme facts. `collect_official_pages.py` now labels these snapshots `shell_only_dynamic_content`, and `extract_facts_for_review.py` excludes them. The resulting `data/official_facts_review.csv` is intentionally empty; the shell-only output must not be treated as scheme evidence. Do not run a full crawl until a supported content source is available.

myScheme is listed as an API Setu partner. Access through the API gateway requires an approved subscription and provider authorization; API Setu then issues the API documentation and credentials. See the [API Setu consumer SOP](https://cdn.apisetu.gov.in/portal/assets/SOP-APISETU.pdf). Use only credentials and endpoints issued for this project; keep secrets outside the repository. Once access is available, adapt the collector to the documented interface or another confirmed official source and preserve language, URL, checked date, and review provenance.

The 250-query benchmark sample has a blank manual-review worksheet at `data/evaluation/manual_review_template.csv`. It is template-generated, not human-authored or reviewed. Leave reviewer outcomes blank until a person checks the source and response.
