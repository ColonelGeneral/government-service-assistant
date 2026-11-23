# Government-Service Assistant — Three-Model Ensemble

This is a new project build based on the supplied workflow. For every identifiable scheme question, the app runs the same query through **all three** models—multilingual-e5-base, BGE-M3, and LaBSE—compares their retrieved passages, and selects reviewed source evidence using reciprocal-rank fusion. It requires the same reviewed fact to appear in at least two models' top results. If the scheme name matches multiple programs, the app asks the user to choose using the title, short title, state, or ministry, then continues the original question with that selection. When a scheme is not in the dataset, it explains in the selected language that the knowledge base is being expanded and includes the scheme URL whenever the dataset provides one. If evidence is missing, the requested language is unreviewed, or sources conflict, it explains the issue instead of guessing.

## Run on Windows

Extract the ZIP, open PowerShell in this folder, and run:

```powershell
py -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe scripts/prepare_project.py
.venv\Scripts\python.exe scripts/make_eval_questions.py
.venv\Scripts\python.exe scripts/build_indexes.py
.venv\Scripts\python.exe -m streamlit run app.py
```

The first index build downloads all 3 model weights. Reserve several GB of disk space and allow time for CPU indexing. Queries then run all three retrievers; the app shows each model's leading passages and the fused source result.

For an NVIDIA GPU, install a CUDA-enabled PyTorch build using the [official PyTorch selector](https://pytorch.org/get-started/locally/) before building indexes. The index builder detects CUDA, uses half precision there, and defaults to 512-token passages; tune with `--batch-size` and `--max-seq-length` if needed. Without CUDA, CPU indexing may take substantially longer.

## Data and validation status

- `data/myscheme/`: the supplied structured English scheme records and FAQs.
- `data/yojana_sahayak/`: downloaded train/eval Q&A partitions. They are included in retrieval context; incomplete/unreviewed Q&A answers are **not** treated as official answer evidence.
- `data/processed/quality_review.csv`: generated flags for missing language pairs, unverified source dates, missing fields, and QA rows needing review.
- `data/source_pages.csv`: English URL rows plus blank Hindi URL slots. Do not guess translated page URLs.
- `data/official_facts_template.csv`: schema for manually verified bilingual facts; copy reviewed rows to `data/official_facts.csv` and set `review_status=approved`, reviewer, official URL, and checked date.

The source CSVs do not provide independent checked-at dates and are English only. Therefore the app marks them as not date-verified. A 10-page static collection attempt returned only the myScheme JavaScript shell, not scheme facts; the collector now flags these pages and the fact extractor excludes them. `data/official_facts_review.csv` currently has no candidates. Do not run a full static crawl until an approved API or another confirmed official content source is available. myScheme is listed as an API Setu partner; access to an API Setu service requires an approved subscription and provider authorization. See the [API Setu consumer SOP](https://cdn.apisetu.gov.in/portal/assets/SOP-APISETU.pdf).

The current data cannot support a promise of zero errors across every scheme; the app instead constrains scheme/intent, shows sources, checks reviewed bilingual numeric conflicts, and abstains when it cannot verify the requested evidence.

## Workflow and model evaluation

See `docs/WORKFLOW.md`. After indexes exist, run `python scripts/benchmark_ensemble.py --max-questions 250`, then `python scripts/make_manual_review_template.py` to create a blank worksheet for the held-out sample. The questions are synthetic and the worksheet is unreviewed; add human-authored test questions and complete `docs/REVIEW_RUBRIC.md` before making accuracy claims. `data/evaluation/model_selection.md` remains pending until that work is done.

The three models are embedding retrievers, not answer-generating LLMs. The returned answer is an extractive official passage so unsupported paraphrases are avoided.

## Analysis notebook

Open `notebooks/government_service_assistant_analysis.ipynb` for the system architecture diagram, use case diagram, benchmark metric calculations, and accuracy, precision, recall, and evidence-retrieval graphs. The notebook embeds its generated diagrams and graphs and can recreate them from the saved evaluation CSVs.

Install notebook dependencies with `python -m pip install -r requirements-notebook.txt`, then launch it from the project folder with `jupyter notebook notebooks/government_service_assistant_analysis.ipynb`. The notebook uses a deterministic sample of 250 generated test questions; its metrics are not human-reviewed answer accuracy.
