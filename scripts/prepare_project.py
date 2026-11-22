"""Normalize both supplied datasets and emit a source/quality review report."""
from pathlib import Path
import json, re, csv, sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
sys.path.insert(0, str(ROOT))
from src.kb import prepare_records


def main():
    records, schemes = prepare_records(DATA)
    processed = DATA / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    with (processed / "records.jsonl").open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (processed / "schemes.json").write_text(json.dumps(schemes, ensure_ascii=False, indent=2), encoding="utf-8")
    issues = []
    for r in records:
        if not r.get("source_url"):
            issues.append({"scheme_id":r["scheme_id"],"field":r["field"],"language":r["language"],"issue":"missing_source_url","record_id":r["record_id"]})
        if not r.get("checked_at"):
            issues.append({"scheme_id":r["scheme_id"],"field":r["field"],"language":r["language"],"issue":"source_date_not_checked","record_id":r["record_id"]})
        if not r.get("usable_as_evidence"):
            issues.append({"scheme_id":r["scheme_id"],"field":r["field"],"language":r["language"],"issue":"qa_needs_fact_verification","record_id":r["record_id"]})
    # Detect absent English/Hindi official fact counterparts by scheme and field.
    facts = [r for r in records if r.get("usable_as_evidence")]
    grouped = {}
    for r in facts:
        grouped.setdefault((r["scheme_id"], r["field"]), set()).add(r["language"])
    for (sid, field), langs in grouped.items():
        if not {"en", "hi"}.issubset(langs):
            issues.append({"scheme_id":sid,"field":field,"language":"/".join(sorted(langs)),"issue":"missing_official_en_hi_pair","record_id":""})
    for s in schemes:
        sid = s["scheme_id"]
        for required in ["eligibility", "benefits", "application_process", "documents_required"]:
            if not any(r["scheme_id"] == sid and r["field"] == required and r.get("usable_as_evidence") for r in records):
                issues.append({"scheme_id":sid,"field":required,"language":"","issue":"required_fact_missing","record_id":""})
    # Curated paired facts get numeric/date consistency checks; flags always need review.
    curated_path = DATA / "official_facts.csv"
    if curated_path.exists():
        curated = pd.read_csv(curated_path, keep_default_na=False)
        for (sid, field), group in curated.groupby(["scheme_id", "field"]):
            en = group[group.language.eq("en")]
            hi = group[group.language.eq("hi")]
            if en.empty or hi.empty:
                issues.append({"scheme_id":sid,"field":field,"language":"/".join(sorted(set(group.language))),"issue":"curated_translation_pair_missing","record_id":""})
                continue
            nums = lambda values: {tuple(re.findall(r"\d[\d,.]*(?:%| rupees| रुपये)?", str(v).lower())) for v in values}
            if nums(en.fact) != nums(hi.fact):
                issues.append({"scheme_id":sid,"field":field,"language":"en/hi","issue":"numeric_or_date_conflict_manual_review","record_id":""})
            for row in group.to_dict("records"):
                checked = pd.to_datetime(row.get("checked_at", ""), errors="coerce", utc=True)
                if pd.isna(checked):
                    issues.append({"scheme_id":sid,"field":field,"language":row.get("language"),"issue":"reviewed_fact_missing_checked_at","record_id":""})
                elif (pd.Timestamp.now(tz="UTC") - checked).days > 180:
                    issues.append({"scheme_id":sid,"field":field,"language":row.get("language"),"issue":"source_check_older_than_180_days","record_id":""})
                until = pd.to_datetime(row.get("valid_until", ""), errors="coerce", utc=True)
                if not pd.isna(until) and until.date() < pd.Timestamp.now(tz="UTC").date():
                    issues.append({"scheme_id":sid,"field":field,"language":row.get("language"),"issue":"fact_past_valid_until_review_needed","record_id":""})
    pd.DataFrame(issues).to_csv(processed / "quality_review.csv", index=False, encoding="utf-8-sig")
    # Seed a manifest with known English myScheme links and an explicit blank Hindi URL slot.
    source_path = DATA / "myscheme" / "schemes.csv"
    df = pd.read_csv(source_path, keep_default_na=False)
    with (DATA / "source_pages.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["scheme_id","language","url","source_type","notes"])
        writer.writeheader()
        for row in df.to_dict("records"):
            sid = str(row.get("slug") or row.get("scheme_name", ""))
            url = str(row.get("source_url", ""))
            if url.startswith("http"):
                writer.writerow({"scheme_id":sid,"language":"en","url":url,"source_type":"myScheme page","notes":"Date not independently checked; review source."})
            writer.writerow({"scheme_id":sid,"language":"hi","url":"","source_type":"official Hindi page","notes":"Add confirmed official Hindi URL; do not infer URL pattern."})
    print(f"Normalized {len(records)} records from both sources; {len(schemes)} schemes.")
    print(f"Wrote {len(issues)} review flags to {processed / 'quality_review.csv'}")


if __name__ == "__main__":
    main()
