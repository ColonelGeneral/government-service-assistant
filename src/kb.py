"""Prepare source records and conservative scheme/intent resolution."""
import json
import re
import unicodedata
from pathlib import Path
import pandas as pd

FIELDS = ["brief_description", "detailed_description", "benefits", "eligibility", "exclusions", "application_process", "documents_required", "scheme_open_date", "scheme_close_date"]
TITLE_STOP = {"pradhan", "mantri", "yojana", "scheme", "the", "for", "and", "of", "ke", "liye", "kaun", "hai", "hain", "who", "is", "eligible", "eligibility", "what", "how", "does", "provide", "apply", "me", "kya", "hai", "hindi"}


def normal(value):
    return re.sub(r"[^a-z0-9]", "", str(value or "").lower())


def words(value):
    return {x.lower() for x in re.findall(r"[a-z0-9]+", str(value or "")) if len(x) > 2 and x.lower() not in TITLE_STOP}


def substantive_answer(answer, title=""):
    answer = re.sub(r"\s+", " ", str(answer or "")).strip()
    if len(answer) < 45 or normal(answer) == normal(title):
        return False
    low = answer.lower()
    if any(low.startswith(prefix) and len(answer) < len(title) * 2 + 60 for prefix in (
        "to be eligible for", "eligibility for", "to be eligible", "benefits of", "the objective of the scheme", "application process for")):
        return False
    # Garbage HTML extraction patterns in some community QA records.
    if re.search(r"eligibilityexclusions|documentsrequired|frequentlyaskedquestions|sourcesand", normal(answer)):
        return False
    return True


def prepare_records(data_dir):
    data_dir = Path(data_dir)
    schemes = pd.read_csv(data_dir / "myscheme" / "schemes.csv", keep_default_na=False).to_dict("records")
    by_slug, by_title = {}, {}
    records = []
    for row in schemes:
        sid, title = str(row.get("slug") or normal(row.get("scheme_name"))), str(row.get("scheme_name", ""))
        aliases = [title, row.get("short_title", ""), row.get("slug", "")]
        info = {"scheme_id": sid, "scheme_name": title, "aliases": [a for a in aliases if str(a).strip()],
                "scheme_url": row.get("source_url", "") or f"https://www.myscheme.gov.in/schemes/{sid}"}
        by_slug[sid] = info
        by_title[normal(title)] = info
        for field in FIELDS:
            text = str(row.get(field, "")).strip()
            if text:
                records.append({"record_id": f"{sid}:{field}:myscheme", **info, "field": field,
                                "language": "en", "text": text, "search_text": " ".join([title, *info["aliases"], field, text]),
                                "source_name": "myScheme structured CSV", "source_url": info["scheme_url"],
                                "checked_at": "", "source_status": "official_url_not_date_checked", "usable_as_evidence": True})
    faq_path = data_dir / "myscheme" / "schemes_faqs.csv"
    if faq_path.exists():
        faqs = pd.read_csv(faq_path, keep_default_na=False).to_dict("records")
        for i, row in enumerate(faqs):
            slug, title = str(row.get("scheme_slug", "")), str(row.get("scheme_name", ""))
            info = by_slug.get(slug) or by_title.get(normal(title))
            if not info:
                continue
            q, a = str(row.get("question", "")).strip(), str(row.get("answer", "")).strip()
            field = "faq"
            if re.search(r"eligible|eligibility|पात्र|योग्य", q, re.I): field = "eligibility"
            elif re.search(r"benefit|amount|लाभ|राशि", q, re.I): field = "benefits"
            elif re.search(r"apply|application|आवेदन|प्रक्रिया", q, re.I): field = "application_process"
            elif re.search(r"document|दस्तावेज", q, re.I): field = "documents_required"
            if substantive_answer(a, info["scheme_name"]):
                records.append({"record_id": f"{slug}:faq:{i}", **info, "field": field, "language": "en",
                                "question": q, "text": a, "search_text": " ".join([info["scheme_name"], *info["aliases"], field, q, a]),
                                "source_name": "myScheme FAQs CSV", "source_url": info["scheme_url"],
                                "checked_at": "", "source_status": "official_url_not_date_checked", "usable_as_evidence": True})
    # Yojana Q&A records are included as multilingual retrieval examples, but not
    # treated as verified answer evidence until a reviewer links and approves them.
    for ypath in [data_dir / "yojana_sahayak" / "train.parquet", data_dir / "yojana_sahayak" / "eval.parquet"]:
        if not ypath.exists():
            continue
        qa = pd.read_parquet(ypath)
        for i, row in enumerate(qa.to_dict("records")):
            title = str(row.get("scheme_name", "")).strip()
            info = by_title.get(normal(title))
            if not info:
                continue
            messages = row.get("messages", [])
            q = next((str(m.get("content", "")).strip() for m in messages if m.get("role") == "user"), "")
            a = next((str(m.get("content", "")).strip() for m in messages if m.get("role") == "assistant"), "")
            if not substantive_answer(a, title):
                continue
            field = str(row.get("field", "faq"))
            # Keep Q&A rows as searchable aliases; final selection prefers official facts.
            records.append({"record_id": f"{info['scheme_id']}:yojana:{ypath.stem}:{i}", **info, "field": field,
                            "language": str(row.get("language", "en")), "question": q, "text": a,
                            "search_text": " ".join([title, *info["aliases"], field, q, a]),
                            "source_name": "Yojana Sahayak (unverified QA)", "source_url": info["scheme_url"],
                            "checked_at": "", "source_status": "review_required", "usable_as_evidence": False})
    # Human-reviewed official bilingual facts can be added to this schema.
    curated = data_dir / "official_facts.csv"
    if curated.exists():
        for i, row in enumerate(pd.read_csv(curated, keep_default_na=False).to_dict("records")):
            sid = str(row.get("scheme_id", "")); info = by_slug.get(sid)
            if not info or row.get("language") not in {"en", "hi"} or not str(row.get("fact", "")).strip():
                continue
            approved = (str(row.get("review_status", "")).lower() in {"approved", "verified", "human_reviewed"}
                        and bool(str(row.get("reviewer", "")).strip())
                        and bool(str(row.get("checked_at", "")).strip())
                        and bool(str(row.get("source_url", "")).strip()))
            records.append({"record_id": f"{sid}:{row.get('field')}:{row.get('language')}:{i}", **info,
                            "field": row.get("field", ""), "language": row.get("language", ""), "text": row.get("fact", ""),
                            "search_text": " ".join([info["scheme_name"], *info["aliases"], row.get("field", ""), row.get("fact", "")]),
                            "source_name": "Reviewed official bilingual fact", "source_url": row.get("source_url", info["scheme_url"]),
                            "checked_at": row.get("checked_at", ""), "source_status": "human_reviewed" if approved else "review_required",
                            "usable_as_evidence": approved})
    return records, list(by_slug.values())


def detect_intent(query):
    q = query.lower()
    rules = [
        ("eligibility", r"eligible|eligibility|who can|patra|पात्र|पात्रता|योग्य|कौन आवेदन"),
        ("benefits", r"benefit|benefits|amount|how much|kitna|labh|लाभ|राशि|कितना मिलेगा"),
        ("documents_required", r"document|documents|dastavez|दस्तावेज|कौन से कागज"),
        ("application_process", r"apply|application|how to|kaise apply|आवेदन|प्रक्रिया|कैसे करें"),
        ("scheme_close_date", r"deadline|last date|close date|आखिरी तारीख|अंतिम तिथि"),
        ("brief_description", r"what is|tell me about|about this scheme|योजना क्या|बताइए"),
    ]
    found = [field for field, pattern in rules if re.search(pattern, q, re.I)]
    return found[0] if found else ""


def resolve_scheme(query, schemes):
    qn, qw = normal(query), words(query)
    exact = []
    overlap = {}
    for scheme in schemes:
        for alias in scheme["aliases"]:
            an = normal(alias)
            if len(an) >= 5 and an in qn:
                exact.append(scheme["scheme_id"])
            score = len(qw & words(alias))
            if score:
                overlap[scheme["scheme_id"]] = max(overlap.get(scheme["scheme_id"], 0), score)
    if exact:
        ids = list(dict.fromkeys(exact))
        return (ids[0], "exact", [ids[0]]) if len(ids) == 1 else (None, "ambiguous", ids)
    if overlap:
        top = max(overlap.values())
        ids = [sid for sid, score in overlap.items() if score == top]
        return (ids[0], "token", [ids[0]]) if len(ids) == 1 else (None, "ambiguous", ids)
    return None, "not_found", []
