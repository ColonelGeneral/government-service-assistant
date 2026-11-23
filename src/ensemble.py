"""Run one query through all three models and choose evidence by rank consensus."""
import gc, hashlib, json, re
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
from src.kb import resolve_scheme, detect_intent

MODEL_IDS = {
    "multilingual-e5-base": "intfloat/multilingual-e5-base",
    "BGE-M3": "BAAI/bge-m3",
    "LaBSE": "sentence-transformers/LaBSE",
}


def _tokens(text):
    return tuple(sorted(set(re.findall(r"\d[\d,.]*(?:%| रुपये| rupees)?", str(text).lower()))))


def retrieve_ensemble(query, language, records, schemes, index_root, top_n=8, selected_scheme_id=None):
    if selected_scheme_id is not None:
        selected = next((s for s in schemes if s["scheme_id"] == selected_scheme_id), None)
        if selected is None:
            return {"status":"invalid_scheme_selection","message":"That scheme selection is no longer available. Please submit the question again.","models":[]}
        scheme_id, resolution, scheme_options = selected_scheme_id, "user_selected", [selected_scheme_id]
    else:
        scheme_id, resolution, scheme_options = resolve_scheme(query, schemes)
    if not scheme_id:
        option_names = [s["scheme_name"] for s in schemes if s["scheme_id"] in scheme_options]
        if resolution == "ambiguous":
            msg = "More than one scheme matches that name. Choose the specific scheme from the options shown."
        else:
            msg = "I couldn't identify a scheme from this question. Include its official scheme name or short title."
        return {"status":resolution,"message":msg,"scheme_options":scheme_options,
                "scheme_option_names":option_names,"models":[]}
    intent = detect_intent(query)
    if not intent:
        return {"status":"intent_missing","message":"Please say what you need: eligibility, benefits, documents, or application process.","models":[]}
    # Query candidates are restricted to the resolved scheme and requested field.
    ids = [i for i,r in enumerate(records) if r["scheme_id"] == scheme_id and r["field"] == intent]
    if not ids:
        return {"status":"field_missing","message":"This dataset has no record for that scheme and topic.","models":[]}
    digest = hashlib.sha256("\n".join(r["record_id"] for r in records).encode()).hexdigest()
    model_results = []
    for name, model_id in MODEL_IDS.items():
        index_path = Path(index_root) / name / "passages.npy"
        metadata_path = Path(index_root) / name / "index.json"
        if not index_path.exists() or not metadata_path.exists():
            return {"status":"indexes_missing","message":"Build all three indexes first: python scripts/build_indexes.py","models":[]}
        meta = json.loads(metadata_path.read_text(encoding="utf-8"))
        if meta.get("rows") != len(records):
            return {"status":"stale_index","message":"Indexes do not match the prepared records. Run scripts/build_indexes.py again.","models":[]}
        if meta.get("records_sha256") != digest:
            return {"status":"stale_index","message":"Indexes were built from a different record version. Run scripts/build_indexes.py again.","models":[]}
        model = None
        try:
            model = SentenceTransformer(model_id)
            query_text = f"query: {query}" if name == "multilingual-e5-base" else query
            qv = model.encode([query_text], normalize_embeddings=True, convert_to_numpy=True)[0]
            matrix = np.load(index_path, mmap_mode="r")
            scores = np.asarray(matrix[ids]) @ qv
        except Exception as exc:
            if model is not None:
                del model
            gc.collect()
            return {"status":"model_unavailable","message":f"{name} could not complete, so the three-model comparison is incomplete and no answer is selected. Check model download/storage and retry. Details: {type(exc).__name__}: {exc}","models":model_results}
        order = np.argsort(-scores)[:top_n]
        ranked = [{"record_id": records[ids[int(j)]]['record_id'], "rank": rank+1,
                   "score": float(scores[int(j)]), "record": records[ids[int(j)]]}
                  for rank,j in enumerate(order)]
        approved_local = [j for j, record_index in enumerate(ids) if records[record_index]["usable_as_evidence"]]
        approved_order = sorted(approved_local, key=lambda j: float(scores[j]), reverse=True)[:top_n]
        evidence_ranked = [{"record_id": records[ids[int(j)]]['record_id'], "rank": rank+1,
                            "score": float(scores[int(j)]), "record": records[ids[int(j)]]}
                           for rank,j in enumerate(approved_order)]
        top = ranked[0]
        # Rank-based fusion is robust to incomparable raw cosine scales.
        model_results.append({"model":name,"top":top,"passages":ranked,"evidence_passages":evidence_ranked})
        del matrix, model, qv, scores
        gc.collect()
    # Never present a community-generated QA response as official evidence.
    official_candidates = [r for r in records if r["scheme_id"] == scheme_id and r["field"] == intent
                           and r["language"] == language and r["usable_as_evidence"]]
    if language == "hi" and not official_candidates:
        english = [r for r in records if r["scheme_id"] == scheme_id and r["field"] == intent
                   and r["language"] == "en" and r["usable_as_evidence"]]
        english_ids = {r["record_id"] for r in english}; ranks = {rid:0.0 for rid in english_ids}
        for model_result in model_results:
            for passage in model_result["evidence_passages"]:
                if passage["record_id"] in english_ids:
                    ranks[passage["record_id"]] += 1.0/(60+passage["rank"])
        by_id = {r["record_id"]:r for r in english}
        ordered = [by_id[rid] for rid in sorted(ranks,key=ranks.get,reverse=True) if ranks[rid] > 0]
        return {"status":"verified_language_missing","message":"No reviewed official Hindi fact is available for this scheme/topic. The models retrieved these English source passages; add and review an official Hindi page before returning a Hindi answer.","scheme_id":scheme_id,"english_evidence":ordered[:3],"models":model_results}
    if not official_candidates:
        return {"status":"evidence_missing","message":"Models found candidate passages, but none is approved as answer evidence.","models":model_results}
    # Rank-fuse only the reviewed, language-matched source records. Require the same
    # source passage to appear in at least two models' top-k results.
    official_ids = {r["record_id"] for r in official_candidates}
    official_votes = {rid: 0.0 for rid in official_ids}
    support_by_id = {rid: 0 for rid in official_ids}
    for result in model_results:
        for p in result["evidence_passages"]:
            if p["record_id"] in official_ids:
                official_votes[p["record_id"]] += 1.0 / (60 + p["rank"])
                support_by_id[p["record_id"]] += 1
    winner_id = max(official_votes, key=official_votes.get)
    support = support_by_id[winner_id]
    if support < 2:
        return {"status":"no_consensus","message":"Fewer than two models retrieved the same reviewed source fact. No answer is returned.","models":model_results}
    winner_fact = next(r for r in official_candidates if r["record_id"] == winner_id)
    # If translated facts are explicitly reviewed, compare amounts/dates and abstain on conflict.
    pairs = [r for r in records if r["scheme_id"] == scheme_id and r["field"] == intent
             and r.get("source_status") == "human_reviewed" and r.get("usable_as_evidence")]
    numeric = {tuple(_tokens(r["text"])) for r in pairs}
    if {r["language"] for r in pairs} >= {"en","hi"} and len(numeric) > 1:
        return {"status":"source_conflict","message":"Reviewed English and Hindi source facts disagree on numbers/dates. No definitive answer is returned.","conflicts":pairs,"models":model_results}
    return {"status":"ok","scheme_id":scheme_id,"intent":intent,"winner":winner_fact,"models":model_results,"support":support}
