"""Three-model evidence ensemble demo. All model results are shown per query."""
import json
import csv
from pathlib import Path
import streamlit as st
from src.ensemble import retrieve_ensemble

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
st.set_page_config(page_title="Government Service Assistant · 3-model ensemble", page_icon="🇮🇳", layout="wide")
st.title("🇮🇳 Multilingual Government-Service Assistant")
st.caption("One query is evaluated by multilingual-e5-base, BGE-M3, and LaBSE. Their evidence rankings are fused; weak or conflicting evidence is surfaced instead of guessed.")

records_file = DATA / "processed" / "records.jsonl"
schemes_file = DATA / "processed" / "schemes.json"
if not records_file.exists() or not schemes_file.exists():
    st.error("Dataset preparation is required. From the project folder run: python scripts/prepare_project.py")
    st.stop()
records = [json.loads(line) for line in records_file.read_text(encoding="utf-8").splitlines() if line]
schemes = json.loads(schemes_file.read_text(encoding="utf-8"))
scheme_by_id = {s["scheme_id"]: s for s in schemes}
source_csv = DATA / "myscheme" / "schemes.csv"
if source_csv.exists():
    with source_csv.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            sid = str(row.get("slug") or "")
            if sid in scheme_by_id:
                # Display context helps people distinguish similarly named schemes.
                scheme_by_id[sid].update({key: row.get(key, "") for key in (
                    "short_title", "level", "state", "ministry", "department", "beneficiary_type", "target_beneficiaries"
                )})

with st.sidebar:
    st.header("Answer settings")
    language = st.selectbox("Answer language", ["English", "Hindi"])
    language_code = "en" if language == "English" else "hi"
    st.write("**Models run for each valid query**")
    st.write("1. multilingual-e5-base\n2. BGE-M3\n3. LaBSE")
    st.caption("First run needs all three model weights and built indexes. This takes disk space and can be slow on CPU.")

st.write("Include a scheme name and the topic you need: eligibility, benefits, required documents, or application process.")
with st.form("query"):
    query = st.text_input("Ask a question", placeholder="Pradhan Mantri Awas Yojana ke liye kaun eligible hai?")
    submitted = st.form_submit_button("Run all 3 models", type="primary")

if submitted:
    if not query.strip():
        st.warning("Question likhiye.")
        st.session_state.pop("pending_query", None)
        st.session_state.pop("selected_scheme_id", None)
        st.session_state.pop("query_result_key", None)
        st.session_state.pop("query_result", None)
    else:
        st.session_state["pending_query"] = query.strip()
        st.session_state.pop("selected_scheme_id", None)
        st.session_state.pop("query_result_key", None)
        st.session_state.pop("query_result", None)

active_query = st.session_state.get("pending_query")
if active_query:
    selected_scheme_id = st.session_state.get("selected_scheme_id")
    result_key = (active_query, language_code, selected_scheme_id)
    if st.session_state.get("query_result_key") != result_key:
        with st.spinner("Running the same question through all three retrieval models and comparing evidence…"):
            result = retrieve_ensemble(
                active_query, language_code, records, schemes, DATA / "indexes",
                selected_scheme_id=selected_scheme_id,
            )
        st.session_state["query_result_key"] = result_key
        st.session_state["query_result"] = result
    else:
        result = st.session_state["query_result"]

    status = result.get("status")
    if status == "ambiguous":
        st.warning("I found several similarly named schemes. Select the one you mean to continue. / मिलते-जुलते नाम वाली कई योजनाएँ मिलीं। आगे बढ़ने के लिए सही योजना चुनें।")
        option_ids = result.get("scheme_options", [])

        def scheme_label(sid):
            item = scheme_by_id.get(sid, {"scheme_name":sid})
            details = [item.get(k, "").strip() for k in ("short_title", "level", "state", "ministry", "department") if item.get(k, "").strip()]
            return " — ".join([item.get("scheme_name", sid), *details, f"ID: {sid}"])

        with st.form("scheme_clarification"):
            chosen_scheme = st.selectbox(
                "Which scheme do you mean? / आप किस योजना के बारे में पूछ रहे हैं?",
                option_ids,
                format_func=scheme_label,
            )
            st.caption("Choose using the scheme title, ministry, state, or department. Your original question will be kept.")
            clarify_submitted = st.form_submit_button("Continue with this scheme / इस योजना के साथ जारी रखें", type="primary")
        if clarify_submitted:
            st.session_state["selected_scheme_id"] = chosen_scheme
            st.session_state.pop("query_result_key", None)
            st.rerun()
    elif status == "ok":
        fact = result["winner"]
        st.subheader(fact["scheme_name"])
        st.write(fact["text"])
        checked = fact.get("checked_at") or "not independently date-checked"
        st.caption(f"Topic: {fact['field'].replace('_',' ').title()} · Source: {fact['source_name']} · Source check: {checked} · Support: {result['support']}/3 models")
        if fact.get("source_url"):
            st.markdown(f"[Open source page]({fact['source_url']})")
    elif status == "verified_language_missing":
        st.warning(result["message"])
        for fact in result.get("english_evidence", []):
            st.markdown(f"**English source evidence · {fact['scheme_name']}**")
            st.write(fact["text"])
            st.caption(f"Source check: {fact.get('checked_at') or 'not independently date-checked'}")
            if fact.get("source_url"):
                st.markdown(f"[Open official source]({fact['source_url']})")
    elif status == "source_conflict":
        st.error(result["message"])
        for fact in result.get("conflicts", []):
            st.markdown(f"**{fact['language'].upper()} source**: {fact['text']}")
            if fact.get("source_url"):
                st.markdown(f"[Open source]({fact['source_url']})")
    else:
        st.warning(result.get("message", "No safe result could be selected."))

    model_results = result.get("models", [])
    if model_results:
        st.subheader("Retrieved passage comparison")
        for model in model_results:
            top = model["top"]
            with st.expander(f"{model['model']} · score {top['score']:.3f}", expanded=True):
                for passage in model["passages"][:3]:
                    rec = passage["record"]
                    label = "approved source" if rec["usable_as_evidence"] else "QA example · not verified evidence"
                    st.markdown(f"**Rank {passage['rank']} · {label} · {rec['scheme_name']} · {rec['field']}**")
                    st.write(rec["text"])
                    st.caption(f"{rec['source_name']} · {rec['source_url']}")

with st.expander("Current source and verification status"):
    st.markdown("""
- The supplied myScheme `schemes.csv` and `schemes_faqs.csv` provide English scheme facts and official page URLs.
- Yojana Sahayak English/Hindi-Hinglish rows are indexed as query/evidence candidates, but community QA rows are **not** treated as official answer evidence until reviewed.
- The Hindi official-fact table is not populated yet. The assistant therefore will not present an unreviewed Hindi QA answer as verified; it can show the English source and report the missing Hindi evidence.
- Source check dates absent from the supplied files are shown as not independently date-checked.
""")
