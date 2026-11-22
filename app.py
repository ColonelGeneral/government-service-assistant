"""Three-model evidence ensemble demo. All model results are shown per query."""
import json
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
        st.stop()
    with st.spinner("Running the same question through all three retrieval models and comparing evidence…"):
        result = retrieve_ensemble(query.strip(), language_code, records, schemes, DATA / "indexes")
    status = result.get("status")
    if status == "ok":
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
