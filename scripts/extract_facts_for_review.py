"""Convert collected page headings into review candidates; never auto-approve them."""
import csv, json, re
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SNAPS=ROOT/"data"/"page_snapshots"
OUT=ROOT/"data"/"official_facts_review.csv"
RULES=[("eligibility",r"eligib|who can|पात्र|योग्य|कौन आवेदन"),
       ("benefits",r"benefit|assistance|amount|लाभ|सहायता|राशि"),
       ("application_process",r"apply|application|प्रक्रिया|आवेदन"),
       ("documents_required",r"document|proof|certificate|दस्तावेज|प्रमाण"),
       ("exclusions",r"exclu|not eligible|अपात्र|शामिल नहीं")]

def main():
    rows=[]
    if SNAPS.exists():
        for p in SNAPS.glob("*.json"):
            data=json.loads(p.read_text(encoding="utf-8"))
            # Never turn a JavaScript shell, an error dialog, or navigation text
            # into fact candidates. These snapshots need a browser/API collector.
            if data.get("collection_status") == "shell_only_dynamic_content":
                continue
            for section in data.get("sections_for_review",[]):
                heading=section.get("heading",""); field=next((f for f,pat in RULES if re.search(pat,heading,re.I)),"unclassified")
                if field == "unclassified":
                    continue
                text=section.get("text","").strip()
                if not text or "Something went wrong. Please try again later." in text:
                    continue
                rows.append({"scheme_id":data.get("scheme_id",""),"language":data.get("language",""),"field":field,
                  "fact":f"{heading}: {text}","source_url":data.get("url",""),"checked_at":data.get("checked_at",""),
                  "valid_from":"","valid_until":"","reviewer":"","review_status":"human_review_required"})
    with OUT.open("w",encoding="utf-8-sig",newline="") as f:
        cols=["scheme_id","language","field","fact","source_url","checked_at","valid_from","valid_until","reviewer","review_status"]
        w=csv.DictWriter(f,fieldnames=cols); w.writeheader(); w.writerows(rows)
    print(f"Wrote {len(rows)} review candidates to {OUT}")

if __name__=="__main__": main()
