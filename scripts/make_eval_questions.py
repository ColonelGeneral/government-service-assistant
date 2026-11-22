"""Generate English, Hindi, and code-mixed retrieval questions; split by scheme ID."""
from pathlib import Path
import hashlib
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
TEMPLATES={
"eligibility": [("en","Who is eligible for {name}?"),("hi-Latn","{name} ke liye kaun eligible hai?"),("hi","{name} के लिए कौन पात्र है?")],
"benefits": [("en","What benefits does {name} provide?"),("hi-Latn","{name} mein kya benefits milte hain?"),("hi","{name} में क्या लाभ मिलते हैं?")],
"documents_required": [("en","Which documents are required for {name}?"),("hi-Latn","{name} ke liye kaun se documents chahiye?"),("hi","{name} के लिए कौन से दस्तावेज़ चाहिए?")],
"application_process": [("en","How do I apply for {name}?"),("hi-Latn","{name} ke liye apply kaise karun?"),("hi","{name} के लिए आवेदन कैसे करें?")],
}
def split(sid):
    n=int(hashlib.sha1(str(sid).encode()).hexdigest()[:8],16)%100
    return "train" if n<70 else ("dev" if n<85 else "test")
def main():
    df=pd.read_csv(ROOT/"data"/"myscheme"/"schemes.csv",keep_default_na=False); rows=[]
    for s in df.to_dict("records"):
        sid=str(s.get("slug") or s.get("scheme_name")); name=str(s.get("scheme_name",""))
        for field,variants in TEMPLATES.items():
            if not str(s.get(field," ")).strip(): continue
            for lang,template in variants:
                rows.append({"scheme_id":sid,"scheme_name":name,"field":field,"language":lang,
                  "question":template.format(name=name),"gold_source_url":s.get("source_url",""),"split":split(sid),"question_origin":"synthetic_template"})
    out=ROOT/"data"/"evaluation"; out.mkdir(parents=True,exist_ok=True)
    x=pd.DataFrame(rows)
    for name in ["train","dev","test"]: x[x.split.eq(name)].to_csv(out/f"{name}_questions.csv",index=False,encoding="utf-8-sig")
    print(f"Created {len(x)} templated queries; held-out test is grouped by scheme.")
if __name__=="__main__": main()
