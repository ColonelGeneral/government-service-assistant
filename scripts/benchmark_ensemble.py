"""Measure each retriever and the three-model RRF ensemble on held-out queries."""
import argparse, gc, json
from pathlib import Path
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

ROOT=Path(__file__).resolve().parents[1]
MODEL_IDS={"multilingual-e5-base":"intfloat/multilingual-e5-base","BGE-M3":"BAAI/bge-m3","LaBSE":"sentence-transformers/LaBSE"}

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--max-questions",type=int,default=250); parser.add_argument("--batch-size",type=int,default=32); args=parser.parse_args()
    data=ROOT/"data"; qpath=data/"evaluation"/"test_questions.csv"; recpath=data/"processed"/"records.jsonl"
    if not qpath.exists() or not recpath.exists(): raise SystemExit("Run prepare_project.py and make_eval_questions.py first.")
    questions=pd.read_csv(qpath,keep_default_na=False)
    if args.max_questions and len(questions)>args.max_questions: questions=questions.sample(args.max_questions,random_state=24).reset_index(drop=True)
    records=[json.loads(x) for x in recpath.read_text(encoding="utf-8").splitlines() if x]
    qn=len(questions); model_top={}; per_model={}; details=[]
    for name,model_id in MODEL_IDS.items():
        index=np.load(data/"indexes"/name/"passages.npy",mmap_mode="r")
        model=SentenceTransformer(model_id); query_texts=questions.question.astype(str).tolist()
        if name=="multilingual-e5-base": query_texts=["query: "+q for q in query_texts]
        ranks=[]; top_rows=[]
        for start in range(0,qn,args.batch_size):
            stop=min(start+args.batch_size,qn)
            qv=model.encode(query_texts[start:stop],normalize_embeddings=True,batch_size=args.batch_size,show_progress_bar=False,convert_to_numpy=True)
            sims=np.asarray(qv) @ np.asarray(index).T
            for local,j in enumerate(range(start,stop)):
                order=np.argsort(-sims[local])
                q=questions.iloc[j]
                gold=np.array([r["scheme_id"]==str(q.scheme_id) and r["field"]==str(q.field) and r["usable_as_evidence"] for r in records])
                pos=np.flatnonzero(gold[order]); rank=int(pos[0]+1) if len(pos) else None
                # The app fuses only approved passages, ranked independently of
                # unverified QA rows, and considers each model's top eight.
                evidence_top=[int(k) for k in order if records[int(k)]["usable_as_evidence"]][:8]
                top_rows.append([(records[k]["record_id"],int(rank0+1)) for rank0,k in enumerate(evidence_top)])
                ranks.append(rank)
                details.append({"model":name,"question_id":q.get("question_id",j),"language":q.language,"rank_of_gold":rank,"top1_approved_evidence":bool(records[int(order[0])]["usable_as_evidence"]),"top1_scheme_id":records[int(order[0])]["scheme_id"],"top1_field":records[int(order[0])]["field"]})
            del qv,sims
        per_model[name]={"ranks":ranks,"top":top_rows}
        del model,index; gc.collect()
    # Fuse the same query's ranked record lists using reciprocal rank; no raw scores compared across models.
    ensemble_ok=[]; ensemble_consensus=[]
    for j in range(qn):
        fusion={}; votes={}
        for name in MODEL_IDS:
            for rid,rank in per_model[name]["top"][j]:
                fusion[rid]=fusion.get(rid,0)+1/(60+rank)
                votes[rid]=votes.get(rid,0)+1
        eligible=[rid for rid in fusion if votes[rid]>=2]
        winner=max(eligible,key=fusion.get) if eligible else None
        q=questions.iloc[j]
        chosen=next((r for r in records if r["record_id"]==winner),None)
        ok=bool(chosen and chosen["scheme_id"]==str(q.scheme_id) and chosen["field"]==str(q.field))
        ensemble_ok.append(ok)
        ensemble_consensus.append(winner is not None)
    summary=[]
    for name in MODEL_IDS:
        ranks=per_model[name]["ranks"]
        summary.append({"setup":name,"queries":qn,"recall_at_1":sum(r==1 for r in ranks)/qn,"recall_at_5":sum(r is not None and r<=5 for r in ranks)/qn,"mrr":sum(1/r for r in ranks if r)/qn,"ensemble_accuracy":""})
    summary.append({"setup":"three-model RRF (approved passages, 2-of-3 support)","queries":qn,
                    "recall_at_1":"","recall_at_5":"","mrr":"",
                    "ensemble_accuracy":sum(ensemble_ok)/qn,"ensemble_coverage":sum(ensemble_consensus)/qn})
    out=data/"evaluation"; out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(summary).to_csv(out/"model_comparison.csv",index=False)
    pd.DataFrame(details).to_csv(out/"model_comparison_details.csv",index=False)
    print(f"Wrote comparison for {qn} held-out queries. Review model_comparison.csv and human language rubric.")

if __name__=="__main__": main()
