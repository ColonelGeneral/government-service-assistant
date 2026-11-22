"""Build aligned dense passage indexes sequentially (one model loaded at a time)."""
import argparse, gc, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
RECORDS = ROOT / "data" / "processed" / "records.jsonl"
MODELS = {
    "multilingual-e5-base": "intfloat/multilingual-e5-base",
    "BGE-M3": "BAAI/bge-m3",
    "LaBSE": "sentence-transformers/LaBSE",
}


def main():
    parser = argparse.ArgumentParser(description="Build aligned passage indexes for all configured models.")
    parser.add_argument("--batch-size", type=int, default=32, help="Embedding batch size (lower this for limited GPU memory).")
    parser.add_argument("--max-seq-length", type=int, default=512, help="Truncate embedded passages at this many tokens.")
    args = parser.parse_args()
    if args.batch_size < 1 or args.max_seq_length < 1:
        raise SystemExit("--batch-size and --max-seq-length must be positive integers.")
    if not RECORDS.exists():
        raise SystemExit("Run python scripts/prepare_project.py first.")
    rows = [json.loads(line) for line in RECORDS.read_text(encoding="utf-8").splitlines() if line]
    passages = [str(r["search_text"]) for r in rows]
    digest = hashlib.sha256("\n".join(r["record_id"] for r in rows).encode()).hexdigest()
    for name, model_id in MODELS.items():
        out = ROOT / "data" / "indexes" / name
        out.mkdir(parents=True, exist_ok=True)
        matrix_path = out / "passages.npy"
        metadata_path = out / "index.json"
        if matrix_path.exists() and metadata_path.exists():
            meta = json.loads(metadata_path.read_text(encoding="utf-8"))
            if meta.get("records_sha256") == digest:
                print(f"{name}: existing index matches records; skipping")
                continue
        print(f"Loading {name}; first run downloads its model weights.")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = SentenceTransformer(model_id, device=device)
        model.max_seq_length = min(model.max_seq_length, args.max_seq_length)
        # Half precision substantially reduces memory use and improves throughput
        # on CUDA devices; saved embeddings are converted back to float32 below.
        if device == "cuda":
            model.half()
        docs = [f"passage: {x}" for x in passages] if name == "multilingual-e5-base" else passages
        vectors = model.encode(docs, normalize_embeddings=True, batch_size=args.batch_size, show_progress_bar=True, convert_to_numpy=True)
        np.save(matrix_path, np.asarray(vectors, dtype=np.float32))
        metadata_path.write_text(json.dumps({"model_name":name,"model_id":model_id,"rows":len(rows),"dimension":int(vectors.shape[1]),
                                            "records_sha256":digest,"device":device,"precision":"float16" if device == "cuda" else "float32",
                                            "max_seq_length":model.max_seq_length,"batch_size":args.batch_size}, indent=2), encoding="utf-8")
        del vectors, model
        gc.collect()
        print(f"Saved {len(rows)} passage vectors for {name}.")
    print("All three model indexes are ready.")


if __name__ == "__main__":
    main()
