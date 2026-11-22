"""Rate-limited collector for the official URL manifest; saves snapshots, not claims."""
import argparse, csv, json, re, time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
ALLOWED = {"myscheme.gov.in", "india.gov.in"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=10, help="Default 10 pages; use 0 for all manifest URLs")
    parser.add_argument("--language", choices=["en", "hi"])
    args = parser.parse_args()
    manifest = ROOT / "data" / "source_pages.csv"
    with manifest.open(encoding="utf-8-sig", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r.get("url", "").startswith("http")]
    if args.language:
        rows = [r for r in rows if r.get("language") == args.language]
    if args.limit:
        rows = rows[:args.limit]
    out = ROOT / "data" / "page_snapshots"
    out.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = "GovernmentServiceAssistantCapstone/1.0 (research; polite rate limit)"
    log = []
    for row in rows:
        url = row["url"].strip()
        host = (urlparse(url).hostname or "").lower()
        if host not in ALLOWED and not any(host.endswith("." + d) for d in ALLOWED):
            log.append({**row, "status": "rejected_non_gov_host", "checked_at": ""})
            continue
        stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        try:
            res = session.get(url, timeout=25); res.raise_for_status()
            if "html" not in res.headers.get("content-type", "").lower():
                raise ValueError("Non-HTML source; review manually")
            soup = BeautifulSoup(res.text, "html.parser")
            sections = []
            for h in soup.find_all(re.compile(r"^h[1-6]$", re.I)):
                body = []
                for node in h.find_all_next(["h1","h2","h3","h4","h5","h6","p","li"]):
                    if node.name.lower().startswith("h"):
                        break
                    txt = node.get_text(" ", strip=True)
                    if txt: body.append(txt)
                if body: sections.append({"heading":h.get_text(" ",strip=True),"text":" ".join(body)})
            for tag in soup(["script","style","noscript"]): tag.decompose()
            page_text = soup.get_text("\n", strip=True)
            has_fact_headings = any(re.search(r"eligib|benefit|application|document|exclu|पात्र|लाभ|आवेदन|दस्तावेज", s["heading"], re.I)
                                    for s in sections)
            shell_only = ("Something went wrong. Please try again later." in page_text
                          or (not has_fact_headings and not (soup.title and soup.title.get_text(strip=True))))
            safe = re.sub(r"[^a-zA-Z0-9_-]", "_", row.get("scheme_id", "scheme"))
            lang = row.get("language", "und")
            path = out / f"{safe}_{lang}.json"
            path.write_text(json.dumps({"scheme_id":row.get("scheme_id"),"language":lang,"url":url,
                "title":soup.title.get_text(" ",strip=True) if soup.title else "",
                "text":page_text,"sections_for_review":sections,"checked_at":stamp,
                "collection_status":"shell_only_dynamic_content" if shell_only else "page_content_captured"},
                ensure_ascii=False,indent=2),encoding="utf-8")
            status = f"shell_only_dynamic_content:{path.name}" if shell_only else f"saved:{path.name}"
            log.append({**row,"status":status,"checked_at":stamp})
        except Exception as exc:
            log.append({**row,"status":f"error:{type(exc).__name__}:{exc}","checked_at":stamp})
        time.sleep(1)
    pdict = out / "collection_log.csv"
    with pdict.open("w",encoding="utf-8-sig",newline="") as f:
        cols=["scheme_id","language","url","source_type","notes","status","checked_at"]
        w=csv.DictWriter(f,fieldnames=cols); w.writeheader(); w.writerows(log)
    print(f"Processed {len(rows)} URLs. Review {pdict}")


if __name__ == "__main__":
    import pandas as pd
    main()
