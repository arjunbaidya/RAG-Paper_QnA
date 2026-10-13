import argparse
import json
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import requests

import config

API_URL = "https://export.arxiv.org/api/query"
ATOM = "{http://www.w3.org/2005/Atom}"
HEADERS = {"User-Agent": "paper-qa/0.1 (portfolio project)"}


def slugify(text):

    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:80]


def query_arxiv(params):

    resp = requests.get(API_URL, params=params, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    root = ET.fromstring(resp.text)

    papers = []
    for entry in root.findall(ATOM + "entry"):

        abs_url = entry.findtext(ATOM + "id", "").strip()
        title = " ".join(entry.findtext(ATOM + "title", "").split())

        pdf_url = None
        for link in entry.findall(ATOM + "link"):
            if link.get("title")=="pdf":
                pdf_url = link.get("href")

        if not abs_url or not title or not pdf_url:
            continue

        papers.append({"arxiv_id": abs_url.split("/abs/")[-1],
                       "title": title,
                       "abs_url": abs_url,
                       "pdf_url": pdf_url,
                       "published": entry.findtext(ATOM + "published", "")[:10]})
    return papers


def search(query, n, sort):

    # "dataset poisoning detection" -> all:dataset AND all:poisoning AND all:detection
    search_query = " AND ".join(["all:" + word for word in query.split()])
    params = {"search_query": search_query, "max_results": n, "sortBy": sort, "sortOrder": "descending"}
    return query_arxiv(params)


def fetch_by_ids(ids):

    papers = []

    # 20 ids at a time otherwise the url becomes too long
    for i in range(0, len(ids), 20):
        batch = ids[i:i+20]
        papers += query_arxiv({"id_list": ",".join(batch), "max_results": len(batch)})
        time.sleep(3)

    return papers


def read_ids_file(path):

    ids = []
    for line in Path(path).read_text().splitlines():
        line = line.split("#")[0].strip()
        if line:
            ids.append(line)
    return ids


parser = argparse.ArgumentParser()
parser.add_argument("query", nargs="?")
parser.add_argument("-n", type=int, default=25)
parser.add_argument("--sort", choices=["relevance", "submittedDate"], default="relevance")
parser.add_argument("--ids-file")
parser.add_argument("--out", default=config.PAPERS_DIR)
args = parser.parse_args()

if not args.query and not args.ids_file:
    parser.error("give a search query or --ids-file")

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)

index_file = out / "index.json"
index = {}
if index_file.exists():
    index = json.loads(index_file.read_text())

if args.ids_file:
    ids = read_ids_file(args.ids_file)
    papers = fetch_by_ids(ids)
    print(f"Found {len(papers)} out of {len(ids)} papers from {args.ids_file}")
else:
    papers = search(args.query, args.n, args.sort)
    print(f"Found {len(papers)} papers for '{args.query}'")

for i,p in enumerate(papers, 1):

    filename = p["arxiv_id"].replace("/", "_") + "_" + slugify(p["title"]) + ".pdf"
    path = out / filename

    if path.exists():
        print(f"[{i}/{len(papers)}] already downloaded: {filename}")
    else:
        print(f"[{i}/{len(papers)}] downloading: {p['title'][:70]}")
        try:
            r = requests.get(p["pdf_url"], headers=HEADERS, timeout=60)
            r.raise_for_status()
            path.write_bytes(r.content)
        except requests.RequestException as e:
            print("    failed:", e)
            continue
        time.sleep(3)   # arXiv asks for 3 sec gap between requests

    index[filename] = {"title": p["title"], "url": p["abs_url"], "pdf_url": p["pdf_url"], "published": p["published"]}

index_file.write_text(json.dumps(index, indent=2))
print(f"Done. {len(index)} papers in {index_file}")
