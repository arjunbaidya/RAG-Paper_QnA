import argparse
import json
import re
from pathlib import Path

import pymupdf
from tqdm import tqdm

import config
from rag import get_collection, get_embedder


def clean(text):

    text = re.sub(r"-\n(\w)", r"\1", text)   # joining the words broken by hyphen at line end
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def read_pdf(path):

    with pymupdf.open(path) as doc:
        pages = [clean(page.get_text()) for page in doc]
        title = (doc.metadata or {}).get("title") or ""
    return title.strip(), pages


# Splitting the text into chunks of chunk_size words with some overlap
# a chunk can go across pages so we store both start and end page for the citations

def chunk_pages(pages, chunk_size, overlap):

    words = []
    page_of = []
    for page_num, text in enumerate(pages, 1):
        for w in text.split():
            words.append(w)
            page_of.append(page_num)

    chunks = []
    step = chunk_size - overlap
    start = 0

    while start < len(words):

        end = min(start+chunk_size, len(words))
        chunks.append({"text": " ".join(words[start:end]),
                       "page_start": page_of[start],
                       "page_end": page_of[end-1]})
        if end==len(words):
            break
        start+=step

    return chunks


parser = argparse.ArgumentParser()
parser.add_argument("--papers", default=config.PAPERS_DIR)
parser.add_argument("--collection", default=config.COLLECTION)
parser.add_argument("--chunk-size", type=int, default=300)   # in words, roughly 400 tokens
parser.add_argument("--overlap", type=int, default=50)
parser.add_argument("--reset", action="store_true")
args = parser.parse_args()

if args.overlap>=args.chunk_size:
    raise SystemExit("overlap should be smaller than chunk size")

pdfs = sorted(Path(args.papers).glob("*.pdf"))
if len(pdfs)==0:
    raise SystemExit(f"No PDFs found in {args.papers}/, add some or run fetch_papers.py first")

# Titles and links saved by fetch_papers.py
index_file = Path(args.papers) / "index.json"
index = {}
if index_file.exists():
    index = json.loads(index_file.read_text())

collection = get_collection(args.collection, reset=args.reset)
embedder = get_embedder()

# Skipping the files which are already in the collection
done = set()
for meta in collection.get(include=["metadatas"])["metadatas"]:
    done.add(meta["source"])

added = 0
for pdf in tqdm(pdfs, desc="Ingesting"):

    if pdf.name in done:
        continue

    try:
        pdf_title, pages = read_pdf(pdf)
    except Exception as e:
        tqdm.write(f"Skipping {pdf.name}: {e}")
        continue

    chunks = chunk_pages(pages, args.chunk_size, args.overlap)
    if len(chunks)==0:
        tqdm.write(f"Skipping {pdf.name}: no text found (maybe a scanned pdf)")
        continue

    info = index.get(pdf.name, {})
    title = info.get("title") or pdf_title or pdf.stem.replace("_", " ")

    texts = [c["text"] for c in chunks]
    vectors = embedder.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False)

    ids = []
    metas = []
    for i,c in enumerate(chunks):
        ids.append(f"{pdf.name}::{i}")
        metas.append({"source": pdf.name,
                      "title": title,
                      "url": info.get("pdf_url", ""),
                      "page_start": c["page_start"],
                      "page_end": c["page_end"]})

    collection.add(ids=ids, documents=texts, embeddings=vectors.tolist(), metadatas=metas)
    added+=len(chunks)

print(f"Added {added} chunks. Collection '{args.collection}' now has {collection.count()} chunks")
