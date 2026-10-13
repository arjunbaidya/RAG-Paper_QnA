import sys
import chromadb
from openai import OpenAI

import config

SYSTEM_PROMPT = """You answer questions about research papers using ONLY the numbered context passages provided.
Rules:
- Cite passages inline like [1] or [2][3], right after the claim they support.
- If the passages don't contain the answer, say you couldn't find it in the indexed papers. Never use outside knowledge.
- Be concise and precise. When comparing methods, name the papers they come from."""

# Models take time to load so loading them only once
embedder = None
reranker = None
client = None
llm = None


def get_embedder():

    global embedder
    if embedder is None:
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer(config.EMBED_MODEL)
    return embedder


def get_reranker():

    global reranker
    if reranker is None:
        from sentence_transformers import CrossEncoder
        reranker = CrossEncoder(config.RERANK_MODEL)
    return reranker


def get_llm():

    global llm
    if llm is None:
        llm = OpenAI(base_url=config.LLM_BASE_URL, api_key=config.LLM_API_KEY)
    return llm


def get_collection(name=config.COLLECTION, reset=False):

    global client
    if client is None:
        client = chromadb.PersistentClient(path=config.DB_DIR)

    if reset:
        try:
            client.delete_collection(name)
        except Exception:
            pass

    return client.get_or_create_collection(name)


def retrieve(question, collection_name=config.COLLECTION, k=5, rerank=False, fetch_k=20):

    collection = get_collection(collection_name)
    total = collection.count()
    if total==0:
        return []

    # For reranking we first take more candidates, then the cross-encoder picks the top k
    n = min(fetch_k if rerank else k, total)

    query_vec = get_embedder().encode([config.QUERY_INSTRUCTION + question], normalize_embeddings=True)
    res = collection.query(query_embeddings=query_vec.tolist(), n_results=n,
                           include=["documents", "metadatas", "distances"])

    hits = []
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        hit = dict(meta)
        hit["text"] = doc
        hit["score"] = 1 - dist/2   # vectors are normalized so this gives the cosine similarity
        hits.append(hit)

    if rerank:
        scores = get_reranker().predict([(question, h["text"]) for h in hits])
        for i,h in enumerate(hits):
            h["score"] = float(scores[i])
        hits = sorted(hits, key=lambda h: h["score"], reverse=True)

    return hits[:k]


def page_label(hit):

    if hit["page_start"]==hit["page_end"]:
        return f"p. {hit['page_start']}"
    return f"pp. {hit['page_start']}-{hit['page_end']}"


def build_context(hits):

    blocks = []
    for i,h in enumerate(hits, 1):
        blocks.append(f"[{i}] {h['title']} ({page_label(h)})\n{h['text']}")
    return "\n\n".join(blocks)


def stream_text(response):

    for chunk in response:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


def answer(question, hits, stream=False):

    if len(hits)==0:
        msg = "No papers are indexed yet. Add PDFs to papers/ and run `python ingest.py`."
        if stream:
            return iter([msg])
        return msg

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context passages:\n\n{build_context(hits)}\n\nQuestion: {question}"}
    ]

    response = get_llm().chat.completions.create(model=config.LLM_MODEL, messages=messages,
                                                 temperature=0.1, stream=stream)
    if stream:
        return stream_text(response)
    return response.choices[0].message.content


# Quick test from terminal: python rag.py "your question"
if __name__ == "__main__":

    if len(sys.argv)<2:
        print('Usage: python rag.py "your question"')
        sys.exit()

    question = " ".join(sys.argv[1:])
    hits = retrieve(question)

    print(answer(question, hits))
    print("\nSources:")
    for i,h in enumerate(hits, 1):
        print(f"  [{i}] {h['title']} ({page_label(h)}), score: {round(h['score'],3)}")
