import argparse
import json
from datetime import datetime
from pathlib import Path

import config
from rag import retrieve, answer, page_label


def load_questions(path):

    questions = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line=="" or line.startswith("//"):
            continue
        q = json.loads(line)
        if isinstance(q["source"], str):
            q["source"] = [q["source"]]
        questions.append(q)
    return questions


# Returns rank of the first correct paper in the results, 0 if it is not there

def first_correct_rank(hits, sources):

    for rank, h in enumerate(hits, 1):
        for s in sources:
            if s in h["source"]:
                return rank
    return 0


def evaluate(questions, collection, k, rerank, verbose):

    n_hits = 0
    rr_sum = 0
    misses = []

    for q in questions:
        hits = retrieve(q["question"], collection, k=k, rerank=rerank)
        rank = first_correct_rank(hits, q["source"])
        if rank>0:
            n_hits+=1
            rr_sum+=1/rank
        else:
            misses.append((q["question"], [h["source"] for h in hits]))

    if verbose and len(misses)>0:
        print(f"\nMissed questions for {collection} (rerank={rerank}):")
        for question, got in misses:
            print(" -", question)
            print("   got:", ", ".join(got[:3]))

    n = len(questions)
    return {"collection": collection, "rerank": rerank, "hit": n_hits/n, "mrr": rr_sum/n}


# Saving the generated answers to a file so that we can check them manually

def write_answers(questions, collection, k, out_file):

    lines = [f"# Answer review: {collection}, k={k}\n"]

    for i,q in enumerate(questions, 1):
        hits = retrieve(q["question"], collection, k=k)
        sources = [f"[{j}] {h['title']} ({page_label(h)})" for j,h in enumerate(hits, 1)]

        lines.append(f"## {i}. {q['question']}\n")
        lines.append(answer(q["question"], hits) + "\n")
        lines.append("Sources: " + "; ".join(sources))
        lines.append("\n- [ ] Faithful to sources\n- [ ] Contains unsupported claims\n- [ ] Says 'not found' when it should\n")

    out_file.write_text("\n".join(lines))
    print("Saved answers to", out_file)


parser = argparse.ArgumentParser()
parser.add_argument("--questions", default="data/eval_questions.jsonl")
parser.add_argument("--collections", nargs="+", default=[config.COLLECTION])
parser.add_argument("-k", type=int, default=5)
parser.add_argument("--with-rerank", action="store_true")
parser.add_argument("--answers", action="store_true")
parser.add_argument("-v", "--verbose", action="store_true")
args = parser.parse_args()

questions = load_questions(args.questions)
print(f"Evaluating {len(questions)} questions, k={args.k}")

results = []
for collection in args.collections:
    results.append(evaluate(questions, collection, args.k, False, args.verbose))
    if args.with_rerank:
        results.append(evaluate(questions, collection, args.k, True, args.verbose))

# Making a markdown table so that it can be pasted in the README directly
table = [f"| Collection | Rerank | Hit@{args.k} | MRR |", "|---|---|---|---|"]
for r in results:
    rerank = "yes" if r["rerank"] else "no"
    table.append(f"| {r['collection']} | {rerank} | {r['hit']:.2f} | {r['mrr']:.2f} |")
table = "\n".join(table)
print("\n" + table)

Path("results").mkdir(exist_ok=True)
stamp = datetime.now().strftime("%Y%m%d_%H%M")

table_file = Path("results") / f"retrieval_{stamp}.md"
table_file.write_text(table + "\n")
print("\nSaved table to", table_file)

if args.answers:
    write_answers(questions, args.collections[0], args.k, Path("results") / f"answers_{stamp}.md")
