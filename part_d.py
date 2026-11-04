import os, glob, json, math, re
import numpy as np
import pandas as pd
import mock_llm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---- Task 1: chunk the documents (40 words, overlap 10) ----
def chunk_words(words, size=40, overlap=10):
    step = size - overlap
    chunks, start = [], 0
    while True:
        chunks.append(words[start:start + size])
        if start + size >= len(words):   # stop after the first chunk whose end reaches the last word
            break
        start += step
    return chunks

chunks = []
for path in sorted(glob.glob("data/policies/*.txt")):
    words = open(path, encoding="utf-8").read().split()
    for c in chunk_words(words):
        chunks.append({"source": os.path.basename(path), "text": " ".join(c)})
print("chunks:", len(chunks))
print(pd.Series([c["source"] for c in chunks]).value_counts().sort_index().to_string())

# ---- Task 2: TF-IDF retrieval ----
vec = TfidfVectorizer()
M = vec.fit_transform([c["text"] for c in chunks])

def retrieve(query, k=3):
    sims = cosine_similarity(vec.transform([query]), M).ravel()
    top = np.argsort(-sims)[:k]
    return [{"source": chunks[i]["source"], "score": float(sims[i]), "text": chunks[i]["text"]} for i in top]

print([r["source"] for r in retrieve("refund policy late delivery")])

# ---- Task 3: augment and answer ----
SYSTEM = ("You are a POLICY ASSISTANT. Answer only from the supplied context. "
          "If the context does not contain the answer, say you could not find the information.")

def answer(question):
    hits = retrieve(question, 3)
    context = "\n".join(f"[{h['source']}]\n{h['text']}" for h in hits)
    user = f"CONTEXT:\n{context}\nQUESTION:\n{question}"
    resp = mock_llm.chat([{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}])
    return resp.choices[0].message.content, [h["source"] for h in hits]

# ---- Task 4: evaluate retrieval and grounding ----
ev = pd.read_csv("data/raw/policy_eval.csv", keep_default_na=False)
def run_eval(label):
    ans_ok = doc_ok = fb = 0
    answerable = ev[ev["expected_doc"] != "NONE"]
    unanswerable = ev[ev["expected_doc"] == "NONE"]
    for _, r in answerable.iterrows():
        text, srcs = answer(r["question"])
        doc_ok += r["expected_doc"] in srcs
        ans_ok += r["expected_fact"] in text
    for _, r in unanswerable.iterrows():
        text, _ = answer(r["question"])
        fb += text.startswith("I could not find")
    print(f"{label}: expected_doc in top 3: {doc_ok}/{len(answerable)}, "
          f"expected_fact in answer: {ans_ok}/{len(answerable)}, fallback: {fb}/{len(unanswerable)}")

run_eval("normal run")

# zero-overlap run: chunks of 40 words with no overlap
chunks_normal, M_normal = chunks, M
chunks = []
for path in sorted(glob.glob("data/policies/*.txt")):
    words = open(path, encoding="utf-8").read().split()
    for i in range(0, len(words), 40):
        chunks.append({"source": os.path.basename(path), "text": " ".join(words[i:i + 40])})
vec = TfidfVectorizer()
M = vec.fit_transform([c["text"] for c in chunks])
print("zero-overlap chunks:", len(chunks))
run_eval("zero overlap")
chunks, M = chunks_normal, M_normal
vec = TfidfVectorizer().fit([c["text"] for c in chunks])
