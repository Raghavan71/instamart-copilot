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

# ================= Tasks 5-8: refund agent =================
import sqlite3, time, shutil

DB = "data/instamart.db"
LEDGER = "outputs/refund_ledger.jsonl"
RUNS = "outputs/agent_runs.jsonl"
STATE_DIR = "state"
os.makedirs("outputs", exist_ok=True)
shutil.rmtree(STATE_DIR, ignore_errors=True); os.makedirs(STATE_DIR)
for f in (LEDGER, RUNS):
    open(f, "w").close()

CURRENT = {"run_id": None}

class SimulatedCrash(Exception):
    pass

# ---- Task 5: tools ----
def _ledger(tool, order_id, amount):
    with open(LEDGER, "a") as f:
        f.write(json.dumps({"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "tool": tool,
                            "run_id": CURRENT["run_id"], "order_id": order_id,
                            "refund_inr": amount}) + "\n")

def get_order(order_id):
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT order_id, customer_id, status, order_amount, delivery_minutes "
                       "FROM orders WHERE order_id = ?", (order_id,)).fetchone()
    conn.close()
    return {"error": f"order {order_id} not found"} if row is None else dict(row)

def search_policy(query):
    return [{"source": h["source"], "text": h["text"]} for h in retrieve(query, 3)]

def calculate_refund(order_amount, delivery_minutes):
    late = delivery_minutes is not None and delivery_minutes > 45
    return {"refund_inr": round(float(order_amount), 2) if late else 0.0}

def issue_refund(order_id, refund_inr):
    _ledger("issue_refund", order_id, refund_inr)
    return {"refund_inr": refund_inr}

def escalate_to_human(order_id, refund_inr):
    _ledger("escalate_to_human", order_id, refund_inr)
    return {"escalated": True, "refund_inr": refund_inr}

TOOL_MAP = {"get_order": get_order, "search_policy": search_policy,
            "calculate_refund": calculate_refund, "issue_refund": issue_refund,
            "escalate_to_human": escalate_to_human}

AGENT_SYSTEM = ("You are a refund AGENT. Look up the order, check the refund rules, "
                "calculate the refund and then issue it or escalate it. "
                "Finish with STATUS: explanation.")

# ---- Tasks 6-8: ReAct loop with limits and checkpoints ----
def _save(state):
    with open(f"{STATE_DIR}/{state['run_id']}.json", "w") as f:
        json.dump(state, f)

def _finish(state, status, answer=None):
    if answer is None:
        done = ", ".join(state["tools_executed"]) or "none"
        answer = (f"HANDOFF TO HUMAN: run stopped with {status} after "
                  f"{len(state['steps'])} tool steps ({done}).")
    state["status"], state["final_answer"] = status, answer
    _save(state)
    return {"run_id": state["run_id"], "scenario_id": state["scenario_id"], "status": status,
            "final_answer": answer, "steps": state["steps"], "llm_calls": state["llm_calls"],
            "tokens_used": state["tokens_used"], "refund_inr": state["refund_inr"],
            "tools_executed": state["tools_executed"]}

def _loop(state, crash_after_tools=None):
    CURRENT["run_id"] = state["run_id"]
    t0 = time.monotonic()
    while True:
        if time.monotonic() - t0 >= state["time_limit_seconds"]:        # (c) wall clock
            return _finish(state, "FALLBACK_TIMEOUT")
        if state["llm_calls"] >= state["max_iterations"]:               # (a) iterations
            return _finish(state, "FALLBACK_MAX_ITERATIONS")
        resp = mock_llm.chat(state["messages"], tools=list(TOOL_MAP))
        state["llm_calls"] += 1
        state["tokens_used"] += resp.usage.total_tokens
        if state["tokens_used"] > state["token_limit"]:                 # (b) tokens, tool NOT run
            return _finish(state, "FALLBACK_TOKEN_LIMIT")
        msg = resp.choices[0].message
        if not msg.tool_calls:
            answer = msg.content
            return _finish(state, answer.split(":")[0].strip(), answer)
        tc = msg.tool_calls[0]
        state["messages"].append({"role": "assistant", "content": None,
                                  "tool_calls": [{"id": tc.id, "name": tc.name, "arguments": tc.arguments}]})
        result = TOOL_MAP[tc.name](**tc.arguments)
        state["messages"].append({"role": "tool", "tool_call_id": tc.id, "name": tc.name,
                                  "content": json.dumps(result)})
        state["steps"].append({"tool": tc.name, "arguments": tc.arguments, "observation": result})
        state["tools_executed"].append(tc.name)
        if tc.name == "calculate_refund":
            state["refund_inr"] = result["refund_inr"]
        _save(state)                                                     # checkpoint
        if crash_after_tools is not None and len(state["tools_executed"]) == crash_after_tools:
            raise SimulatedCrash(f"simulated crash after tool execution {crash_after_tools}")

def run_agent(user_message, scenario_id, max_iterations=6, token_limit=6000,
              time_limit_seconds=10, run_id=None, crash_after_tools=None):
    state = {"run_id": run_id or scenario_id, "scenario_id": scenario_id,
             "messages": [{"role": "system", "content": AGENT_SYSTEM},
                          {"role": "user", "content": user_message}],
             "tokens_used": 0, "llm_calls": 0, "steps": [], "tools_executed": [], "refund_inr": 0.0,
             "max_iterations": max_iterations, "token_limit": token_limit,
             "time_limit_seconds": time_limit_seconds}
    return _loop(state, crash_after_tools)

def resume_agent(run_id):
    with open(f"{STATE_DIR}/{run_id}.json") as f:
        state = json.load(f)
    return _loop(state)

def log_run(res, label):
    with open(RUNS, "a") as f:
        f.write(json.dumps({"scenario_id": res["scenario_id"], "run_id": res["run_id"], "label": label,
                            "status": res["status"], "final_answer": res["final_answer"],
                            "llm_calls": res["llm_calls"], "tokens_used": res["tokens_used"],
                            "steps": res["steps"]}) + "\n")

# ---- run all seven scenarios ----
sc = pd.read_csv("data/raw/agent_scenarios.csv")
table = []
for _, r in sc.iterrows():
    res = run_agent(r["message"], r["scenario_id"])
    log_run(res, "scenario")
    ok = (res["status"].startswith(r["expected_outcome"]) and
          abs(res["refund_inr"] - float(r["expected_refund_inr"])) <= 0.01)
    table.append({"scenario": r["scenario_id"], "status": res["status"], "refund_inr": res["refund_inr"],
                  "model_calls": res["llm_calls"], "tokens": res["tokens_used"],
                  "expected": r["expected_outcome"], "matches": ok})
print(pd.DataFrame(table).to_string(index=False))

# ---- task 7: stop-safely demonstrations on S1 ----
s1 = sc.loc[sc["scenario_id"] == "S1", "message"].iloc[0]
d1 = run_agent(s1, "S1", token_limit=200, run_id="S1-token-limit"); log_run(d1, "demo token_limit=200")
d2 = run_agent(s1, "S1", time_limit_seconds=0, run_id="S1-timeout"); log_run(d2, "demo time_limit_seconds=0")
print("token_limit=200 demo status:", d1["status"], "| llm_calls:", d1["llm_calls"])
print("time_limit_seconds=0 demo status:", d2["status"], "| llm_calls:", d2["llm_calls"])

# ---- task 8: crash and resume on S1 ----
try:
    run_agent(s1, "S1", run_id="S1-crash", crash_after_tools=2)
except SimulatedCrash as e:
    print("crash:", e)
res = resume_agent("S1-crash")
log_run(res, "demo crash and resume")
print("resumed status:", res["status"], "| refund:", res["refund_inr"])
print("tools executed across crash and resume:", res["tools_executed"])
