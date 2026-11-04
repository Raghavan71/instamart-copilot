import json, os, random, re, time
import pandas as pd
import mock_llm
from mock_llm import est_tokens, TransientLLMError

ALLOWED = ["billing", "delivery", "technical", "fraud"]
MAX_INPUT_TOKENS = 400
BASE_DELAY = 0.05

tickets = pd.read_csv("data/raw/tickets.csv", keep_default_na=False)
os.makedirs("outputs", exist_ok=True)

# ---- Task 1: input guardrails ----
def guard(text):
    if text.strip() == "":
        return "skipped_empty"
    if est_tokens(text) > MAX_INPUT_TOKENS:
        return "flagged_long"
    return ""

tickets["guard"] = [guard(t) for t in tickets["text"]]
print(tickets["guard"].value_counts().to_string())
print("long tickets est_tokens:", [est_tokens(t) for t in tickets.loc[tickets["guard"] == "flagged_long", "text"]])

# ---- Task 2: redact personal information ----
EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
PHONE = re.compile(r"\b[6-9]\d{9}\b")
ORDER = re.compile(r"ORD-\d{6}")
n_email = n_phone = n_order = 0
redacted = []
for text in tickets["text"]:
    text, a = EMAIL.subn("[EMAIL]", text)
    text, b = PHONE.subn("[PHONE]", text)
    text, c = ORDER.subn("[ORDER_ID]", text)
    n_email += a; n_phone += b; n_order += c
    redacted.append(text)
tickets["redacted"] = redacted
print("substitutions: emails", n_email, "phones", n_phone, "order ids", n_order,
      "total", n_email + n_phone + n_order)
left = sum(len(p.findall(t)) for t in redacted for p in (EMAIL, PHONE, ORDER))
print("left after re-applying the patterns:", left)

# ---- Task 3: two prompts ----
ZERO_SHOT = ("You are a support assistant. CLASSIFY the support ticket into exactly one of these "
             "categories: billing, delivery, technical, fraud. "
             'Reply with JSON only, in the form {"category": "...", "reason": "..."}.')
FEW_SHOT = (ZERO_SHOT + "\nEXAMPLES:\n"
            'Ticket: I was charged twice for order [ORDER_ID]. -> {"category": "billing", "reason": "duplicate charge"}\n'
            'Ticket: Someone used my card without my permission. -> {"category": "fraud", "reason": "unauthorized transaction"}')
assert "EXAMPLES:" not in ZERO_SHOT and "EXAMPLES:" in FEW_SHOT

# ---- Task 4: resilient calls ----
def call_with_retry(messages, ticket_id):
    for attempt in range(4):              # first try + at most 3 retries
        try:
            return mock_llm.chat(messages), attempt
        except TransientLLMError:
            if attempt == 3:
                raise
            wait = BASE_DELAY * 2 ** attempt * random.uniform(0.9, 1.15)
            print(f"retry log: {ticket_id} retry {attempt} wait {wait:.4f}s")
            time.sleep(wait)

# ---- Task 5: validate every answer ----
def validate(reply):
    try:
        data = json.loads(reply)
        if isinstance(data, dict) and data.get("category") in ALLOWED:
            return "classified_json", data["category"]
    except json.JSONDecodeError:
        words = set(re.findall(r"[a-z]+", reply.lower()))
        hits = [c for c in ALLOWED if c in words]
        if len(hits) == 1:
            return "classified_keyword", hits[0]
    return "flagged_invalid_output", ""

# ---- full pipeline (used by tasks 4-7) ----
def run_pipeline(system_prompt, token_budget=None, label=""):
    mock_llm.reset()
    random.seed(42)
    retries = total_tokens = 0
    rows = []
    for _, t in tickets.iterrows():
        row = {"ticket_id": t["ticket_id"], "customer_id": t["customer_id"],
               "true_category": t["true_category"], "status": "", "category": "", "tokens_used": 0}
        if t["guard"] != "":
            row["status"] = t["guard"]
        elif token_budget is not None and total_tokens >= token_budget:
            row["status"] = "budget_exhausted"
        else:
            user_text = t["redacted"]
            assert not (EMAIL.search(user_text) or PHONE.search(user_text) or ORDER.search(user_text)), \
                "personal data reached the prompt"
            messages = [{"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_text}]
            resp, n = call_with_retry(messages, t["ticket_id"])
            retries += n
            used = resp.usage.total_tokens
            total_tokens += used
            row["tokens_used"] = used
            row["status"], row["category"] = validate(resp.choices[0].message.content)
        rows.append(row)
    df = pd.DataFrame(rows)
    print(f"[{label}] retries: {retries}, total tokens: {total_tokens}")
    print(df["status"].value_counts().to_string())
    return df

# ---- Task 6: zero-shot vs few-shot ----
def evaluate(df, name):
    ok = df[df["status"].isin(["classified_json", "classified_keyword"])]
    right = (ok["category"] == ok["true_category"]).sum()
    total = df["tokens_used"].sum()
    print(f"{name}: classified {len(ok)}, accuracy {right / len(ok):.1%} ({right} of {len(ok)}), "
          f"total tokens {total}, tokens per classified ticket {total / len(ok):.1f}")
    return ok

zero = run_pipeline(ZERO_SHOT, label="zero-shot")
ok_zero = evaluate(zero, "zero-shot")
print(pd.crosstab(ok_zero["true_category"], ok_zero["category"]))
few = run_pipeline(FEW_SHOT, label="few-shot")
evaluate(few, "few-shot")

# ---- Task 7a: token budget demo (zero-shot) ----
TOKEN_BUDGET = 6000
demo = run_pipeline(ZERO_SHOT, token_budget=TOKEN_BUDGET, label="budget demo")
demo.to_csv("outputs/triage_budget_demo.csv", index=False)
print("budget_exhausted:", (demo["status"] == "budget_exhausted").sum(), "with TOKEN_BUDGET =", TOKEN_BUDGET)

# ---- Task 7b: final run (few-shot) with routing ----
final = run_pipeline(FEW_SHOT, token_budget=100000, label="final few-shot")
seg = pd.read_csv("outputs/customer_segments.csv")[["customer_id", "segment_name", "is_premium"]]
final = final.merge(seg, on="customer_id", how="left")
final["segment_name"] = final["segment_name"].fillna("")
final["is_premium"] = final["is_premium"].fillna(False).astype(bool)

HUMAN = {"skipped_empty", "flagged_long", "flagged_invalid_output", "budget_exhausted"}
def route(r):
    if r["status"] in HUMAN:
        return "human_review", "NORMAL"
    if r["category"] == "fraud":
        return "security_team", "HIGH"
    if r["is_premium"]:
        return "premium_desk", "HIGH"
    return "support_queue", "NORMAL"

rt = final.apply(route, axis=1, result_type="expand")
final["route"], final["priority"] = rt[0], rt[1]
final[["ticket_id", "customer_id", "status", "category", "segment_name",
       "priority", "route", "tokens_used"]].to_csv("outputs/triage_results.csv", index=False)
print(final["route"].value_counts().to_string())
print("triage_results.csv rows:", len(final))
