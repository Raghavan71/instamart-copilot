# Instamart Customer Intelligence & Support Copilot

## Overview

Four connected parts built on synthetic, fictional Instamart data (all amounts in INR):

- Part A cleans the raw exports into a SQLite database and answers business questions in SQL and pandas.
- Part B segments customers (RFM + K-means) and predicts who will lapse.
- Part C triages support tickets with privacy, retry, validation and budget guardrails, using the offline `mock_llm.py`.
- Part D answers policy questions with TF-IDF retrieval and runs a refund agent (ReAct loop, stop limits, checkpoints).

No API key or network is used anywhere.

## Reproduce

Python 3.11+ (tested on 3.12). From the repository root:

1. `python3 -m venv venv`
2. `source venv/bin/activate`
3. `pip install -r requirements.txt`
4. `python generate_data.py`
5. `python audit.py`
6. `python clean.py`
7. `python validate.py`
8. `python run_queries.py`
9. `python analysis.py`
10. `python part_b.py`
11. `python part_c.py`
12. `python part_d.py`

## Part A: Data Foundation

**Raw audit** (`audit.py`): shape, data types and missing values

| Table     | Shape      | Missing values                                                     |
| --------- | ---------- | ------------------------------------------------------------------ |
| customers | (1200, 5)  | age 61 (5.08%), gender 30 (2.50%), others 0                        |
| products  | (40, 4)    | none                                                               |
| orders    | (13084, 9) | order_amount 53 (0.41%), delivery_minutes 2,541 (19.42%), others 0 |

Data types: customers `customer_id`, `city`, `signup_date`, `gender` text, `age` float; products `product_id`, `product_name`, `category` text, `price_inr` int; orders `order_id`, `customer_id`, `product_id`, `order_ts`, `status`, `order_amount` text, `quantity` and `discount_pct` int, `delivery_minutes` float. (`order_amount` is text because some entries are `N/A` or `error`.)

| Audit figure                        | Value  |
| ----------------------------------- | ------ |
| raw order rows                      | 13,084 |
| exact duplicate rows                | 119    |
| pending rows                        | 656    |
| distinct order_id                   | 12,309 |
| non-numeric order_amount            | 120    |
| negative delivery_minutes           | 106    |
| blank delivery_minutes              | 2,541  |
| blank age                           | 61     |
| age outside 10-100                  | 6      |
| distinct raw city spellings         | 24     |
| distinct non-blank gender spellings | 10     |
| blank gender                        | 30     |

**Cleaning rules and rows changed** (`clean.py`)

- customers.city: strip spaces + title case; 476 values changed; 6 cities remain (Pune 227, Delhi 210, Bengaluru 197, Chennai 191, Hyderabad 191, Mumbai 184)
- customers.gender: mapped to male / female / unknown; 950 values changed (female 610, male 560, unknown 30)
- customers.age: values outside 10-100 set to missing, then all 67 missing ages (61 blank + 6 invalid) filled with the median age of the customer's city; 0 missing, all ages 18-60
- customers.signup_date converted to datetime
- orders.order_ts converted to datetime
- orders: 119 exact duplicate rows removed
- orders: 656 older records of the same order_id removed (latest order_ts kept)
- orders.order_amount: 112 values filled as price x quantity x (1 - discount/100)
- orders.delivery_minutes: 105 negative values set to missing
- orders.delivery_minutes: 774 missing values of completed orders filled with the median of the order's product category among completed orders; cancelled orders stay missing (1,199)
- Result: 12,309 orders (11,110 completed, 1,199 cancelled, 0 pending); completed order_amount sums to INR 2,998,668.80
- `validate.py` asserts 16 reference checkpoints and prints `ALL CHECKPOINTS PASSED`

**SQL results** (`sql/queries.sql`, run from Python by `run_queries.py`)

| Query | Technique                      | Result                                                                                                                                                                                                                  |
| ----- | ------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Q1    | WHERE then HAVING              | 30 customers                                                                                                                                                                                                            |
| Q2    | three-table join               | 30 rows, total INR 2,998,668.80, largest cell Bengaluru / Electronics INR 233,909.45                                                                                                                                    |
| Q3    | scalar subquery                | 11 products above the average price                                                                                                                                                                                     |
| Q4    | LEFT JOIN                      | 25 customers with no completed order                                                                                                                                                                                    |
| Q5    | CTE                            | 550 customers above the average of INR 2,552.06                                                                                                                                                                         |
| Q6    | RANK / DENSE_RANK / ROW_NUMBER | top per city: Bengaluru C0082 (12,856.20), Chennai C0930 (10,151.40), Delhi C0291 (13,742.60), Hyderabad C0743 (12,730.05), Mumbai C0693 (14,766.55), Pune C0426 (10,377.10); 8 rows where RANK differs from ROW_NUMBER |
| Q7    | running total                  | 183 daily rows, last cumulative revenue INR 2,998,668.80                                                                                                                                                                |

`RANK()` assigns the same rank to customers with tied spending and skips subsequent rank numbers, whereas `ROW_NUMBER()` gives every customer a unique sequential number regardless of ties.

**pandas cross-check and EDA** (`analysis.py`)

- pivot_table (margins=True) grand total INR 2,998,668.80 equals the SQL total

| City      | sum        | mean   | count |
| --------- | ---------- | ------ | ----- |
| Bengaluru | 523,643.40 | 276.47 | 1,894 |
| Chennai   | 463,629.15 | 272.72 | 1,700 |
| Delhi     | 502,829.80 | 243.62 | 2,064 |
| Hyderabad | 528,589.90 | 284.95 | 1,855 |
| Mumbai    | 479,058.50 | 297.00 | 1,613 |
| Pune      | 500,918.05 | 252.48 | 1,984 |

- Completed order_amount: mean 269.91, median 135.00, skewness 4.61, Q1 64.80, Q3 297.00, 1.5xIQR fences -283.50 and 645.30, 897 outliers above and 0 below
- Correlation (order_amount, discount_pct, quantity, delivery_minutes): largest in absolute value is order_amount with discount_pct at -0.095; order_amount with quantity is +0.087

**EDA interpretation**

This dataset represents fictional Instamart customer, product and order transactions across six Indian cities. The completed order amounts are strongly right-skewed, with a mean of INR 269.91 compared with a median of INR 135.00, indicating that most orders are relatively small while a smaller number have much higher values. The distribution contains 897 high-value outliers above the upper IQR fence of INR 645.30, with no outliers below the lower fence. These may represent unusually large orders or premium purchases and should be investigated rather than automatically removed. The largest city-level revenue is Hyderabad at INR 528,589.90, while Mumbai has the highest average order amount at INR 297.00. The strongest observed correlation is between order amount and discount percentage (-0.095), followed by order amount and quantity (+0.087). Both relationships are weak, and correlations of this size do not establish causation.

## Part B: Customer Segments and Lapse Prediction

**RFM** (snapshot 2025-10-01): 1,175 customers; recency median 19 (min 0, max 180); frequency mean 9.46 (max 33); monetary mean 2,552.06 (max 14,766.55).

**K-means** (standardised RFM, `n_init=10`, `random_state=42`)

| K   | inertia | silhouette |
| --- | ------- | ---------- |
| 2   | 1820.8  | 0.465      |
| 3   | 1268.2  | 0.445      |
| 4   | 743.1   | 0.526      |
| 5   | 617.4   | 0.480      |
| 6   | 503.2   | 0.482      |
| 7   | 408.5   | 0.416      |
| 8   | 366.5   | 0.412      |

**K selection:** I selected K=4 because it achieved the highest silhouette score of 0.526, indicating better separation and cohesion between clusters among the tested values. The inertia also shows a noticeable reduction up to K=4, after which the improvements become smaller, suggesting an elbow around four clusters. K=8 always has lower inertia than K=2 because adding more clusters gives K-means more flexibility to minimise within-cluster distances. However, lower inertia alone does not mean better clustering, since too many clusters can create unnecessary complexity and less interpretable customer groups.

**Segments (K=4)** (cluster numbers are only labels and have no ordering)

| id  | name                               | size | mean recency | mean frequency | mean monetary | action                              |
| --- | ---------------------------------- | ---- | ------------ | -------------- | ------------- | ----------------------------------- |
| 0   | Occasional low spenders            | 436  | 28.86        | 3.77           | 1,057.62      | free-delivery nudge                 |
| 1   | Premium big spenders (**Premium**) | 138  | 24.12        | 6.77           | 7,129.70      | VIP perks, early electronics access |
| 2   | Loyal regulars                     | 419  | 8.77         | 19.40          | 3,410.02      | subscription / bundle offers        |
| 3   | At-risk / fading                   | 182  | 109.04       | 2.20           | 685.99        | win-back coupon                     |

PCA on the scaled RFM matrix: explained*variance_ratio* 0.634 and 0.218, total 0.852. Output: `outputs/customer_segments.csv`.

**Lapse problem** (cutoff 2025-09-01): cohort 1,156 customers, 443 lapsed (38.32%); stratified split with `random_state=42` gives 924 train / 232 test. Features use only orders before the cutoff. The segment labels are not used as features, because both would leak the answer. Encoders and scaler are fitted on the training set only. Always predicting "not lapsed" gives accuracy 0.616 and lapsed-class F1 0.000.

**Tuning and test results** (lapsed = positive class)

| Model               | best params                      | best CV F1 | test accuracy | precision | recall | F1    | confusion [[TN, FP], [FN, TP]] |
| ------------------- | -------------------------------- | ---------- | ------------- | --------- | ------ | ----- | ------------------------------ |
| Logistic regression | C=1                              | 0.655      | 0.694         | 0.594     | 0.640  | 0.616 | [[104, 39], [32, 57]]          |
| Random forest       | max_depth=5, min_samples_leaf=10 | 0.687      | 0.724         | 0.637     | 0.652  | 0.644 | [[110, 33], [31, 58]]          |

**Model selection:** I selected Random Forest because it achieved a test F1 score of 0.644, outperforming Logistic Regression's 0.616 and comfortably exceeding the required 0.55 threshold. It also achieved better test accuracy, precision and recall, making it the stronger overall model in this experiment. Missing a genuinely lapsing customer could mean losing future revenue and a valuable customer relationship. However, wrongly targeting a loyal customer wastes retention incentives and marketing resources. The selected model provides a reasonable balance between these two costs.

**Class-balance experiment** (training set over-sampled to 570 / 570, logistic regression, original test set): recall 0.640 -> 0.854, precision 0.594 -> 0.585, F1 0.616 -> 0.694.

**Class-balance interpretation:** Recall improved the most, increasing from 0.640 to 0.854, meaning the model identified a much larger proportion of genuinely lapsing customers. Precision decreased slightly from 0.594 to 0.585, indicating a small increase in false-positive retention targets. The F1 score also improved from 0.616 to 0.694. For the retention team, this means fewer at-risk customers are missed, although more customers may receive unnecessary retention offers.

**Interpretation**

- Top 5 random-forest importances: frequency 0.343, monetary 0.276, recency 0.084, cancel_rate 0.066, avg_delivery 0.048
- Logistic regression coefficients (standardised features): recency +0.041, frequency -1.546, monetary +0.174

**Coefficient interpretation:**

- Recency (+0.041): A higher number of days since the last order slightly increases the predicted likelihood of lapsing.
- Frequency (-1.546): Customers with more frequent orders have substantially lower predicted odds of lapsing.
- Monetary (+0.174): Higher spending slightly increases the predicted likelihood of lapsing when the other features are held constant.

**Persistence and retention list:** `outputs/lapse_model.joblib` reloads and reproduces the test predictions exactly. `outputs/retention_list.csv` holds the top 10 of the 40 Premium customers in the test set by predicted lapse probability; 1 of the 10 truly lapsed.

The retention list is illustrative rather than a leakage-free evaluation because the Premium flag is derived from whole-window RFM data, which includes the September period used to define the lapse label.

## Part C: Support Ticket Triage with Guardrails

- Input guardrails: 4 empty tickets (`skipped_empty`) and 4 long tickets (`flagged_long`, 734 estimated tokens each); none sent to the model
- Redaction: 55 emails, 91 phones, 132 order ids = 278 substitutions; re-applying the three patterns to the redacted texts finds 0
- Retries (max 3, exponential backoff with jitter, BASE_DELAY 0.05): 19 `TransientLLMError`s in each full run, all recovered; every wait between 0.045 and 0.0575 s
- Output validation over the 156 eligible tickets: 135 `classified_json`, 12 `classified_keyword`, 9 `flagged_invalid_output`

| Prompt    | accuracy over 147 classified | total tokens | tokens per classified ticket |
| --------- | ---------------------------- | ------------ | ---------------------------- |
| zero-shot | 85.7% (126 of 147)           | 10,492       | 71.4                         |
| few-shot  | 100% (147 of 147)            | 16,420       | 111.7                        |

Zero-shot errors: 11 fraud tickets predicted billing and 10 technical tickets predicted billing.

**Prompt comparison:** The few-shot prompt is worth the additional token cost when accurate ticket routing is important, particularly for fraud and technical issues where incorrect classification can delay resolution. It achieved 100% accuracy compared with 85.7% for zero-shot prompting, eliminating all 21 observed classification errors. The trade-off is an increase of approximately 40.3 tokens per classified ticket (111.7 versus 71.4). For a support system where incorrect routing creates operational and customer-service costs, this accuracy improvement justifies the additional token usage.

- Budget demo (zero-shot, `TOKEN_BUDGET = 6000`): 66 tickets ended `budget_exhausted` (`outputs/triage_budget_demo.csv`)
- Final run (few-shot, budget 100,000): `outputs/triage_results.csv` with 164 rows. Routes: support_queue 102, security_team 34, human_review 17, premium_desk 11
- Rule-validated outputs: the 147 `classified_json` and `classified_keyword` results. Sent to humans: the empty, long, invalid-output and budget-exhausted tickets (17)

## Part D: Policy RAG Assistant and Refund Agent

**RAG:** 15 chunks of 40 words with overlap 10 (refund 4, delivery 3, escalation 3, payment 3, privacy 2). For the query "refund policy late delivery" the top three chunks all come from `refund_policy.txt`.

**TF-IDF explanation:** Term Frequency (TF) measures how often a word appears in a document, while Inverse Document Frequency (IDF) reduces the importance of words that appear frequently across many documents. One limitation of TF-IDF is that it relies on exact word matching and may fail to recognise synonyms or different expressions with the same meaning, which can reduce retrieval quality for customer policy questions.

| Evaluation               | expected doc in top 3 | expected fact in answer | fallback on unanswerable |
| ------------------------ | --------------------- | ----------------------- | ------------------------ |
| normal (overlap 10)      | 12/12                 | 12/12                   | 2/2                      |
| zero overlap (13 chunks) | 12/12                 | 10/12                   | 2/2                      |

With zero overlap two answers lose their fact because it is cut across a chunk boundary.

**Refund agent, seven scenarios**

| Scenario | status                  | refund (INR) | model calls | tokens |
| -------- | ----------------------- | ------------ | ----------- | ------ |
| S1       | REFUND                  | 128.25       | 5           | 993    |
| S2       | REFUND                  | 135.00       | 5           | 993    |
| S3       | ESCALATE                | 1999.00      | 5           | 983    |
| S4       | DENIED                  | 0.00         | 4           | 685    |
| S5       | DENIED                  | 0.00         | 3           | 401    |
| S6       | NOT_FOUND               | 0.00         | 2           | 119    |
| S7       | FALLBACK_MAX_ITERATIONS | 0.00         | 6           | 3003   |

**Stop-safely demonstrations (S1):** `token_limit=200` ends with `FALLBACK_TOKEN_LIMIT` (3 model calls); `time_limit_seconds=0` ends with `FALLBACK_TIMEOUT` (0 model calls).

**Crash and resume (S1):** a simulated crash after the second tool execution, then `resume_agent`. Tools executed across both parts, in order: `get_order`, `search_policy`, `calculate_refund`, `issue_refund` (four executions, none repeated). Final status `REFUND` with 128.25.

Outputs: `outputs/agent_runs.jsonl`, `outputs/refund_ledger.jsonl`, `state/`.

**ReAct versus plan-and-execute**

ReAct uses a repeated reasoning-and-action loop, allowing the agent to inspect each tool result and decide what to do next. In this experiment, S1 required 5 model calls and 993 tokens, S5 required 3 calls and 401 tokens, and S6 required 2 calls and 119 tokens. This shows that the number of calls and token usage can vary depending on the complexity of the scenario.

A plan-and-execute design would create a sequence of actions in advance and then execute that plan. It can be more predictable for straightforward refund cases, but may be less flexible when unexpected observations occur. For example, a `NOT_FOUND` order should stop the refund process rather than continue with calculations or payment actions. Similarly, a cancelled order may require a different decision based on policy and order status.

For this workflow, I would choose ReAct because refund decisions depend on order details, policy retrieval and validation of intermediate tool results. Its ability to adapt to unexpected observations, combined with iteration limits, token and time limits, checkpoints, and crash recovery, makes it safer for financial operations. Although it can require more model calls, correctness and controlled execution are more important than minimising token usage alone.

## Limitations and Next Steps

**Limitations**

- **Synthetic data:** The project uses fictional Instamart data, so its results may not represent actual customer behaviour, order patterns or support operations.
- **Keyword-based mock model:** The offline `mock_llm.py` provides a controlled simulation but does not capture the full reasoning and language capabilities of a production LLM.
- **TF-IDF retrieval:** The RAG system relies on keyword matching and may miss synonyms, contextual meaning and semantically related policy questions.
- **Single train/test split:** The lapse prediction results come from one stratified train/test split, so model performance may vary on different customer samples or future data.
- **Retention-list caveat:** The Premium flag used to create the retention list comes from whole-window RFM data, which includes the period defining the lapse label. Therefore, the list is illustrative and should not be treated as a leakage-free evaluation.

**Next Steps**

1. **Use real, anonymised data:** Test the cleaning, segmentation and prediction pipelines on representative production data, with appropriate privacy controls and monitoring.
2. **Improve modelling and retrieval:** Evaluate additional models using repeated cross-validation and time-based validation, and replace or complement TF-IDF with semantic embeddings to improve policy retrieval.
3. **Strengthen production readiness:** Integrate a real LLM with secure API handling, monitoring, human approval for high-value refunds, and continuous evaluation of ticket classification and refund decisions.

---

_Built as an end-to-end customer intelligence and support automation project using Python, SQL, SQLite, pandas, scikit-learn, TF-IDF and agentic workflows._
