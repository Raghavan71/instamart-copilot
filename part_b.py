import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import sqlite3

conn = sqlite3.connect("data/instamart.db")
orders = pd.read_sql_query("SELECT * FROM orders", conn)
customers = pd.read_sql_query("SELECT * FROM customers", conn)
conn.close()
orders["order_ts"] = pd.to_datetime(orders["order_ts"])
customers["signup_date"] = pd.to_datetime(customers["signup_date"])
completed = orders[orders["status"] == "completed"]

# ---- Task 1: RFM table as of 2025-10-01 ----
snapshot = pd.Timestamp("2025-10-01 00:00:00")
rfm = completed.groupby("customer_id").agg(
    last_order_ts=("order_ts", "max"),
    frequency=("order_id", "nunique"),
    monetary=("order_amount", "sum"),
).reset_index()
rfm["recency"] = (snapshot - rfm["last_order_ts"]).dt.days
rfm = rfm[["customer_id", "recency", "frequency", "monetary"]]
print("customers:", len(rfm))
print(rfm.describe().round(2))

# ---- Task 2: K-means for K = 2..8 ----
X = StandardScaler().fit_transform(rfm[["recency", "frequency", "monetary"]])
rows = []
for k in range(2, 9):
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    rows.append((k, round(km.inertia_, 1), round(silhouette_score(X, km.labels_), 3)))
print(pd.DataFrame(rows, columns=["K", "inertia", "silhouette"]).to_string(index=False))
