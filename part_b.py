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

import os
from sklearn.decomposition import PCA

os.makedirs("outputs", exist_ok=True)
pd.set_option("display.width", 200)

# ---- Task 3: profile, name and project the segments (K = 4) ----
K = 4
km = KMeans(n_clusters=K, n_init=10, random_state=42).fit(X)
rfm["segment_id"] = km.labels_
profile = rfm.groupby("segment_id").agg(
    size=("customer_id", "count"),
    mean_recency=("recency", "mean"),
    mean_frequency=("frequency", "mean"),
    mean_monetary=("monetary", "mean"),
).round(2)
print(profile.to_string())

# name segments from their profile (cluster numbers are only labels, with no order)
left = set(profile.index)
premium_id = profile["mean_monetary"].idxmax(); left.discard(premium_id)
loyal_id = profile.loc[list(left), "mean_frequency"].idxmax(); left.discard(loyal_id)
risk_id = profile.loc[list(left), "mean_recency"].idxmax(); left.discard(risk_id)
(occasional_id,) = left

names = {premium_id: "Premium big spenders", loyal_id: "Loyal regulars",
         risk_id: "At-risk / fading", occasional_id: "Occasional low spenders"}
actions = {premium_id: "VIP perks and early access to electronics launches",
           loyal_id: "Subscription / bundle offers to protect repeat habit",
           risk_id: "Win-back discount coupon sent this week",
           occasional_id: "Free-delivery nudge to raise order frequency"}
rfm["segment_name"] = rfm["segment_id"].map(names)
rfm["is_premium"] = rfm["segment_id"] == premium_id
for sid in profile.index:
    print(sid, "|", names[sid], "|", actions[sid])

# PCA on the scaled RFM matrix
pca = PCA(n_components=2).fit(X)
print("explained_variance_ratio_:", pca.explained_variance_ratio_.round(3),
      "total:", round(pca.explained_variance_ratio_.sum(), 3))

rfm[["customer_id", "recency", "frequency", "monetary",
     "segment_id", "segment_name", "is_premium"]].to_csv("outputs/customer_segments.csv", index=False)
print("saved outputs/customer_segments.csv:", len(rfm), "rows")
