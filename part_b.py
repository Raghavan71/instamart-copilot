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

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# ---- Task 4: lapse problem, features from orders BEFORE 2025-09-01 only ----
cutoff = pd.Timestamp("2025-09-01")
pre = completed[completed["order_ts"] < cutoff]
sept = completed[(completed["order_ts"] >= cutoff) & (completed["order_ts"] < pd.Timestamp("2025-10-01"))]

cohort_ids = pre["customer_id"].unique()
lapsed = (~pd.Series(cohort_ids).isin(sept["customer_id"].unique())).astype(int).values

feat = pre.groupby("customer_id").agg(
    last_order=("order_ts", "max"),
    frequency=("order_id", "nunique"),
    monetary=("order_amount", "sum"),
    avg_discount=("discount_pct", "mean"),
    avg_delivery=("delivery_minutes", "mean"),
).reset_index()
feat["recency"] = (cutoff - feat["last_order"]).dt.days

# cancellation rate: cancelled / all orders with a final status, before the cutoff
all_pre = orders[orders["order_ts"] < cutoff]
cancel = (all_pre.assign(c=(all_pre["status"] == "cancelled").astype(int))
          .groupby("customer_id")["c"].mean().rename("cancel_rate"))
feat = feat.merge(cancel, on="customer_id", how="left")

# most frequent product category before the cutoff
products = pd.read_sql_query("SELECT * FROM products", sqlite3.connect("data/instamart.db"))
pre_cat = pre.merge(products[["product_id", "category"]], on="product_id")
top_cat = (pre_cat.groupby("customer_id")["category"]
           .agg(lambda s: s.value_counts().index[0]).rename("top_category"))
feat = feat.merge(top_cat, on="customer_id", how="left")

feat = feat.merge(customers[["customer_id", "city", "age", "gender", "signup_date"]], on="customer_id", how="left")
feat["tenure_days"] = (cutoff - feat["signup_date"]).dt.days
feat["lapsed"] = (~feat["customer_id"].isin(sept["customer_id"].unique())).astype(int)

print("cohort:", len(feat), "lapsed:", feat["lapsed"].sum(), f"({feat['lapsed'].mean():.2%})")

# ---- Task 5: split first, then fit encoders and scaler on TRAIN only ----
num_cols = ["recency", "frequency", "monetary", "avg_discount", "avg_delivery",
            "cancel_rate", "age", "tenure_days"]
cat_cols = ["top_category", "city", "gender"]
X_all = feat[num_cols + cat_cols]
y = feat["lapsed"]
X_train, X_test, y_train, y_test = train_test_split(
    X_all, y, test_size=0.2, stratify=y, random_state=42)
print("train:", len(X_train), "test:", len(X_test))
print("baseline accuracy (always 'not lapsed'):", round((y_test == 0).mean(), 4))

enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(X_train[cat_cols])
scaler = StandardScaler().fit(X_train[num_cols])
def prep(Xd):
    return np.hstack([scaler.transform(Xd[num_cols]), enc.transform(Xd[cat_cols])])
Xtr, Xte = prep(X_train), prep(X_test)
print("feature matrix shapes:", Xtr.shape, Xte.shape)
