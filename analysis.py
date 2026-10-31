import sqlite3
import pandas as pd

conn = sqlite3.connect("data/instamart.db")
orders = pd.read_sql_query("SELECT * FROM orders", conn)
customers = pd.read_sql_query("SELECT * FROM customers", conn)
products = pd.read_sql_query("SELECT * FROM products", conn)
conn.close()

completed = orders[orders["status"] == "completed"].copy()

# (a) pivot_table with margins vs SQL total
merged = completed.merge(customers[["customer_id", "city"]], on="customer_id").merge(
    products[["product_id", "category"]], on="product_id")
pivot = pd.pivot_table(merged, values="order_amount", index="city", columns="category",
                       aggfunc="sum", margins=True, margins_name="Total")
print(pivot.round(2))
print("pivot grand total:", round(pivot.loc["Total", "Total"], 2), "(SQL total: 2998668.80)")

# (a) groupby with named aggregation, per city
by_city = merged.groupby("city").agg(
    total=("order_amount", "sum"), mean=("order_amount", "mean"), count=("order_amount", "count")
).round(2)
print(by_city)

# (b) distribution statistics for completed order_amount
s = completed["order_amount"]
q1, q3 = s.quantile(0.25), s.quantile(0.75)
iqr = q3 - q1
lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
print("mean:", round(s.mean(), 2), "median:", s.median(), "skew:", round(s.skew(), 2))
print("Q1:", q1, "Q3:", q3, "fences:", lo, hi)
print("outliers above:", (s > hi).sum(), "below:", (s < lo).sum())

# correlation matrix
cols = ["order_amount", "discount_pct", "quantity", "delivery_minutes"]
print(completed[cols].corr().round(3))
