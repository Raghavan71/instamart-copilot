import sqlite3
import pandas as pd

conn = sqlite3.connect("data/instamart.db")
orders = pd.read_sql_query("SELECT * FROM orders", conn)
customers = pd.read_sql_query("SELECT * FROM customers", conn)
conn.close()

completed = orders[orders["status"] == "completed"]
cancelled = orders[orders["status"] == "cancelled"]

# ---- orders ----
assert len(orders) == 12309, "orders: row count"
assert orders["order_id"].nunique() == 12309, "orders: one row per order_id"
assert len(completed) == 11110, "orders: completed count"
assert len(cancelled) == 1199, "orders: cancelled count"
assert not (orders["status"] == "pending").any(), "orders: no pending rows left"
assert orders["order_amount"].isna().sum() == 0, "orders: no missing order_amount"
assert completed["delivery_minutes"].isna().sum() == 0, "orders: completed delivery filled"
assert cancelled["delivery_minutes"].isna().all(), "orders: cancelled delivery left missing"
assert (orders["delivery_minutes"].dropna() >= 0).all(), "orders: no negative delivery times"
assert abs(completed["order_amount"].sum() - 2998668.80) < 0.01, "orders: completed amount sum"
assert orders["order_ts"].str.match(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$").all(), "orders: order_ts format"

# ---- customers ----
assert len(customers) == 1200, "customers: row count"
assert customers["city"].value_counts().to_dict() == {
    "Pune": 227, "Delhi": 210, "Bengaluru": 197,
    "Chennai": 191, "Hyderabad": 191, "Mumbai": 184,
}, "customers: city counts"
assert customers["gender"].value_counts().to_dict() == {
    "female": 610, "male": 560, "unknown": 30,
}, "customers: gender counts"
assert customers["age"].isna().sum() == 0, "customers: no missing age"
assert customers["age"].between(18, 60).all(), "customers: ages between 18 and 60"

print("ALL CHECKPOINTS PASSED")