import pandas as pd

customers = pd.read_csv("data/raw/customers.csv")
raw = customers.copy()  # keep the raw version to count what changed

# (a) city: strip spaces, then title case
customers["city"] = customers["city"].str.strip().str.title()

# (b) gender: map every spelling to male / female / unknown (blank -> unknown)
gender_map = {"male": "male", "m": "male", "female": "female", "f": "female"}
customers["gender"] = (
    customers["gender"].str.strip().str.lower().map(gender_map).fillna("unknown")
)

# (c) age: out-of-range ages become missing, then fill ALL missing ages
#     with the median age of that customer's city
customers["age"] = customers["age"].where(customers["age"].between(10, 100))
city_median = customers.groupby("city")["age"].transform("median")
customers["age"] = customers["age"].fillna(city_median)

# (d) signup_date to datetime
customers["signup_date"] = pd.to_datetime(customers["signup_date"])

# ---- checks against the brief ----
print(customers["city"].value_counts())
print(customers["gender"].value_counts())
print("missing ages:", customers["age"].isna().sum())
print("age min/max:", customers["age"].min(), customers["age"].max())

# ---- counts for the README ----
print("city values changed:", (raw["city"] != customers["city"]).sum())
print("gender values changed:", (raw["gender"].fillna("") != customers["gender"]).sum())
print("ages filled (blank + out of range):", raw["age"].isna().sum() + (~raw["age"].dropna().between(10, 100)).sum())
# ================= Task 4: clean orders =================
orders = pd.read_csv("data/raw/orders.csv")
products = pd.read_csv("data/raw/products.csv")
n_raw = len(orders)

# 1. order_ts to datetime
orders["order_ts"] = pd.to_datetime(orders["order_ts"])

# 2. remove exact duplicate rows
orders = orders.drop_duplicates()
n_after_exact = len(orders)

# 3. same order_id more than once: keep only the row with the latest order_ts
orders = orders.sort_values("order_ts").drop_duplicates(subset="order_id", keep="last")
n_after_latest = len(orders)

# 4. order_amount to numeric (bad values -> NaN), then fill from the product price
orders["order_amount"] = pd.to_numeric(orders["order_amount"], errors="coerce")
orders = orders.merge(products[["product_id", "price_inr", "category"]], on="product_id", how="left")
calc_amount = (orders["price_inr"] * orders["quantity"] * (1 - orders["discount_pct"] / 100)).round(2)
n_amount_filled = orders["order_amount"].isna().sum()
orders["order_amount"] = orders["order_amount"].fillna(calc_amount)

# 5. negative delivery_minutes -> missing
n_negative = (orders["delivery_minutes"] < 0).sum()
orders["delivery_minutes"] = orders["delivery_minutes"].where(orders["delivery_minutes"] >= 0)

# 6. completed orders: fill missing delivery_minutes with the median of that
#    product category among completed orders; cancelled orders stay missing
completed = orders["status"] == "completed"
medians = orders[completed].groupby("category")["delivery_minutes"].median()
n_delivery_filled = (completed & orders["delivery_minutes"].isna()).sum()
orders.loc[completed, "delivery_minutes"] = orders.loc[completed, "delivery_minutes"].fillna(
    orders.loc[completed, "category"].map(medians)
)

# ---- checks ----
print("rows:", len(orders))
print(orders["status"].value_counts())
print("missing order_amount:", orders["order_amount"].isna().sum())
print("missing delivery (completed):", orders[orders["status"] == "completed"]["delivery_minutes"].isna().sum())
print("missing delivery (cancelled):", orders[orders["status"] == "cancelled"]["delivery_minutes"].isna().sum())
print("completed order_amount sum:", round(orders[orders["status"] == "completed"]["order_amount"].sum(), 2))

# ---- counts for the README ----
print("exact duplicates removed:", n_raw - n_after_exact)
print("older (pending) rows removed:", n_after_exact - n_after_latest)
print("order_amount values filled:", n_amount_filled)
print("negative delivery_minutes blanked:", n_negative)
print("completed delivery_minutes filled with category median:", n_delivery_filled)
# ================= Task 5: store =================
import os, sqlite3

os.makedirs("data/clean", exist_ok=True)

customers_out = customers.copy()
customers_out["signup_date"] = customers_out["signup_date"].dt.strftime("%Y-%m-%d")

# helper columns from the merge are not part of the clean orders table
orders_out = orders.drop(columns=["price_inr", "category"]).copy()
orders_out["order_ts"] = orders_out["order_ts"].dt.strftime("%Y-%m-%d %H:%M:%S")

customers_out.to_csv("data/clean/customers.csv", index=False)
products.to_csv("data/clean/products.csv", index=False)
orders_out.to_csv("data/clean/orders.csv", index=False)

conn = sqlite3.connect("data/instamart.db")
customers_out.to_sql("customers", conn, if_exists="replace", index=False)
products.to_sql("products", conn, if_exists="replace", index=False)
orders_out.to_sql("orders", conn, if_exists="replace", index=False)
conn.commit()
conn.close()

print("saved:", len(customers_out), "customers,", len(products), "products,", len(orders_out), "orders")