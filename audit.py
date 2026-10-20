import pandas as pd

customers = pd.read_csv("data/raw/customers.csv")
products = pd.read_csv("data/raw/products.csv")
orders = pd.read_csv("data/raw/orders.csv")

for name, df in [("customers", customers), ("products", products), ("orders", orders)]:
    print(f"\n=== {name} ===")
    print("shape:", df.shape)
    print(df.dtypes)
    missing = df.isna().sum()
    print(pd.DataFrame({"missing": missing, "pct": (missing / len(df) * 100).round(2)}))
    # ---- end of the for loop above ----

print("\n=== Specific audit figures ===")

# orders
bad_amount = pd.to_numeric(orders["order_amount"], errors="coerce").isna().sum()
print("order_amount not convertible to number:", bad_amount)
print("raw order rows:", len(orders))
print("exact duplicate rows:", orders.duplicated().sum())
print("pending rows:", (orders["status"] == "pending").sum())
print("distinct order_id:", orders["order_id"].nunique())
print("negative delivery_minutes:", (orders["delivery_minutes"] < 0).sum())
print("blank delivery_minutes:", orders["delivery_minutes"].isna().sum())

# customers
print("blank age:", customers["age"].isna().sum())
print("age outside 10-100:", (customers["age"] < 10).sum() + (customers["age"] > 100).sum())
print("distinct raw city spellings:", customers["city"].nunique())
print("distinct non-blank gender spellings:", customers["gender"].nunique())
print("blank gender:", customers["gender"].isna().sum())             