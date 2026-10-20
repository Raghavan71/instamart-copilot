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
