import re
import sqlite3
import pandas as pd

conn = sqlite3.connect("data/instamart.db")
text = open("sql/queries.sql").read()

parts = re.split(r"(?m)^-- (Q\d)\b.*$", text)
for name, body in zip(parts[1::2], parts[2::2]):
    sql = body.strip().rstrip(";")
    if not sql:
        continue
    df = pd.read_sql_query(sql, conn)
    print(f"\n=== {name}: {len(df)} rows ===")
    print(df.head(10).to_string(index=False))

conn.close()
