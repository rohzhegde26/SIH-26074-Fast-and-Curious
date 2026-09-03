import duckdb

con = duckdb.connect()
res = con.execute("""
    SELECT stname, dtname, count(*), count(DISTINCT gpcode)
    FROM 'data/raw/geodata/LGD_panchayats.parquet'
    WHERE upper(stname) LIKE '%KARNATAKA%' AND upper(dtname) LIKE '%MANDYA%'
    GROUP BY stname, dtname
""").fetchall()
print("Mandya records in parquet:", res)

res_mysuru = con.execute("""
    SELECT stname, dtname, count(*), count(DISTINCT gpcode)
    FROM 'data/raw/geodata/LGD_panchayats.parquet'
    WHERE upper(stname) LIKE '%KARNATAKA%' AND upper(dtname) LIKE '%MYS%'
    GROUP BY stname, dtname
""").fetchall()
print("Mysuru records in parquet:", res_mysuru)
