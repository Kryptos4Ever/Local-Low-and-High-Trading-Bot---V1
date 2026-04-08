import sqlite3
conn = sqlite3.connect(r"C:\Estrategias de trading automatizado\DB\btc_hourly.db")
cur = conn.cursor()
cur.execute("""
    SELECT 
        COUNT(*) as total,
        SUM(CASE WHEN taker_buy_base_volume IS NULL THEN 1 ELSE 0 END) as null_taker,
        SUM(CASE WHEN quote_volume IS NULL THEN 1 ELSE 0 END) as null_quote,
        MIN(datetime), MAX(datetime)
    FROM btc_hourly
""")
print(cur.fetchone())