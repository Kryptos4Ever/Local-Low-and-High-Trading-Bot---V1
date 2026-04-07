"""
config_local.py — Configuración del entorno local
═══════════════════════════════════════════════════
Rutas a archivos locales y parámetros compartidos entre estrategias.
Este archivo cambia entre máquinas — no commitear con rutas absolutas
de una máquina específica en un equipo.
"""

# ── Rutas ──────────────────────────────────────────────────────────────────────
DB_PATH      = r"C:\Estrategias de trading automatizado\DB\btc_hourly.db"
DB_TABLE     = "btc_hourly"
RESULTS_JSON = "backtest_results.json"
STATE_PATH   = "state/trading_state.jsonl"

# ── Rango de fechas ────────────────────────────────────────────────────────────
# Formato: 'YYYY-MM-DD'  |  None = desde el inicio / hasta el final del dataset
FECHA_INICIO = '2021-11-10'
FECHA_FIN    = '2025-10-06'

# Referencias útiles:
#   Bottom Bear 2018  : '2018-12-10'
#   Pre COVID         : '2019-06-27'
#   Inicio Bull 2020  : '2020-03-17'
#   TOP1 2021         : '2021-04-14'
#   TOP2 2021         : '2021-11-10'
#   Bottom Bear 2022  : '2022-11-22'
#   Inicio Bull 2023  : '2023-01-01'
#   Inicio Bull 2024  : '2024-01-01'
#   TOP 2025          : '2025-10-06'

# ── Parámetros compartidos de simulación ──────────────────────────────────────
SYMBOL             = "BTCUSDT"
SALDO_USDT_INICIAL = 1000.0
MAX_POSICIONES     = 10
COMMISSION_PCT     = 0.1       # % (Binance Spot maker/taker)

# ── Salida ─────────────────────────────────────────────────────────────────────
DARK_MODE  = True
OUTPUT_PNG = "analisis_estrategia.png"
DPI        = 150
