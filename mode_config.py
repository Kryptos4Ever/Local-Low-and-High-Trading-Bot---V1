"""
mode_config.py — Selectores de modo de operación
══════════════════════════════════════════════════
Único archivo que decide qué implementación usa cada actor.
Cambiar estos booleanos es todo lo que se necesita para pasar
de simulación a producción.

Regla: True = mundo real (Binance), False = simulación local.
"""

# ── Actores ────────────────────────────────────────────────────────────────────
USE_LIVE_FEED       = False   # True → BinanceWSFeed   | False → SQLiteFeed
USE_LIVE_WALLET     = False   # True → BinanceWallet   | False → JSONWallet
USE_LIVE_ORDERBOOK  = False   # True → BinanceOrderBook| False → SimulatedOrderBook
USE_LIVE_CLOCK      = False   # True → LiveClock       | False → LocalClock

# ── Wallet local ───────────────────────────────────────────────────────────────
USE_MEMORY_WALLET   = False   # True → MemoryWallet (sin JSON, para grid search)

# ── Risk Manager ──────────────────────────────────────────────────────────────
ENABLE_RISK_MANAGER = False   # True → activa límites de riesgo

# Límites (solo aplican si ENABLE_RISK_MANAGER = True)
RISK_MAX_DRAWDOWN_PCT    = 20.0    # detiene si el portfolio cae X% desde el pico
RISK_MAX_DAILY_LOSS_USDT = 100.0   # detiene si la pérdida del día supera Y USDT
RISK_MAX_ORDER_USDT      = 300.0   # rechaza órdenes individuales > Z USDT
RISK_MIN_ORDER_USDT      = 5.0     # rechaza órdenes < mínimo operativo
RISK_DEDUP_WINDOW_S      = 0       # 0 = deduplicación desactivada

# ── Logger ────────────────────────────────────────────────────────────────────
LOG_TO_FILE = False    # True → escribe logs en logs/YYYY-MM-DD_<modulo>.jsonl
LOG_LEVEL   = "INFO"   # DEBUG | INFO | WARNING | ERROR | CRITICAL
LOG_DIR     = "logs"
