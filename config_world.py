"""
config_world.py — Configuración de conexiones externas
════════════════════════════════════════════════════════
Endpoints, timeouts y parámetros de red para producción.
Este archivo NO contiene credenciales (van en .env).
Puede commitearse sin riesgo.
"""

# ── Binance REST API ───────────────────────────────────────────────────────────
BINANCE_BASE_URL      = "https://api.binance.com"
BINANCE_TESTNET_URL   = "https://testnet.binance.vision"   # para pruebas sin dinero real
USE_TESTNET           = True    # True = testnet, False = producción real

# ── Binance WebSocket ──────────────────────────────────────────────────────────
BINANCE_WS_URL        = "wss://stream.binance.com:9443/ws"
BINANCE_WS_TESTNET = "wss://stream.testnet.binance.vision/ws"

# ── Timeouts y reintentos ──────────────────────────────────────────────────────
REQUEST_TIMEOUT_S     = 10      # timeout por request REST (segundos)
WS_RECONNECT_DELAY_S  = 5       # espera antes de reconectar WebSocket
MAX_RETRIES           = 3       # reintentos ante error de red

# ── Símbolo e intervalo ────────────────────────────────────────────────────────
SYMBOL                = "BTCUSDT"
KLINE_INTERVAL        = "1h"    # intervalo de velas: 1m, 5m, 15m, 1h, 4h, 1d

# ── Orden ─────────────────────────────────────────────────────────────────────
ORDER_TYPE            = "MARKET"   # MARKET | LIMIT
RECV_WINDOW_MS        = 5000       # ventana de validez de la firma (ms)
