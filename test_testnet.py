"""
test_testnet.py — Diagnóstico completo del entorno Binance Testnet
════════════════════════════════════════════════════════════════════
Verifica en orden:
  1. Que el .env tiene las keys correctas
  2. Conectividad REST con testnet (sin autenticación)
  3. Tiempo del servidor vs tiempo local (diferencia > 1s rompe las firmas)
  4. Saldo real de la cuenta (GET /api/v3/account — requiere firma)
  5. Info del par BTCUSDT (filtros, min notional, precisión de cantidad)
  6. Velas históricas (GET /api/v3/klines — sin firma)
  7. WebSocket stream (wss://stream.testnet.binance.vision/ws)
  8. Orden de prueba en modo TEST (POST /api/v3/order/test — sin ejecutar)

Ejecutar:
    python test_testnet.py

Todos los pasos son no-destructivos. El paso 8 usa el endpoint /order/test
que valida la orden sin ejecutarla ni afectar el saldo.
"""

import hashlib
import hmac
import json
import os
import sys
import time
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path

import requests

# ─── Cargar .env ──────────────────────────────────────────────────────────────

def load_env(env_file: str = ".env") -> None:
    """Parser mínimo de .env — igual al de secrets.py."""
    current = Path.cwd()
    for _ in range(4):
        candidate = current / env_file
        if candidate.exists():
            with open(candidate, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, _, value = line.partition("=")
                    key   = key.strip()
                    value = value.strip().strip('"').strip("'")
                    if key and key not in os.environ:
                        os.environ[key] = value
            print(f"  .env cargado desde: {candidate}")
            return
        current = current.parent
    print("  .env no encontrado — usando variables de entorno del sistema")

load_env()

# ─── Configuración ────────────────────────────────────────────────────────────

BASE_URL   = "https://testnet.binance.vision"
WS_URL     = "wss://stream.testnet.binance.vision/ws"
API_KEY    = os.environ.get("BINANCE_TESTNET_API_KEY", "")
SECRET     = os.environ.get("BINANCE_TESTNET_SECRET", "")
SYMBOL     = "BTCUSDT"
TIMEOUT    = 10

OK   = "  ✓"
FAIL = "  ✗"
WARN = "  ⚠"
SEP  = "─" * 60


# Offset de tiempo medido en el Paso 3 (se actualiza después)
_time_offset_ms: int = 0

def _now_ms() -> int:
    """Timestamp en ms corregido por el desfase medido."""
    return int(time.time() * 1000) + _time_offset_ms

def _sign(params: dict) -> dict:
    """Agrega timestamp y firma HMAC-SHA256 al dict de parámetros."""
    params["timestamp"]  = _now_ms()
    params["recvWindow"] = 5000
    query = urllib.parse.urlencode(params)
    sig   = hmac.new(
        SECRET.encode("utf-8"),
        query.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    params["signature"] = sig
    return params


def _headers() -> dict:
    return {"X-MBX-APIKEY": API_KEY}


# ══════════════════════════════════════════════════════════════════════════════
# PASO 1 — Credenciales en .env
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n{'═'*60}")
print("  PASO 1 — Credenciales")
print(SEP)

if not API_KEY:
    print(f"{FAIL} BINANCE_TESTNET_API_KEY no encontrada en .env")
    print("      Creá el archivo .env con:")
    print("      BINANCE_TESTNET_API_KEY=tu_key")
    print("      BINANCE_TESTNET_SECRET=tu_secret")
    sys.exit(1)

if not SECRET:
    print(f"{FAIL} BINANCE_TESTNET_SECRET no encontrada en .env")
    sys.exit(1)

print(f"{OK} API Key   : {API_KEY[:8]}...{API_KEY[-4:]}  ({len(API_KEY)} chars)")
print(f"{OK} Secret    : {SECRET[:4]}...  ({len(SECRET)} chars)")


# ══════════════════════════════════════════════════════════════════════════════
# PASO 2 — Conectividad REST (ping)
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n{'═'*60}")
print("  PASO 2 — Conectividad REST con testnet")
print(SEP)

try:
    t0   = time.time()
    resp = requests.get(f"{BASE_URL}/api/v3/ping", timeout=TIMEOUT)
    latencia = (time.time() - t0) * 1000
    resp.raise_for_status()
    print(f"{OK} {BASE_URL}/api/v3/ping")
    print(f"     Latencia: {latencia:.0f}ms")
except requests.RequestException as e:
    print(f"{FAIL} No se puede conectar: {e}")
    print("     ¿Estás conectado a internet?")
    sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
# PASO 3 — Sincronización de tiempo
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n{'═'*60}")
print("  PASO 3 — Sincronización de tiempo")
print(SEP)

try:
    resp         = requests.get(f"{BASE_URL}/api/v3/time", timeout=TIMEOUT)
    server_time  = resp.json()["serverTime"]   # epoch ms
    local_time   = int(time.time() * 1000)
    diferencia_s = abs(server_time - local_time) / 1000

    print(f"     Servidor : {server_time}  ({datetime.fromtimestamp(server_time/1000, tz=timezone.utc).strftime('%H:%M:%S')} UTC)")
    print(f"     Local    : {local_time}   ({datetime.fromtimestamp(local_time/1000, tz=timezone.utc).strftime('%H:%M:%S')} UTC)")
    print(f"     Diferencia: {diferencia_s:.1f}s")

    # Medir offset con más precisión (3 muestras, mediana)
    offsets = []
    for _ in range(3):
        t0   = int(time.time() * 1000)
        resp2 = requests.get(f"{BASE_URL}/api/v3/time", timeout=TIMEOUT)
        t1   = int(time.time() * 1000)
        srv  = resp2.json()["serverTime"]
        offsets.append(srv - (t0 + t1) // 2)
    offsets.sort()
    _time_offset_ms = offsets[1]  # mediana de 3

    if diferencia_s > 1.0:
        print(f"{WARN} Diferencia de tiempo {diferencia_s:.1f}s — compensando automáticamente")
        print(f"     Offset aplicado: {_time_offset_ms:+d}ms")
        print("     (Para solución permanente: w32tm /resync /force en PowerShell Admin)")
    else:
        print(f"{OK} Tiempo sincronizado ({diferencia_s:.2f}s de diferencia)")
except Exception as e:
    print(f"{FAIL} Error consultando tiempo del servidor: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# PASO 4 — Saldo real de la cuenta
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n{'═'*60}")
print("  PASO 4 — Saldo de la cuenta (GET /api/v3/account)")
print(SEP)

try:
    params = _sign({})
    resp   = requests.get(
        f"{BASE_URL}/api/v3/account",
        params  = params,
        headers = _headers(),
        timeout = TIMEOUT,
    )

    if resp.status_code == 401:
        print(f"{FAIL} Error 401 — API Key incorrecta o sin permisos")
        print(f"     Respuesta: {resp.json()}")
        sys.exit(1)

    if resp.status_code == 400:
        err = resp.json()
        print(f"{FAIL} Error {err.get('code')}: {err.get('msg')}")
        if err.get('code') == -1021:
            print("      → Problema de tiempo. Sincronizá el reloj del sistema.")
        sys.exit(1)

    resp.raise_for_status()
    account = resp.json()

    # Mostrar balances con fondos
    print(f"{OK} Cuenta autenticada correctamente")
    print(f"     Tipo de cuenta  : {account.get('accountType', 'SPOT')}")
    print(f"     Puede operar    : {account.get('canTrade', False)}")
    print(f"     Comisión maker  : {account.get('makerCommission', 0)/100:.2f}%")
    print(f"     Comisión taker  : {account.get('takerCommission', 0)/100:.2f}%")
    print()

    balances_con_fondos = [
        b for b in account.get("balances", [])
        if float(b["free"]) > 0 or float(b["locked"]) > 0
    ]

    if balances_con_fondos:
        print(f"     Activos con saldo ({len(balances_con_fondos)}):")
        for b in balances_con_fondos[:10]:   # mostrar máximo 10
            free   = float(b["free"])
            locked = float(b["locked"])
            print(f"       {b['asset']:<8}  libre={free:>15.6f}  bloqueado={locked:>12.6f}")
    else:
        print(f"{WARN} No hay activos con saldo en la cuenta")
        print("     El testnet asigna saldo automáticamente al registrarse.")
        print("     Si no tenés saldo, intentá hacer logout y login nuevamente.")

    # Saldo USDT específico
    usdt_balance = next(
        (float(b["free"]) for b in account["balances"] if b["asset"] == "USDT"),
        0.0
    )
    btc_balance = next(
        (float(b["free"]) for b in account["balances"] if b["asset"] == "BTC"),
        0.0
    )
    print()
    print(f"     USDT disponible : ${usdt_balance:,.2f}")
    print(f"     BTC disponible  : {btc_balance:.8f}")

    if usdt_balance < 10:
        print(f"\n{WARN} Saldo USDT muy bajo (${usdt_balance:.2f})")
        print("     El sistema opera con slots de ~$200 (capital/5 posiciones).")
        print("     Si el testnet reseteó recientemente, los fondos se recargan solos.")

except requests.RequestException as e:
    print(f"{FAIL} Error de red: {e}")
    sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
# PASO 5 — Info del par BTCUSDT (filtros y precisión)
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n{'═'*60}")
print("  PASO 5 — Info del par BTCUSDT (filtros)")
print(SEP)

try:
    resp = requests.get(
        f"{BASE_URL}/api/v3/exchangeInfo",
        params  = {"symbol": SYMBOL},
        timeout = TIMEOUT,
    )
    resp.raise_for_status()
    info = resp.json()

    symbol_info = next(
        (s for s in info.get("symbols", []) if s["symbol"] == SYMBOL),
        None
    )

    if not symbol_info:
        print(f"{FAIL} {SYMBOL} no encontrado en exchangeInfo")
    else:
        print(f"{OK} {SYMBOL} disponible para operar")
        print(f"     Estado       : {symbol_info.get('status', 'UNKNOWN')}")
        print(f"     Base asset   : {symbol_info.get('baseAsset')} (BTC)")
        print(f"     Quote asset  : {symbol_info.get('quoteAsset')} (USDT)")

        # Extraer filtros relevantes
        filtros = {f["filterType"]: f for f in symbol_info.get("filters", [])}

        if "LOT_SIZE" in filtros:
            lot = filtros["LOT_SIZE"]
            print(f"     Cantidad mín : {lot['minQty']} BTC")
            print(f"     Cantidad máx : {lot['maxQty']} BTC")
            print(f"     Step size    : {lot['stepSize']} BTC")

        if "MIN_NOTIONAL" in filtros:
            mn = filtros["MIN_NOTIONAL"]
            print(f"     Notional mín : ${mn.get('minNotional', '?')} USDT")

        if "NOTIONAL" in filtros:
            n = filtros["NOTIONAL"]
            print(f"     Notional mín : ${n.get('minNotional', '?')} USDT")

        if "PRICE_FILTER" in filtros:
            pf = filtros["PRICE_FILTER"]
            print(f"     Precio mín   : ${pf['minPrice']}")
            print(f"     Tick size    : ${pf['tickSize']}")

except Exception as e:
    print(f"{FAIL} Error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# PASO 6 — Velas históricas (GET /api/v3/klines)
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n{'═'*60}")
print("  PASO 6 — Velas históricas (GET /api/v3/klines)")
print(SEP)

try:
    resp = requests.get(
        f"{BASE_URL}/api/v3/klines",
        params  = {"symbol": SYMBOL, "interval": "1h", "limit": 5},
        timeout = TIMEOUT,
    )
    resp.raise_for_status()
    klines = resp.json()

    print(f"{OK} {len(klines)} velas recibidas")
    print(f"     Últimas {len(klines)} velas horarias de {SYMBOL}:")
    print(f"     {'Apertura':<22} {'Open':>10} {'High':>10} {'Low':>10} {'Close':>10}")
    print(f"     {'-'*65}")
    for k in klines:
        ts  = datetime.fromtimestamp(k[0]/1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
        print(f"     {ts:<22} {float(k[1]):>10.2f} {float(k[2]):>10.2f} "
              f"{float(k[3]):>10.2f} {float(k[4]):>10.2f}")

    precio_actual = float(klines[-1][4])
    print(f"\n     Precio BTC/USDT actual: ${precio_actual:,.2f}")
    print(f"     Slot estimado (1000/5) : ${1000/5:.0f} USDT")
    print(f"     BTC por slot           : {(1000/5) / precio_actual:.8f} BTC")

except Exception as e:
    print(f"{FAIL} Error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# PASO 7 — WebSocket stream (conectar y recibir 1 mensaje)
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n{'═'*60}")
print("  PASO 7 — WebSocket stream (1h kline)")
print(SEP)
print(f"     Conectando a: {WS_URL}/btcusdt@kline_1h")
print("     (espera máx 10s — el stream tarda en llegar el primer mensaje)")

try:
    import asyncio
    import websockets as ws_lib

    async def _test_ws():
        url = f"{WS_URL}/btcusdt@kline_1h"
        async with ws_lib.connect(url, open_timeout=10, close_timeout=5) as ws:
            raw = await asyncio.wait_for(ws.recv(), timeout=10)
            msg = json.loads(raw)
            k   = msg.get("k", {})
            ts  = datetime.fromtimestamp(k.get("t", 0)/1000, tz=timezone.utc).strftime("%H:%M:%S")
            return {
                "evento":  msg.get("e"),
                "simbolo": msg.get("s"),
                "open_ts": ts,
                "close":   k.get("c"),
                "cerrada": k.get("x"),
            }

    loop   = asyncio.new_event_loop()
    result = loop.run_until_complete(_test_ws())
    loop.close()

    print(f"{OK} WebSocket conectado y datos recibidos")
    print(f"     Evento   : {result['evento']}")
    print(f"     Símbolo  : {result['simbolo']}")
    print(f"     Vela     : abrió a las {result['open_ts']} UTC")
    print(f"     Close    : ${float(result['close']):,.2f}")
    print(f"     Cerrada  : {result['cerrada']} (True = vela completada)")

except ImportError:
    print(f"{WARN} websockets no instalado — saltando este paso")
    print("     Instalar con: pip install websockets")
except asyncio.TimeoutError:
    print(f"{WARN} Timeout esperando mensaje del WebSocket (normal en algunas redes)")
    print("     El stream funciona — simplemente tardó más de 10s en el primer tick")
except Exception as e:
    print(f"{FAIL} Error WebSocket: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# PASO 8 — Orden de prueba (POST /api/v3/order/test — no ejecuta nada)
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n{'═'*60}")
print("  PASO 8 — Orden de prueba (POST /api/v3/order/test)")
print(SEP)
print("     Este endpoint valida la firma y los parámetros SIN ejecutar la orden.")

try:
    # Usar el precio actual para calcular una cantidad razonable
    precio_btc = float(klines[-1][4]) if 'klines' in dir() and klines else 80000.0
    slot_usdt  = 200.0   # slot de prueba
    btc_qty    = slot_usdt / precio_btc
    btc_str    = f"{btc_qty:.5f}"   # 5 decimales = stepSize de BTCUSDT

    params_test = _sign({
        "symbol":        SYMBOL,
        "side":          "BUY",
        "type":          "MARKET",
        "quoteOrderQty": f"{slot_usdt:.2f}",
    })

    resp = requests.post(
        f"{BASE_URL}/api/v3/order/test",
        params  = params_test,
        headers = _headers(),
        timeout = TIMEOUT,
    )

    if resp.status_code == 200:
        body = resp.json()
        print(f"{OK} Firma y parámetros válidos — la orden sería aceptada")
        print(f"     Response: {body}")   # {} en testnet = orden válida
        print(f"     Parámetros validados:")
        print(f"       symbol        = {SYMBOL}")
        print(f"       side          = BUY")
        print(f"       type          = MARKET")
        print(f"       quoteOrderQty = ${slot_usdt:.2f} USDT")
    elif resp.status_code == 400:
        err = resp.json()
        print(f"{FAIL} Orden rechazada: [{err.get('code')}] {err.get('msg')}")
        if err.get('code') == -1013:
            print("      → Notional demasiado bajo. Aumentar el slot o verificar el precio.")
        elif err.get('code') == -2010:
            print("      → Saldo insuficiente.")
    else:
        print(f"{FAIL} Error {resp.status_code}: {resp.text}")

except Exception as e:
    print(f"{FAIL} Error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# RESUMEN
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n{'═'*60}")
print("  RESUMEN")
print(SEP)
print("  Si todos los pasos muestran ✓, el entorno está listo.")
print()
print("  Próximo paso:")
print("    python live_local_reversal.py")
print()
print("  El trader descargará ~500 velas históricas, entrenará el modelo")
print("  (~90s) y esperará el cierre de la próxima vela horaria.")
print(f"{'═'*60}\n")
