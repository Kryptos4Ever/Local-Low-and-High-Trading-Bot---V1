"""
live_local_reversal.py — Runner live con dashboard en tiempo real
═════════════════════════════════════════════════════════════════
Ejecuta LocalReversalStrategy contra Binance Testnet con:

  · Limpieza automática de BTC al arrancar (vende BTC libre pre-existente)
  · Dashboard en consola que se refresca en cada tick
  · Dashboard HTML en live_dashboard.html (abrir en browser, refresca solo)

Flujo completo
───────────────
  1. Cargar credenciales (.env)
  2. Medir desfase de reloj vs Binance (compensación automática)
  3. Vender BTC libre pre-existente
  4. Inicializar actores (Wallet, OrderBook, Feed, Clock)
  5. Warmup: ~500 velas históricas + entrenar modelo (~90s)
  6. Loop: tick → señal → ejecución → dashboards → checkpoint
  7. Shutdown limpio (Ctrl+C)

Parámetros configurables
─────────────────────────
  THR_B, THR_T, MAX_POSICIONES, WARMUP_CANDLES
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import signal
import sys
import time
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests

sys.path.insert(0, str(Path(__file__).parent))

from actors.binance_feed       import BinanceRESTFeed, BinanceWSFeed
from actors.binance_order_book import BinanceOrderBook
from actors.binance_wallet     import BinanceWallet
from actors.live_clock         import LiveClock
from actors.order_book         import OrderSide
from actors.price_feed         import Candle
from state.state_manager       import JSONStateManager, Checkpoint
from strategies.base_strategy  import SignalSide
from strategies.local_reversal import LocalReversalStrategy
from support.logger            import get_logger
from support.secrets           import secrets
from support.time_utils        import now_epoch_s, to_iso

log = get_logger("live_trader")


# ════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ════════════════════════════════════════════════════════════════════════════

THR_B             = 0.50
THR_T             = 0.50
MAX_POSICIONES    = 5
COMMISSION_PCT    = 0.1
SYMBOL            = "BTCUSDT"
WARMUP_CANDLES    = 500
LIVE_RESULTS_JSON = "live_results.json"
STATE_PATH        = "state/live_trading_state.jsonl"
CACHE_DIR         = ".cache_local_reversal"
DASHBOARD_HTML    = "live_dashboard.html"
DASHBOARD_REFRESH = 10     # segundos entre refresh del HTML
MAX_TRADE_LOG     = 50
CHART_CANDLES     = 48     # velas a mostrar en el gráfico (~2 días)


# ════════════════════════════════════════════════════════════════════════════
# SIGNER — HMAC + compensación de reloj
# ════════════════════════════════════════════════════════════════════════════

def _get_config() -> dict:
    try:
        import config_world as CW
        return {
            "base_url":    CW.BINANCE_TESTNET_URL if CW.USE_TESTNET else CW.BINANCE_BASE_URL,
            "timeout":     CW.REQUEST_TIMEOUT_S,
            "recv_window": CW.RECV_WINDOW_MS,
        }
    except ImportError:
        return {"base_url": "https://testnet.binance.vision",
                "timeout": 10, "recv_window": 5000}


class _Signer:
    """Firma requests HMAC-SHA256 compensando el desfase de reloj local."""

    def __init__(self, api_key: str, secret: str,
                 base_url: str, recv_window: int, timeout: int) -> None:
        self.api_key      = api_key
        self.secret       = secret
        self.base_url     = base_url
        self.recv_window  = recv_window
        self.timeout      = timeout
        self._offset_ms   = self._measure_offset()
        if abs(self._offset_ms) > 500:
            log.warning("desfase de reloj detectado",
                        offset_s=f"{self._offset_ms/1000:.1f}s",
                        accion="compensando automáticamente")

    def _measure_offset(self) -> int:
        offsets = []
        for _ in range(3):
            try:
                t0  = int(time.time() * 1000)
                r   = requests.get(f"{self.base_url}/api/v3/time",
                                   timeout=self.timeout)
                t1  = int(time.time() * 1000)
                offsets.append(r.json()["serverTime"] - (t0 + t1) // 2)
            except Exception:
                pass
        if not offsets:
            return 0
        offsets.sort()
        return offsets[len(offsets) // 2]

    def _now_ms(self) -> int:
        return int(time.time() * 1000) + self._offset_ms

    def _sign(self, params: dict) -> dict:
        p = dict(params)
        p["timestamp"]  = self._now_ms()
        p["recvWindow"] = self.recv_window
        qs  = urllib.parse.urlencode(p)
        sig = hmac.new(self.secret.encode(), qs.encode(), hashlib.sha256).hexdigest()
        p["signature"] = sig
        return p

    def _hdrs(self) -> dict:
        return {"X-MBX-APIKEY": self.api_key}

    def get(self, endpoint: str, params: dict = None) -> dict:
        p    = self._sign(params or {})
        resp = requests.get(f"{self.base_url}{endpoint}",
                            params=p, headers=self._hdrs(),
                            timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def post(self, endpoint: str, params: dict) -> dict:
        p    = self._sign(params)
        resp = requests.post(f"{self.base_url}{endpoint}",
                             params=p, headers=self._hdrs(),
                             timeout=self.timeout)
        return resp.json()

    def price(self, symbol: str) -> float:
        r = requests.get(f"{self.base_url}/api/v3/ticker/price",
                         params={"symbol": symbol}, timeout=self.timeout)
        return float(r.json()["price"])

    def step_size(self, symbol: str) -> str:
        r = requests.get(f"{self.base_url}/api/v3/exchangeInfo",
                         params={"symbol": symbol}, timeout=self.timeout)
        sym = next(s for s in r.json()["symbols"] if s["symbol"] == symbol)
        return {f["filterType"]: f for f in sym["filters"]}["LOT_SIZE"]["stepSize"]


# ════════════════════════════════════════════════════════════════════════════
# LIMPIEZA DE BTC PRE-EXISTENTE
# ════════════════════════════════════════════════════════════════════════════

def _truncate(qty: float, step: str) -> str:
    dec = len(step.rstrip("0").split(".")[-1])
    return f"{int(qty * 10**dec) / 10**dec:.{dec}f}"


def sell_preexisting_btc(signer: _Signer, symbol: str = "BTCUSDT") -> None:
    """
    Vende todo el BTC libre de la cuenta a mercado antes de arrancar.
    Garantiza que el trader parte de un estado limpio (solo USDT).
    Se ejecuta en cada arranque — idempotente si no hay BTC libre.
    """
    print("\n  [limpieza] Verificando BTC libre...", end=" ", flush=True)

    account  = signer.get("/api/v3/account")
    btc_free = next(
        (float(b["free"]) for b in account["balances"] if b["asset"] == "BTC"),
        0.0,
    )

    if btc_free < 0.00001:
        print("ninguno. OK")
        return

    precio    = signer.price(symbol)
    valor_est = btc_free * precio
    step      = signer.step_size(symbol)
    qty_str   = _truncate(btc_free, step)

    print(f"encontrado {btc_free:.8f} BTC (~${valor_est:,.2f} USDT)")
    print(f"  [limpieza] Vendiendo {qty_str} BTC...", end=" ", flush=True)

    result = signer.post("/api/v3/order", {
        "symbol": symbol, "side": "SELL",
        "type": "MARKET", "quantity": qty_str,
    })

    if result.get("status") in ("FILLED", "PARTIALLY_FILLED"):
        usdt_rec = float(result.get("cummulativeQuoteQty", 0))
        avg_p    = usdt_rec / max(float(result.get("executedQty", btc_free)), 1e-9)
        print(f"OK → recibidos ${usdt_rec:,.2f} USDT @ ${avg_p:,.2f}")
    elif "code" in result:
        print(f"no ejecutado — {result.get('msg')} (continuando de todas formas)")
    else:
        print(f"respuesta inesperada: {result}")


# ════════════════════════════════════════════════════════════════════════════
# DASHBOARD — CONSOLA
# ════════════════════════════════════════════════════════════════════════════

G = "\033[92m"; R = "\033[91m"; Y = "\033[93m"
C = "\033[96m"; D = "\033[2m";  B = "\033[1m"; X = "\033[0m"
CLR = "\033[2J\033[H"


def _pnl_color(v: float, sign: bool = True) -> str:
    col = G if v >= 0 else R
    s   = ("+" if v >= 0 else "") if sign else ""
    return f"{col}{s}{v:.2f}%{X}"


def _countdown(next_ts: int) -> str:
    rem = max(0, next_ts - int(time.time()))
    h, r = divmod(rem, 3600); m, s = divmod(r, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _next_hour() -> int:
    return (int(time.time()) // 3600 + 1) * 3600


def render_console(st: dict) -> None:
    now_s = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    pnl_p = (st["portfolio"] / st["usdt_ini"] - 1) * 100 if st["usdt_ini"] else 0
    pnl_u = st["portfolio"] - st["usdt_ini"]

    print(CLR, end="")
    print(f"{B}{'═'*68}{X}")
    print(f"{B}  LOCAL REVERSAL — LIVE TESTNET{X}  "
          f"{D}{now_s}{X}")
    print(f"{B}{'═'*68}{X}")

    # Portfolio
    print(f"\n  {B}Portfolio{X}")
    pc = G if pnl_p >= 0 else R
    print(f"    Total   : {pc}${st['portfolio']:>12,.2f}{X}  "
          f"({_pnl_color(pnl_p)}  {pc}{'+' if pnl_u>=0 else ''}${pnl_u:,.2f}{X})")
    print(f"    USDT lib: ${st['usdt_lib']:>12,.2f}    "
          f"Slot: ${st['slot']:,.2f}")

    # Posiciones
    px = st["precio"]
    print(f"\n  {B}Posiciones abiertas  ({st['n_pos']}/{MAX_POSICIONES}){X}")
    if not st["pos"]:
        print(f"    {D}Sin posiciones{X}")
    else:
        print(f"    {'#':<3}{'Entrada':>10}{'Actual':>10}{'BTC':>12}{'P&L':>9}")
        print(f"    {'─'*44}")
        for i, p in enumerate(st["pos"], 1):
            pe  = p["entry_price"]
            pct = (px / pe - 1) * 100 if pe else 0
            col = G if pct >= 0 else R
            print(f"    {i:<3}${pe:>9,.2f}${px:>9,.2f}"
                  f"{p['btc']:>12.6f} "
                  f"{col}{'+' if pct>=0 else ''}{pct:.2f}%{X}")

    # Señal del modelo
    pb, pt = st["pb"], st["pt"]
    ts_s   = (datetime.fromtimestamp(st["last_ts"], tz=timezone.utc)
              .strftime("%H:%M UTC") if st["last_ts"] else "—")
    print(f"\n  {B}Señal del modelo{X}  vela {ts_s}  close=${px:,.2f}")
    cb = G if pb >= THR_B else (Y if pb >= THR_B*0.8 else D)
    ct = R if pt >= THR_T else (Y if pt >= THR_T*0.8 else D)
    bb = "█"*int(pb*20) + "░"*(20-int(pb*20))
    bt = "█"*int(pt*20) + "░"*(20-int(pt*20))
    print(f"    prob_bottom: {cb}{bb} {pb:.3f}{X}  (thr={THR_B})")
    print(f"    prob_top   : {ct}{bt} {pt:.3f}{X}  (thr={THR_T})")
    print(f"    Señal: {B}{st['sig']}{X}")

    # Countdown
    print(f"\n  {B}Próxima vela en{X}  {C}{_countdown(st['next_ts'])}{X}")

    # Últimas 5 operaciones
    print(f"\n  {B}Operaciones{X}  "
          f"(buy={st['nb']}  sell={st['ns']}  ign={st['ni']})")
    trades = st["tlog"][-5:][::-1]
    if not trades:
        print(f"    {D}Sin operaciones aún{X}")
    else:
        print(f"    {'Hora':>6}{'Tipo':>6}{'Precio':>11}{'Cantidad':>13}{'Info':>10}")
        print(f"    {'─'*46}")
        for t in trades:
            col = G if t["tipo"]=="BUY" else (R if t["tipo"]=="SELL" else D)
            print(f"    {t['hora']:>6}{col}{t['tipo']:>6}{X}"
                  f"${t['precio']:>10,.2f}{t['qty']:>13}{t['info']:>10}")

    print(f"\n{D}  THR_B={THR_B}  THR_T={THR_T}  Ctrl+C para detener{X}")
    print(f"{B}{'═'*68}{X}")


# ════════════════════════════════════════════════════════════════════════════
# DASHBOARD — HTML
# ════════════════════════════════════════════════════════════════════════════

def render_html(st: dict) -> None:
    import json as _json
    now_s   = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    pnl_p   = (st["portfolio"] / st["usdt_ini"] - 1) * 100 if st["usdt_ini"] else 0
    pnl_u   = st["portfolio"] - st["usdt_ini"]
    pnl_col = "#22c55e" if pnl_p >= 0 else "#ef4444"
    px      = st["precio"]
    pb, pt  = st["pb"], st["pt"]
    pb_w    = f"{pb*100:.1f}%"
    pt_w    = f"{pt*100:.1f}%"
    pb_col  = "#22c55e" if pb>=THR_B else ("#eab308" if pb>=THR_B*0.8 else "#6b7280")
    pt_col  = "#ef4444" if pt>=THR_T else ("#eab308" if pt>=THR_T*0.8 else "#6b7280")
    sig_cls = "buy" if st["sig"]=="BUY" else ("sell" if st["sig"]=="SELL" else "hold")
    rem = max(0, st["next_ts"] - int(time.time()))
    h, r = divmod(rem, 3600); m, s = divmod(r, 60)
    cd  = f"{h:02d}:{m:02d}:{s:02d}"
    ts_s = (datetime.fromtimestamp(st["last_ts"], tz=timezone.utc)
            .strftime("%H:%M UTC") if st["last_ts"] else "—")
    pos_html = ""
    if st["pos"]:
        rows = ""
        for i, p in enumerate(st["pos"], 1):
            pe  = p["entry_price"]
            pct = (px / pe - 1) * 100 if pe else 0
            pc  = "#22c55e" if pct >= 0 else "#ef4444"
            rows += (f"<tr><td>{i}</td><td>${pe:,.2f}</td>"
                     f"<td>${px:,.2f}</td><td>{p['btc']:.6f}</td>"
                     f"<td style='color:{pc};font-weight:600'>"
                     f"{'+' if pct>=0 else ''}{pct:.2f}%</td></tr>")
        pos_html = (f"<table><thead><tr><th>#</th><th>Entrada</th>"
                    f"<th>Actual</th><th>BTC</th><th>P&L</th></tr></thead>"
                    f"<tbody>{rows}</tbody></table>")
    else:
        pos_html = "<p class='muted'>Sin posiciones abiertas</p>"
    t_rows = ""
    for t in st["tlog"][-5:][::-1]:
        tc = ("#22c55e" if t["tipo"]=="BUY" else
              "#ef4444" if t["tipo"]=="SELL" else "#6b7280")
        t_rows += (f"<tr><td>{t['hora']}</td>"
                   f"<td style='color:{tc};font-weight:600'>{t['tipo']}</td>"
                   f"<td>${t['precio']:,.2f}</td>"
                   f"<td>{t['qty']}</td><td>{t['info']}</td></tr>")
    if not t_rows:
        t_rows = "<tr><td colspan='5' class='muted'>Sin operaciones aún</td></tr>"
    chart     = st.get("chart", [])
    n_chart   = len(chart)
    ohlc_data = _json.dumps([{"x": c["ts"], "o": c["o"], "h": c["h"],
                               "l": c["l"], "c": c["c"]} for c in chart])
    pb_data   = _json.dumps([{"x": c["ts"], "y": c["pb"]} for c in chart])
    pt_data   = _json.dumps([{"x": c["ts"], "y": c["pt"]} for c in chart])
    buy_m     = _json.dumps([{"x": c["ts"], "y": c["l"] * 0.9994}
                              for c in chart if c["sig"] == "BUY"])
    sell_m    = _json.dumps([{"x": c["ts"], "y": c["h"] * 1.0006}
                              for c in chart if c["sig"] == "SELL"])
    thr_b_l   = _json.dumps([{"x": c["ts"], "y": THR_B} for c in chart])
    thr_t_l   = _json.dumps([{"x": c["ts"], "y": THR_T} for c in chart])
    no_data   = ("" if chart else
                 "<p class='muted' style='text-align:center;padding:40px'>"
                 "Esperando velas del stream...</p>")
    html = (
        "<!DOCTYPE html><html lang='es'><head>\n"
        "<meta charset='UTF-8'>\n"
        f"<meta http-equiv='refresh' content='{DASHBOARD_REFRESH}'>\n"
        "<title>Live Dashboard</title>\n"
        "<script src='https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js'></script>\n"
        "<script src='https://cdn.jsdelivr.net/npm/chartjs-chart-financial@0.1.1/dist/chartjs-chart-financial.min.js'></script>\n"
        "<script src='https://cdn.jsdelivr.net/npm/luxon@3/build/global/luxon.min.js'></script>\n"
        "<script src='https://cdn.jsdelivr.net/npm/chartjs-adapter-luxon@1/dist/chartjs-adapter-luxon.umd.min.js'></script>\n"
        "<style>\n"
        "*{box-sizing:border-box;margin:0;padding:0}\n"
        "body{font-family:'Segoe UI',system-ui,sans-serif;background:#0f172a;"
        "color:#e2e8f0;padding:20px;font-size:14px;line-height:1.6}\n"
        "h1{font-size:18px;font-weight:600;color:#f8fafc}\n"
        "h2{font-size:11px;font-weight:600;color:#94a3b8;text-transform:uppercase;"
        "letter-spacing:.06em;margin:20px 0 8px}\n"
        ".hdr{display:flex;justify-content:space-between;align-items:flex-start;"
        "margin-bottom:20px;padding-bottom:14px;border-bottom:1px solid #1e293b}\n"
        ".ts{font-size:11px;color:#64748b;text-align:right}\n"
        ".grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));"
        "gap:10px;margin-bottom:4px}\n"
        ".card{background:#1e293b;border-radius:10px;padding:14px 18px;"
        "border:1px solid #334155}\n"
        ".card .lbl{font-size:10px;color:#64748b;text-transform:uppercase;letter-spacing:.06em}\n"
        ".card .val{font-size:20px;font-weight:700;color:#f8fafc;margin-top:2px}\n"
        ".card .sub{font-size:11px;color:#94a3b8;margin-top:1px}\n"
        f".pnl{{color:{pnl_col}}}\n"
        ".sec{background:#1e293b;border-radius:10px;padding:14px 18px;"
        "border:1px solid #334155;margin-bottom:10px}\n"
        "table{width:100%;border-collapse:collapse}\n"
        "th{font-size:10px;color:#64748b;text-transform:uppercase;"
        "letter-spacing:.06em;padding:5px 8px;text-align:left;"
        "border-bottom:1px solid #0f172a}\n"
        "td{padding:7px 8px;border-bottom:1px solid #0f172a;color:#cbd5e1}\n"
        "tr:last-child td{border-bottom:none}\n"
        "tr:hover td{background:#243044}\n"
        ".bar-row{display:flex;align-items:center;gap:8px;margin:5px 0}\n"
        ".bar-lbl{width:88px;font-size:11px;color:#94a3b8}\n"
        ".bar-trk{flex:1;background:#0f172a;border-radius:3px;height:7px;overflow:hidden}\n"
        ".bar-fil{height:100%;border-radius:3px;transition:width .4s}\n"
        ".bar-val{width:40px;text-align:right;font-size:11px;font-weight:600}\n"
        ".badge{display:inline-block;padding:2px 10px;border-radius:20px;"
        "font-size:11px;font-weight:700}\n"
        ".buy{background:#14532d;color:#86efac}\n"
        ".sell{background:#450a0a;color:#fca5a5}\n"
        ".hold{background:#1e293b;color:#94a3b8}\n"
        ".cd{font-size:26px;font-weight:700;color:#38bdf8;font-variant-numeric:tabular-nums}\n"
        ".muted{color:#475569;font-style:italic;font-size:12px}\n"
        ".cw{position:relative;height:280px;margin-bottom:0}\n"
        ".cws{position:relative;height:130px}\n"
        ".legend{display:flex;gap:16px;margin-top:8px;font-size:11px;color:#64748b;"
        "flex-wrap:wrap;padding:0 4px}\n"
        ".leg-dot{width:9px;height:9px;border-radius:50%;display:inline-block}\n"
        ".leg-line{width:18px;height:2px;display:inline-block}\n"
        ".leg-dash{width:18px;height:0;display:inline-block;"
        "border-top:2px dashed currentColor}\n"
        "</style></head><body>\n"
        "<div class='hdr'>\n"
        "  <div><h1>Local Reversal — Live Testnet</h1>\n"
        f"  <div class='ts' style='text-align:left;margin-top:4px'>"
        f"THR_B={THR_B} · THR_T={THR_T} · Max pos={MAX_POSICIONES} · {SYMBOL}</div></div>\n"
        f"  <div class='ts'>Actualizado: {now_s}<br>"
        f"  <span style='color:#475569'>Auto-refresca cada {DASHBOARD_REFRESH}s</span></div>\n"
        "</div>\n"
        "<div class='grid'>\n"
        f"  <div class='card'><div class='lbl'>Portfolio total</div>"
        f"<div class='val'>${st['portfolio']:,.2f}</div>"
        f"<div class='sub'>Inicial: ${st['usdt_ini']:,.2f}</div></div>\n"
        f"  <div class='card'><div class='lbl'>P&amp;L sesión</div>"
        f"<div class='val pnl'>{'+' if pnl_p>=0 else ''}{pnl_p:.2f}%</div>"
        f"<div class='sub pnl'>{'+' if pnl_u>=0 else ''}${pnl_u:,.2f} USDT</div></div>\n"
        f"  <div class='card'><div class='lbl'>USDT libre</div>"
        f"<div class='val'>${st['usdt_lib']:,.2f}</div>"
        f"<div class='sub'>Slot: ${st['slot']:,.2f}</div></div>\n"
        f"  <div class='card'><div class='lbl'>Posiciones</div>"
        f"<div class='val'>{st['n_pos']} / {MAX_POSICIONES}</div>"
        f"<div class='sub'>BTC @ ${px:,.2f}</div></div>\n"
        f"  <div class='card'><div class='lbl'>Operaciones</div>"
        f"<div class='val'>{st['nb']+st['ns']}</div>"
        f"<div class='sub'>Buy {st['nb']} · Sell {st['ns']} · Ign {st['ni']}</div></div>\n"
        f"  <div class='card'><div class='lbl'>Próxima vela en</div>"
        f"<div class='cd'>{cd}</div>"
        f"<div class='sub'>Último tick: {ts_s}</div></div>\n"
        "</div>\n"
        f"<h2>Gráfico BTC/USDT — últimas {n_chart} velas horarias</h2>\n"
        "<div class='sec' style='padding:12px 14px'>\n"
        f"  {no_data}\n"
        "  <div class='cw'><canvas id='cc'></canvas></div>\n"
        "  <div style='height:8px'></div>\n"
        "  <div class='cws'><canvas id='pc'></canvas></div>\n"
        "  <div class='legend'>\n"
        "    <span style='display:flex;align-items:center;gap:5px'>"
        "<span class='leg-dot' style='background:#22c55e'></span>BUY señal</span>\n"
        "    <span style='display:flex;align-items:center;gap:5px'>"
        "<span class='leg-dot' style='background:#ef4444'></span>SELL señal</span>\n"
        "    <span style='display:flex;align-items:center;gap:5px'>"
        "<span class='leg-line' style='background:#22c55e;height:2px'></span>prob_bottom</span>\n"
        "    <span style='display:flex;align-items:center;gap:5px'>"
        "<span class='leg-line' style='background:#ef4444;height:2px'></span>prob_top</span>\n"
        "    <span style='display:flex;align-items:center;gap:5px'>"
        "<span class='leg-dash' style='color:#94a3b8'></span>umbrales</span>\n"
        "  </div>\n"
        "</div>\n"
        "<h2>Señal del modelo — último tick</h2>\n"
        "<div class='sec'>\n"
        "  <div style='display:flex;align-items:center;gap:10px;margin-bottom:10px'>\n"
        f"    <span>Señal emitida:</span><span class='badge {sig_cls}'>{st['sig']}</span>\n"
        f"    <span style='color:#64748b;font-size:11px'>Close: ${px:,.2f}</span></div>\n"
        "  <div class='bar-row'><span class='bar-lbl'>prob_bottom</span>\n"
        f"    <div class='bar-trk'><div class='bar-fil' style='width:{pb_w};background:{pb_col}'></div></div>\n"
        f"    <span class='bar-val' style='color:{pb_col}'>{pb:.3f}</span></div>\n"
        f"  <div style='font-size:10px;color:#475569;margin:0 0 6px 96px'>umbral BUY = {THR_B}</div>\n"
        "  <div class='bar-row'><span class='bar-lbl'>prob_top</span>\n"
        f"    <div class='bar-trk'><div class='bar-fil' style='width:{pt_w};background:{pt_col}'></div></div>\n"
        f"    <span class='bar-val' style='color:{pt_col}'>{pt:.3f}</span></div>\n"
        f"  <div style='font-size:10px;color:#475569;margin-top:2px;margin-left:96px'>umbral SELL = {THR_T}</div>\n"
        "</div>\n"
        f"<h2>Posiciones abiertas ({st['n_pos']})</h2>\n"
        f"<div class='sec'>{pos_html}</div>\n"
        f"<h2>Últimas operaciones (buy={st['nb']} sell={st['ns']} ign={st['ni']})</h2>\n"
        "<div class='sec'><table><thead><tr>"
        "<th>Hora</th><th>Tipo</th><th>Precio</th><th>Cantidad</th><th>Info</th>"
        f"</tr></thead><tbody>{t_rows}</tbody></table></div>\n"
        "<script>\n"
        f"const OHLC={ohlc_data};\n"
        f"const PBD={pb_data};\n"
        f"const PTD={pt_data};\n"
        f"const BUY_M={buy_m};\n"
        f"const SELL_M={sell_m};\n"
        f"const TBL={thr_b_l};\n"
        f"const TTL={thr_t_l};\n"
        "const GR={grid:'rgba(255,255,255,0.06)',tick:'#64748b',border:'#1e293b'};\n"
        "const XS={type:'time',time:{unit:'hour',displayFormats:{hour:'dd HH:mm'}},"
        "grid:{color:GR.grid},ticks:{color:GR.tick,maxTicksLimit:8,maxRotation:0},"
        "border:{color:GR.border}};\n"
        "const OPT={responsive:true,maintainAspectRatio:false,animation:false,"
        "plugins:{legend:{display:false},tooltip:{mode:'index',intersect:false}}};\n"
        "const cc=document.getElementById('cc');\n"
        "if(cc&&OHLC.length>0){\n"
        "  new Chart(cc,{type:'candlestick',data:{datasets:[\n"
        "    {label:'BTCUSDT',data:OHLC,"
        "color:{up:'#22c55e',down:'#ef4444',unchanged:'#94a3b8'},"
        "borderColor:{up:'#22c55e',down:'#ef4444',unchanged:'#94a3b8'}},\n"
        "    {type:'scatter',data:BUY_M,pointStyle:'triangle',rotation:0,"
        "pointRadius:9,pointHoverRadius:11,backgroundColor:'#22c55e',"
        "borderColor:'#16a34a',borderWidth:1.5},\n"
        "    {type:'scatter',data:SELL_M,pointStyle:'triangle',rotation:180,"
        "pointRadius:9,pointHoverRadius:11,backgroundColor:'#ef4444',"
        "borderColor:'#dc2626',borderWidth:1.5},\n"
        "  ]},options:{...OPT,scales:{x:XS,y:{position:'right',"
        "grid:{color:GR.grid},"
        "ticks:{color:GR.tick,callback:v=>'$'+v.toLocaleString('en',{maximumFractionDigits:0})},"
        "border:{color:GR.border}}}}});\n"
        "}\n"
        "const pc=document.getElementById('pc');\n"
        "if(pc&&PBD.length>0){\n"
        "  new Chart(pc,{type:'line',data:{datasets:[\n"
        "    {label:'pb',data:PBD,borderColor:'#22c55e',fill:false,"
        "borderWidth:1.5,pointRadius:2,tension:0.3},\n"
        "    {label:'pt',data:PTD,borderColor:'#ef4444',fill:false,"
        "borderWidth:1.5,pointRadius:2,tension:0.3},\n"
        "    {label:'thr_b',data:TBL,borderColor:'rgba(34,197,94,.4)',"
        "borderWidth:1,borderDash:[4,4],pointRadius:0,fill:false,tension:0},\n"
        "    {label:'thr_t',data:TTL,borderColor:'rgba(239,68,68,.4)',"
        "borderWidth:1,borderDash:[4,4],pointRadius:0,fill:false,tension:0},\n"
        "  ]},options:{...OPT,scales:{x:XS,y:{position:'right',min:0,max:1,"
        "grid:{color:GR.grid},"
        "ticks:{color:GR.tick,stepSize:.25,callback:v=>v.toFixed(2)},"
        "border:{color:GR.border}}}}});\n"
        "}\n"
        "</script></body></html>\n"
    )
    try:
        Path(DASHBOARD_HTML).write_text(html, encoding="utf-8")
    except Exception as e:
        log.warning("no se pudo escribir HTML", error=str(e))


def _build_state(wallet, nb, ns, ni, candle, pb, pt, sig, tlog, usdt_ini, chart_hist=None) -> dict:
    px = candle.close if candle else 0.0
    return {
        "portfolio": round(wallet.portfolio_value(px), 2),
        "usdt_lib":  round(wallet.get_usdt_balance(), 2),
        "slot":      round(wallet.get_slot_usdt(), 2),
        "n_pos":     wallet.positions_count,
        "pos":       [{"entry_price": p.entry_price, "btc": p.btc}
                      for p in wallet.get_positions()],
        "precio":    px,
        "pb": pb, "pt": pt, "sig": sig,
        "last_ts":   candle.ts if candle else 0,
        "next_ts":   _next_hour(),
        "nb": nb, "ns": ns, "ni": ni,
        "tlog":      tlog,
        "usdt_ini":  usdt_ini,
        "chart":     chart_hist or [],
    }


def _extract_probs(signal) -> tuple[float, float]:
    """Extrae prob_bottom y prob_top del reason o score de la señal."""
    pb = pt = 0.0
    if signal.score is not None:
        if signal.side == SignalSide.BUY:
            pb = signal.score
        elif signal.side == SignalSide.SELL:
            pt = signal.score
    reason = signal.reason or ""
    if "prob_bottom=" in reason:
        try:
            pb = float(reason.split("prob_bottom=")[1].split(">=")[0])
        except Exception:
            pass
    if "prob_top=" in reason:
        try:
            pt = float(reason.split("prob_top=")[1].split(">=")[0])
        except Exception:
            pass
    return pb, pt


# ════════════════════════════════════════════════════════════════════════════
# LIVE TRADER
# ════════════════════════════════════════════════════════════════════════════

class LiveTrader:

    def __init__(self) -> None:
        self._clock:     Optional[LiveClock]             = None
        self._wallet:    Optional[BinanceWallet]         = None
        self._ob:        Optional[BinanceOrderBook]      = None
        self._strategy:  Optional[LocalReversalStrategy] = None
        self._state_mgr: Optional[JSONStateManager]      = None
        self._signer:    Optional[_Signer]               = None
        self._running    = False
        self._nb = self._ns = self._ni = 0
        self._t_start    = 0.0
        self._last_candle: Optional[Candle] = None
        self._pb = self._pt = 0.0
        self._sig        = "—"
        self._tlog:      list = []
        self._usdt_ini   = 0.0
        self._chart_hist: list = []  # {ts, o, h, l, c, pb, pt, signal}

    def run(self) -> None:
        self._t_start = time.time()
        self._setup_signals()
        print("╔══════════════════════════════════════════════════════════╗")
        print("║       LIVE TRADER — LOCAL REVERSAL — TESTNET            ║")
        print("╚══════════════════════════════════════════════════════════╝")
        print(f"  THR_B={THR_B}  THR_T={THR_T}  Max pos={MAX_POSICIONES}")
        print(f"  HTML dashboard: abrir '{DASHBOARD_HTML}' en el browser")
        print("─" * 60)
        try:
            self._setup()
            self._warmup()
            self._loop()
        except KeyboardInterrupt:
            pass
        finally:
            self._shutdown()

    # ── Setup ─────────────────────────────────────────────────────────────

    def _setup(self) -> None:
        cfg = _get_config()
        self._signer = _Signer(
            api_key     = secrets.get("BINANCE_TESTNET_API_KEY"),
            secret      = secrets.get("BINANCE_TESTNET_SECRET"),
            base_url    = cfg["base_url"],
            recv_window = cfg["recv_window"],
            timeout     = cfg["timeout"],
        )
        sell_preexisting_btc(self._signer, SYMBOL)
        self._state_mgr = JSONStateManager(STATE_PATH)
        self._wallet = BinanceWallet.from_account(
            max_posiciones=MAX_POSICIONES,
            json_path=LIVE_RESULTS_JSON,
            state_path=STATE_PATH,
        )
        self._usdt_ini = self._wallet.get_usdt_balance()
        self._ob = BinanceOrderBook(
            max_posiciones=MAX_POSICIONES,
            commission_pct=COMMISSION_PCT,
        )
        feed = BinanceWSFeed()
        self._clock = LiveClock(feed=feed, symbol=SYMBOL)
        self._strategy = LocalReversalStrategy(
            thr_b=THR_B, thr_t=THR_T,
            cache_dir=CACHE_DIR, force_recompute=False,
        )
        print(f"\n  Wallet: ${self._wallet.get_usdt_balance():,.2f} USDT libre  "
              f"| {self._wallet.positions_count} pos. abiertas")

    def _warmup(self) -> None:
        print(f"\n  Entrenando modelo ({WARMUP_CANDLES} velas históricas, ~90s)...")
        rest = BinanceRESTFeed()
        self._strategy.on_start(
            wallet=self._wallet, feed=rest,
            start=now_epoch_s() - WARMUP_CANDLES * 3600,
            end="now", symbol=SYMBOL,
        )
        print("  Modelo listo. Esperando el cierre de la próxima vela...\n")
        self._refresh()

    # ── Loop ──────────────────────────────────────────────────────────────

    def _loop(self) -> None:
        self._running = True
        for candle in self._clock:
            if not self._running:
                break
            self._last_candle = candle
            signal = self._strategy._tick(candle, self._wallet)
            self._pb, self._pt = _extract_probs(signal)
            self._sig = signal.side.value

            # Acumular vela en el historial del gráfico
            self._chart_hist.append({
                "ts":  candle.ts * 1000,   # ms para Chart.js
                "o":   candle.open,
                "h":   candle.high,
                "l":   candle.low,
                "c":   candle.close,
                "pb":  round(self._pb, 4),
                "pt":  round(self._pt, 4),
                "sig": signal.side.value,
            })
            if len(self._chart_hist) > CHART_CANDLES:
                self._chart_hist.pop(0)

            if signal.is_actionable:
                order = self._ob.execute_with_guards(
                    side=signal.to_order_side(),
                    price=candle.close,
                    wallet=self._wallet,
                    candle_ts=candle.ts,
                )
                if order.is_filled:
                    side_s = order.side.value
                    t = order.trade
                    qty  = (f"{t.btc_bought:.6f} BTC" if t and t.btc_bought
                            else f"{t.btc_sold:.6f} BTC" if t and t.btc_sold
                            else "")
                    info = (f"${t.usdt_received:,.2f}" if t and t.usdt_received
                            else f"${t.usdt_spent:,.2f}" if t and t.usdt_spent
                            else "")
                    self._tlog.append({
                        "hora":   datetime.fromtimestamp(
                                      candle.ts, tz=timezone.utc
                                  ).strftime("%H:%M"),
                        "tipo":   side_s,
                        "precio": candle.close,
                        "qty":    qty,
                        "info":   info,
                    })
                    if len(self._tlog) > MAX_TRADE_LOG:
                        self._tlog.pop(0)
                    if side_s == "BUY":
                        self._nb += 1
                    else:
                        self._ns += 1
                    # scores en el JSON
                    log_e = self._wallet.get_trade_log()
                    if log_e:
                        log_e[-1]["score_bot"] = self._pb if side_s=="BUY"  else 0.0
                        log_e[-1]["score_top"] = self._pt if side_s=="SELL" else 0.0
                elif order.is_ignored:
                    self._ni += 1

            self._state_mgr.save(Checkpoint.from_wallet(
                self._wallet, candle.ts, candle.close,
                metadata={"thr_b": THR_B, "thr_t": THR_T},
            ))
            self._refresh()

    # ── Dashboard ─────────────────────────────────────────────────────────

    def _refresh(self) -> None:
        st = _build_state(
            self._wallet, self._nb, self._ns, self._ni,
            self._last_candle, self._pb, self._pt, self._sig,
            self._tlog, self._usdt_ini,
            chart_hist=self._chart_hist,
        )
        render_console(st)
        render_html(st)

    # ── Shutdown ──────────────────────────────────────────────────────────

    def _shutdown(self) -> None:
        self._running = False
        if self._clock:
            self._clock.stop()
        if self._strategy and self._wallet:
            self._strategy.on_stop(self._wallet)
        if self._wallet and self._wallet.get_trade_log():
            px = self._last_candle.close if self._last_candle else 0
            self._wallet.flush({
                "estrategia": "LocalReversal-GBM", "modo": "LIVE_TESTNET",
                "fecha_inicio": to_iso(int(self._t_start)),
                "fecha_fin": to_iso(now_epoch_s()),
                "symbol": SYMBOL, "thr_b": THR_B, "thr_t": THR_T,
                "total_compras": self._nb, "total_ventas": self._ns,
                "total_ignorados": self._ni,
                "saldo_inicial_usdt": self._usdt_ini,
                "usdt_balance_final": round(self._wallet.get_usdt_balance(), 8),
                "portfolio_value_final": round(
                    self._wallet.portfolio_value(px), 4),
            })
        if self._wallet:
            self._refresh()
        log.info("live trader detenido",
                 compras=self._nb, ventas=self._ns,
                 sesion_h=f"{(time.time()-self._t_start)/3600:.1f}h")

    def _setup_signals(self) -> None:
        def _h(sig, frame):
            print("\n\nCtrl+C — cerrando limpiamente...")
            self._running = False
            if self._clock:
                self._clock.stop()
        signal.signal(signal.SIGINT,  _h)
        signal.signal(signal.SIGTERM, _h)


# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    LiveTrader().run()
