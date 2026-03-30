"""
live_zigzag_reversal.py — Runner Live de ZigZag Reversal Strategy
══════════════════════════════════════════════════════════════════
Ejecuta ZigZagReversalStrategy en producción contra Binance (testnet o real).

Diferencia clave vs live_local_reversal.py:
  · NO requiere caché ni entrenamiento previo.
  · La estrategia arranca en segundos — todos los indicadores se calculan
    en línea sobre el buffer de velas que llega del WebSocket.
  · El warmup de indicadores (55 velas) ocurre automáticamente durante
    las primeras horas de operación; en ese período no se emiten señales.

Flujo completo
───────────────
  1. Cargar credenciales (.env)
  2. Medir desfase de reloj vs Binance
  3. Vender BTC libre pre-existente (estado limpio)
  4. Inicializar actores (Wallet, OrderBook, Feed, Clock)
  5. Inicializar estrategia (sin entrenamiento)
  6. Loop: tick → señal → ejecución → dashboard → checkpoint
  7. Shutdown limpio (Ctrl+C)

Actores utilizados
───────────────────
    PriceFeed  : BinanceWSFeed    (stream en tiempo real)
    Wallet     : BinanceWallet    (sincronizada con cuenta real)
    OrderBook  : BinanceOrderBook (órdenes reales)
    Clock      : LiveClock
    State      : JSONStateManager

Parámetros configurables
─────────────────────────
  MIN_PUNTOS_BUY, MIN_PUNTOS_SELL, STOP_LOSS_PCT
  Los umbrales RSI y MA20 también se pueden ajustar aquí.
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

from actors.binance_feed       import BinanceWSFeed
from actors.binance_order_book import BinanceOrderBook
from actors.binance_wallet     import BinanceWallet
from actors.live_clock         import LiveClock
from actors.order_book         import OrderSide
from actors.price_feed         import Candle
from state.state_manager       import JSONStateManager, Checkpoint
from strategies.base_strategy  import SignalSide
from strategies.zigzag_reversal import ZigZagReversalStrategy
from support.logger            import get_logger
from support.secrets           import secrets
from support.time_utils        import now_epoch_s, to_iso

log = get_logger("live_zigzag")


# ════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ════════════════════════════════════════════════════════════════════

# Indicadores
RSI_PERIOD      = 14
MA_SHORT        = 20
MA_LONG         = 50
WINDOW          = 24
LAST_N          = 5

# Señal BUY
RSI_BUY_STRONG  = 30.0
RSI_BUY_WEAK    = 40.0
MA20_BUY_STRONG = -3.5
MA20_BUY_WEAK   = -1.0

# Señal SELL
RSI_SELL_STRONG = 65.0
RSI_SELL_WEAK   = 55.0
MA20_SELL_STRONG= 2.5
MA20_SELL_WEAK  = 1.0

# Puntaje mínimo
MIN_PUNTOS_BUY  = 5
MIN_PUNTOS_SELL = 5

# Gestión de riesgo
STOP_LOSS_PCT   = 0.04
MAX_POSICIONES  = 5
COMMISSION_PCT  = 0.1
SYMBOL          = "BTCUSDT"

# Archivos
LIVE_RESULTS_JSON = "live_results_zigzag.json"
STATE_PATH        = "state/live_zigzag_state.jsonl"
DASHBOARD_HTML    = "live_dashboard_zigzag.html"
DASHBOARD_REFRESH = 10
MAX_TRADE_LOG     = 50
CHART_CANDLES     = 48


# ════════════════════════════════════════════════════════════════════
# SIGNER (idéntico a live_local_reversal.py)
# ════════════════════════════════════════════════════════════════════

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
    def __init__(self, api_key, secret, base_url, recv_window, timeout):
        self.api_key     = api_key
        self.secret      = secret
        self.base_url    = base_url
        self.recv_window = recv_window
        self.timeout     = timeout
        self._offset_ms  = self._measure_offset()

    def _measure_offset(self):
        offsets = []
        for _ in range(3):
            try:
                t0 = int(time.time() * 1000)
                r  = requests.get(f"{self.base_url}/api/v3/time", timeout=self.timeout)
                t1 = int(time.time() * 1000)
                offsets.append(r.json()["serverTime"] - (t0 + t1) // 2)
            except Exception:
                pass
        offsets.sort()
        return offsets[len(offsets) // 2] if offsets else 0

    def _now_ms(self):
        return int(time.time() * 1000) + self._offset_ms

    def _sign(self, params):
        p = dict(params)
        p["timestamp"]  = self._now_ms()
        p["recvWindow"] = self.recv_window
        qs  = urllib.parse.urlencode(p)
        sig = hmac.new(self.secret.encode(), qs.encode(), hashlib.sha256).hexdigest()
        p["signature"] = sig
        return p

    def _hdrs(self):
        return {"X-MBX-APIKEY": self.api_key}

    def get(self, endpoint, params=None):
        p    = self._sign(params or {})
        resp = requests.get(f"{self.base_url}{endpoint}",
                            params=p, headers=self._hdrs(), timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def post(self, endpoint, params):
        p    = self._sign(params)
        resp = requests.post(f"{self.base_url}{endpoint}",
                             params=p, headers=self._hdrs(), timeout=self.timeout)
        return resp.json()

    def price(self, symbol):
        r = requests.get(f"{self.base_url}/api/v3/ticker/price",
                         params={"symbol": symbol}, timeout=self.timeout)
        return float(r.json()["price"])

    def step_size(self, symbol):
        r = requests.get(f"{self.base_url}/api/v3/exchangeInfo",
                         params={"symbol": symbol}, timeout=self.timeout)
        sym = next(s for s in r.json()["symbols"] if s["symbol"] == symbol)
        return {f["filterType"]: f for f in sym["filters"]}["LOT_SIZE"]["stepSize"]


def _truncate(qty, step):
    dec = len(step.rstrip("0").split(".")[-1])
    return f"{int(qty * 10**dec) / 10**dec:.{dec}f}"


def sell_preexisting_btc(signer, symbol="BTCUSDT"):
    print("\n  [limpieza] Verificando BTC libre...", end=" ", flush=True)
    account  = signer.get("/api/v3/account")
    btc_free = next((float(b["free"]) for b in account["balances"] if b["asset"] == "BTC"), 0.0)
    if btc_free < 0.00001:
        print("ninguno. OK")
        return
    precio    = signer.price(symbol)
    step      = signer.step_size(symbol)
    qty_str   = _truncate(btc_free, step)
    print(f"encontrado {btc_free:.8f} BTC (~${btc_free*precio:,.2f})")
    print(f"  [limpieza] Vendiendo {qty_str}...", end=" ", flush=True)
    result = signer.post("/api/v3/order", {
        "symbol": symbol, "side": "SELL", "type": "MARKET", "quantity": qty_str,
    })
    if result.get("status") in ("FILLED", "PARTIALLY_FILLED"):
        usdt_rec = float(result.get("cummulativeQuoteQty", 0))
        print(f"OK → ${usdt_rec:,.2f} USDT recibidos")
    else:
        print(f"no ejecutado — {result.get('msg','?')}")


# ════════════════════════════════════════════════════════════════════
# DASHBOARD CONSOLA
# ════════════════════════════════════════════════════════════════════

G = "\033[92m"; R = "\033[91m"; C = "\033[96m"
D = "\033[2m";  B = "\033[1m"; X = "\033[0m"
CLR = "\033[2J\033[H"


def _countdown(next_ts):
    rem = max(0, next_ts - int(time.time()))
    h, r = divmod(rem, 3600); m, s = divmod(r, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _next_hour():
    return (int(time.time()) // 3600 + 1) * 3600


def render_console(st: dict) -> None:
    now_s = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    pnl_p = (st["portfolio"] / st["usdt_ini"] - 1) * 100 if st["usdt_ini"] else 0
    pnl_u = st["portfolio"] - st["usdt_ini"]

    print(CLR, end="")
    print(f"{B}{'═'*68}{X}")
    print(f"{B}  ZIGZAG REVERSAL — LIVE{X}  {D}{now_s}{X}")
    print(f"{B}{'═'*68}{X}")

    pc = G if pnl_p >= 0 else R
    print(f"\n  {B}Portfolio{X}")
    print(f"    Total   : {pc}${st['portfolio']:>12,.2f}{X}  ({pc}{pnl_p:+.2f}%  ${pnl_u:+,.2f}{X})")
    print(f"    USDT lib: ${st['usdt_lib']:>12,.2f}    Slot: ${st['slot']:,.2f}")

    print(f"\n  {B}Posiciones ({st['n_pos']}/{MAX_POSICIONES}){X}")
    if not st["pos"]:
        print(f"    {D}Sin posiciones{X}")
    else:
        px = st["precio"]
        print(f"    {'#':<3}{'Entrada':>10}{'Actual':>10}{'BTC':>12}{'P&L':>9}")
        print(f"    {'─'*44}")
        for i, p in enumerate(st["pos"], 1):
            pe  = p["entry_price"]
            pct = (px / pe - 1) * 100 if pe else 0
            col = G if pct >= 0 else R
            print(f"    {i:<3}${pe:>9,.0f}${px:>9,.0f}{p['btc']:>12.6f} {col}{pct:+.2f}%{X}")

    print(f"\n  {B}Señal — última vela{X}  close=${st['precio']:,.0f}")
    puntos_b = st.get("puntos_buy", 0)
    puntos_s = st.get("puntos_sell", 0)
    sig      = st.get("sig", "HOLD")
    rsi      = st.get("rsi", None)
    rsi_s    = f"RSI={rsi:.1f}" if rsi is not None else "RSI=N/A"
    print(f"    Señal: {B}{sig}{X}  pts_buy={puntos_b}/{MIN_PUNTOS_BUY}  pts_sell={puntos_s}/{MIN_PUNTOS_SELL}  {rsi_s}")

    print(f"\n  {B}Próxima vela en{X}  {C}{_countdown(st['next_ts'])}{X}")

    print(f"\n  {B}Operaciones{X}  (buy={st['nb']}  sell={st['ns']}  stop={st['nsl']}  ign={st['ni']})")
    trades = st["tlog"][-5:][::-1]
    if not trades:
        print(f"    {D}Sin operaciones aún{X}")
    else:
        for t in trades:
            col = G if t["tipo"] == "BUY" else R
            print(f"    {t['hora']:>6} {col}{t['tipo']:>4}{X} ${t['precio']:>10,.0f}  {t['qty']:>14}  {t['info']:>12}")

    print(f"\n{D}  SL={STOP_LOSS_PCT*100:.0f}%  Ctrl+C para detener{X}")
    print(f"{B}{'═'*68}{X}")


def render_html(st: dict) -> None:
    """Dashboard HTML simplificado compatible con el formato del live_local_reversal."""
    now_s = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    pnl_p = (st["portfolio"] / st["usdt_ini"] - 1) * 100 if st["usdt_ini"] else 0
    pnl_u = st["portfolio"] - st["usdt_ini"]
    pnl_col = "#22c55e" if pnl_p >= 0 else "#ef4444"
    px = st["precio"]
    rem = max(0, st["next_ts"] - int(time.time()))
    h, r = divmod(rem, 3600); m, s = divmod(r, 60)
    cd = f"{h:02d}:{m:02d}:{s:02d}"

    pos_rows = ""
    for i, p in enumerate(st["pos"], 1):
        pe  = p["entry_price"]
        pct = (px / pe - 1) * 100 if pe else 0
        pc  = "#22c55e" if pct >= 0 else "#ef4444"
        pos_rows += (f"<tr><td>{i}</td><td>${pe:,.0f}</td>"
                     f"<td>${px:,.0f}</td><td>{p['btc']:.6f}</td>"
                     f"<td style='color:{pc}'>{pct:+.2f}%</td></tr>")
    pos_html = (f"<table><thead><tr><th>#</th><th>Entrada</th>"
                f"<th>Actual</th><th>BTC</th><th>P&L</th></tr></thead>"
                f"<tbody>{pos_rows or '<tr><td colspan=5>Sin posiciones</td></tr>'}</tbody></table>")

    t_rows = ""
    for t in st["tlog"][-8:][::-1]:
        tc = "#22c55e" if t["tipo"] == "BUY" else "#ef4444"
        t_rows += (f"<tr><td>{t['hora']}</td>"
                   f"<td style='color:{tc}'>{t['tipo']}</td>"
                   f"<td>${t['precio']:,.0f}</td>"
                   f"<td>{t['qty']}</td><td>{t['info']}</td></tr>")
    if not t_rows:
        t_rows = "<tr><td colspan='5'>Sin operaciones</td></tr>"

    html = f"""<!DOCTYPE html><html lang='es'><head>
<meta charset='UTF-8'><meta http-equiv='refresh' content='{DASHBOARD_REFRESH}'>
<title>ZigZag Live</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:'Segoe UI',sans-serif;background:#0f172a;color:#e2e8f0;padding:20px;font-size:14px}}
h1{{font-size:18px;font-weight:600;color:#f8fafc}}
h2{{font-size:11px;font-weight:600;color:#94a3b8;text-transform:uppercase;margin:20px 0 8px;border-left:3px solid #94a3b8;padding-left:8px}}
.hdr{{display:flex;justify-content:space-between;margin-bottom:20px;padding-bottom:14px;border-bottom:1px solid #1e293b}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px;margin-bottom:4px}}
.card{{background:#1e293b;border-radius:10px;padding:14px 18px;border:1px solid #334155}}
.card .lbl{{font-size:10px;color:#64748b;text-transform:uppercase}}
.card .val{{font-size:20px;font-weight:700;color:#f8fafc;margin-top:2px}}
.card .sub{{font-size:11px;color:#94a3b8}}
.pnl{{color:{pnl_col}}}
.sec{{background:#1e293b;border-radius:10px;padding:14px 18px;border:1px solid #334155;margin-bottom:10px}}
table{{width:100%;border-collapse:collapse;font-size:12px}}
th{{background:#252836;color:#94a3b8;padding:6px;text-align:left}}
td{{padding:6px;border-bottom:1px solid #0f172a;color:#cbd5e1}}
.cd{{font-size:26px;font-weight:700;color:#38bdf8}}
.ts{{font-size:11px;color:#64748b;text-align:right}}
</style></head><body>
<div class='hdr'>
  <div><h1>ZigZag Reversal — Live</h1>
  <div class='ts' style='text-align:left;margin-top:4px'>
    pts_BUY≥{MIN_PUNTOS_BUY} · pts_SELL≥{MIN_PUNTOS_SELL} · SL={STOP_LOSS_PCT*100:.0f}% · {SYMBOL}
  </div></div>
  <div class='ts'>Actualizado: {now_s}<br>Auto-refresh {DASHBOARD_REFRESH}s</div>
</div>
<div class='grid'>
  <div class='card'><div class='lbl'>Portfolio total</div>
    <div class='val'>${st['portfolio']:,.2f}</div>
    <div class='sub'>Inicial: ${st['usdt_ini']:,.2f}</div></div>
  <div class='card'><div class='lbl'>P&amp;L sesión</div>
    <div class='val pnl'>{pnl_p:+.2f}%</div>
    <div class='sub pnl'>{pnl_u:+,.2f} USDT</div></div>
  <div class='card'><div class='lbl'>USDT libre</div>
    <div class='val'>${st['usdt_lib']:,.2f}</div>
    <div class='sub'>Slot: ${st['slot']:,.2f}</div></div>
  <div class='card'><div class='lbl'>Posiciones</div>
    <div class='val'>{st['n_pos']} / {MAX_POSICIONES}</div>
    <div class='sub'>BTC @ ${px:,.0f}</div></div>
  <div class='card'><div class='lbl'>Señal última vela</div>
    <div class='val'>{st.get('sig','HOLD')}</div>
    <div class='sub'>buy={st.get('puntos_buy',0)} sell={st.get('puntos_sell',0)}</div></div>
  <div class='card'><div class='lbl'>Próxima vela</div>
    <div class='cd'>{cd}</div></div>
</div>
<h2>Posiciones abiertas</h2>
<div class='sec'>{pos_html}</div>
<h2>Últimas operaciones (buy={st['nb']} sell={st['ns']} stop={st['nsl']} ign={st['ni']})</h2>
<div class='sec'><table><thead><tr>
  <th>Hora</th><th>Tipo</th><th>Precio</th><th>Cantidad</th><th>Info</th>
</tr></thead><tbody>{t_rows}</tbody></table></div>
</body></html>"""

    try:
        Path(DASHBOARD_HTML).write_text(html, encoding="utf-8")
    except Exception as e:
        log.warning("HTML no pudo escribirse", error=str(e))


# ════════════════════════════════════════════════════════════════════
# LIVE TRADER
# ════════════════════════════════════════════════════════════════════

class LiveTrader:

    def __init__(self):
        self._clock:     Optional[LiveClock]              = None
        self._wallet:    Optional[BinanceWallet]          = None
        self._ob:        Optional[BinanceOrderBook]       = None
        self._strategy:  Optional[ZigZagReversalStrategy] = None
        self._state_mgr: Optional[JSONStateManager]       = None
        self._signer:    Optional[_Signer]                = None
        self._running    = False
        self._nb = self._ns = self._ni = self._nsl = 0
        self._t_start    = 0.0
        self._last_candle: Optional[Candle] = None
        self._tlog: list = []
        self._usdt_ini   = 0.0
        self._chart_hist: list = []
        self._last_puntos_buy  = 0
        self._last_puntos_sell = 0
        self._last_rsi: Optional[float] = None
        self._last_sig = "HOLD"

    def run(self):
        self._t_start = time.time()
        self._setup_signals()
        print("╔══════════════════════════════════════════════════════════╗")
        print("║       LIVE TRADER — ZIGZAG REVERSAL                     ║")
        print("╚══════════════════════════════════════════════════════════╝")
        print(f"  pts_BUY≥{MIN_PUNTOS_BUY}  pts_SELL≥{MIN_PUNTOS_SELL}  SL={STOP_LOSS_PCT*100:.0f}%")
        print(f"  HTML dashboard: abrir '{DASHBOARD_HTML}' en el browser")
        print("─" * 60)
        try:
            self._setup()
            self._loop()
        except KeyboardInterrupt:
            pass
        finally:
            self._shutdown()

    def _setup(self):
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
            max_posiciones = MAX_POSICIONES,
            json_path      = LIVE_RESULTS_JSON,
            state_path     = STATE_PATH,
        )
        self._usdt_ini = self._wallet.get_usdt_balance()
        self._ob = BinanceOrderBook(
            max_posiciones = MAX_POSICIONES,
            commission_pct = COMMISSION_PCT,
        )
        feed         = BinanceWSFeed()
        self._clock  = LiveClock(feed=feed, symbol=SYMBOL)

        # La estrategia no necesita feed ni entrenamiento
        self._strategy = ZigZagReversalStrategy(
            rsi_period       = RSI_PERIOD,
            ma_short         = MA_SHORT,
            ma_long          = MA_LONG,
            window           = WINDOW,
            last_n           = LAST_N,
            rsi_buy_strong   = RSI_BUY_STRONG,
            rsi_buy_weak     = RSI_BUY_WEAK,
            ma20_buy_strong  = MA20_BUY_STRONG,
            ma20_buy_weak    = MA20_BUY_WEAK,
            rsi_sell_strong  = RSI_SELL_STRONG,
            rsi_sell_weak    = RSI_SELL_WEAK,
            ma20_sell_strong = MA20_SELL_STRONG,
            ma20_sell_weak   = MA20_SELL_WEAK,
            min_puntos_buy   = MIN_PUNTOS_BUY,
            min_puntos_sell  = MIN_PUNTOS_SELL,
            stop_loss_pct    = STOP_LOSS_PCT,
        )
        self._strategy.on_start(self._wallet)

        print(f"\n  Wallet: ${self._wallet.get_usdt_balance():,.2f} USDT  "
              f"| {self._wallet.positions_count} pos.")
        print(f"  Esperando primera vela horaria...\n")
        self._refresh()

    def _loop(self):
        self._running = True
        for candle in self._clock:
            if not self._running:
                break
            self._last_candle = candle
            signal = self._strategy._tick(candle, self._wallet)

            # Guardar estado de indicadores para el dashboard
            # (internos a la estrategia; los extraemos del reason si está disponible)
            self._last_sig = signal.side.value
            # Intentar parsear puntos del reason
            if "pts=" in (signal.reason or ""):
                try:
                    pts_str = signal.reason.split("pts=")[1].split("/")[0]
                    pts_val = int(pts_str)
                    if signal.side == SignalSide.BUY:
                        self._last_puntos_buy  = pts_val
                        self._last_puntos_sell = 0
                    elif signal.side == SignalSide.SELL:
                        self._last_puntos_sell = pts_val
                        self._last_puntos_buy  = 0
                    else:
                        self._last_puntos_buy = self._last_puntos_sell = 0
                except Exception:
                    pass

            self._chart_hist.append({
                "ts":  candle.ts * 1000,
                "o": candle.open, "h": candle.high,
                "l": candle.low,  "c": candle.close,
                "sig": signal.side.value,
            })
            if len(self._chart_hist) > CHART_CANDLES:
                self._chart_hist.pop(0)

            if signal.is_actionable:
                order = self._ob.execute_with_guards(
                    side      = signal.to_order_side(),
                    price     = signal.price or candle.close,
                    wallet    = self._wallet,
                    candle_ts = candle.ts,
                )
                if order.is_filled:
                    side_s = order.side.value
                    t      = order.trade
                    qty    = (f"{t.btc_bought:.6f} BTC" if t and t.btc_bought
                              else f"{t.btc_sold:.6f} BTC" if t and t.btc_sold else "")
                    info   = (f"${t.usdt_received:,.2f}" if t and t.usdt_received
                              else f"${t.usdt_spent:,.2f}" if t and t.usdt_spent else "")
                    self._tlog.append({
                        "hora":   datetime.fromtimestamp(candle.ts, tz=timezone.utc).strftime("%H:%M"),
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
                        if signal.score == 0.0:
                            self._nsl += 1
                    if self._wallet.get_trade_log():
                        lt = self._wallet.get_trade_log()[-1]
                        lt["score_bot"] = signal.score if side_s == "BUY"  else 0.0
                        lt["score_top"] = signal.score if side_s == "SELL" else 0.0
                elif order.is_ignored:
                    self._ni += 1

            self._state_mgr.save(Checkpoint.from_wallet(
                self._wallet, candle.ts, candle.close,
                metadata={"pts_buy": MIN_PUNTOS_BUY, "pts_sell": MIN_PUNTOS_SELL},
            ))
            self._refresh()

    def _refresh(self):
        px = self._last_candle.close if self._last_candle else 0.0
        st = {
            "portfolio": round(self._wallet.portfolio_value(px), 2),
            "usdt_lib":  round(self._wallet.get_usdt_balance(), 2),
            "slot":      round(self._wallet.get_slot_usdt(), 2),
            "n_pos":     self._wallet.positions_count,
            "pos":       [{"entry_price": p.entry_price, "btc": p.btc}
                          for p in self._wallet.get_positions()],
            "precio":    px,
            "sig":       self._last_sig,
            "puntos_buy":  self._last_puntos_buy,
            "puntos_sell": self._last_puntos_sell,
            "rsi":       self._last_rsi,
            "next_ts":   _next_hour(),
            "nb": self._nb, "ns": self._ns,
            "ni": self._ni, "nsl": self._nsl,
            "tlog":      self._tlog,
            "usdt_ini":  self._usdt_ini,
        }
        render_console(st)
        render_html(st)

    def _shutdown(self):
        self._running = False
        if self._clock:
            self._clock.stop()
        if self._strategy and self._wallet:
            self._strategy.on_stop(self._wallet)
        if self._wallet and self._wallet.get_trade_log():
            px = self._last_candle.close if self._last_candle else 0
            self._wallet.flush({
                "estrategia":            "ZigZagReversal",
                "modo":                  "LIVE",
                "fecha_inicio":          to_iso(int(self._t_start)),
                "fecha_fin":             to_iso(now_epoch_s()),
                "symbol":                SYMBOL,
                "min_puntos_buy":        MIN_PUNTOS_BUY,
                "min_puntos_sell":       MIN_PUNTOS_SELL,
                "stop_loss_pct":         STOP_LOSS_PCT,
                "total_compras":         self._nb,
                "total_ventas":          self._ns,
                "total_stoploss":        self._nsl,
                "total_ignorados":       self._ni,
                "saldo_inicial_usdt":    self._usdt_ini,
                "usdt_balance_final":    round(self._wallet.get_usdt_balance(), 8),
                "portfolio_value_final": round(self._wallet.portfolio_value(px), 4),
            })
        if self._wallet:
            self._refresh()
        log.info("live trader detenido",
                 compras=self._nb, ventas=self._ns, stops=self._nsl,
                 sesion_h=f"{(time.time()-self._t_start)/3600:.1f}h")

    def _setup_signals(self):
        def _h(sig, frame):
            print("\n\nCtrl+C — cerrando limpiamente...")
            self._running = False
            if self._clock:
                self._clock.stop()
        signal.signal(signal.SIGINT,  _h)
        signal.signal(signal.SIGTERM, _h)


# ════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    LiveTrader().run()
