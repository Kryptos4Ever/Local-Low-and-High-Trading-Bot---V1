"""
live_dual_reversal.py — Runner Live de Dual Reversal Strategy
═════════════════════════════════════════════════════════════
Ejecuta DualReversalStrategy en tiempo real sobre Binance (testnet o real).

Ventaja vs otras estrategias:
  · Sin entrenamiento ni caché — arranca en segundos.
  · Los indicadores se calculan en línea sobre cada vela nueva del WebSocket.
  · Durante el warmup (≈60 velas = primeras 60h) no emite señales.

Configurar los parámetros óptimos obtenidos del grid search antes de correr.
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

from actors.binance_feed        import BinanceWSFeed
from actors.binance_order_book  import BinanceOrderBook
from actors.binance_wallet      import BinanceWallet
from actors.live_clock          import LiveClock
from actors.order_book          import OrderSide
from actors.price_feed          import Candle
from state.state_manager        import JSONStateManager, Checkpoint
from strategies.base_strategy   import SignalSide
from strategies.dual_reversal   import DualReversalStrategy
from support.logger             import get_logger
from support.secrets            import secrets
from support.time_utils         import now_epoch_s, to_iso

log = get_logger("live_dual_reversal")


# ════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN — pegar aquí la combinación ganadora del grid search
# ════════════════════════════════════════════════════════════════════

# Indicadores
RSI_PERIOD      = 14
MA_SHORT        = 20
MA_LONG         = 50
WINDOW          = 24
LAST_N          = 5

# Contexto
CTX_RSI_BUY     = 40.0    # ctx_rsi_sell = 100 - CTX_RSI_BUY = 60.0
CTX_MA20_BUY    = -2.0    # ctx_ma20_sell = 2.0
CTX_MIN_PTS     = 3

# Disparador
TRIG_RSI_SLOPE  = 0.03
TRIG_WICK       = 0.28
TRIG_MIN_PTS    = 2

# Sistema
MAX_POSICIONES  = 5
COMMISSION_PCT  = 0.1
SYMBOL          = "BTCUSDT"

LIVE_RESULTS_JSON = "live_results_dual_reversal.json"
STATE_PATH        = "state/live_dual_reversal_state.jsonl"
DASHBOARD_HTML    = "live_dashboard_dual_reversal.html"
DASHBOARD_REFRESH = 10
MAX_TRADE_LOG     = 50
CHART_CANDLES     = 48


# ════════════════════════════════════════════════════════════════════
# SIGNER
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
        self.api_key = api_key; self.secret = secret
        self.base_url = base_url; self.recv_window = recv_window
        self.timeout = timeout
        self._offset_ms = self._measure_offset()

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

    def _now_ms(self): return int(time.time() * 1000) + self._offset_ms

    def _sign(self, params):
        p = dict(params)
        p["timestamp"] = self._now_ms(); p["recvWindow"] = self.recv_window
        qs = urllib.parse.urlencode(p)
        p["signature"] = hmac.new(self.secret.encode(), qs.encode(),
                                   hashlib.sha256).hexdigest()
        return p

    def _hdrs(self): return {"X-MBX-APIKEY": self.api_key}

    def get(self, ep, params=None):
        r = requests.get(f"{self.base_url}{ep}", params=self._sign(params or {}),
                         headers=self._hdrs(), timeout=self.timeout)
        r.raise_for_status(); return r.json()

    def post(self, ep, params):
        r = requests.post(f"{self.base_url}{ep}", params=self._sign(params),
                          headers=self._hdrs(), timeout=self.timeout)
        return r.json()

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
    print("\n  [limpieza] BTC libre...", end=" ", flush=True)
    btc = next((float(b["free"]) for b in signer.get("/api/v3/account")["balances"]
                if b["asset"] == "BTC"), 0.0)
    if btc < 0.00001:
        print("ninguno. OK"); return
    qty = _truncate(btc, signer.step_size(symbol))
    print(f"{btc:.8f} BTC  vendiendo {qty}...", end=" ", flush=True)
    r = signer.post("/api/v3/order",
                    {"symbol": symbol, "side": "SELL", "type": "MARKET", "quantity": qty})
    usdt = float(r.get("cummulativeQuoteQty", 0))
    print(f"OK → ${usdt:,.2f} USDT" if r.get("status") in ("FILLED","PARTIALLY_FILLED")
          else f"no ejecutado — {r.get('msg','?')}")


# ════════════════════════════════════════════════════════════════════
# DASHBOARD CONSOLA
# ════════════════════════════════════════════════════════════════════

G = "\033[92m"; R = "\033[91m"; C = "\033[96m"
D = "\033[2m";  B = "\033[1m"; X = "\033[0m"
CLR = "\033[2J\033[H"


def _next_hour(): return (int(time.time()) // 3600 + 1) * 3600


def _cd(next_ts):
    s = max(0, next_ts - int(time.time()))
    return f"{s//3600:02d}:{s%3600//60:02d}:{s%60:02d}"


def render_console(st: dict) -> None:
    pnl_p = (st["portfolio"] / st["usdt_ini"] - 1) * 100 if st["usdt_ini"] else 0
    pnl_u = st["portfolio"] - st["usdt_ini"]
    now_s = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(CLR, end="")
    print(f"{B}{'═'*70}{X}")
    print(f"{B}  DUAL REVERSAL — LIVE{X}  {D}{now_s}{X}")
    print(f"{B}{'═'*70}{X}")
    pc = G if pnl_p >= 0 else R
    print(f"\n  {B}Portfolio{X}")
    print(f"    Total   : {pc}${st['portfolio']:>12,.2f}{X}  ({pc}{pnl_p:+.2f}%  ${pnl_u:+,.2f}{X})")
    print(f"    USDT    : ${st['usdt_lib']:>12,.2f}    Slot: ${st['slot']:,.2f}")
    px = st["precio"]
    print(f"\n  {B}Posiciones ({st['n_pos']}/{MAX_POSICIONES}){X}")
    if st["pos"]:
        print(f"    {'#':<3}{'Entrada':>10}{'Actual':>10}{'BTC':>12}{'P&L':>9}")
        print(f"    {'─'*44}")
        for i, p in enumerate(st["pos"], 1):
            pe  = p["entry_price"]
            pct = (px/pe - 1)*100 if pe else 0
            col = G if pct >= 0 else R
            print(f"    {i:<3}${pe:>9,.0f}${px:>9,.0f}{p['btc']:>12.6f} {col}{pct:+.2f}%{X}")
    else:
        print(f"    {D}Sin posiciones{X}")

    sig = st.get("sig", "HOLD")
    ctx = st.get("ctx_pts", 0)
    trg = st.get("trig_pts", 0)
    print(f"\n  {B}Última vela{X}  close=${px:,.0f}  señal={B}{sig}{X}")
    print(f"    Contexto : {ctx}/{CTX_MIN_PTS} pts    Disparador: {trg}/{TRIG_MIN_PTS} pts")
    print(f"\n  {B}Próxima vela{X}  {C}{_cd(st['next_ts'])}{X}")
    print(f"\n  {B}Ops{X}  buy={st['nb']} sell={st['ns']} ign={st['ni']}")
    for t in st["tlog"][-5:][::-1]:
        col = G if t["tipo"]=="BUY" else R
        print(f"    {t['hora']:>6} {col}{t['tipo']:>4}{X} ${t['precio']:>10,.0f}  {t['qty']:>14}  {t['info']}")
    print(f"\n{D}  RSI({RSI_PERIOD}) MA{MA_SHORT}/{MA_LONG} W={WINDOW} N={LAST_N}  Ctrl+C para detener{X}")
    print(f"{B}{'═'*70}{X}")


def render_html(st: dict) -> None:
    pnl_p = (st["portfolio"] / st["usdt_ini"] - 1) * 100 if st["usdt_ini"] else 0
    pnl_u = st["portfolio"] - st["usdt_ini"]
    pnl_col = "#22c55e" if pnl_p >= 0 else "#ef4444"
    px = st["precio"]
    pos_rows = "".join(
        f"<tr><td>{i}</td><td>${p['entry_price']:,.0f}</td>"
        f"<td>${px:,.0f}</td><td>{p['btc']:.6f}</td>"
        f"<td style='color:{'#22c55e' if (px/p['entry_price']-1)>=0 else '#ef4444'}'>"
        f"{(px/p['entry_price']-1)*100:+.2f}%</td></tr>"
        for i, p in enumerate(st["pos"], 1)
    ) or "<tr><td colspan='5'>Sin posiciones</td></tr>"

    t_rows = "".join(
        f"<tr><td>{t['hora']}</td>"
        f"<td style='color:{'#22c55e' if t['tipo']=='BUY' else '#ef4444'}'>{t['tipo']}</td>"
        f"<td>${t['precio']:,.0f}</td><td>{t['qty']}</td><td>{t['info']}</td></tr>"
        for t in st["tlog"][-8:][::-1]
    ) or "<tr><td colspan='5'>Sin operaciones</td></tr>"

    s = max(0, st["next_ts"] - int(time.time()))
    cd = f"{s//3600:02d}:{s%3600//60:02d}:{s%60:02d}"
    now_s = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    html = f"""<!DOCTYPE html><html lang='es'><head>
<meta charset='UTF-8'><meta http-equiv='refresh' content='{DASHBOARD_REFRESH}'>
<title>Dual Reversal Live</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:'Segoe UI',sans-serif;background:#0f172a;color:#e2e8f0;padding:20px;font-size:14px}}
h1{{font-size:18px;font-weight:600;color:#f8fafc}}
h2{{font-size:11px;font-weight:600;color:#94a3b8;text-transform:uppercase;margin:18px 0 8px;border-left:3px solid #94a3b8;padding-left:8px}}
.hdr{{display:flex;justify-content:space-between;margin-bottom:18px;padding-bottom:12px;border-bottom:1px solid #1e293b}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:10px;margin-bottom:4px}}
.card{{background:#1e293b;border-radius:10px;padding:14px 18px;border:1px solid #334155}}
.card .lbl{{font-size:10px;color:#64748b;text-transform:uppercase}}
.card .val{{font-size:18px;font-weight:700;color:#f8fafc;margin-top:2px}}
.card .sub{{font-size:11px;color:#94a3b8;margin-top:2px}}
.pnl{{color:{pnl_col}}}
.sec{{background:#1e293b;border-radius:10px;padding:14px 18px;border:1px solid #334155;margin-bottom:10px}}
table{{width:100%;border-collapse:collapse;font-size:12px}}
th{{background:#252836;color:#94a3b8;padding:6px;text-align:left}}
td{{padding:6px;border-bottom:1px solid #0f172a;color:#cbd5e1}}
.cd{{font-size:24px;font-weight:700;color:#38bdf8}}
.ts{{font-size:11px;color:#64748b;text-align:right}}
.badge{{display:inline-block;padding:2px 10px;border-radius:12px;font-size:12px;font-weight:700}}
.buy{{background:#14532d;color:#86efac}}.sell{{background:#450a0a;color:#fca5a5}}
.hold{{background:#1e293b;color:#94a3b8}}
</style></head><body>
<div class='hdr'>
  <div><h1>Dual Reversal — Live</h1>
  <div class='ts' style='text-align:left;margin-top:4px'>
    RSI({RSI_PERIOD}) MA{MA_SHORT}/{MA_LONG} W={WINDOW} N={LAST_N} |
    ctx≥{CTX_MIN_PTS}pts trig≥{TRIG_MIN_PTS}pts | {SYMBOL}
  </div></div>
  <div class='ts'>Actualizado: {now_s}<br>Auto-refresh {DASHBOARD_REFRESH}s</div>
</div>
<div class='grid'>
  <div class='card'><div class='lbl'>Portfolio</div>
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
  <div class='card'><div class='lbl'>Señal</div>
    <div class='val'><span class='badge {st.get("sig","HOLD").lower()}'>{st.get("sig","HOLD")}</span></div>
    <div class='sub'>ctx={st.get("ctx_pts",0)}/{CTX_MIN_PTS} trig={st.get("trig_pts",0)}/{TRIG_MIN_PTS}</div></div>
  <div class='card'><div class='lbl'>Próxima vela</div>
    <div class='cd'>{cd}</div>
    <div class='sub'>ops: {st['nb']}B {st['ns']}S {st['ni']}ign</div></div>
</div>
<h2>Posiciones abiertas ({st['n_pos']})</h2>
<div class='sec'><table><thead><tr>
  <th>#</th><th>Entrada</th><th>Actual</th><th>BTC</th><th>P&L</th>
</tr></thead><tbody>{pos_rows}</tbody></table></div>
<h2>Últimas operaciones</h2>
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
        self._clock:     Optional[LiveClock]             = None
        self._wallet:    Optional[BinanceWallet]         = None
        self._ob:        Optional[BinanceOrderBook]      = None
        self._strategy:  Optional[DualReversalStrategy]  = None
        self._state_mgr: Optional[JSONStateManager]      = None
        self._signer:    Optional[_Signer]               = None
        self._running = False
        self._nb = self._ns = self._ni = 0
        self._t_start = 0.0
        self._last_candle: Optional[Candle] = None
        self._tlog: list = []
        self._usdt_ini = 0.0
        self._last_sig = "HOLD"
        self._last_ctx_pts = 0
        self._last_trig_pts = 0

    def run(self):
        self._t_start = time.time()
        self._setup_signals()
        print("╔══════════════════════════════════════════════════════════╗")
        print("║       LIVE TRADER — DUAL REVERSAL                       ║")
        print("╚══════════════════════════════════════════════════════════╝")
        print(f"  RSI({RSI_PERIOD}) MA{MA_SHORT}/{MA_LONG} W={WINDOW} N={LAST_N}")
        print(f"  ctx≥{CTX_MIN_PTS}pts  trig≥{TRIG_MIN_PTS}pts")
        print(f"  Dashboard: abrir '{DASHBOARD_HTML}' en el browser")
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
            max_posiciones=MAX_POSICIONES,
            json_path=LIVE_RESULTS_JSON,
            state_path=STATE_PATH,
        )
        self._usdt_ini = self._wallet.get_usdt_balance()
        self._ob = BinanceOrderBook(max_posiciones=MAX_POSICIONES,
                                    commission_pct=COMMISSION_PCT)

        warmup = max(MA_LONG + LAST_N + 4, RSI_PERIOD + LAST_N + 4, WINDOW + LAST_N + 4) + 4
        self._strategy = DualReversalStrategy(
            rsi_period     = RSI_PERIOD,
            ma_short       = MA_SHORT,
            ma_long        = MA_LONG,
            window         = WINDOW,
            last_n         = LAST_N,
            ctx_rsi_buy    = CTX_RSI_BUY,
            ctx_rsi_sell   = 100.0 - CTX_RSI_BUY,
            ctx_ma20_buy   = CTX_MA20_BUY,
            ctx_ma20_sell  = abs(CTX_MA20_BUY),
            ctx_min_pts    = CTX_MIN_PTS,
            trig_rsi_slope = TRIG_RSI_SLOPE,
            trig_wick      = TRIG_WICK,
            trig_min_pts   = TRIG_MIN_PTS,
            warmup         = warmup,
        )
        self._strategy.on_start(self._wallet)

        feed = BinanceWSFeed()
        self._clock = LiveClock(feed=feed, symbol=SYMBOL)
        print(f"\n  Wallet: ${self._wallet.get_usdt_balance():,.2f} USDT  "
              f"| {self._wallet.positions_count} posiciones")
        print(f"  Warmup: {warmup} velas  ({warmup}h hasta primera señal posible)")
        print(f"  Esperando primera vela horaria...\n")
        self._refresh()

    def _loop(self):
        self._running = True
        for candle in self._clock:
            if not self._running:
                break
            self._last_candle = candle
            signal = self._strategy._tick(candle, self._wallet)

            # Extraer info del reason para el dashboard
            self._last_sig = signal.side.value
            if signal.reason and "ctx=" in signal.reason:
                try:
                    ctx_s  = signal.reason.split("ctx=")[1].split("/")[0]
                    trig_s = signal.reason.split("trig=")[1].split("/")[0]
                    self._last_ctx_pts  = int(ctx_s)
                    self._last_trig_pts = int(trig_s)
                except Exception:
                    pass
            elif not signal.is_actionable:
                self._last_ctx_pts = self._last_trig_pts = 0

            if signal.is_actionable:
                order = self._ob.execute_with_guards(
                    side      = signal.to_order_side(),
                    price     = signal.price or candle.close,
                    wallet    = self._wallet,
                    candle_ts = candle.ts,
                )
                if order.is_filled:
                    t = order.trade
                    qty  = (f"{t.btc_bought:.6f} BTC" if t and t.btc_bought
                            else f"{t.btc_sold:.6f} BTC"   if t and t.btc_sold else "")
                    info = (f"${t.usdt_received:,.2f}" if t and t.usdt_received
                            else f"${t.usdt_spent:,.2f}" if t and t.usdt_spent else "")
                    self._tlog.append({
                        "hora":   datetime.fromtimestamp(candle.ts, tz=timezone.utc).strftime("%H:%M"),
                        "tipo":   order.side.value,
                        "precio": candle.close,
                        "qty":    qty,
                        "info":   info,
                    })
                    if len(self._tlog) > MAX_TRADE_LOG:
                        self._tlog.pop(0)
                    if order.side.value == "BUY":
                        self._nb += 1
                    else:
                        self._ns += 1
                    if self._wallet.get_trade_log():
                        lt = self._wallet.get_trade_log()[-1]
                        lt["score_bot"] = signal.score if signal.side == SignalSide.BUY  else 0.0
                        lt["score_top"] = signal.score if signal.side == SignalSide.SELL else 0.0
                elif order.is_ignored:
                    self._ni += 1

            self._state_mgr.save(Checkpoint.from_wallet(
                self._wallet, candle.ts, candle.close,
                metadata={"ctx_pts": CTX_MIN_PTS, "trig_pts": TRIG_MIN_PTS},
            ))
            self._refresh()

    def _refresh(self):
        px = self._last_candle.close if self._last_candle else 0.0
        st = {
            "portfolio":  round(self._wallet.portfolio_value(px), 2),
            "usdt_lib":   round(self._wallet.get_usdt_balance(), 2),
            "slot":       round(self._wallet.get_slot_usdt(), 2),
            "n_pos":      self._wallet.positions_count,
            "pos":        [{"entry_price": p.entry_price, "btc": p.btc}
                           for p in self._wallet.get_positions()],
            "precio":     px,
            "sig":        self._last_sig,
            "ctx_pts":    self._last_ctx_pts,
            "trig_pts":   self._last_trig_pts,
            "next_ts":    _next_hour(),
            "nb": self._nb, "ns": self._ns, "ni": self._ni,
            "tlog":       self._tlog,
            "usdt_ini":   self._usdt_ini,
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
                "estrategia":            "DualReversal-2Capas",
                "modo":                  "LIVE",
                "fecha_inicio":          to_iso(int(self._t_start)),
                "fecha_fin":             to_iso(now_epoch_s()),
                "symbol":                SYMBOL,
                "ctx_min_pts":           CTX_MIN_PTS,
                "trig_min_pts":          TRIG_MIN_PTS,
                "total_compras":         self._nb,
                "total_ventas":          self._ns,
                "total_ignorados":       self._ni,
                "saldo_inicial_usdt":    self._usdt_ini,
                "usdt_balance_final":    round(self._wallet.get_usdt_balance(), 8),
                "portfolio_value_final": round(self._wallet.portfolio_value(px), 4),
            })
        if self._wallet:
            self._refresh()
        log.info("live dual reversal detenido",
                 compras=self._nb, ventas=self._ns,
                 sesion_h=f"{(time.time()-self._t_start)/3600:.1f}h")

    def _setup_signals(self):
        def _h(sig, frame):
            print("\n\nCtrl+C — cerrando limpiamente...")
            self._running = False
            if self._clock:
                self._clock.stop()
        signal.signal(signal.SIGINT,  _h)
        signal.signal(signal.SIGTERM, _h)


if __name__ == "__main__":
    LiveTrader().run()
