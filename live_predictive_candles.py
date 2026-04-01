"""
live_predictive_candles.py — Runner Live de PredictiveCandles
══════════════════════════════════════════════════════════════
Ejecuta PredictiveCandlesStrategy en tiempo real contra Binance
(testnet o producción).

Uso:
    python live_predictive_candles.py

Configuración:
    · config_world.py  →  endpoints, USE_TESTNET
    · config_local.py  →  rutas, capital, comisiones
    · Sección CONFIG   →  predictores, umbrales y cooldowns (editar aquí)
"""

from __future__ import annotations

import hashlib
import hmac
import signal
import sys
import time
import urllib.parse
from datetime import datetime, timezone
from pathlib  import Path
from typing   import Optional

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
from strategies.predictive_candles import PredictiveCandlesStrategy
from support.logger            import get_logger
from support.secrets           import secrets
from support.time_utils        import now_epoch_s, to_iso

log = get_logger("live_predictive")


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG — Editar aquí antes de cada ejecución
# ══════════════════════════════════════════════════════════════════════════════

VENTANA      = 10    # velas hacia atrás para calcular factores  [10, 16]
UMBRAL_BOT   = 0.85  # score mínimo para señal BUY               (0, 1]
UMBRAL_TOP   = 0.70  # score mínimo para señal SELL              (0, 1]
COOLDOWN_BOT = 0     # velas mínimas entre señales BUY  (0 = desactivado)
COOLDOWN_TOP = 0     # velas mínimas entre señales SELL (0 = desactivado)

# ── Predictores activos para BOTTOM (señal BUY) ───────────────────────────────
USE_BOT_CLOSE_POSITION = True   # AUC=0.854 ▼
USE_BOT_BB_POSITION    = True   # AUC=0.833 ▼
USE_BOT_RECOVERY_PCT   = True   # AUC=0.823 ▼

# ── Predictores activos para TOP (señal SELL) ─────────────────────────────────
USE_TOP_CLOSE_POSITION = True   # AUC=0.839 ▲
USE_TOP_DRAWDOWN_PCT   = True   # AUC=0.826 ▼
USE_TOP_BB_POSITION    = True   # AUC=0.821 ▲

# ── Parámetros de sesión ──────────────────────────────────────────────────────
MAX_POSICIONES    = 4
COMMISSION_PCT    = 0.1
SYMBOL            = "BTCUSDT"
LIVE_RESULTS_JSON = "live_predictive_results.json"
STATE_PATH        = "state/live_predictive_state.jsonl"
DASHBOARD_REFRESH = 10
MAX_TRADE_LOG     = 50


# ══════════════════════════════════════════════════════════════════════════════
# SIGNER
# ══════════════════════════════════════════════════════════════════════════════

def _get_cfg() -> dict:
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

    def _measure_offset(self) -> int:
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

    def _sign(self, p):
        p = dict(p); p["timestamp"] = self._now_ms(); p["recvWindow"] = self.recv_window
        qs = urllib.parse.urlencode(p)
        p["signature"] = hmac.new(self.secret.encode(), qs.encode(), hashlib.sha256).hexdigest()
        return p

    def _h(self): return {"X-MBX-APIKEY": self.api_key}

    def get(self, ep, params=None):
        p = self._sign(params or {})
        r = requests.get(f"{self.base_url}{ep}", params=p, headers=self._h(), timeout=self.timeout)
        r.raise_for_status(); return r.json()

    def post(self, ep, params):
        p = self._sign(params)
        r = requests.post(f"{self.base_url}{ep}", params=p, headers=self._h(), timeout=self.timeout)
        return r.json()

    def price(self, sym):
        r = requests.get(f"{self.base_url}/api/v3/ticker/price",
                         params={"symbol": sym}, timeout=self.timeout)
        return float(r.json()["price"])

    def step_size(self, sym):
        r = requests.get(f"{self.base_url}/api/v3/exchangeInfo",
                         params={"symbol": sym}, timeout=self.timeout)
        s = next(s for s in r.json()["symbols"] if s["symbol"] == sym)
        return {f["filterType"]: f for f in s["filters"]}["LOT_SIZE"]["stepSize"]


def _truncate(qty, step):
    dec = len(step.rstrip("0").split(".")[-1])
    return f"{int(qty * 10**dec) / 10**dec:.{dec}f}"


def sell_preexisting_btc(signer: _Signer, symbol: str) -> None:
    print("  [limpieza] Verificando BTC libre...", end=" ", flush=True)
    account  = signer.get("/api/v3/account")
    btc_free = next((float(b["free"]) for b in account["balances"]
                     if b["asset"] == "BTC"), 0.0)
    if btc_free < 0.00001:
        print("ninguno. OK"); return
    precio  = signer.price(symbol)
    qty_str = _truncate(btc_free, signer.step_size(symbol))
    print(f"encontrado {btc_free:.8f} BTC (~${btc_free*precio:,.2f})")
    print(f"  [limpieza] Vendiendo {qty_str} BTC...", end=" ", flush=True)
    result = signer.post("/api/v3/order", {
        "symbol": symbol, "side": "SELL", "type": "MARKET", "quantity": qty_str,
    })
    if result.get("status") in ("FILLED", "PARTIALLY_FILLED"):
        usdt_rec = float(result.get("cummulativeQuoteQty", 0))
        print(f"OK → ${usdt_rec:,.2f} USDT")
    else:
        print(f"no ejecutado — {result.get('msg', '?')} (continuando)")


# ══════════════════════════════════════════════════════════════════════════════
# DASHBOARD — CONSOLA
# ══════════════════════════════════════════════════════════════════════════════

G = "\033[92m"; R = "\033[91m"; Y = "\033[93m"
C = "\033[96m"; D = "\033[2m";  B = "\033[1m"; X = "\033[0m"
CLR = "\033[2J\033[H"

_bot_on = [k for k, v in {
    "close_pos": USE_BOT_CLOSE_POSITION,
    "bb_pos":    USE_BOT_BB_POSITION,
    "recovery":  USE_BOT_RECOVERY_PCT,
}.items() if v]

_top_on = [k for k, v in {
    "close_pos": USE_TOP_CLOSE_POSITION,
    "drawdown":  USE_TOP_DRAWDOWN_PCT,
    "bb_pos":    USE_TOP_BB_POSITION,
}.items() if v]


def _next_hour() -> int:
    return (int(time.time()) // 3600 + 1) * 3600


def _countdown(ts: int) -> str:
    rem = max(0, ts - int(time.time()))
    h, r = divmod(rem, 3600); m, s = divmod(r, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def render_console(st: dict) -> None:
    pnl_p = (st["portfolio"] / st["usdt_ini"] - 1) * 100 if st["usdt_ini"] else 0
    pnl_u = st["portfolio"] - st["usdt_ini"]
    now_s = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    print(CLR, end="")
    print(f"{B}{'═'*70}{X}")
    print(f"{B}  PREDICTIVE CANDLES — LIVE{X}  {D}{now_s}{X}")
    print(f"{B}{'═'*70}{X}")
    print(f"\n  {B}Config{X}  vent={VENTANA}  thr_b={UMBRAL_BOT}  thr_t={UMBRAL_TOP}  "
          f"cd_b={'off' if not COOLDOWN_BOT else COOLDOWN_BOT}  "
          f"cd_t={'off' if not COOLDOWN_TOP else COOLDOWN_TOP}")
    print(f"  BOT: {', '.join(_bot_on) or 'NINGUNO'}   "
          f"TOP: {', '.join(_top_on) or 'NINGUNO'}")

    pc = G if pnl_p >= 0 else R
    print(f"\n  {B}Portfolio{X}")
    print(f"    Total   : {pc}${st['portfolio']:>12,.2f}{X}  "
          f"({pc}{'+' if pnl_p>=0 else ''}{pnl_p:.2f}%  "
          f"{'+' if pnl_u>=0 else ''}${pnl_u:,.2f}{X})")
    print(f"    USDT lib: ${st['usdt_lib']:>12,.2f}    Slot: ${st['slot']:,.2f}")

    px = st["precio"]
    print(f"\n  {B}Posiciones ({st['n_pos']}/{MAX_POSICIONES}){X}")
    if not st["pos"]:
        print(f"    {D}Sin posiciones{X}")
    else:
        print(f"    {'#':<3}{'Entrada':>10}{'Actual':>10}{'BTC':>12}{'P&L':>9}")
        for i, p in enumerate(st["pos"], 1):
            pe  = p["entry_price"]
            pct = (px / pe - 1) * 100 if pe else 0
            col = G if pct >= 0 else R
            print(f"    {i:<3}${pe:>9,.2f}${px:>9,.2f}"
                  f"{p['btc']:>12.6f} {col}{'+' if pct>=0 else ''}{pct:.2f}%{X}")

    sb, st_ = st["score_bot"], st["score_top"]
    print(f"\n  {B}Score del modelo{X}  close=${px:,.2f}")
    cb = G if sb >= UMBRAL_BOT else (Y if sb >= UMBRAL_BOT * 0.8 else D)
    ct = R if st_ >= UMBRAL_TOP else (Y if st_ >= UMBRAL_TOP * 0.8 else D)
    bb_bar = "█" * int(sb * 20) + "░" * (20 - int(sb * 20))
    bt_bar = "█" * int(st_ * 20) + "░" * (20 - int(st_ * 20))
    print(f"    score_bot: {cb}{bb_bar} {sb:.3f}{X}  (thr={UMBRAL_BOT}  "
          f"cd={'✓' if st.get('cd_ok_bot') else '⏸'})")
    print(f"    score_top: {ct}{bt_bar} {st_:.3f}{X}  (thr={UMBRAL_TOP}  "
          f"cd={'✓' if st.get('cd_ok_top') else '⏸'})")

    pv = st.get("pred_values", {})
    if any(v is not None for v in pv.values()):
        def _fv(k): v = pv.get(k); return f"{v:.3f}" if v is not None else "N/A"
        print(f"    {D}cp={_fv('close_position')}  bb={_fv('bb_position')}  "
              f"rc={_fv('recovery_pct'):.3s}%  dw={_fv('drawdown_pct'):.3s}%{X}")

    print(f"\n  {B}Próxima vela en{X}  {C}{_countdown(st['next_ts'])}{X}")

    print(f"\n  {B}Operaciones{X}  (buy={st['nb']}  sell={st['ns']}  ign={st['ni']})")
    for t in st["tlog"][-5:][::-1]:
        col = G if t["tipo"] == "BUY" else (R if t["tipo"] == "SELL" else D)
        print(f"    {t['hora']:>6} {col}{t['tipo']:>4}{X}  "
              f"${t['precio']:>10,.2f}  {t['qty']:>14}  {t['info']}")

    print(f"\n{D}  Ctrl+C para detener{X}")
    print(f"{B}{'═'*70}{X}")


# ══════════════════════════════════════════════════════════════════════════════
# LIVE TRADER
# ══════════════════════════════════════════════════════════════════════════════

class LivePredictiveTrader:

    def __init__(self) -> None:
        self._clock:     Optional[LiveClock]                 = None
        self._wallet:    Optional[BinanceWallet]             = None
        self._ob:        Optional[BinanceOrderBook]          = None
        self._strategy:  Optional[PredictiveCandlesStrategy] = None
        self._state_mgr: Optional[JSONStateManager]          = None
        self._signer:    Optional[_Signer]                   = None
        self._running    = False
        self._nb = self._ns = self._ni = 0
        self._t_start    = 0.0
        self._last_candle: Optional[Candle] = None
        self._tlog: list  = []
        self._usdt_ini    = 0.0

    def run(self) -> None:
        self._t_start = time.time()
        self._setup_signals()

        print("╔══════════════════════════════════════════════════════════╗")
        print("║   LIVE PREDICTIVE CANDLES — BTC/USDT                    ║")
        print("╚══════════════════════════════════════════════════════════╝")
        print(f"  Ventana: {VENTANA}  |  Umbral BOT: {UMBRAL_BOT}  TOP: {UMBRAL_TOP}")
        print(f"  Cooldown BOT: {'off' if not COOLDOWN_BOT else str(COOLDOWN_BOT)+' velas'}  "
              f"TOP: {'off' if not COOLDOWN_TOP else str(COOLDOWN_TOP)+' velas'}")
        print(f"  BOT activos: {', '.join(_bot_on) or 'NINGUNO'}")
        print(f"  TOP activos: {', '.join(_top_on) or 'NINGUNO'}")
        print("─" * 62)

        try:
            self._setup()
            self._loop()
        except KeyboardInterrupt:
            pass
        finally:
            self._shutdown()

    def _setup(self) -> None:
        cfg = _get_cfg()
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
        feed        = BinanceWSFeed()
        self._clock = LiveClock(feed=feed, symbol=SYMBOL)

        self._strategy = PredictiveCandlesStrategy(
            ventana                = VENTANA,
            umbral_bot             = UMBRAL_BOT,
            umbral_top             = UMBRAL_TOP,
            use_bot_close_position = USE_BOT_CLOSE_POSITION,
            use_bot_bb_position    = USE_BOT_BB_POSITION,
            use_bot_recovery_pct   = USE_BOT_RECOVERY_PCT,
            use_top_close_position = USE_TOP_CLOSE_POSITION,
            use_top_drawdown_pct   = USE_TOP_DRAWDOWN_PCT,
            use_top_bb_position    = USE_TOP_BB_POSITION,
            cooldown_bot           = COOLDOWN_BOT,
            cooldown_top           = COOLDOWN_TOP,
        )
        self._strategy.on_start(self._wallet)

        print(f"\n  Wallet: ${self._wallet.get_usdt_balance():,.2f} USDT libre  "
              f"| {self._wallet.positions_count} pos. abiertas")
        print("  Esperando cierre de próxima vela...\n")
        self._refresh()

    def _loop(self) -> None:
        self._running = True
        for candle in self._clock:
            if not self._running:
                break
            self._last_candle = candle
            signal = self._strategy._tick(candle, self._wallet)

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
                              else f"{t.btc_sold:.6f} BTC"   if t and t.btc_sold else "")
                    info   = (f"${t.usdt_received:,.2f}" if t and t.usdt_received
                              else f"${t.usdt_spent:,.2f}"   if t and t.usdt_spent else "")
                    self._tlog.append({
                        "hora":   datetime.fromtimestamp(candle.ts, tz=timezone.utc)
                                  .strftime("%H:%M"),
                        "tipo":   side_s, "precio": candle.close,
                        "qty":    qty,    "info":   info,
                    })
                    if len(self._tlog) > MAX_TRADE_LOG:
                        self._tlog.pop(0)

                    # Enriquecer último trade con pred_* y scores
                    log_e = self._wallet.get_trade_log()
                    if log_e:
                        last = log_e[-1]
                        last["score_bot"]           = self._strategy.last_score_bot
                        last["score_top"]           = self._strategy.last_score_top
                        last["pred_close_position"] = self._strategy.last_pred_values.get("close_position")
                        last["pred_bb_position"]    = self._strategy.last_pred_values.get("bb_position")
                        last["pred_recovery_pct"]   = self._strategy.last_pred_values.get("recovery_pct")
                        last["pred_drawdown_pct"]   = self._strategy.last_pred_values.get("drawdown_pct")

                    if side_s == "BUY": self._nb += 1
                    else:               self._ns += 1
                elif order.is_ignored:
                    self._ni += 1

            self._state_mgr.save(Checkpoint.from_wallet(
                self._wallet, candle.ts, candle.close,
                metadata={
                    "ventana":      VENTANA,
                    "umbral_bot":   UMBRAL_BOT,
                    "umbral_top":   UMBRAL_TOP,
                    "cooldown_bot": COOLDOWN_BOT,
                    "cooldown_top": COOLDOWN_TOP,
                },
            ))
            self._refresh()

    def _refresh(self) -> None:
        px = self._last_candle.close if self._last_candle else 0.0
        st = {
            "portfolio":  round(self._wallet.portfolio_value(px), 2),
            "usdt_lib":   round(self._wallet.get_usdt_balance(), 2),
            "slot":       round(self._wallet.get_slot_usdt(), 2),
            "n_pos":      self._wallet.positions_count,
            "pos":        [{"entry_price": p.entry_price, "btc": p.btc}
                           for p in self._wallet.get_positions()],
            "precio":     px,
            "score_bot":  self._strategy.last_score_bot,
            "score_top":  self._strategy.last_score_top,
            "cd_ok_bot":  self._strategy.last_cooldown_ok_bot,
            "cd_ok_top":  self._strategy.last_cooldown_ok_top,
            "pred_values":self._strategy.last_pred_values,
            "next_ts":    _next_hour(),
            "nb":         self._nb,
            "ns":         self._ns,
            "ni":         self._ni,
            "tlog":       self._tlog,
            "usdt_ini":   self._usdt_ini,
        }
        render_console(st)

    def _shutdown(self) -> None:
        self._running = False
        if self._clock:
            self._clock.stop()
        if self._strategy and self._wallet:
            self._strategy.on_stop(self._wallet)
        if self._wallet and self._wallet.get_trade_log():
            px = self._last_candle.close if self._last_candle else 0
            self._wallet.flush({
                "estrategia":            "PredictiveCandles",
                "modo":                  "LIVE",
                "ventana":               VENTANA,
                "umbral_bot":            UMBRAL_BOT,
                "umbral_top":            UMBRAL_TOP,
                "cooldown_bot":          COOLDOWN_BOT,
                "cooldown_top":          COOLDOWN_TOP,
                "active_bot":            _bot_on,
                "active_top":            _top_on,
                "fecha_inicio":          to_iso(int(self._t_start)),
                "fecha_fin":             to_iso(now_epoch_s()),
                "symbol":                SYMBOL,
                "total_compras":         self._nb,
                "total_ventas":          self._ns,
                "total_ignorados":       self._ni,
                "saldo_inicial_usdt":    self._usdt_ini,
                "usdt_balance_final":    round(self._wallet.get_usdt_balance(), 8),
                "portfolio_value_final": round(self._wallet.portfolio_value(px), 4),
            })
        log.info("live trader detenido",
                 compras=self._nb, ventas=self._ns,
                 sesion_h=f"{(time.time()-self._t_start)/3600:.1f}h")

    def _setup_signals(self) -> None:
        def _h(sig, frame):
            print("\n\nCtrl+C — cerrando limpiamente...")
            self._running = False
            if self._clock: self._clock.stop()
        signal.signal(signal.SIGINT,  _h)
        signal.signal(signal.SIGTERM, _h)


if __name__ == "__main__":
    LivePredictiveTrader().run()