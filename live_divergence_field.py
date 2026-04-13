"""
live_divergence_field.py — Live Trader: DivergenceFieldStrategy  v2
════════════════════════════════════════════════════════════════════
Modos:
  python live_divergence_field.py
      Órdenes reales en Binance Testnet.

  python live_divergence_field.py --paper
      Paper trading con precios REALES de Binance producción.
      Órdenes simuladas. Sin dinero real. Velas no manipuladas.

  python live_divergence_field.py --paper --testnet
      Paper trading con precios del testnet.

  python live_divergence_field.py --paper --capital 5000
      Paper mode con capital inicial personalizado.

BUGS CORREGIDOS vs v1
──────────────────────
  BUG 1 — Slot vs RiskManager:
    v1: RiskConfig.conservative() → max_order_usdt=$300 fijo.
        Slot real testnet = $10,000 / 10 = $1,000 → TODA orden bloqueada.
    Fix: RiskConfig se construye DESPUÉS de leer el wallet real.
         max_order = slot × 1.05. Se loggea al inicio.

  BUG 2 — capital_inicial con BTC pre-existente:
    v1: capital_inicial = wallet.portfolio_value(última vela warmup).
        Testnet tiene 1 BTC libre → inflaba el capital base.
    Fix: capital_inicial = wallet.get_usdt_balance() únicamente.

  BUG 3 — Velas testnet artificiales:
    Testnet genera velas con rangos de ±20% en una hora (sintéticas).
    --paper conecta al feed REAL de Binance producción.
    El live imprime advertencia si detecta rangos > 5% en las últimas velas.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib  import Path
from typing   import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent))

import config_local  as CL
import config_world  as CW

from actors.price_feed         import Candle
from actors.wallet             import JSONWallet, TradeRecord
from actors.order_book         import SimulatedOrderBook, OrderSide
from risk.risk_manager         import RiskManager, RiskConfig
from state.state_manager       import JSONStateManager, Checkpoint
from strategies.divergence_field_strategy import (
    DivergenceFieldStrategy, DFConfig,
    TEEstimator, WindowMode, FieldDefinition,
    CMIRegimes, ThresholdMode, SinkMode,
)
from support.logger import get_logger

log = get_logger("live_divfield")


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG — Pegar resultado del deep-grid aquí
# ══════════════════════════════════════════════════════════════════════════════

CONFIG = DFConfig(
    te_estimator        = TEEstimator.BINNING,
    window_mode         = WindowMode.FIXED,
    window_size         = 14,
    field_def           = FieldDefinition.JACOBIAN,
    cmi_regimes         = CMIRegimes.TERNARY,
    threshold_mode      = ThresholdMode.FIXED,
    sink_mode           = SinkMode.FILTER_AND,
    score_threshold_bot = 0.70,
    score_threshold_top = 0.65,
    cooldown            = 0,
    k_bins              = 4,
    k_nn                = 3,
    n_norm              = 200,
    w_te                = 0.40,
    w_cmi               = 0.30,
    w_field             = 0.20,
    w_sink              = 0.10,
    sink_threshold      = 1.20,
    sink_window         = 5,
)

# ── Parámetros ────────────────────────────────────────────────────────────────
MAX_POSICIONES    = CL.MAX_POSICIONES
COMMISSION_PCT    = CL.COMMISSION_PCT
SYMBOL            = CL.SYMBOL
WARMUP_CANDLES    = max(CONFIG.window_size * 3, 60)
LIVE_RESULTS_JSON = "live_divfield_results.json"
STATE_PATH        = "state/live_divfield_state.jsonl"
DASHBOARD_HTML    = "live_divfield_dashboard.html"
DASHBOARD_REFRESH = 10
CHART_CANDLES     = 48

# Risk limits — escalados desde el slot real en main()
RISK_MAX_DD_PCT       = 15.0
RISK_DAILY_SLOTS      = 3      # pérdida máxima diaria = N × slot
RISK_ORDER_MARGIN     = 1.05   # max_order = slot × margin
RISK_MIN_ORDER        = 5.0


# ══════════════════════════════════════════════════════════════════════════════
# ARGS
# ══════════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Live Trader — DivergenceFieldStrategy v2")
    p.add_argument("--paper",    action="store_true",
                   help="Paper trading: simula órdenes, usa precios reales")
    p.add_argument("--testnet",  action="store_true",
                   help="Forzar feed del testnet (incluso en paper mode)")
    p.add_argument("--capital",  type=float, default=None,
                   help="Capital para paper trading (default: CL.SALDO_USDT_INICIAL)")
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# ESTADO GLOBAL
# ══════════════════════════════════════════════════════════════════════════════

class LiveState:
    def __init__(self, paper_mode: bool = False):
        self.paper_mode        = paper_mode
        self.started_at        = _now_iso()
        self.last_tick_at      = "—"
        self.next_candle_at    = "—"
        self.candles_seen      = 0
        self.signals_total     = 0
        self.te_raw:           Optional[float] = None
        self.te_norm:          float = 0.0
        self.cmi_raw:          Optional[float] = None
        self.cmi_norm:         float = 0.0
        self.field_price_div:  Optional[float] = None
        self.field_vol_div:    Optional[float] = None
        self.field_curl:       Optional[float] = None
        self.field_norm:       float = 0.0
        self.sink_raw:         Optional[float] = None
        self.sink_norm:        float = 0.0
        self.is_bot_pattern:   bool  = False
        self.is_top_pattern:   bool  = False
        self.score_bot:        float = 0.0
        self.score_top:        float = 0.0
        self.last_signal:      str   = "HOLD"
        self.last_signal_price: Optional[float] = None
        self.last_signal_at:   str   = "—"
        self.last_signal_reason: str = "—"
        self.last_price:       float = 0.0
        self.portfolio_value:  float = 0.0
        self.usdt_balance:     float = 0.0
        self.btc_in_positions: float = 0.0
        self.positions_count:  int   = 0
        self.slot_usdt:        float = 0.0
        self.pnl_pct:          float = 0.0
        self.pnl_usdt:         float = 0.0
        self.capital_inicial:  float = 0.0
        self.candle_history:   List[dict] = []
        self.trade_history:    List[dict] = []
        self.metric_history:   List[dict] = []
        self.config_dict:      dict = {}
        self.warmup_complete:  bool = False
        self.risk_max_order:   float = 0.0
        self.risk_max_daily:   float = 0.0

    def update_from_candle(self, candle: Candle, strategy: DivergenceFieldStrategy,
                            wallet) -> None:
        self.last_tick_at      = _now_iso()
        self.candles_seen      = strategy.candles_seen
        self.last_price        = candle.close
        self.portfolio_value   = wallet.portfolio_value(candle.close)
        self.usdt_balance      = wallet.get_usdt_balance()
        self.btc_in_positions  = wallet.btc_en_posiciones()
        self.positions_count   = wallet.positions_count
        self.slot_usdt         = wallet.get_slot_usdt()
        self.pnl_usdt          = self.portfolio_value - self.capital_inicial
        self.pnl_pct           = (self.pnl_usdt / self.capital_inicial * 100.0
                                  if self.capital_inicial > 0 else 0.0)
        self.te_raw            = strategy.last_te
        self.te_norm           = strategy.last_te_norm
        self.cmi_raw           = strategy.last_cmi
        self.cmi_norm          = strategy.last_cmi_norm
        self.field_price_div   = strategy.last_field_price
        self.field_vol_div     = strategy.last_field_vol
        self.field_curl        = strategy.last_field_curl
        self.field_norm        = strategy.last_field_norm
        self.sink_raw          = strategy.last_sink
        self.sink_norm         = strategy.last_sink_norm
        self.score_bot         = strategy.last_score_bot
        self.score_top         = strategy.last_score_top
        self.is_bot_pattern    = strategy.last_is_bot_pattern
        self.is_top_pattern    = strategy.last_is_top_pattern
        self.warmup_complete   = strategy.candles_seen >= strategy.cfg.window_size

        self.candle_history.append({
            "ts": candle.ts, "open": candle.open, "high": candle.high,
            "low": candle.low, "close": candle.close, "vol": candle.volume,
            "taker": (candle.taker_buy_base_vol / candle.volume
                      if candle.taker_buy_base_vol and candle.volume > 0 else 0.5),
        })
        if len(self.candle_history) > CHART_CANDLES:
            self.candle_history.pop(0)

        self.metric_history.append({
            "ts": candle.ts, "te_norm": self.te_norm, "cmi_norm": self.cmi_norm,
            "field_norm": self.field_norm, "sink_norm": self.sink_norm,
            "score_bot": self.score_bot, "score_top": self.score_top,
            "is_bot": self.is_bot_pattern, "is_top": self.is_top_pattern,
        })
        if len(self.metric_history) > CHART_CANDLES:
            self.metric_history.pop(0)

    def update_from_signal(self, sig: str, price: float, reason: str) -> None:
        self.last_signal        = sig
        self.last_signal_price  = price
        self.last_signal_at     = _now_iso()
        self.last_signal_reason = reason
        if sig != "HOLD":
            self.signals_total += 1

    def add_trade(self, td: dict) -> None:
        self.trade_history.append(td)
        if len(self.trade_history) > 50:
            self.trade_history.pop(0)

    def estimate_next_candle(self) -> None:
        now_s  = int(time.time())
        next_h = ((now_s // 3600) + 1) * 3600
        m, s   = divmod(next_h - now_s, 60)
        self.next_candle_at = f"{m}m {s:02d}s"


# ══════════════════════════════════════════════════════════════════════════════
# RISK CONFIG — escalado desde slot real
# ══════════════════════════════════════════════════════════════════════════════

def _build_risk_config(slot_usdt: float, capital: float) -> RiskConfig:
    """FIX BUG 1: max_order escalado desde el slot real, no hardcodeado."""
    max_order = round(slot_usdt * RISK_ORDER_MARGIN,   2)
    max_daily = round(slot_usdt * RISK_DAILY_SLOTS,    2)
    log.info("RiskConfig escalado desde slot real",
             slot=f"${slot_usdt:.2f}", max_order=f"${max_order:.2f}",
             max_daily=f"${max_daily:.2f}", max_dd=f"{RISK_MAX_DD_PCT}%")
    return RiskConfig(
        max_drawdown_pct    = RISK_MAX_DD_PCT,
        max_daily_loss_usdt = max_daily,
        max_order_usdt      = max_order,
        min_order_usdt      = RISK_MIN_ORDER,
        dedup_window_s      = 60,
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAPER WALLET
# ══════════════════════════════════════════════════════════════════════════════

class PaperJSONWallet(JSONWallet):
    """JSONWallet que marca cada trade como 'paper_trade': True."""
    def update(self, trade: TradeRecord) -> None:
        super().update(trade)
        entries = self.get_trade_log()
        if entries:
            entries[-1]["paper_trade"] = True


# ══════════════════════════════════════════════════════════════════════════════
# DASHBOARD HTML
# ══════════════════════════════════════════════════════════════════════════════

def generate_dashboard(state: LiveState) -> str:
    data = json.dumps({
        "paper_mode":        state.paper_mode,
        "started_at":        state.started_at,
        "last_tick_at":      state.last_tick_at,
        "next_candle_at":    state.next_candle_at,
        "candles_seen":      state.candles_seen,
        "signals_total":     state.signals_total,
        "warmup_complete":   state.warmup_complete,
        "warmup_size":       CONFIG.window_size,
        "te_raw":            state.te_raw,
        "te_norm":           round(state.te_norm, 4),
        "cmi_raw":           state.cmi_raw,
        "cmi_norm":          round(state.cmi_norm, 4),
        "field_price_div":   state.field_price_div,
        "field_vol_div":     state.field_vol_div,
        "field_curl":        state.field_curl,
        "field_norm":        round(state.field_norm, 4),
        "sink_raw":          state.sink_raw,
        "sink_norm":         round(state.sink_norm, 4),
        "is_bot_pattern":    state.is_bot_pattern,
        "is_top_pattern":    state.is_top_pattern,
        "score_bot":         round(state.score_bot, 4),
        "score_top":         round(state.score_top, 4),
        "last_signal":       state.last_signal,
        "last_signal_price": state.last_signal_price,
        "last_signal_at":    state.last_signal_at,
        "last_signal_reason":state.last_signal_reason,
        "last_price":        state.last_price,
        "portfolio_value":   round(state.portfolio_value, 2),
        "usdt_balance":      round(state.usdt_balance, 2),
        "btc_in_positions":  state.btc_in_positions,
        "positions_count":   state.positions_count,
        "slot_usdt":         round(state.slot_usdt, 2),
        "pnl_pct":           round(state.pnl_pct, 2),
        "pnl_usdt":          round(state.pnl_usdt, 2),
        "capital_inicial":   state.capital_inicial,
        "thr_bot":           CONFIG.score_threshold_bot,
        "thr_top":           CONFIG.score_threshold_top,
        "sink_thr":          CONFIG.sink_threshold,
        "risk_max_order":    state.risk_max_order,
        "risk_max_daily":    state.risk_max_daily,
        "risk_max_dd":       RISK_MAX_DD_PCT,
        "candles":           state.candle_history,
        "metrics":           state.metric_history,
        "trades":            state.trade_history[-10:],
        "config":            state.config_dict,
    }, default=str)

    paper_tag = "📄 PAPER" if state.paper_mode else "🔴 LIVE"
    mode_color = "#d29922" if state.paper_mode else "#3fb950"
    dot_anim  = "" if state.paper_mode else "animation:pulse 1.5s infinite;"

    # ── Inline style + HTML ───────────────────────────────────────────────────
    css = """
:root{--bg:#0d1117;--bg2:#161b22;--bg3:#21262d;--bd:#30363d;--tx:#e6edf3;--mu:#8b949e;
      --g:#3fb950;--r:#f85149;--y:#d29922;--b:#58a6ff;--p:#a371f7;--o:#fb8500;}
*{box-sizing:border-box;margin:0;padding:0;}
body{background:var(--bg);color:var(--tx);font-family:'Segoe UI',monospace;font-size:13px;}
h1{font-size:15px;font-weight:600;color:var(--b);}
h2{font-size:10px;font-weight:600;color:var(--mu);text-transform:uppercase;letter-spacing:1px;margin-bottom:7px;}
.hdr{background:var(--bg2);border-bottom:1px solid var(--bd);padding:8px 14px;
     display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px;}
.hdr-meta{display:flex;gap:14px;font-size:10px;color:var(--mu);flex-wrap:wrap;}
@keyframes pulse{0%,100%{opacity:1;}50%{opacity:.3;}}
.layout{display:grid;grid-template-columns:265px 1fr 265px;gap:0;height:calc(100vh - 50px);overflow:hidden;}
.col{overflow-y:auto;padding:9px;display:flex;flex-direction:column;gap:7px;}
.col-l{border-right:1px solid var(--bd);}
.col-r{border-left:1px solid var(--bd);}
.card{background:var(--bg2);border:1px solid var(--bd);border-radius:5px;padding:9px;}
.cb{border-left:3px solid var(--g);} .ct{border-left:3px solid var(--r);}
.cbl{border-left:3px solid var(--b);} .cp{border-left:3px solid var(--p);}
.cy{border-left:3px solid var(--y);}
.pg{display:grid;grid-template-columns:1fr 1fr;gap:5px;}
.pc{background:var(--bg3);border:1px solid var(--bd);border-radius:4px;padding:6px 8px;}
.pc .lb{font-size:9px;color:var(--mu);margin-bottom:1px;}
.pc .vl{font-size:14px;font-weight:700;}
.pos{color:var(--g);}.neg{color:var(--r);}.neu{color:var(--tx);}
.mr{display:flex;align-items:center;gap:6px;margin-bottom:5px;}
.ml{width:68px;font-size:10px;color:var(--mu);flex-shrink:0;}
.mb{flex:1;background:var(--bg3);border-radius:3px;height:12px;position:relative;overflow:hidden;}
.mf{height:100%;border-radius:3px;transition:width .4s;}
.mv{width:40px;text-align:right;font-size:10px;font-family:monospace;flex-shrink:0;}
.mt{position:absolute;top:0;bottom:0;width:2px;background:rgba(255,255,255,.3);}
.bte{background:linear-gradient(90deg,#1f6feb,#58a6ff);}
.bci{background:linear-gradient(90deg,#6e40c9,#a371f7);}
.bfd{background:linear-gradient(90deg,#fb8500,#ffd166);}
.bsk{background:linear-gradient(90deg,#0e9c6c,#3fb950);}
.bsb{background:linear-gradient(90deg,#0e9c6c,#3fb950);}
.bst{background:linear-gradient(90deg,#b31d28,#f85149);}
.sb{display:inline-block;padding:3px 10px;border-radius:10px;font-size:11px;font-weight:700;letter-spacing:1px;}
.sb-BUY{background:rgba(63,185,80,.2);color:var(--g);border:1px solid var(--g);}
.sb-SELL{background:rgba(248,81,73,.2);color:var(--r);border:1px solid var(--r);}
.sb-HOLD{background:rgba(139,148,158,.1);color:var(--mu);border:1px solid var(--bd);}
.fvr{display:flex;align-items:center;justify-content:space-around;background:var(--bg3);border-radius:4px;padding:6px;margin-top:5px;}
.fvc{text-align:center;}
.fvl{font-size:9px;color:var(--mu);}
.fva{font-size:14px;}
.fvv{font-size:13px;font-weight:700;}
.ib{background:rgba(88,166,255,.05);border:1px solid rgba(88,166,255,.15);border-radius:4px;
    padding:6px;font-size:10px;color:var(--mu);line-height:1.8;}
.kv{display:flex;justify-content:space-between;}
.kv .k{color:var(--mu);}.kv .v{color:var(--tx);font-family:monospace;}
table{width:100%;border-collapse:collapse;font-size:10px;}
th{text-align:left;color:var(--mu);padding:3px 4px;border-bottom:1px solid var(--bd);font-weight:400;}
td{padding:3px 4px;border-bottom:1px solid rgba(48,54,61,.4);}
tr:last-child td{border-bottom:none;}
.tbu{color:var(--g);font-weight:600;}.tse{color:var(--r);font-weight:600;}.tig{color:var(--mu);}
.cfg-r{display:flex;justify-content:space-between;padding:2px 0;border-bottom:1px solid rgba(48,54,61,.3);font-size:10px;}
.cfg-k{color:var(--mu);}.cfg-v{color:var(--b);font-family:monospace;}
.pr{background:var(--bg3);border:1px solid var(--bd);border-radius:4px;padding:6px 8px;margin-bottom:5px;}
.prh{display:flex;justify-content:space-between;align-items:center;}
.prd{font-size:9px;color:var(--mu);margin-top:2px;}
.rr{display:flex;align-items:center;gap:7px;padding:3px 0;font-size:11px;}
.rl{width:100px;color:var(--mu);flex-shrink:0;}
.rv{color:var(--tx);font-family:monospace;}
.rok{color:var(--g);}.rwn{color:var(--y);}.rbd{color:var(--r);}
canvas{width:100%!important;}
::-webkit-scrollbar{width:4px;}
::-webkit-scrollbar-thumb{background:var(--bd);border-radius:2px;}
"""

    html_body = f"""
<div class="hdr">
  <div style="display:flex;align-items:center;gap:9px;">
    <span style="width:8px;height:8px;border-radius:50%;background:{mode_color};{dot_anim}display:inline-block;"></span>
    <h1 id="title-h1">{paper_tag} — Divergence Field</h1>
    {"<span style='background:rgba(210,153,34,.15);border:1px solid #d29922;border-radius:4px;padding:3px 10px;color:#d29922;font-size:11px;font-weight:600;'>📄 PAPER TRADING — Precios reales · Sin dinero real</span>" if state.paper_mode else ""}
  </div>
  <div class="hdr-meta">
    <span>🕐 Prox.vela: <b id="h-next">—</b></span>
    <span>📊 Velas: <b id="h-cv">0</b></span>
    <span>🔁 Señales: <b id="h-sg">0</b></span>
    <span>💰 Slot: <b id="h-slot">—</b></span>
    <span>⏱ <b id="h-start">—</b></span>
  </div>
</div>

<div class="layout">
<div class="col col-l">

  <div class="card cbl">
    <h2>Portfolio</h2>
    <div class="pg">
      <div class="pc"><div class="lb">Total</div><div class="vl neu" id="pf-t">$0</div></div>
      <div class="pc"><div class="lb">P&L Sesión</div><div class="vl" id="pf-pnl">+0%</div></div>
      <div class="pc"><div class="lb">USDT Libre</div><div class="vl neu" id="pf-u">$0</div></div>
      <div class="pc"><div class="lb">Posiciones</div><div class="vl neu" id="pf-ps">0</div></div>
    </div>
    <div style="margin-top:6px;" class="ib">
      <div class="kv"><span class="k">BTC precio</span><span class="v" id="pf-p">—</span></div>
      <div class="kv"><span class="k">BTC posic.</span><span class="v" id="pf-b">—</span></div>
      <div class="kv"><span class="k">P&L USDT</span><span class="v" id="pf-pu">—</span></div>
      <div class="kv"><span class="k">Slot actual</span><span class="v" id="pf-sl">—</span></div>
      <div class="kv"><span class="k">Capital base</span><span class="v" id="pf-cap">—</span></div>
    </div>
  </div>

  <div class="card cy">
    <h2>Risk Monitor</h2>
    <div class="rr"><span class="rl">Max/orden</span><span class="rv" id="r-mo">—</span></div>
    <div class="rr"><span class="rl">Max/día</span><span class="rv" id="r-md">—</span></div>
    <div class="rr"><span class="rl">Drawdown actual</span><span class="rv" id="r-dd">—</span></div>
    <div class="rr"><span class="rl">Drawdown límite</span><span class="rv">{RISK_MAX_DD_PCT}%</span></div>
    <div class="rr"><span class="rl">Slot margin</span><span class="rv">×{RISK_ORDER_MARGIN}</span></div>
  </div>

  <div class="card">
    <h2>Última Señal</h2>
    <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">
      <span class="sb sb-HOLD" id="sig-b">HOLD</span>
      <span style="font-size:10px;color:var(--mu)" id="sig-t">—</span>
    </div>
    <div class="ib">
      <div class="kv"><span class="k">Precio</span><span class="v" id="sig-p">—</span></div>
      <div class="kv"><span class="k">Motivo</span><span class="v" id="sig-r" style="max-width:145px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">—</span></div>
      <div class="kv"><span class="k">score_bot</span><span class="v" id="sig-sb">—</span></div>
      <div class="kv"><span class="k">score_top</span><span class="v" id="sig-st">—</span></div>
    </div>
  </div>

  <div class="card">
    <h2>Posiciones Abiertas</h2>
    <div id="pos-c"><div style="color:var(--mu);font-size:10px;">Sin posiciones</div></div>
  </div>

  <div class="card">
    <h2>Últimas Operaciones</h2>
    <table><thead><tr><th>Tipo</th><th>Precio</th><th>USDT</th><th>Hora</th></tr></thead>
    <tbody id="tr-body"><tr><td colspan="4" style="color:var(--mu)">—</td></tr></tbody></table>
  </div>

  <div id="wu-div" style="display:none;">
    <div style="background:rgba(210,153,34,.15);border:1px solid var(--y);border-radius:4px;
                padding:5px 9px;color:var(--y);font-size:10px;">
      ⏳ Warmup: <b id="wu-n">0</b>/{CONFIG.window_size} velas
    </div>
  </div>

</div>

<div class="col">
  <div class="card" style="height:265px;position:relative;"><h2>BTC/USDT — Precio + Señales</h2><canvas id="c-price" height="225"></canvas></div>
  <div class="card" style="height:165px;position:relative;"><h2>Score BOT (verde) · Score TOP (rojo) · Umbrales</h2><canvas id="c-score" height="125"></canvas></div>
  <div class="card" style="height:165px;position:relative;"><h2>Métricas IT Normalizadas — TE · CMI · Field · Sink</h2><canvas id="c-metr" height="125"></canvas></div>
  <div class="card" style="height:145px;position:relative;"><h2>Taker Ratio (azul) · Volumen relativo (gris)</h2><canvas id="c-takr" height="105"></canvas></div>
</div>

<div class="col col-r">

  <div class="card cbl">
    <h2>Transfer Entropy — TE(taker→precio)</h2>
    <div class="mr"><div class="ml">TE norm.</div>
      <div class="mb"><div class="mf bte" id="b-te" style="width:0%"></div><div class="mt" id="t-te" style="left:55%"></div></div>
      <div class="mv" id="v-te">0.00</div></div>
    <div class="ib" style="margin-top:4px;">
      <div class="kv"><span class="k">TE crudo</span><span class="v" id="te-r">—</span></div>
      <div class="kv"><span class="k">Estimador</span><span class="v">{CONFIG.te_estimator.value}</span></div>
      <div class="kv"><span class="k">Ventana</span><span class="v">{CONFIG.window_size} ({CONFIG.window_mode.value})</span></div>
    </div>
  </div>

  <div class="card cp">
    <h2>CMI(RSI; vol_accel | price_vs_MA{int(CONFIG.cmi_regimes)})</h2>
    <div class="mr"><div class="ml">CMI norm.</div>
      <div class="mb"><div class="mf bci" id="b-ci" style="width:0%"></div><div class="mt" style="left:55%"></div></div>
      <div class="mv" id="v-ci">0.00</div></div>
    <div class="ib" style="margin-top:4px;">
      <div class="kv"><span class="k">CMI crudo</span><span class="v" id="ci-r">—</span></div>
      <div class="kv"><span class="k">Regímenes</span><span class="v">{int(CONFIG.cmi_regimes)} ({"binario" if CONFIG.cmi_regimes==CMIRegimes.BINARY else "ternario"})</span></div>
    </div>
  </div>

  <div class="card cbl">
    <h2>Divergence Field ({CONFIG.field_def.value})</h2>
    <div class="mr"><div class="ml">|Div| norm.</div>
      <div class="mb"><div class="mf bfd" id="b-fd" style="width:0%"></div><div class="mt" style="left:55%"></div></div>
      <div class="mv" id="v-fd">0.00</div></div>
    <div class="fvr">
      <div class="fvc"><div class="fvl">Δprice</div><div class="fva" id="fa-p">→</div><div class="fvv" id="fv-p">0.000</div></div>
      <div style="color:var(--bd);font-size:14px;">⊕</div>
      <div class="fvc"><div class="fvl">Δvol</div><div class="fva" id="fa-v">→</div><div class="fvv" id="fv-v">0.000</div></div>
      <div style="color:var(--bd);font-size:14px;">⊕</div>
      <div class="fvc"><div class="fvl">curl</div><div class="fva" id="fa-c">→</div><div class="fvv" id="fv-c">0.000</div></div>
    </div>
    <div class="ib" style="margin-top:4px;font-size:10px;">
      <div class="kv"><span class="k">Patrón BOT</span><span class="v" id="pat-b">—</span></div>
      <div class="kv"><span class="k">Patrón TOP</span><span class="v" id="pat-t">—</span></div>
      <div class="kv"><span class="k">BOT espera</span><span class="v" style="color:var(--mu)">Δp↓+Δv↑</span></div>
      <div class="kv"><span class="k">TOP espera</span><span class="v" style="color:var(--mu)">Δp↑+Δv↓</span></div>
    </div>
  </div>

  <div class="card">
    <h2>Sink Condition (vol_last{CONFIG.sink_window}/vol_avg)</h2>
    <div class="mr"><div class="ml">Sink norm.</div>
      <div class="mb"><div class="mf bsk" id="b-sk" style="width:0%"></div><div class="mt" style="left:55%"></div></div>
      <div class="mv" id="v-sk">0.00</div></div>
    <div class="ib" style="margin-top:4px;">
      <div class="kv"><span class="k">Ratio crudo</span><span class="v" id="sk-r">—</span></div>
      <div class="kv"><span class="k">Modo</span><span class="v">{CONFIG.sink_mode.value}</span></div>
      <div class="kv"><span class="k">Threshold</span><span class="v">{CONFIG.sink_threshold:.2f}×</span></div>
    </div>
  </div>

  <div class="card">
    <h2>Score Compuesto Final</h2>
    <div class="mr"><div class="ml">score_bot</div>
      <div class="mb"><div class="mf bsb" id="b-sb" style="width:0%"></div>
        <div class="mt" id="t-sb" style="left:{int(CONFIG.score_threshold_bot*100)}%"></div></div>
      <div class="mv" id="v-sb">0.00</div></div>
    <div class="mr"><div class="ml">score_top</div>
      <div class="mb"><div class="mf bst" id="b-st" style="width:0%"></div>
        <div class="mt" id="t-st" style="left:{int(CONFIG.score_threshold_top*100)}%"></div></div>
      <div class="mv" id="v-st">0.00</div></div>
    <div class="ib" style="margin-top:4px;">
      <div class="kv"><span class="k">Pesos TE/CMI/Fld/Snk</span>
        <span class="v">{CONFIG.w_te:.0%}/{CONFIG.w_cmi:.0%}/{CONFIG.w_field:.0%}/{CONFIG.w_sink:.0%}</span></div>
      <div class="kv"><span class="k">Umbral BOT</span><span class="v" style="color:var(--g)">{CONFIG.score_threshold_bot:.2f}</span></div>
      <div class="kv"><span class="k">Umbral TOP</span><span class="v" style="color:var(--r)">{CONFIG.score_threshold_top:.2f}</span></div>
      <div class="kv"><span class="k">Cooldown</span><span class="v">{CONFIG.cooldown or "off"}</span></div>
    </div>
  </div>

  <div class="card">
    <h2>Config Activa</h2>
    <div id="cfg-c"></div>
  </div>

</div>
</div>
"""

    js = f"""
const D = {data};
const $ = id => document.getElementById(id);
const f2=v=>v==null?'—':Number(v).toFixed(2);
const f4=v=>v==null?'—':Number(v).toFixed(4);
const fU=v=>'$'+Number(v).toLocaleString('en-US',{{minimumFractionDigits:2,maximumFractionDigits:2}});
const fP=v=>(v>=0?'+':'')+Number(v).toFixed(2)+'%';
function bar(id,v){{const e=$(id);if(e)e.style.width=Math.round(Math.min(Math.max(v,0),1)*100)+'%';}}
function arw(v){{if(v==null)return{{a:'→',c:'neu'}};if(v>0.05)return{{a:'↑',c:'pos'}};if(v<-0.05)return{{a:'↓',c:'neg'}};return{{a:'→',c:'neu'}};}}

// Header
$('h-next').textContent   = D.next_candle_at;
$('h-cv').textContent     = D.candles_seen;
$('h-sg').textContent     = D.signals_total;
$('h-slot').textContent   = fU(D.slot_usdt);
$('h-start').textContent  = (D.started_at||'').slice(11,19)||'—';

// Portfolio
$('pf-t').textContent  = fU(D.portfolio_value);
$('pf-u').textContent  = fU(D.usdt_balance);
$('pf-ps').textContent = D.positions_count;
$('pf-p').textContent  = fU(D.last_price);
$('pf-b').textContent  = (D.btc_in_positions||0).toFixed(8)+' BTC';
$('pf-pu').textContent = (D.pnl_usdt>=0?'+':'')+fU(D.pnl_usdt);
$('pf-sl').textContent = fU(D.slot_usdt);
$('pf-cap').textContent= fU(D.capital_inicial);
const pe=$('pf-pnl');pe.textContent=fP(D.pnl_pct);
pe.className='vl '+(D.pnl_pct>0?'pos':D.pnl_pct<0?'neg':'neu');

// Risk
$('r-mo').textContent = fU(D.risk_max_order||D.slot_usdt*{RISK_ORDER_MARGIN});
$('r-md').textContent = fU(D.risk_max_daily||D.slot_usdt*{RISK_DAILY_SLOTS});
const ddA=D.capital_inicial>0?(1-D.portfolio_value/D.capital_inicial)*100:0;
const dde=$('r-dd');dde.textContent=fP(-ddA);
dde.className='rv '+(ddA<5?'rok':ddA<10?'rwn':'rbd');

// Signal
const sb=$('sig-b');sb.textContent=D.last_signal;sb.className='sb sb-'+D.last_signal;
$('sig-t').textContent = (D.last_signal_at||'').slice(11,19)||'—';
$('sig-p').textContent = D.last_signal_price?fU(D.last_signal_price):'—';
$('sig-r').textContent = D.last_signal_reason||'—';
$('sig-sb').textContent= f4(D.score_bot)+' ≥ '+D.thr_bot+'?';
$('sig-st').textContent= f4(D.score_top)+' ≥ '+D.thr_top+'?';

// Warmup
if(!D.warmup_complete){{$('wu-div').style.display='block';$('wu-n').textContent=D.candles_seen;}}

// Metric bars
bar('b-te',D.te_norm);bar('b-ci',D.cmi_norm);bar('b-fd',D.field_norm);
bar('b-sk',D.sink_norm);bar('b-sb',D.score_bot);bar('b-st',D.score_top);
$('v-te').textContent=f4(D.te_norm);$('v-ci').textContent=f4(D.cmi_norm);
$('v-fd').textContent=f4(D.field_norm);$('v-sk').textContent=f4(D.sink_norm);
$('v-sb').textContent=f4(D.score_bot);$('v-st').textContent=f4(D.score_top);
$('te-r').textContent=D.te_raw!=null?f4(D.te_raw)+' bits':'—';
$('ci-r').textContent=D.cmi_raw!=null?f4(D.cmi_raw)+' bits':'—';
$('sk-r').textContent=D.sink_raw!=null?f2(D.sink_raw)+'×':'—';

// Field vector
[['fa-p','fv-p',D.field_price_div],['fa-v','fv-v',D.field_vol_div],['fa-c','fv-c',D.field_curl]]
.forEach(([a,v,val])=>{{const i=arw(val);$(a).textContent=i.a;$(a).className='fva '+i.c;$(v).textContent=val!=null?Number(val).toFixed(3):'—';$(v).className='fvv '+i.c;}});
$('pat-b').textContent=D.is_bot_pattern?'✅ Activo':'⭕ Inactivo';
$('pat-b').style.color=D.is_bot_pattern?'var(--g)':'var(--mu)';
$('pat-t').textContent=D.is_top_pattern?'✅ Activo':'⭕ Inactivo';
$('pat-t').style.color=D.is_top_pattern?'var(--r)':'var(--mu)';

// Positions
const pc=$('pos-c');
const buys=D.trades.filter(t=>t.type==='BUY'&&!t.ignorado);
pc.innerHTML=buys.length?buys.slice(-5).map(t=>{{
  const pp=D.last_price>0?((D.last_price-t.price)/t.price*100):0;
  return `<div class="pr"><div class="prh"><span style="color:var(--g);font-weight:600">BUY @ ${{Number(t.price).toLocaleString()}}</span>
  <span class="${{pp>=0?'pos':'neg'}}">${{pp>=0?'+':''}}${{pp.toFixed(2)}}%</span></div>
  <div class="prd">${{(t.datetime||'').slice(0,16)}} ${{t.paper_trade?'[P]':''}} · ${{t.usdt_spent?'$'+Number(t.usdt_spent).toFixed(0):'—'}}</div></div>`;
}}).join(''):'<div style="color:var(--mu);font-size:10px;">Sin posiciones</div>';

// Trades
const tb=$('tr-body');
const trs=[...D.trades].reverse().slice(0,10);
tb.innerHTML=trs.length?trs.map(t=>{{
  const cls=t.ignorado?'tig':t.type==='BUY'?'tbu':'tse';
  const u=t.usdt_spent||t.usdt_received||0;
  return `<tr><td class="${{cls}}">${{t.ignorado?'✗':''}}${{t.type}}${{t.paper_trade?' [P]':''}}</td>
  <td>${{Number(t.price||0).toLocaleString('en-US',{{maximumFractionDigits:0}})}}</td>
  <td>${{u?'$'+Number(u).toFixed(0):'—'}}</td>
  <td>${{(t.datetime||'').slice(11,16)||'—'}}</td></tr>`;
}}).join(''):'<tr><td colspan="4" style="color:var(--mu)">—</td></tr>';

// Config
$('cfg-c').innerHTML=Object.entries(D.config)
  .map(([k,v])=>`<div class="cfg-r"><span class="cfg-k">${{k}}</span><span class="cfg-v">${{v}}</span></div>`).join('');

// ── GRÁFICOS ──────────────────────────────────────────────────────────────────
function rC(c){{c.width=c.parentElement.clientWidth-18;c.height=parseInt(c.getAttribute('height'));}}

// Precio
(function(){{
  const cv=$('c-price');rC(cv);
  const ctx=cv.getContext('2d'),W=cv.width,H=cv.height,cs=D.candles;
  if(!cs||cs.length<2)return;
  ctx.fillStyle='#0d1117';ctx.fillRect(0,0,W,H);
  const ps=cs.flatMap(c=>[c.high,c.low]),mn=Math.min(...ps),mx=Math.max(...ps),rng=mx-mn||1;
  const pL=46,pR=8,pT=7,pB=16;
  const py=p=>pT+(1-(p-mn)/rng)*(H-pT-pB);
  const cx=i=>pL+(i+.5)*(W-pL-pR)/cs.length;
  const cW=Math.max(2,(W-pL-pR)/cs.length-1);
  for(let g=0;g<4;g++){{
    const y=pT+g*(H-pT-pB)/3;
    ctx.strokeStyle='#21262d';ctx.lineWidth=.5;ctx.beginPath();ctx.moveTo(pL,y);ctx.lineTo(W-pR,y);ctx.stroke();
    ctx.fillStyle='#8b949e';ctx.font='9px monospace';
    ctx.fillText('$'+Math.round(mx-g*rng/3).toLocaleString(),1,y+3);
  }}
  cs.forEach((c,i)=>{{
    const x=cx(i),col=c.close>=c.open?'#3fb950':'#f85149';
    ctx.strokeStyle=col;ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(x,py(c.high));ctx.lineTo(x,py(c.low));ctx.stroke();
    const bY=py(Math.max(c.open,c.close)),bH=Math.max(1,Math.abs(py(c.open)-py(c.close)));
    ctx.fillStyle=col;ctx.fillRect(x-cW/2,bY,cW,bH);
  }});
  D.trades.forEach(t=>{{
    if(t.ignorado)return;
    const idx=D.candles.findIndex(c=>Math.abs(c.ts-t.ts)<7200);
    if(idx<0)return;
    const x=cx(idx),y=py(t.price||0);
    ctx.beginPath();
    if(t.type==='BUY'){{ctx.fillStyle='#3fb950';ctx.moveTo(x,y+9);ctx.lineTo(x-5,y+16);ctx.lineTo(x+5,y+16);}}
    else{{ctx.fillStyle='#f85149';ctx.moveTo(x,y-9);ctx.lineTo(x-5,y-16);ctx.lineTo(x+5,y-16);}}
    ctx.closePath();ctx.fill();
  }});
}})();

// Scores
(function(){{
  const cv=$('c-score');rC(cv);
  const ctx=cv.getContext('2d'),W=cv.width,H=cv.height,ms=D.metrics;
  if(!ms||ms.length<2)return;
  ctx.fillStyle='#0d1117';ctx.fillRect(0,0,W,H);
  const pL=6,pR=28,pT=5,pB=13;
  const px=i=>pL+i*(W-pL-pR)/(ms.length-1);
  const py=v=>pT+(1-Math.min(Math.max(v,0),1))*(H-pT-pB);
  [[D.thr_bot,'#3fb950'],[D.thr_top,'#f85149']].forEach(([thr,col])=>{{
    ctx.strokeStyle=col+'55';ctx.lineWidth=1;ctx.setLineDash([4,4]);
    ctx.beginPath();ctx.moveTo(pL,py(thr));ctx.lineTo(W-pR,py(thr));ctx.stroke();
    ctx.setLineDash([]);ctx.fillStyle=col+'99';ctx.font='9px monospace';
    ctx.fillText(thr.toFixed(2),W-pR+2,py(thr)+3);
  }});
  [['score_bot','#3fb950'],['score_top','#f85149']].forEach(([k,col])=>{{
    ctx.strokeStyle=col;ctx.lineWidth=2;ctx.beginPath();
    ms.forEach((m,i)=>{{const x=px(i),y=py(m[k]||0);i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);}});
    ctx.stroke();
  }});
  ctx.fillStyle='#8b949e';ctx.font='9px monospace';
  [0,Math.floor(ms.length/2),ms.length-1].forEach(i=>{{
    const ts=ms[i]&&ms[i].ts;
    if(ts)ctx.fillText(new Date(ts*1000).toLocaleTimeString('es').slice(0,5),px(i)-10,H-2);
  }});
}})();

// Métricas IT
(function(){{
  const cv=$('c-metr');rC(cv);
  const ctx=cv.getContext('2d'),W=cv.width,H=cv.height,ms=D.metrics;
  if(!ms||ms.length<2)return;
  ctx.fillStyle='#0d1117';ctx.fillRect(0,0,W,H);
  const pL=6,pR=6,pT=5,pB=18;
  const px=i=>pL+i*(W-pL-pR)/(ms.length-1);
  const py=v=>pT+(1-Math.min(Math.max(v,0),1))*(H-pT-pB);
  [['te_norm','#58a6ff','TE'],['cmi_norm','#a371f7','CMI'],
   ['field_norm','#fb8500','Field'],['sink_norm','#3fb950','Sink']].forEach(([k,col,lbl],i)=>{{
    ctx.strokeStyle=col;ctx.lineWidth=1.5;ctx.beginPath();
    ms.forEach((m,j)=>{{const x=px(j),y=py(m[k]||0);j===0?ctx.moveTo(x,y):ctx.lineTo(x,y);}});
    ctx.stroke();
    ctx.fillStyle=col;ctx.fillRect(pL+i*50,H-13,6,6);
    ctx.fillStyle='#8b949e';ctx.font='9px sans-serif';ctx.fillText(lbl,pL+i*50+8,H-5);
  }});
}})();

// Taker
(function(){{
  const cv=$('c-takr');rC(cv);
  const ctx=cv.getContext('2d'),W=cv.width,H=cv.height,cs=D.candles;
  if(!cs||cs.length<2)return;
  ctx.fillStyle='#0d1117';ctx.fillRect(0,0,W,H);
  const pL=6,pR=6,pT=5,pB=15;
  const px=i=>pL+i*(W-pL-pR)/(cs.length-1);
  const pyt=v=>pT+(1-Math.min(Math.max(v,0),1))*(H-pT-pB);
  const vls=cs.map(c=>c.vol||0),avg=vls.reduce((a,b)=>a+b,0)/vls.length||1;
  const mxVR=Math.max(...vls.map(v=>v/avg),2);
  vls.forEach((v,i)=>{{
    const x=px(i),y=pyt(v/avg/mxVR);
    ctx.fillStyle='#8b949e22';ctx.fillRect(x-2,y,4,H-pB-y);
  }});
  ctx.strokeStyle='#58a6ff';ctx.lineWidth=1.5;ctx.beginPath();
  cs.forEach((c,i)=>{{const x=px(i),y=pyt(c.taker||.5);i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);}});
  ctx.stroke();
  ctx.strokeStyle='rgba(255,255,255,.07)';ctx.lineWidth=1;ctx.setLineDash([3,3]);
  const mid=pyt(.5);ctx.beginPath();ctx.moveTo(pL,mid);ctx.lineTo(W-pR,mid);ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle='#8b949e';ctx.font='9px monospace';ctx.fillText('0.5',1,mid+3);
}})();
"""

    return (f"<!DOCTYPE html><html lang='es'><head><meta charset='UTF-8'>"
            f"<meta http-equiv='refresh' content='{DASHBOARD_REFRESH}'>"
            f"<title>{paper_tag} — Divergence Field</title>"
            f"<style>{css}</style></head><body>"
            f"{html_body}"
            f"<script>{js}</script></body></html>")


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

def _write_dashboard(state: LiveState) -> None:
    try:
        tmp = DASHBOARD_HTML + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(generate_dashboard(state))
        os.replace(tmp, DASHBOARD_HTML)
    except Exception as e:
        log.warning("error dashboard", error=str(e))

def _print_tick(state: LiveState, candle: Candle) -> None:
    pnl_s  = f"{'+' if state.pnl_pct>=0 else ''}{state.pnl_pct:.2f}%"
    pat    = "▼BOT" if state.is_bot_pattern else "▲TOP" if state.is_top_pattern else " — "
    sig_s  = {"BUY":"🟢 BUY","SELL":"🔴 SELL","HOLD":"⬜ hold"}.get(state.last_signal,"⬜")
    pm     = "[P]" if state.paper_mode else ""
    print(f"{pm}[{_now_iso()[11:]}] ${candle.close:>10,.2f} | "
          f"TE={state.te_norm:.2f} CMI={state.cmi_norm:.2f} "
          f"Fld={state.field_norm:.2f} Snk={state.sink_norm:.2f} | "
          f"sBot={state.score_bot:.3f} sTop={state.score_top:.3f} | "
          f"{pat} | {sig_s} | "
          f"${state.portfolio_value:,.2f} ({pnl_s}) | "
          f"Pos={state.positions_count} Slot=${state.slot_usdt:,.0f} | "
          f"Prox:{state.next_candle_at}")

def _make_trade_dict(t: TradeRecord, strategy: DivergenceFieldStrategy,
                     paper: bool) -> dict:
    return {
        "ts": t.ts,
        "datetime": datetime.fromtimestamp(t.ts, tz=timezone.utc)
                    .strftime("%Y-%m-%d %H:%M:%S"),
        "type": t.side, "price": t.price,
        "usdt_spent": t.usdt_spent, "usdt_received": t.usdt_received,
        "btc_bought": t.btc_bought, "btc_sold": t.btc_sold,
        "commission": t.commission, "ignorado": False, "paper_trade": paper,
        "score_bot": strategy.last_score_bot, "score_top": strategy.last_score_top,
        "te_norm": strategy.last_te_norm, "cmi_norm": strategy.last_cmi_norm,
        "field_norm": strategy.last_field_norm, "sink_norm": strategy.last_sink_norm,
    }


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    args       = _parse_args()
    paper_mode = args.paper
    # feed_real = True → feed de Binance producción (precios NO sintéticos)
    feed_real  = paper_mode and not args.testnet

    print("╔══════════════════════════════════════════════════════════╗")
    if paper_mode:
        print("║   PAPER TRADER — DivergenceField  BTC/USDT             ║")
    else:
        print("║   LIVE TRADER  — DivergenceField  BTC/USDT  Testnet    ║")
    print("╚══════════════════════════════════════════════════════════╝")
    if paper_mode:
        src = "Binance PRODUCCIÓN (real)" if feed_real else "Binance TESTNET"
        print(f"  Modo           : 📄 PAPER TRADING")
        print(f"  Feed precios   : {src}")
        print(f"  Órdenes        : SIMULADAS")
    else:
        print(f"  Modo           : 🔴 LIVE TESTNET")
    print(f"  TE estimador   : {CONFIG.te_estimator.value}")
    print(f"  Ventana        : {CONFIG.window_size} ({CONFIG.window_mode.value})")
    print(f"  Score BUY/SELL : {CONFIG.score_threshold_bot}/{CONFIG.score_threshold_top}")
    print(f"  Warmup         : {WARMUP_CANDLES} velas")
    print("─" * 60)

    live_state             = LiveState(paper_mode=paper_mode)
    live_state.config_dict = CONFIG.to_dict()

    # ── 1. Feed ───────────────────────────────────────────────────────────────
    print("[1/5] Inicializando PriceFeed...")
    if feed_real:
        original_testnet = CW.USE_TESTNET
        CW.USE_TESTNET = False
    from actors.binance_feed import BinanceWSFeed
    feed = BinanceWSFeed()
    if feed_real:
        CW.USE_TESTNET = original_testnet

    warmup_start = datetime.fromtimestamp(
        int(time.time()) - WARMUP_CANDLES * 3600, tz=timezone.utc
    ).strftime("%Y-%m-%d")

    try:
        hist = feed.get_candles(warmup_start, "now", SYMBOL)
        print(f"   ✓ {len(hist)} velas cargadas")
        if hist and len(hist) >= 5:
            ranges = [(c.high - c.low) / max(c.close, 1) * 100 for c in hist[-5:]]
            avg_rng = sum(ranges) / len(ranges)
            flag = "⚠ ANÓMALO — velas sintéticas testnet" if avg_rng > 5 else "✓ Normal"
            print(f"   ℹ Rango promedio últimas 5 velas: {avg_rng:.2f}%  [{flag}]")
            if avg_rng > 5 and not paper_mode:
                print("   💡 Consejo: usar --paper para conectar al feed real de Binance")
    except Exception as e:
        log.error("error cargando warmup", error=str(e))
        hist = []

    # ── 2. Wallet ─────────────────────────────────────────────────────────────
    print("[2/5] Inicializando Wallet...")
    Path("state").mkdir(exist_ok=True)

    if paper_mode:
        capital = args.capital if args.capital else CL.SALDO_USDT_INICIAL
        print(f"   ℹ Capital paper: ${capital:,.2f} USDT"
              + (" (--capital)" if args.capital else " (CL.SALDO_USDT_INICIAL)"))
        wallet = PaperJSONWallet(
            usdt_inicial=capital, max_posiciones=MAX_POSICIONES,
            json_path=LIVE_RESULTS_JSON,
        )
        # FIX BUG 2: capital_inicial = USDT puro (sin BTC pre-existente)
        live_state.capital_inicial = capital
    else:
        from actors.binance_wallet import BinanceWallet
        wallet = BinanceWallet.from_account(
            max_posiciones=MAX_POSICIONES, json_path=LIVE_RESULTS_JSON,
            state_path=STATE_PATH, commission_pct=COMMISSION_PCT,
        )
        # FIX BUG 2: solo USDT libre, no incluye BTC pre-existente del testnet
        live_state.capital_inicial = wallet.get_usdt_balance()
        print(f"   ✓ USDT={wallet.get_usdt_balance():.2f}  "
              f"Pos={wallet.positions_count}  Slot=${wallet.get_slot_usdt():.2f}")

    # ── 3. OrderBook + Risk (FIX BUG 1) ─────────────────────────────────────
    print("[3/5] Inicializando OrderBook y RiskManager...")
    slot = wallet.get_slot_usdt()

    if paper_mode:
        ob = SimulatedOrderBook(commission_pct=COMMISSION_PCT,
                                max_posiciones=MAX_POSICIONES)
    else:
        from actors.binance_order_book import BinanceOrderBook
        ob = BinanceOrderBook(max_posiciones=MAX_POSICIONES,
                              commission_pct=COMMISSION_PCT)

    risk_cfg = _build_risk_config(slot, live_state.capital_inicial)
    risk     = RiskManager(config=risk_cfg, usdt_inicial=live_state.capital_inicial)
    state_mgr= JSONStateManager(STATE_PATH)

    live_state.risk_max_order = risk_cfg.max_order_usdt
    live_state.risk_max_daily = risk_cfg.max_daily_loss_usdt

    print(f"   ✓ max_order=${risk_cfg.max_order_usdt:.2f}  "
          f"max_daily=${risk_cfg.max_daily_loss_usdt:.2f}  "
          f"max_dd={RISK_MAX_DD_PCT}%")

    # ── 4. Estrategia + warmup ────────────────────────────────────────────────
    print("[4/5] Iniciando estrategia y procesando warmup...")
    strategy = DivergenceFieldStrategy(CONFIG)
    strategy.on_start(wallet)
    for c in hist:
        strategy._tick(c, wallet)
    print(f"   ✓ {strategy.candles_seen} velas warmup procesadas")

    # ── 5. LiveClock ──────────────────────────────────────────────────────────
    print("[5/5] Conectando LiveClock...")
    if feed_real:
        CW.USE_TESTNET = False
    from actors.live_clock import LiveClock
    clock = LiveClock(feed=feed, symbol=SYMBOL)
    if feed_real:
        CW.USE_TESTNET = original_testnet

    print(f"   ✓ Esperando próxima vela horaria...")
    print(f"   Dashboard: {DASHBOARD_HTML}")
    print("─" * 60)
    _write_dashboard(live_state)

    stop_req = [False]
    def _stop(sig, frame):
        print("\n[STOP] Cerrando...")
        stop_req[0] = True
        clock.stop()
    signal.signal(signal.SIGINT,  _stop)
    signal.signal(signal.SIGTERM, _stop)

    # ═════════════════════════════════════════════════════════════════════════
    # LOOP PRINCIPAL
    # ═════════════════════════════════════════════════════════════════════════
    try:
        for candle in clock:
            if stop_req[0]:
                break

            signal_out = strategy._tick(candle, wallet)
            live_state.update_from_candle(candle, strategy, wallet)
            live_state.estimate_next_candle()

            if not signal_out.is_actionable:
                live_state.update_from_signal("HOLD", candle.close, "sin señal")
                _print_tick(live_state, candle)
                _write_dashboard(live_state)
                continue

            order_side = signal_out.to_order_side()
            live_state.update_from_signal(
                signal_out.side.value, signal_out.price, signal_out.reason)

            risk_reason = risk.check(order_side, signal_out.price, wallet, candle)
            if risk_reason:
                log.warning("señal bloqueada por riesgo", reason=risk_reason)
                wallet.update(TradeRecord(
                    ts=candle.ts, side=order_side.value, price=signal_out.price,
                    ignored=True, ignore_reason=f"RISK:{risk_reason}",
                ))
                live_state.update_from_signal("HOLD", candle.close,
                                               f"BLOQUEADO: {risk_reason}")
                _print_tick(live_state, candle)
                _write_dashboard(live_state)
                continue

            order = ob.execute_with_guards(
                order_side, signal_out.price, wallet, candle_ts=candle.ts)

            if order.is_filled and order.trade:
                live_state.add_trade(
                    _make_trade_dict(order.trade, strategy, paper_mode))
                log.info("orden ejecutada", side=order.trade.side,
                         price=f"{order.trade.price:.2f}", paper=paper_mode)
            elif order.is_ignored:
                live_state.add_trade({
                    "ts": candle.ts, "datetime": _now_iso(),
                    "type": order_side.value, "price": signal_out.price,
                    "ignorado": True, "motivo": order.reject_reason,
                    "paper_trade": paper_mode,
                })
                live_state.update_from_signal("HOLD", candle.close,
                                               f"IGN:{order.reject_reason}")

            live_state.update_from_candle(candle, strategy, wallet)
            live_state.estimate_next_candle()
            risk.update_peak(wallet.portfolio_value(candle.close))

            if not paper_mode:
                state_mgr.save(Checkpoint.from_wallet(
                    wallet, candle.ts, candle.close,
                    metadata={"estrategia": strategy.name}))

            _print_tick(live_state, candle)
            _write_dashboard(live_state)

    except Exception as e:
        log.error("error en loop", error=str(e))
        traceback.print_exc()
    finally:
        strategy.on_stop(wallet)
        precio_final = live_state.last_price or 0.0
        port_final   = wallet.portfolio_value(precio_final)
        pnl_pct      = ((port_final / live_state.capital_inicial - 1) * 100
                        if live_state.capital_inicial > 0 else 0.0)
        wallet.flush({
            "estrategia": strategy.name, "paper_mode": paper_mode,
            "started_at": live_state.started_at, "stopped_at": _now_iso(),
            "capital_inicial": live_state.capital_inicial,
            "portfolio_final": round(port_final, 4),
            "pnl_pct": round(pnl_pct, 4),
            "candles": live_state.candles_seen,
            "signals": live_state.signals_total,
            "parametros": CONFIG.to_dict(),
        })
        _write_dashboard(live_state)
        clock.stop()
        print(f"\n  Portfolio: ${port_final:,.2f} | PnL: {'+' if pnl_pct>=0 else ''}{pnl_pct:.2f}%")
        print(f"  Velas: {live_state.candles_seen} | Señales: {live_state.signals_total}")
        print(f"  JSON: {LIVE_RESULTS_JSON}")
        print("✓ Cerrado limpiamente.")


if __name__ == "__main__":
    main()
