"""
live_divergence_field.py — Live Trader: DivergenceFieldStrategy en Binance Testnet
═══════════════════════════════════════════════════════════════════════════════════
Ejecuta DivergenceFieldStrategy en tiempo real contra Binance Testnet (o producción).

Arquitectura idéntica al resto del sistema:
  BinanceWSFeed  →  LiveClock  →  DivergenceFieldStrategy  →  BinanceOrderBook
  BinanceWallet  ←  TradeRecord  ←────────────────────────────────────────────┘

Flujo de arranque:
  1. Verificar credenciales y conectividad REST
  2. Medir desfase de reloj vs Binance
  3. Cargar ~{WARMUP_CANDLES} velas históricas (warmup del modelo)
  4. Inicializar actores (Wallet reconciliada con checkpoint)
  5. Conectar WebSocket y arrancar LiveClock
  6. Loop: tick → señal → ejecución → dashboard → checkpoint

Dashboard HTML:
  Se regenera en cada vela. Abrir live_divfield_dashboard.html en el browser.
  Auto-refresh cada {DASHBOARD_REFRESH}s. Muestra todos los indicadores de
  teoría de la información + estado del portfolio en tiempo real.

Uso:
  python live_divergence_field.py

Detener limpiamente:
  Ctrl+C  →  cierra WS, guarda estado, escribe JSON final.

Configuración:
  · USE_TESTNET en config_world.py (True = testnet, False = producción REAL)
  · CONFIG al inicio de este archivo (DFConfig)
  · Los umbrales óptimos se obtienen corriendo backtest_divergence_field.py --deep-grid
"""

from __future__ import annotations

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

import config_local as CL

from actors.binance_feed       import BinanceWSFeed
from actors.binance_wallet     import BinanceWallet
from actors.binance_order_book import BinanceOrderBook
from actors.live_clock         import LiveClock
from actors.order_book         import OrderSide
from actors.wallet             import TradeRecord
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
# CONFIG — Pegar aquí el resultado del deep-grid optimizer
# ══════════════════════════════════════════════════════════════════════════════

CONFIG = DFConfig(
    te_estimator        = TEEstimator.KDE,
    window_mode         = WindowMode.FIXED,
    window_size         = 10,
    field_def           = FieldDefinition.JACOBIAN,
    cmi_regimes         = CMIRegimes.TERNARY,
    threshold_mode      = ThresholdMode.ADAPTIVE_PERCENTILE,
    sink_mode           = SinkMode.FILTER_AND,
    score_threshold_bot = 0.55,
    score_threshold_top = 0.55,
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

# ── Parámetros del live trader ────────────────────────────────────────────────
MAX_POSICIONES    = CL.MAX_POSICIONES
COMMISSION_PCT    = CL.COMMISSION_PCT
SYMBOL            = CL.SYMBOL
WARMUP_CANDLES    = max(CONFIG.window_size * 3, 60)  # velas históricas para warmup
LIVE_RESULTS_JSON = "live_divfield_results.json"
STATE_PATH        = "state/live_divfield_state.jsonl"
DASHBOARD_HTML    = "live_divfield_dashboard.html"
DASHBOARD_REFRESH = 10      # segundos entre auto-refresh del HTML
CHART_CANDLES     = 48      # velas en el gráfico de precio (~2 días)
LOG_EVERY_N       = 1       # loggear métricas cada N velas (1 = siempre)


# ══════════════════════════════════════════════════════════════════════════════
# ESTADO GLOBAL DEL LIVE TRADER
# (se actualiza en cada tick y se vuelca al dashboard)
# ══════════════════════════════════════════════════════════════════════════════

class LiveState:
    """
    Objeto central que agrega toda la información relevante del trader
    y la expone al generador del dashboard.
    """
    def __init__(self):
        self.started_at:      str   = _now_iso()
        self.last_tick_at:    str   = "—"
        self.next_candle_at:  str   = "—"
        self.candles_seen:    int   = 0
        self.signals_total:   int   = 0

        # Métricas de teoría de la información (última vela)
        self.te_raw:          Optional[float] = None
        self.te_norm:         float = 0.0
        self.cmi_raw:         Optional[float] = None
        self.cmi_norm:        float = 0.0
        self.field_price_div: Optional[float] = None
        self.field_vol_div:   Optional[float] = None
        self.field_curl:      Optional[float] = None
        self.field_norm:      float = 0.0
        self.sink_raw:        Optional[float] = None
        self.sink_norm:       float = 0.0
        self.is_bot_pattern:  bool  = False
        self.is_top_pattern:  bool  = False

        # Score y señal
        self.score_bot:       float = 0.0
        self.score_top:       float = 0.0
        self.last_signal:     str   = "HOLD"
        self.last_signal_price: Optional[float] = None
        self.last_signal_at:  str   = "—"
        self.last_signal_reason: str = "—"

        # Precio y portfolio
        self.last_price:      float = 0.0
        self.portfolio_value: float = 0.0
        self.usdt_balance:    float = 0.0
        self.btc_in_positions: float = 0.0
        self.positions_count: int   = 0
        self.pnl_pct:         float = 0.0
        self.pnl_usdt:        float = 0.0
        self.capital_inicial: float = 0.0

        # Historial de velas y trades (para el gráfico)
        self.candle_history:  List[dict] = []   # últimas CHART_CANDLES velas
        self.trade_history:   List[dict] = []   # últimas 20 operaciones
        self.metric_history:  List[dict] = []   # últimas 48 velas de métricas

        # Config visible
        self.config_dict:     dict = {}
        self.warmup_complete: bool = False

    def update_from_candle(self, candle, strategy, wallet) -> None:
        """Actualiza el estado con los datos de la vela más reciente."""
        self.last_tick_at   = _now_iso()
        self.candles_seen   = strategy.candles_seen
        self.last_price     = candle.close
        self.portfolio_value= wallet.portfolio_value(candle.close)
        self.usdt_balance   = wallet.get_usdt_balance()
        self.btc_in_positions = wallet.btc_en_posiciones()
        self.positions_count  = wallet.positions_count
        self.pnl_usdt       = self.portfolio_value - self.capital_inicial
        self.pnl_pct        = (self.pnl_usdt / self.capital_inicial * 100.0
                               if self.capital_inicial > 0 else 0.0)

        # Métricas IT
        self.te_raw          = strategy.last_te
        self.te_norm         = strategy.last_te_norm
        self.cmi_raw         = strategy.last_cmi
        self.cmi_norm        = strategy.last_cmi_norm
        self.field_price_div = strategy.last_field_price
        self.field_vol_div   = strategy.last_field_vol
        self.field_curl      = strategy.last_field_curl
        self.field_norm      = strategy.last_field_norm
        self.sink_raw        = strategy.last_sink
        self.sink_norm       = strategy.last_sink_norm
        self.score_bot       = strategy.last_score_bot
        self.score_top       = strategy.last_score_top
        self.is_bot_pattern  = strategy.last_is_bot_pattern
        self.is_top_pattern  = strategy.last_is_top_pattern
        self.warmup_complete = strategy.candles_seen >= strategy.cfg.window_size

        # Historial de velas
        self.candle_history.append({
            "ts":    candle.ts,
            "open":  candle.open,
            "high":  candle.high,
            "low":   candle.low,
            "close": candle.close,
            "vol":   candle.volume,
            "taker": (candle.taker_buy_base_vol / candle.volume
                      if candle.taker_buy_base_vol and candle.volume > 0 else 0.5),
        })
        if len(self.candle_history) > CHART_CANDLES:
            self.candle_history.pop(0)

        # Historial de métricas
        self.metric_history.append({
            "ts":        candle.ts,
            "te_norm":   self.te_norm,
            "cmi_norm":  self.cmi_norm,
            "field_norm":self.field_norm,
            "sink_norm": self.sink_norm,
            "score_bot": self.score_bot,
            "score_top": self.score_top,
            "is_bot":    self.is_bot_pattern,
            "is_top":    self.is_top_pattern,
        })
        if len(self.metric_history) > CHART_CANDLES:
            self.metric_history.pop(0)

    def update_from_signal(self, signal_type: str, price: float, reason: str) -> None:
        self.last_signal        = signal_type
        self.last_signal_price  = price
        self.last_signal_at     = _now_iso()
        self.last_signal_reason = reason
        if signal_type != "HOLD":
            self.signals_total += 1

    def add_trade(self, trade_dict: dict) -> None:
        self.trade_history.append(trade_dict)
        if len(self.trade_history) > 20:
            self.trade_history.pop(0)

    def estimate_next_candle(self) -> None:
        """Estima el timestamp de cierre de la vela actual (horaria)."""
        now_s   = int(time.time())
        next_h  = ((now_s // 3600) + 1) * 3600
        minutes = (next_h - now_s) // 60
        seconds = (next_h - now_s) % 60
        self.next_candle_at = f"{minutes}m {seconds:02d}s"


# ══════════════════════════════════════════════════════════════════════════════
# DASHBOARD HTML
# ══════════════════════════════════════════════════════════════════════════════

def generate_dashboard(state: LiveState) -> str:
    """
    Genera el HTML completo del dashboard.
    Todos los datos están embebidos como JSON para que el JS los renderice.
    """
    data_json = json.dumps({
        "started_at":        state.started_at,
        "last_tick_at":      state.last_tick_at,
        "next_candle_at":    state.next_candle_at,
        "candles_seen":      state.candles_seen,
        "signals_total":     state.signals_total,
        "warmup_complete":   state.warmup_complete,
        "te_raw":            state.te_raw,
        "te_norm":           round(state.te_norm,  4),
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
        "pnl_pct":           round(state.pnl_pct, 2),
        "pnl_usdt":          round(state.pnl_usdt, 2),
        "capital_inicial":   state.capital_inicial,
        "thr_bot":           CONFIG.score_threshold_bot,
        "thr_top":           CONFIG.score_threshold_top,
        "sink_threshold":    CONFIG.sink_threshold,
        "candles":           state.candle_history,
        "metrics":           state.metric_history,
        "trades":            state.trade_history[-10:],
        "config":            state.config_dict,
        "refresh_s":         DASHBOARD_REFRESH,
    }, default=str)

    return f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta http-equiv="refresh" content="{DASHBOARD_REFRESH}">
<title>Divergence Field — Live Dashboard</title>
<style>
  :root {{
    --bg:      #0d1117; --bg2: #161b22; --bg3: #21262d;
    --border:  #30363d; --text: #e6edf3; --muted: #8b949e;
    --green:   #3fb950; --red: #f85149; --yellow: #d29922;
    --blue:    #58a6ff; --purple: #a371f7; --orange: #fb8500;
    --cyan:    #39d353; --teal: #1f6feb;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: var(--bg); color: var(--text); font-family: 'Segoe UI', monospace; font-size: 13px; }}
  h1   {{ font-size: 16px; font-weight: 600; color: var(--blue); }}
  h2   {{ font-size: 12px; font-weight: 600; color: var(--muted); text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px; }}

  .header {{ background: var(--bg2); border-bottom: 1px solid var(--border);
             padding: 10px 20px; display: flex; align-items: center; justify-content: space-between; }}
  .header-meta {{ display: flex; gap: 20px; font-size: 11px; color: var(--muted); }}
  .header-meta span {{ display: flex; align-items: center; gap: 4px; }}
  .dot-live {{ width: 8px; height: 8px; border-radius: 50%; background: var(--green);
               animation: pulse 1.5s infinite; display: inline-block; }}
  @keyframes pulse {{ 0%,100% {{ opacity:1; }} 50% {{ opacity:0.3; }} }}

  .layout {{ display: grid; grid-template-columns: 280px 1fr 280px; gap: 0; height: calc(100vh - 47px); overflow: hidden; }}
  .col-left  {{ border-right: 1px solid var(--border); overflow-y: auto; padding: 12px; display: flex; flex-direction: column; gap: 10px; }}
  .col-mid   {{ overflow-y: auto; padding: 12px; display: flex; flex-direction: column; gap: 10px; }}
  .col-right {{ border-left: 1px solid var(--border); overflow-y: auto; padding: 12px; display: flex; flex-direction: column; gap: 10px; }}

  .card {{ background: var(--bg2); border: 1px solid var(--border); border-radius: 6px; padding: 12px; }}
  .card-accent-bot {{ border-left: 3px solid var(--green); }}
  .card-accent-top {{ border-left: 3px solid var(--red); }}
  .card-accent-blue{{ border-left: 3px solid var(--blue); }}
  .card-accent-purple{{ border-left: 3px solid var(--purple); }}

  /* Portfolio cards */
  .pf-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }}
  .pf-card  {{ background: var(--bg3); border: 1px solid var(--border); border-radius: 4px;
               padding: 8px 10px; }}
  .pf-card .label {{ font-size: 10px; color: var(--muted); margin-bottom: 2px; }}
  .pf-card .value {{ font-size: 16px; font-weight: 700; }}
  .pf-card .value.pos {{ color: var(--green); }}
  .pf-card .value.neg {{ color: var(--red); }}
  .pf-card .value.neu {{ color: var(--text); }}

  /* Gauge bars */
  .metric-row {{ display: flex; align-items: center; gap: 8px; margin-bottom: 6px; }}
  .metric-label {{ width: 80px; font-size: 11px; color: var(--muted); flex-shrink: 0; }}
  .metric-bar-bg {{ flex: 1; background: var(--bg3); border-radius: 3px; height: 14px; position: relative; overflow: hidden; }}
  .metric-bar-fill {{ height: 100%; border-radius: 3px; transition: width 0.5s ease; }}
  .metric-val {{ width: 44px; text-align: right; font-size: 11px; font-family: monospace; flex-shrink: 0; }}
  .metric-threshold {{ position: absolute; top: 0; bottom: 0; width: 2px; background: rgba(255,255,255,0.4); }}

  .bar-te    {{ background: linear-gradient(90deg, #1f6feb, #58a6ff); }}
  .bar-cmi   {{ background: linear-gradient(90deg, #6e40c9, #a371f7); }}
  .bar-field {{ background: linear-gradient(90deg, #fb8500, #ffd166); }}
  .bar-sink  {{ background: linear-gradient(90deg, #0e9c6c, #3fb950); }}
  .bar-sbot  {{ background: linear-gradient(90deg, #0e9c6c, #3fb950); }}
  .bar-stop  {{ background: linear-gradient(90deg, #b31d28, #f85149); }}

  /* Pattern indicator */
  .pattern-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }}
  .pattern-card {{ background: var(--bg3); border: 1px solid var(--border); border-radius: 4px;
                   padding: 8px; text-align: center; }}
  .pattern-card.active-bot {{ border-color: var(--green); background: rgba(63,185,80,0.08); }}
  .pattern-card.active-top {{ border-color: var(--red);   background: rgba(248,81,73,0.08);  }}
  .pattern-label {{ font-size: 10px; color: var(--muted); margin-bottom: 4px; }}
  .pattern-icon  {{ font-size: 22px; }}
  .pattern-desc  {{ font-size: 9px; color: var(--muted); margin-top: 2px; }}

  /* Signal badge */
  .signal-badge {{ display: inline-block; padding: 4px 12px; border-radius: 12px;
                   font-size: 12px; font-weight: 700; letter-spacing: 1px; }}
  .signal-BUY  {{ background: rgba(63,185,80,0.2);  color: var(--green); border: 1px solid var(--green); }}
  .signal-SELL {{ background: rgba(248,81,73,0.2);  color: var(--red);   border: 1px solid var(--red);   }}
  .signal-HOLD {{ background: rgba(139,148,158,0.1); color: var(--muted); border: 1px solid var(--border); }}

  /* Field vector display */
  .field-vector {{ display: flex; align-items: center; justify-content: space-around;
                   background: var(--bg3); border-radius: 4px; padding: 8px; margin-top: 6px; }}
  .fv-comp {{ text-align: center; }}
  .fv-label {{ font-size: 9px; color: var(--muted); }}
  .fv-value {{ font-size: 15px; font-weight: 700; margin: 2px 0; }}
  .fv-arrow {{ font-size: 18px; }}
  .pos {{ color: var(--green); }}
  .neg {{ color: var(--red);   }}
  .neu {{ color: var(--muted); }}

  /* Charts */
  canvas {{ width: 100% !important; }}
  .chart-container {{ position: relative; }}

  /* Trade table */
  table {{ width: 100%; border-collapse: collapse; font-size: 11px; }}
  th    {{ text-align: left; color: var(--muted); padding: 4px 6px; border-bottom: 1px solid var(--border); font-weight: 400; }}
  td    {{ padding: 3px 6px; border-bottom: 1px solid rgba(48,54,61,0.5); }}
  tr:last-child td {{ border-bottom: none; }}
  .td-buy  {{ color: var(--green); font-weight: 600; }}
  .td-sell {{ color: var(--red);   font-weight: 600; }}
  .td-ign  {{ color: var(--muted); }}

  /* Config table */
  .cfg-row {{ display: flex; justify-content: space-between; padding: 3px 0;
              border-bottom: 1px solid rgba(48,54,61,0.3); font-size: 11px; }}
  .cfg-key {{ color: var(--muted); }}
  .cfg-val {{ color: var(--blue); font-family: monospace; }}

  /* Positions */
  .pos-row {{ background: var(--bg3); border: 1px solid var(--border); border-radius: 4px;
              padding: 8px 10px; margin-bottom: 6px; }}
  .pos-header {{ display: flex; justify-content: space-between; align-items: center; }}
  .pos-detail {{ font-size: 10px; color: var(--muted); margin-top: 3px; }}

  /* Warmup overlay */
  .warmup-badge {{ background: rgba(210,153,34,0.15); border: 1px solid var(--yellow);
                   border-radius: 4px; padding: 6px 10px; color: var(--yellow); font-size: 11px;
                   display: flex; align-items: center; gap: 6px; }}

  /* Info tooltip area */
  .info-box {{ background: rgba(88,166,255,0.05); border: 1px solid rgba(88,166,255,0.2);
               border-radius: 4px; padding: 8px; font-size: 10px; color: var(--muted); line-height: 1.6; }}
  .kv {{ display: flex; justify-content: space-between; }}
  .kv .k {{ color: var(--muted); }}
  .kv .v {{ color: var(--text); font-family: monospace; }}

  ::-webkit-scrollbar {{ width: 4px; }} ::-webkit-scrollbar-track {{ background: var(--bg); }}
  ::-webkit-scrollbar-thumb {{ background: var(--border); border-radius: 2px; }}
</style>
</head>
<body>
<div class="header">
  <div style="display:flex;align-items:center;gap:10px;">
    <span class="dot-live"></span>
    <h1>Divergence Field — Live Trader</h1>
  </div>
  <div class="header-meta">
    <span>🕐 Próxima vela: <b id="hdr-next">—</b></span>
    <span>📊 Velas procesadas: <b id="hdr-candles">0</b></span>
    <span>🔁 Señales emitidas: <b id="hdr-signals">0</b></span>
    <span>⏱ Iniciado: <b id="hdr-started">—</b></span>
    <span style="color:var(--muted)">Refresh: {DASHBOARD_REFRESH}s</span>
  </div>
</div>

<div class="layout">

  <!-- ══ COLUMNA IZQUIERDA: Portfolio + Señal + Posiciones ══ -->
  <div class="col-left">

    <!-- Portfolio -->
    <div class="card card-accent-blue">
      <h2>Portfolio</h2>
      <div class="pf-grid">
        <div class="pf-card">
          <div class="label">Portfolio Total</div>
          <div class="value neu" id="pf-total">$0.00</div>
        </div>
        <div class="pf-card">
          <div class="label">P&L Sesión</div>
          <div class="value" id="pf-pnl">+0.00%</div>
        </div>
        <div class="pf-card">
          <div class="label">USDT Libre</div>
          <div class="value neu" id="pf-usdt">$0.00</div>
        </div>
        <div class="pf-card">
          <div class="label">Posiciones</div>
          <div class="value neu" id="pf-pos">0</div>
        </div>
      </div>
      <div style="margin-top:8px;" class="info-box">
        <div class="kv"><span class="k">Precio BTC</span><span class="v" id="pf-price">—</span></div>
        <div class="kv"><span class="k">BTC en posic.</span><span class="v" id="pf-btc">—</span></div>
        <div class="kv"><span class="k">P&L USDT</span><span class="v" id="pf-pnl-usdt">—</span></div>
        <div class="kv"><span class="k">Capital inicial</span><span class="v" id="pf-cap-ini">—</span></div>
      </div>
    </div>

    <!-- Señal activa -->
    <div class="card" id="signal-card">
      <h2>Última Señal</h2>
      <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;">
        <span class="signal-badge signal-HOLD" id="sig-badge">HOLD</span>
        <span style="font-size:11px;color:var(--muted)" id="sig-at">—</span>
      </div>
      <div class="info-box" style="font-size:10px;">
        <div class="kv"><span class="k">Precio</span><span class="v" id="sig-price">—</span></div>
        <div class="kv"><span class="k">Motivo</span><span class="v" id="sig-reason" style="max-width:160px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">—</span></div>
        <div class="kv"><span class="k">score_bot</span><span class="v" id="sig-sbot">—</span></div>
        <div class="kv"><span class="k">score_top</span><span class="v" id="sig-stop">—</span></div>
      </div>
    </div>

    <!-- Posiciones abiertas -->
    <div class="card">
      <h2>Posiciones Abiertas</h2>
      <div id="positions-container">
        <div style="color:var(--muted);font-size:11px;">Sin posiciones</div>
      </div>
    </div>

    <!-- Warmup -->
    <div id="warmup-container" style="display:none;">
      <div class="warmup-badge">
        ⏳ Warmup en progreso — acumulando <b id="warmup-progress">0</b>/<b>{CONFIG.window_size}</b> velas
      </div>
    </div>

    <!-- Últimas operaciones -->
    <div class="card">
      <h2>Últimas Operaciones</h2>
      <table>
        <thead><tr>
          <th>Tipo</th><th>Precio</th><th>USDT</th><th>Hora</th>
        </tr></thead>
        <tbody id="trades-body">
          <tr><td colspan="4" style="color:var(--muted)">Sin operaciones</td></tr>
        </tbody>
      </table>
    </div>

  </div>

  <!-- ══ COLUMNA CENTRAL: Gráficos ══ -->
  <div class="col-mid">

    <!-- Gráfico precio + señales -->
    <div class="card chart-container" style="height:280px;">
      <h2>BTC/USDT — Precio + Señales</h2>
      <canvas id="chart-price" height="240"></canvas>
    </div>

    <!-- Gráfico scores -->
    <div class="card chart-container" style="height:180px;">
      <h2>Score BOT (verde) vs Score TOP (rojo) + Umbrales</h2>
      <canvas id="chart-scores" height="140"></canvas>
    </div>

    <!-- Gráfico métricas IT normalizadas -->
    <div class="card chart-container" style="height:180px;">
      <h2>Métricas IT Normalizadas — TE · CMI · Field · Sink</h2>
      <canvas id="chart-metrics" height="140"></canvas>
    </div>

    <!-- Gráfico taker ratio + volumen relativo -->
    <div class="card chart-container" style="height:160px;">
      <h2>Taker Ratio (azul) + Vol. Relativo al Promedio (gris)</h2>
      <canvas id="chart-taker" height="120"></canvas>
    </div>

  </div>

  <!-- ══ COLUMNA DERECHA: Métricas IT detalle + Config ══ -->
  <div class="col-right">

    <!-- Transfer Entropy -->
    <div class="card card-accent-blue">
      <h2>Transfer Entropy  TE(taker→precio)</h2>
      <div class="metric-row">
        <div class="metric-label">TE norm.</div>
        <div class="metric-bar-bg">
          <div class="metric-bar-fill bar-te" id="bar-te" style="width:0%"></div>
          <div class="metric-threshold" id="thr-te" style="left:55%"></div>
        </div>
        <div class="metric-val" id="val-te">0.00</div>
      </div>
      <div class="info-box" style="font-size:10px;margin-top:6px;">
        <div class="kv"><span class="k">TE crudo (bits)</span><span class="v" id="te-raw-val">—</span></div>
        <div class="kv"><span class="k">Estimador</span><span class="v" id="te-estimator">{CONFIG.te_estimator.value}</span></div>
        <div class="kv"><span class="k">Ventana</span><span class="v">{CONFIG.window_size} ({CONFIG.window_mode.value})</span></div>
        <div class="kv"><span class="k">Interpretación</span><span class="v">↑ = vol causa precio</span></div>
      </div>
    </div>

    <!-- CMI -->
    <div class="card card-accent-purple">
      <h2>CMI(RSI; vol_accel | price_vs_MA{int(CONFIG.cmi_regimes)})</h2>
      <div class="metric-row">
        <div class="metric-label">CMI norm.</div>
        <div class="metric-bar-bg">
          <div class="metric-bar-fill bar-cmi" id="bar-cmi" style="width:0%"></div>
          <div class="metric-threshold" id="thr-cmi" style="left:55%"></div>
        </div>
        <div class="metric-val" id="val-cmi">0.00</div>
      </div>
      <div class="info-box" style="font-size:10px;margin-top:6px;">
        <div class="kv"><span class="k">CMI crudo (bits)</span><span class="v" id="cmi-raw-val">—</span></div>
        <div class="kv"><span class="k">Regímenes MA</span><span class="v">{int(CONFIG.cmi_regimes)} ({"binario" if CONFIG.cmi_regimes==CMIRegimes.BINARY else "ternario"})</span></div>
        <div class="kv"><span class="k">Interpretación</span><span class="v">↑ = RSI/vol acoplados</span></div>
      </div>
    </div>

    <!-- Divergence Field -->
    <div class="card card-accent-blue" id="field-card">
      <h2>Divergence Field ({CONFIG.field_def.value})</h2>
      <div class="metric-row">
        <div class="metric-label">|Div| norm.</div>
        <div class="metric-bar-bg">
          <div class="metric-bar-fill bar-field" id="bar-field" style="width:0%"></div>
          <div class="metric-threshold" id="thr-field" style="left:55%"></div>
        </div>
        <div class="metric-val" id="val-field">0.00</div>
      </div>
      <div class="field-vector">
        <div class="fv-comp">
          <div class="fv-label">Δprice</div>
          <div class="fv-arrow" id="fv-price-arrow">→</div>
          <div class="fv-value" id="fv-price-val">0.00</div>
        </div>
        <div style="color:var(--border);font-size:18px;">⊕</div>
        <div class="fv-comp">
          <div class="fv-label">Δvol</div>
          <div class="fv-arrow" id="fv-vol-arrow">→</div>
          <div class="fv-value" id="fv-vol-val">0.00</div>
        </div>
        <div style="color:var(--border);font-size:18px;">⊕</div>
        <div class="fv-comp">
          <div class="fv-label">curl</div>
          <div class="fv-arrow" id="fv-curl-arrow">→</div>
          <div class="fv-value" id="fv-curl-val">0.00</div>
        </div>
      </div>
      <div style="margin-top:6px;" class="info-box" style="font-size:10px;">
        <div class="kv" style="font-size:10px;"><span class="k">BOTTOM pattern</span><span class="v" id="pattern-bot-lbl">—</span></div>
        <div class="kv" style="font-size:10px;"><span class="k">TOP pattern</span><span class="v" id="pattern-top-lbl">—</span></div>
        <div class="kv" style="font-size:10px;"><span class="k">Esperado BOTTOM</span><span class="v" style="color:var(--muted)">Δprice↓ + Δvol↑</span></div>
        <div class="kv" style="font-size:10px;"><span class="k">Esperado TOP</span><span class="v" style="color:var(--muted)">Δprice↑ + Δvol↓</span></div>
      </div>
    </div>

    <!-- Sink -->
    <div class="card">
      <h2>Sink Condition  (vol_last{CONFIG.sink_window} / vol_avg)</h2>
      <div class="metric-row">
        <div class="metric-label">Sink norm.</div>
        <div class="metric-bar-bg">
          <div class="metric-bar-fill bar-sink" id="bar-sink" style="width:0%"></div>
          <div class="metric-threshold" id="thr-sink" style="left:55%"></div>
        </div>
        <div class="metric-val" id="val-sink">0.00</div>
      </div>
      <div class="info-box" style="font-size:10px;margin-top:6px;">
        <div class="kv"><span class="k">Ratio crudo</span><span class="v" id="sink-raw-val">—</span></div>
        <div class="kv"><span class="k">Modo</span><span class="v">{CONFIG.sink_mode.value}</span></div>
        <div class="kv"><span class="k">Threshold</span><span class="v">{CONFIG.sink_threshold:.2f}x</span></div>
        <div class="kv"><span class="k">Interpretación</span><span class="v">>1 = vol activo</span></div>
      </div>
    </div>

    <!-- Score compuesto -->
    <div class="card">
      <h2>Score Compuesto Final</h2>
      <div class="metric-row">
        <div class="metric-label">score_bot</div>
        <div class="metric-bar-bg">
          <div class="metric-bar-fill bar-sbot" id="bar-sbot" style="width:0%"></div>
          <div class="metric-threshold" id="thr-sbot" style="left:{int(CONFIG.score_threshold_bot*100)}%"></div>
        </div>
        <div class="metric-val" id="val-sbot">0.00</div>
      </div>
      <div class="metric-row">
        <div class="metric-label">score_top</div>
        <div class="metric-bar-bg">
          <div class="metric-bar-fill bar-stop" id="bar-stop" style="width:0%"></div>
          <div class="metric-threshold" id="thr-stop" style="left:{int(CONFIG.score_threshold_top*100)}%"></div>
        </div>
        <div class="metric-val" id="val-stop">0.00</div>
      </div>
      <div class="info-box" style="font-size:10px;margin-top:6px;">
        <div class="kv"><span class="k">Pesos (TE/CMI/Field/Sink)</span>
          <span class="v">{CONFIG.w_te:.0%}/{CONFIG.w_cmi:.0%}/{CONFIG.w_field:.0%}/{CONFIG.w_sink:.0%}</span></div>
        <div class="kv"><span class="k">Umbral BOT</span><span class="v" style="color:var(--green)">{CONFIG.score_threshold_bot:.2f}</span></div>
        <div class="kv"><span class="k">Umbral TOP</span><span class="v" style="color:var(--red)">{CONFIG.score_threshold_top:.2f}</span></div>
        <div class="kv"><span class="k">Cooldown</span><span class="v">{CONFIG.cooldown if CONFIG.cooldown else "off"}</span></div>
      </div>
    </div>

    <!-- Config completa -->
    <div class="card">
      <h2>Configuración Activa</h2>
      <div id="config-container"></div>
    </div>

  </div>
</div>

<script>
const D = {data_json};

// ── Helpers ───────────────────────────────────────────────────────────────────
const $ = id => document.getElementById(id);
const fmt2  = v => v == null ? '—' : Number(v).toFixed(2);
const fmt4  = v => v == null ? '—' : Number(v).toFixed(4);
const fmtUSD= v => '$' + Number(v).toLocaleString('en-US', {{minimumFractionDigits:2, maximumFractionDigits:2}});
const fmtPct= v => (v >= 0 ? '+' : '') + Number(v).toFixed(2) + '%';
const fmtTS = ts => ts ? new Date(ts * 1000).toLocaleTimeString('es') : '—';

function setBar(id, pct, min=0, max=1) {{
  const el = $(id);
  if (el) el.style.width = Math.round(Math.min(Math.max((pct - min) / (max - min), 0), 1) * 100) + '%';
}}

function arrowSign(val) {{
  if (val == null) return {{arrow:'→', cls:'neu'}};
  if (val  >  0.05) return {{arrow:'↑', cls:'pos'}};
  if (val  < -0.05) return {{arrow:'↓', cls:'neg'}};
  return {{arrow:'→', cls:'neu'}};
}}

// ── Header ────────────────────────────────────────────────────────────────────
$('hdr-next').textContent    = D.next_candle_at;
$('hdr-candles').textContent = D.candles_seen;
$('hdr-signals').textContent = D.signals_total;
$('hdr-started').textContent = D.started_at.slice(11,19) || '—';

// ── Portfolio ─────────────────────────────────────────────────────────────────
$('pf-total').textContent  = fmtUSD(D.portfolio_value);
$('pf-usdt').textContent   = fmtUSD(D.usdt_balance);
$('pf-pos').textContent    = D.positions_count;
$('pf-price').textContent  = fmtUSD(D.last_price);
$('pf-btc').textContent    = (D.btc_in_positions || 0).toFixed(8) + ' BTC';
$('pf-pnl-usdt').textContent = (D.pnl_usdt >= 0 ? '+' : '') + fmtUSD(D.pnl_usdt);
$('pf-cap-ini').textContent = fmtUSD(D.capital_inicial);
const pnlEl = $('pf-pnl');
pnlEl.textContent = fmtPct(D.pnl_pct);
pnlEl.className   = 'value ' + (D.pnl_pct > 0 ? 'pos' : D.pnl_pct < 0 ? 'neg' : 'neu');

// ── Signal ────────────────────────────────────────────────────────────────────
const sigBadge = $('sig-badge');
sigBadge.textContent = D.last_signal;
sigBadge.className   = 'signal-badge signal-' + D.last_signal;
$('sig-at').textContent     = D.last_signal_at.slice(11,19) || '—';
$('sig-price').textContent  = D.last_signal_price ? fmtUSD(D.last_signal_price) : '—';
$('sig-reason').textContent = D.last_signal_reason || '—';
$('sig-sbot').textContent   = fmt4(D.score_bot) + ' (thr=' + D.thr_bot + ')';
$('sig-stop').textContent   = fmt4(D.score_top) + ' (thr=' + D.thr_top + ')';

// ── Warmup ────────────────────────────────────────────────────────────────────
const wc = $('warmup-container');
if (!D.warmup_complete) {{
  wc.style.display = 'block';
  $('warmup-progress').textContent = D.candles_seen;
}}

// ── Métricas IT — barras ──────────────────────────────────────────────────────
setBar('bar-te',    D.te_norm);
setBar('bar-cmi',   D.cmi_norm);
setBar('bar-field', D.field_norm);
setBar('bar-sink',  D.sink_norm);
setBar('bar-sbot',  D.score_bot);
setBar('bar-stop',  D.score_top);

$('val-te').textContent    = fmt4(D.te_norm);
$('val-cmi').textContent   = fmt4(D.cmi_norm);
$('val-field').textContent = fmt4(D.field_norm);
$('val-sink').textContent  = fmt4(D.sink_norm);
$('val-sbot').textContent  = fmt4(D.score_bot);
$('val-stop').textContent  = fmt4(D.score_top);

$('te-raw-val').textContent  = D.te_raw  != null ? fmt4(D.te_raw)  + ' bits' : '—';
$('cmi-raw-val').textContent = D.cmi_raw != null ? fmt4(D.cmi_raw) + ' bits' : '—';
$('sink-raw-val').textContent= D.sink_raw != null ? fmt2(D.sink_raw) + 'x' : '—';

// ── Field vector ──────────────────────────────────────────────────────────────
const pa = arrowSign(D.field_price_div);
const va = arrowSign(D.field_vol_div);
const ca = arrowSign(D.field_curl);
[['fv-price-arrow','fv-price-val', D.field_price_div, pa],
 ['fv-vol-arrow',  'fv-vol-val',   D.field_vol_div,   va],
 ['fv-curl-arrow', 'fv-curl-val',  D.field_curl,       ca]
].forEach(([aId, vId, val, info]) => {{
  const aEl = $(aId), vEl = $(vId);
  if (aEl) {{ aEl.textContent = info.arrow; aEl.className = 'fv-arrow ' + info.cls; }}
  if (vEl) {{ vEl.textContent = val != null ? Number(val).toFixed(3) : '—'; vEl.className = 'fv-value ' + info.cls; }}
}});

// Pattern labels
$('pattern-bot-lbl').textContent = D.is_bot_pattern ? '✅ Activo (Δp↓+Δv↑)' : '⭕ Inactivo';
$('pattern-top-lbl').textContent = D.is_top_pattern ? '✅ Activo (Δp↑+Δv↓)' : '⭕ Inactivo';
$('pattern-bot-lbl').style.color = D.is_bot_pattern ? 'var(--green)' : 'var(--muted)';
$('pattern-top-lbl').style.color = D.is_top_pattern ? 'var(--red)'   : 'var(--muted)';

// ── Positions ────────────────────────────────────────────────────────────────
// (rendered from trade_history BUY entries not yet paired with SELL)
const posContainer = $('positions-container');
const openBuys = D.trades.filter(t => t.type === 'BUY' && !t.ignorado);
if (openBuys.length === 0) {{
  posContainer.innerHTML = '<div style="color:var(--muted);font-size:11px;">Sin posiciones</div>';
}} else {{
  posContainer.innerHTML = openBuys.slice(-5).map(t => {{
    const entryPrice = t.price || 0;
    const pnlPos = D.last_price > 0 ? ((D.last_price - entryPrice) / entryPrice * 100) : 0;
    return `<div class="pos-row">
      <div class="pos-header">
        <span style="color:var(--green);font-weight:600">BUY @ ${{entryPrice.toLocaleString()}}</span>
        <span class="${{pnlPos>=0?'pos':'neg'}}">${{pnlPos>=0?'+':''}}${{pnlPos.toFixed(2)}}%</span>
      </div>
      <div class="pos-detail">${{t.datetime ? t.datetime.slice(0,16) : '—'}} · ${{t.usdt_spent ? '$'+Number(t.usdt_spent).toFixed(2) : '—'}}</div>
    </div>`;
  }}).join('');
}}

// ── Trades table ──────────────────────────────────────────────────────────────
const tbody = $('trades-body');
const trades = [...D.trades].reverse().slice(0, 10);
if (trades.length === 0) {{
  tbody.innerHTML = '<tr><td colspan="4" style="color:var(--muted)">Sin operaciones</td></tr>';
}} else {{
  tbody.innerHTML = trades.map(t => {{
    const cls = t.ignorado ? 'td-ign' : (t.type==='BUY' ? 'td-buy' : 'td-sell');
    const usd = t.usdt_spent || t.usdt_received || 0;
    return `<tr>
      <td class="${{cls}}">${{t.ignorado ? '✗ '+t.type : t.type}}</td>
      <td>${{Number(t.price||0).toLocaleString('en-US',{{maximumFractionDigits:0}})}}</td>
      <td>${{usd ? '$'+Number(usd).toFixed(2) : '—'}}</td>
      <td>${{t.datetime ? t.datetime.slice(11,16) : '—'}}</td>
    </tr>`;
  }}).join('');
}}

// ── Config ────────────────────────────────────────────────────────────────────
const cfgContainer = $('config-container');
cfgContainer.innerHTML = Object.entries(D.config).map(([k,v]) =>
  `<div class="cfg-row"><span class="cfg-key">${{k}}</span><span class="cfg-val">${{v}}</span></div>`
).join('');

// ══════════════════════════════════════════════════════════════════════════════
// GRÁFICOS (Canvas 2D puro — sin dependencias externas)
// ══════════════════════════════════════════════════════════════════════════════

function resizeCanvas(canvas) {{
  canvas.width  = canvas.parentElement.clientWidth - 24;
  canvas.height = parseInt(canvas.getAttribute('height'));
}}

// ── Gráfico precio ─────────────────────────────────────────────────────────────
(function() {{
  const canvas = $('chart-price');
  resizeCanvas(canvas);
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  const candles = D.candles;
  if (!candles || candles.length < 2) return;

  ctx.fillStyle = '#0d1117'; ctx.fillRect(0,0,W,H);

  const prices = candles.flatMap(c => [c.high, c.low]);
  const minP = Math.min(...prices), maxP = Math.max(...prices);
  const rangeP = maxP - minP || 1;
  const padL=50, padR=20, padT=10, padB=20;
  const cW = Math.max(3, (W - padL - padR) / candles.length - 1);

  const py = p => padT + (1 - (p - minP) / rangeP) * (H - padT - padB);
  const cx = i => padL + (i + 0.5) * (W - padL - padR) / candles.length;

  // Grid lines
  ctx.strokeStyle = '#21262d'; ctx.lineWidth = 0.5;
  for (let g=0; g<4; g++) {{
    const y = padT + g * (H-padT-padB)/3;
    ctx.beginPath(); ctx.moveTo(padL,y); ctx.lineTo(W-padR,y); ctx.stroke();
    const price = maxP - g * rangeP/3;
    ctx.fillStyle='#8b949e'; ctx.font='9px monospace';
    ctx.fillText('$'+Math.round(price).toLocaleString(), 2, y+3);
  }}

  // Velas
  candles.forEach((c, i) => {{
    const x = cx(i);
    const isGreen = c.close >= c.open;
    const color = isGreen ? '#3fb950' : '#f85149';
    ctx.strokeStyle = color; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(x, py(c.high)); ctx.lineTo(x, py(c.low)); ctx.stroke();
    const bodyY = py(Math.max(c.open, c.close));
    const bodyH = Math.max(1, Math.abs(py(c.open) - py(c.close)));
    ctx.fillStyle = color; ctx.fillRect(x - cW/2, bodyY, cW, bodyH);
  }});

  // Señales del trade_history
  D.trades.forEach(t => {{
    if (t.ignorado) return;
    const ts = t.ts;
    const idx = candles.findIndex(c => Math.abs(c.ts - ts) < 7200);
    if (idx < 0) return;
    const x = cx(idx), price = t.price || 0;
    const y = py(price);
    ctx.beginPath();
    if (t.type === 'BUY') {{
      ctx.fillStyle = '#3fb950';
      ctx.moveTo(x, y+12); ctx.lineTo(x-7, y+22); ctx.lineTo(x+7, y+22); ctx.closePath();
    }} else {{
      ctx.fillStyle = '#f85149';
      ctx.moveTo(x, y-12); ctx.lineTo(x-7, y-22); ctx.lineTo(x+7, y-22); ctx.closePath();
    }}
    ctx.fill();
  }});

  // Etiqueta última vela
  if (candles.length > 0) {{
    const last = candles[candles.length-1];
    const x = cx(candles.length-1);
    ctx.strokeStyle='rgba(88,166,255,0.4)'; ctx.lineWidth=1; ctx.setLineDash([3,3]);
    ctx.beginPath(); ctx.moveTo(x, padT); ctx.lineTo(x, H-padB); ctx.stroke();
    ctx.setLineDash([]);
  }}
}})();

// ── Gráfico scores ─────────────────────────────────────────────────────────────
(function() {{
  const canvas = $('chart-scores');
  resizeCanvas(canvas);
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  const mets = D.metrics;
  if (!mets || mets.length < 2) return;

  ctx.fillStyle = '#0d1117'; ctx.fillRect(0,0,W,H);
  const padL=10, padR=10, padT=8, padB=15;
  const IW = W-padL-padR, IH = H-padT-padB;
  const px = i => padL + i * IW / (mets.length-1);
  const py = v => padT + (1 - Math.min(Math.max(v,0),1)) * IH;

  // Threshold lines
  [['#3fb950', D.thr_bot], ['#f85149', D.thr_top]].forEach(([col, thr]) => {{
    ctx.strokeStyle = col+'66'; ctx.lineWidth=1; ctx.setLineDash([4,4]);
    const y = py(thr);
    ctx.beginPath(); ctx.moveTo(padL,y); ctx.lineTo(W-padR,y); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle=col+'99'; ctx.font='9px monospace';
    ctx.fillText(thr.toFixed(2), W-padR-24, y-2);
  }});

  // Lines
  [['score_bot','#3fb950',2],['score_top','#f85149',2]].forEach(([key,col,lw]) => {{
    ctx.strokeStyle=col; ctx.lineWidth=lw; ctx.beginPath();
    mets.forEach((m,i) => {{
      const x=px(i), y=py(m[key]||0);
      i===0 ? ctx.moveTo(x,y) : ctx.lineTo(x,y);
    }});
    ctx.stroke();
  }});

  // X labels
  ctx.fillStyle='#8b949e'; ctx.font='9px monospace';
  [0, Math.floor(mets.length/2), mets.length-1].forEach(i => {{
    const ts = mets[i] && mets[i].ts;
    if (ts) ctx.fillText(new Date(ts*1000).toLocaleTimeString('es').slice(0,5), px(i)-10, H-3);
  }});
}})();

// ── Gráfico métricas IT ────────────────────────────────────────────────────────
(function() {{
  const canvas = $('chart-metrics');
  resizeCanvas(canvas);
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  const mets = D.metrics;
  if (!mets || mets.length < 2) return;

  ctx.fillStyle = '#0d1117'; ctx.fillRect(0,0,W,H);
  const padL=10, padR=10, padT=8, padB=15;
  const IW = W-padL-padR, IH = H-padT-padB;
  const px = i => padL + i * IW / (mets.length-1);
  const py = v => padT + (1 - Math.min(Math.max(v,0),1)) * IH;

  const series = [
    ['te_norm',   '#58a6ff', 'TE'],
    ['cmi_norm',  '#a371f7', 'CMI'],
    ['field_norm','#fb8500', 'Field'],
    ['sink_norm', '#3fb950', 'Sink'],
  ];

  series.forEach(([key,col,label]) => {{
    ctx.strokeStyle=col; ctx.lineWidth=1.5; ctx.beginPath();
    mets.forEach((m,i) => {{
      const x=px(i), y=py(m[key]||0);
      i===0 ? ctx.moveTo(x,y) : ctx.lineTo(x,y);
    }});
    ctx.stroke();
  }});

  // Legend
  series.forEach(([,col,label], i) => {{
    ctx.fillStyle=col; ctx.fillRect(padL+i*55, H-12, 8, 8);
    ctx.fillStyle='#8b949e'; ctx.font='9px sans-serif';
    ctx.fillText(label, padL+i*55+10, H-5);
  }});
}})();

// ── Gráfico taker + vol relativo ───────────────────────────────────────────────
(function() {{
  const canvas = $('chart-taker');
  resizeCanvas(canvas);
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  const candles = D.candles;
  if (!candles || candles.length < 2) return;

  ctx.fillStyle = '#0d1117'; ctx.fillRect(0,0,W,H);
  const padL=10, padR=10, padT=8, padB=15;
  const IW = W-padL-padR, IH = H-padT-padB;
  const px = i => padL + i * IW / (candles.length-1);

  // Taker ratio [0,1]
  ctx.strokeStyle='#58a6ff'; ctx.lineWidth=1.5; ctx.beginPath();
  const py_t = v => padT + (1 - Math.min(Math.max(v,0),1)) * IH;
  candles.forEach((c,i) => {{ i===0 ? ctx.moveTo(px(i),py_t(c.taker||0.5)) : ctx.lineTo(px(i),py_t(c.taker||0.5)); }});
  ctx.stroke();

  // Vol relativo al promedio
  const vols = candles.map(c=>c.vol||0);
  const avgVol = vols.reduce((a,b)=>a+b,0)/vols.length || 1;
  const maxVolRatio = Math.max(...vols.map(v=>v/avgVol), 2);
  const py_v = v => padT + (1 - Math.min(v/maxVolRatio, 1)) * IH;
  ctx.strokeStyle='#8b949e66'; ctx.lineWidth=1;
  vols.forEach((v,i) => {{
    const x=px(i), y=py_v(v/avgVol);
    ctx.fillStyle='#8b949e33';
    ctx.fillRect(x-2, y, 4, H-padB-y);
  }});

  // 0.5 line (neutral taker)
  ctx.strokeStyle='rgba(255,255,255,0.1)'; ctx.lineWidth=1; ctx.setLineDash([3,3]);
  const midY = padT + IH/2;
  ctx.beginPath(); ctx.moveTo(padL, midY); ctx.lineTo(W-padR, midY); ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle='#8b949e'; ctx.font='9px monospace'; ctx.fillText('0.5', 2, midY+3);

  // Legend
  ctx.fillStyle='#58a6ff'; ctx.fillRect(padL, H-12, 8,8);
  ctx.fillStyle='#8b949e'; ctx.font='9px sans-serif'; ctx.fillText('Taker Ratio', padL+10, H-5);
  ctx.fillStyle='#8b949e66'; ctx.fillRect(padL+90, H-12, 8,8);
  ctx.fillText('Vol/Avg', padL+100, H-5);
}})();

</script>
</body>
</html>"""


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _write_dashboard(state: LiveState) -> None:
    try:
        html = generate_dashboard(state)
        tmp  = DASHBOARD_HTML + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(html)
        os.replace(tmp, DASHBOARD_HTML)
    except Exception as e:
        log.warning("error escribiendo dashboard", error=str(e))


def _print_tick_summary(state: LiveState, candle) -> None:
    """Resumen en consola para monitoreo sin browser."""
    pnl_s = f"{'+' if state.pnl_pct>=0 else ''}{state.pnl_pct:.2f}%"
    pattern = ("▼BOT" if state.is_bot_pattern else
                "▲TOP" if state.is_top_pattern else " — ")
    sig_s = {"BUY":"🟢 BUY ","SELL":"🔴 SELL","HOLD":"⬜ hold"}.get(state.last_signal, "⬜")
    print(
        f"[{_now_iso()[11:]}] "
        f"${candle.close:>10,.2f} | "
        f"TE={state.te_norm:.2f} CMI={state.cmi_norm:.2f} "
        f"Fld={state.field_norm:.2f} Snk={state.sink_norm:.2f} | "
        f"sBot={state.score_bot:.3f} sTop={state.score_top:.3f} | "
        f"{pattern} | {sig_s} | "
        f"Port=${state.portfolio_value:,.2f} ({pnl_s}) | "
        f"Pos={state.positions_count} | "
        f"Prox.vela: {state.next_candle_at}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   LIVE TRADER — DivergenceField  BTC/USDT  Testnet      ║")
    print("╚══════════════════════════════════════════════════════════╝")

    cfg_d = CONFIG.to_dict()
    print(f"  TE estimador   : {cfg_d['te_estimator']}")
    print(f"  Ventana        : {cfg_d['window_size']} ({cfg_d['window_mode']})")
    print(f"  Campo          : {cfg_d['field_def']}")
    print(f"  CMI regímenes  : {cfg_d['cmi_regimes']}")
    print(f"  Umbral mode    : {cfg_d['threshold_mode']}")
    print(f"  Score BUY/SELL : {CONFIG.score_threshold_bot} / {CONFIG.score_threshold_top}")
    print(f"  Warmup velas   : {WARMUP_CANDLES}")
    print(f"  Dashboard HTML : {DASHBOARD_HTML}")
    print("─" * 60)

    # ── Inicializar estado global ──────────────────────────────────────────────
    live_state             = LiveState()
    live_state.config_dict = cfg_d

    # ── 1. PriceFeed + warmup histórico ───────────────────────────────────────
    print("[1/5] Inicializando PriceFeed y cargando warmup...")
    feed = BinanceWSFeed()
    warmup_end = "now"
    warmup_start_epoch = int(time.time()) - WARMUP_CANDLES * 3600
    warmup_start = datetime.fromtimestamp(warmup_start_epoch,
                                           tz=timezone.utc).strftime("%Y-%m-%d")

    try:
        hist_candles = feed.get_candles(warmup_start, warmup_end, SYMBOL)
        print(f"   ✓ {len(hist_candles)} velas históricas cargadas")
    except Exception as e:
        log.error("error cargando warmup", error=str(e))
        hist_candles = []

    # ── 2. Wallet reconciliada ────────────────────────────────────────────────
    print("[2/5] Inicializando BinanceWallet...")
    Path("state").mkdir(exist_ok=True)
    wallet = BinanceWallet.from_account(
        max_posiciones = MAX_POSICIONES,
        json_path      = LIVE_RESULTS_JSON,
        state_path     = STATE_PATH,
        commission_pct = COMMISSION_PCT,
    )
    live_state.capital_inicial = wallet.portfolio_value(
        hist_candles[-1].close if hist_candles else 0.0
    )
    print(f"   ✓ USDT={wallet.get_usdt_balance():.2f}  "
          f"Posiciones={wallet.positions_count}  "
          f"Slot=${wallet.get_slot_usdt():.2f}")

    # ── 3. OrderBook y RiskManager ────────────────────────────────────────────
    print("[3/5] Inicializando OrderBook y RiskManager...")
    ob    = BinanceOrderBook(max_posiciones=MAX_POSICIONES,
                              commission_pct=COMMISSION_PCT)
    risk  = RiskManager(config=RiskConfig.conservative(),
                        usdt_inicial=live_state.capital_inicial)
    state_mgr = JSONStateManager(STATE_PATH)

    # ── 4. Estrategia + warmup ────────────────────────────────────────────────
    print("[4/5] Inicializando estrategia y procesando warmup...")
    strategy = DivergenceFieldStrategy(CONFIG)
    strategy.on_start(wallet)

    # Alimentar velas históricas para warm-up (sin ejecutar órdenes)
    for candle in hist_candles:
        strategy._tick(candle, wallet)

    print(f"   ✓ Warmup completado — {strategy.candles_seen} velas procesadas")

    # ── 5. LiveClock ──────────────────────────────────────────────────────────
    print("[5/5] Conectando WebSocket LiveClock...")
    clock = LiveClock(feed=feed, symbol=SYMBOL)
    print(f"   ✓ Esperando cierre de la próxima vela horaria...")
    print(f"   Dashboard → abrir en browser: {DASHBOARD_HTML}")
    print("─" * 60)

    # Dashboard inicial (con warmup hecho)
    _write_dashboard(live_state)

    # ── Manejo de señal Ctrl+C ────────────────────────────────────────────────
    stop_requested = [False]

    def _handle_stop(sig, frame):
        print("\n\n[STOP] Señal de parada recibida — cerrando limpiamente...")
        stop_requested[0] = True
        clock.stop()

    signal.signal(signal.SIGINT,  _handle_stop)
    signal.signal(signal.SIGTERM, _handle_stop)

    # ══════════════════════════════════════════════════════════════════════════
    # LOOP PRINCIPAL
    # ══════════════════════════════════════════════════════════════════════════
    try:
        for candle in clock:
            if stop_requested[0]:
                break

            # ── Procesar vela ─────────────────────────────────────────────────
            signal_out = strategy._tick(candle, wallet)

            # ── Actualizar estado de display ──────────────────────────────────
            live_state.update_from_candle(candle, strategy, wallet)
            live_state.estimate_next_candle()

            if not signal_out.is_actionable:
                live_state.update_from_signal("HOLD", candle.close, "sin señal")
                _print_tick_summary(live_state, candle)
                _write_dashboard(live_state)
                continue

            order_side = signal_out.to_order_side()
            live_state.update_from_signal(
                signal_out.side.value,
                signal_out.price,
                signal_out.reason,
            )

            # ── Risk check ────────────────────────────────────────────────────
            risk_reason = risk.check(order_side, signal_out.price, wallet, candle)
            if risk_reason:
                log.warning("señal bloqueada por riesgo", reason=risk_reason)
                wallet.update(TradeRecord(
                    ts=candle.ts, side=order_side.value, price=signal_out.price,
                    ignored=True, ignore_reason=f"RISK:{risk_reason}",
                ))
                live_state.update_from_signal("HOLD", candle.close,
                                               f"BLOQUEADO: {risk_reason}")
                _print_tick_summary(live_state, candle)
                _write_dashboard(live_state)
                continue

            # ── Ejecutar orden ────────────────────────────────────────────────
            order = ob.execute_with_guards(
                order_side, signal_out.price, wallet, candle_ts=candle.ts
            )

            if order.is_filled and order.trade:
                t   = order.trade
                log.info(
                    "orden ejecutada",
                    side  = t.side,
                    price = f"{t.price:.2f}",
                    usdt  = f"{t.usdt_spent or t.usdt_received:.2f}",
                )
                trade_dict = {
                    "ts":           t.ts,
                    "datetime":     datetime.fromtimestamp(t.ts, tz=timezone.utc)
                                    .strftime("%Y-%m-%d %H:%M:%S"),
                    "type":         t.side,
                    "price":        t.price,
                    "usdt_spent":   t.usdt_spent,
                    "usdt_received":t.usdt_received,
                    "btc_bought":   t.btc_bought,
                    "btc_sold":     t.btc_sold,
                    "commission":   t.commission,
                    "ignorado":     False,
                    "score_bot":    strategy.last_score_bot,
                    "score_top":    strategy.last_score_top,
                    "te_norm":      strategy.last_te_norm,
                    "cmi_norm":     strategy.last_cmi_norm,
                    "field_norm":   strategy.last_field_norm,
                    "sink_norm":    strategy.last_sink_norm,
                }
                live_state.add_trade(trade_dict)

            elif order.is_ignored:
                live_state.add_trade({
                    "ts":       candle.ts,
                    "datetime": _now_iso(),
                    "type":     order_side.value,
                    "price":    signal_out.price,
                    "ignorado": True,
                    "motivo":   order.reject_reason,
                })
                live_state.update_from_signal("HOLD", candle.close,
                                               f"IGN: {order.reject_reason}")

            # ── Actualizar estado y dashboard ─────────────────────────────────
            live_state.update_from_candle(candle, strategy, wallet)
            live_state.estimate_next_candle()
            risk.update_peak(wallet.portfolio_value(candle.close))

            # ── Checkpoint ───────────────────────────────────────────────────
            state_mgr.save(Checkpoint.from_wallet(
                wallet, candle.ts, candle.close,
                metadata={"estrategia": strategy.name, **cfg_d},
            ))

            _print_tick_summary(live_state, candle)
            _write_dashboard(live_state)

    except Exception as e:
        log.error("error en loop principal", error=str(e))
        traceback.print_exc()

    finally:
        # ── Cierre limpio ─────────────────────────────────────────────────────
        print("\n[CIERRE] Guardando estado final...")
        strategy.on_stop(wallet)

        # Guardar JSON de resultados
        precio_final  = live_state.last_price or 0.0
        port_final    = wallet.portfolio_value(precio_final)
        pnl_pct       = ((port_final / live_state.capital_inicial - 1) * 100
                         if live_state.capital_inicial > 0 else 0.0)

        summary = {
            "estrategia":            strategy.name,
            "started_at":            live_state.started_at,
            "stopped_at":            _now_iso(),
            "capital_inicial":       live_state.capital_inicial,
            "portfolio_value_final": round(port_final, 4),
            "pnl_pct":               round(pnl_pct, 4),
            "usdt_balance_final":    round(wallet.get_usdt_balance(), 8),
            "btc_en_posiciones":     round(wallet.btc_en_posiciones(), 10),
            "positions_count_final": wallet.positions_count,
            "candles_procesadas":    live_state.candles_seen,
            "signals_total":         live_state.signals_total,
            "trades_total":          len(live_state.trade_history),
            "parametros":            cfg_d,
        }
        wallet.flush(summary)

        # Dashboard final
        _write_dashboard(live_state)
        clock.stop()

        print(f"  Portfolio final : ${port_final:,.2f} USDT")
        print(f"  PnL sesión      : {'+' if pnl_pct>=0 else ''}{pnl_pct:.2f}%")
        print(f"  Velas           : {live_state.candles_seen}")
        print(f"  Señales         : {live_state.signals_total}")
        print(f"  JSON guardado   : {LIVE_RESULTS_JSON}")
        print(f"  Dashboard final : {DASHBOARD_HTML}")
        print("✓ Live trader cerrado limpiamente.")


if __name__ == "__main__":
    main()
