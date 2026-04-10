"""
optimize_divergence_field.py — Optimizador Bifásico de DivergenceFieldStrategy
════════════════════════════════════════════════════════════════════════════════

POR QUÉ EL OPTIMIZADOR ANTERIOR ERA LENTO
──────────────────────────────────────────
  Cada backtest completo tarda ~50–90 s con 38 500 velas.
  69 120 combinaciones × 60 s / 3 workers = ~384 horas.
  P2/P3 reducían el tiempo por backtest ~3×, pero el problema es
  estructural: se hacen 69 120 backtests cuando solo existen 160
  configuraciones de cómputo distintas.

ARQUITECTURA BIFÁSICA
──────────────────────
  Fase 1 — 160 configs de cómputo  (te × win × field × cmi × threshold × sink)
    Ejecuta el backtest real con _FastDivergenceField.
    Guarda, por vela, (global_idx, ts, price, score_bot, score_top).

  Fase 2 — 432 combos de decisión por config  (thr_bot × thr_top × cd × mp)
    Reproduce la señal usando los scores de Fase 1.
    Usa el MISMO SimulatedOrderBook + MemoryWallet que el backtest completo.
    No recalcula TE, CMI, campo ni normalización.

SPEEDUP: 69 120 backtests → 160 backtests + 69 120 replays ≈ ×400

GARANTÍA DE CORRECTITUD
────────────────────────
  score_bot[i] y score_top[i] son calculados ANTES del chequeo de
  umbral/cooldown en on_candle(), y NUNCA leen el estado de la wallet.
  Son idénticos para cualquier combinación de (thr_bot, thr_top, cd, mp).

  _replay_decision reproduce fielmente on_candle():
    · SELL prioridad sobre BUY (mismo orden de if/elif)
    · last_*_idx actualizado ANTES de execute_with_guards
    · global_idx 1-based coincide con _candles_seen de la estrategia
    · Mismo SimulatedOrderBook + MemoryWallet

  Verificación: la mejor config del optimizador corrida en
  backtest_divergence_field.py debe dar exactamente el mismo PnL y WR%.

USO
───
  python optimize_divergence_field.py              # todo
  python optimize_divergence_field.py --no-knn     # excluye KNN (~3× más rápido)
  python optimize_divergence_field.py --workers 6  # forzar N procesos
  python optimize_divergence_field.py --resume     # continuar tras interrupción
  python optimize_divergence_field.py --top 30
  python optimize_divergence_field.py --min-trades 20
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import multiprocessing
import os
import sys
import time
from dataclasses import dataclass, asdict
from itertools   import product
from pathlib     import Path
from typing      import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from strategies.divergence_field_strategy import (
    DivergenceFieldStrategy, DFConfig,
    TEEstimator, WindowMode, FieldDefinition,
    CMIRegimes, ThresholdMode, SinkMode,
    te_binning, te_kde, te_knn, cmi_binning,
    _digitize_pct,
)


# ══════════════════════════════════════════════════════════════════════════════
# ESPACIOS DE BÚSQUEDA
# ══════════════════════════════════════════════════════════════════════════════

GRID_SCORE_BOT       = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
GRID_SCORE_TOP       = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
GRID_COOLDOWNS       = [0, 24, 48, 96]
DEEP_TE_ESTIMATORS   = ["binning", "kde", "knn"]
DEEP_WINDOWS         = [8, 10, 14, 20, 28]
DEEP_FIELD_DEFS      = ["analogical", "jacobian"]
DEEP_CMI_REGIMES     = [2, 3]
DEEP_THRESHOLD_MODES = ["adaptive_percentile", "fixed"]
DEEP_SINK_MODES      = ["score_component", "filter_and"]
GRID_MAX_POS         = [5, 7, 9]

KNN_MIN_WINDOW = 15

# Solo guardamos velas donde algún score supera el mínimo del grid.
# No descarta ninguna combinación de decisión.
MIN_SCORE = min(GRID_SCORE_BOT + GRID_SCORE_TOP) - 1e-9

OUT_CSV        = "opt_divfield_results.csv"
OUT_JSON       = "opt_divfield_results.json"
CHECKPOINT_CSV = "opt_divfield_checkpoint.csv"
SAVE_EVERY     = 10  # checkpoint cada N configs de cómputo

CSV_FIELDS = [
    "te_estimator", "window_size", "field_def", "cmi_regimes",
    "threshold_mode", "sink_mode", "thr_bot", "thr_top",
    "cooldown", "max_posiciones",
    "pnl_pct", "portfolio_final", "bh_pnl", "alpha_bh",
    "n_compras", "n_ventas", "n_trades", "n_ignorados",
    "win_rate", "pos_finales", "valid",
]


# ══════════════════════════════════════════════════════════════════════════════
# PRECOMPUTACIÓN DE SERIES (una vez por worker)
# ══════════════════════════════════════════════════════════════════════════════

def _precompute_rsi_series(closes: np.ndarray, period: int = 14) -> np.ndarray:
    """RSI rolling O(n) con cumsum. Idéntico a [_rsi(closes[:i+1]) for i]."""
    n      = len(closes)
    result = np.full(n, 50.0, dtype=np.float64)
    if n < period + 1:
        return result
    deltas = np.diff(closes)
    gains  = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    cg     = np.concatenate([[0.0], np.cumsum(gains)])
    cl     = np.concatenate([[0.0], np.cumsum(losses)])
    for i in range(period, n):
        ag        = (cg[i] - cg[i - period]) / period
        al        = (cl[i] - cl[i - period]) / period
        result[i] = 100.0 if al < 1e-10 else 100.0 - 100.0 / (1.0 + ag / al)
    return result


def _precompute_ma_series(closes: np.ndarray, period: int = 20) -> np.ndarray:
    """MA20 rolling O(n) con cumsum."""
    n      = len(closes)
    cum    = np.concatenate([[0.0], np.cumsum(closes)])
    result = np.empty(n, dtype=np.float64)
    for i in range(n):
        s         = max(0, i - period + 1)
        result[i] = (cum[i + 1] - cum[s]) / (i - s + 1)
    return result


# ══════════════════════════════════════════════════════════════════════════════
# CAMPO VECTORIAL SOBRE ARRAYS (sin objetos Candle)
# ══════════════════════════════════════════════════════════════════════════════

def _snorm(arr: np.ndarray) -> np.ndarray:
    s = arr.std()
    return arr / s if s > 1e-10 else arr


def _field_analogical_arr(c, v, t):
    if len(c) < 5:
        return 0.0, 0.0, 0.0
    dp = _snorm(np.diff(c)); dv = _snorm(np.diff(v)); dt = _snorm(np.diff(t))
    if len(dp) < 4:
        return 0.0, 0.0, 0.0
    ddp = np.diff(dp); ddv = np.diff(dv); ddt = np.diff(dt)
    pd  = float(np.mean(ddp))
    vtd = float(np.mean(ddv) + np.mean(ddt))
    try:
        m    = min(len(dp) - 1, len(dv) - 1)
        curl = (float(np.corrcoef(dv[:m], np.diff(dp[:m+1]))[0,1]
                      - np.corrcoef(dp[:m], np.diff(dv[:m+1]))[0,1])
                if len(dp) >= 5 else 0.0)
    except Exception:
        curl = 0.0
    return (0.0 if not np.isfinite(pd)   else pd,
            0.0 if not np.isfinite(vtd)  else vtd,
            0.0 if not np.isfinite(curl) else curl)


def _field_jacobian_arr(c, v, t):
    if len(c) < 6:
        return 0.0, 0.0, 0.0
    dp = _snorm(np.diff(c)); dv = _snorm(np.diff(v)); dt = _snorm(np.diff(t))
    m  = min(len(dp), len(dv), len(dt))
    if m < 5:
        return 0.0, 0.0, 0.0
    try:
        cov = np.cov(np.vstack([dp[:m], dv[:m], dt[:m]]))
        if not np.all(np.isfinite(cov)):
            return 0.0, 0.0, 0.0
        vals, vecs = np.linalg.eigh(cov)
        d = vecs[:, int(np.argmax(np.abs(vals)))]
        pc  = float(d[0]); vtc = float(d[1]+d[2])
        curl = float(np.corrcoef(dp[:m], dv[:m])[0,1])
        return (0.0 if not np.isfinite(pc)   else pc,
                0.0 if not np.isfinite(vtc)  else vtc,
                0.0 if not np.isfinite(curl) else curl)
    except (np.linalg.LinAlgError, ValueError):
        return 0.0, 0.0, 0.0


# ══════════════════════════════════════════════════════════════════════════════
# TIPOS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ComputeConfig:
    """Los 6 parámetros que determinan completamente el valor de los scores."""
    te_estimator:   str
    window_size:    int
    field_def:      str
    cmi_regimes:    int
    threshold_mode: str
    sink_mode:      str

    def key(self) -> str:
        return (f"{self.te_estimator}|{self.window_size}|{self.field_def}|"
                f"{self.cmi_regimes}|{self.threshold_mode}|{self.sink_mode}")

    def to_dfconfig(self) -> DFConfig:
        """Thresholds neutros: no afectan el cálculo de scores."""
        return DFConfig(
            te_estimator        = TEEstimator(self.te_estimator),
            window_mode         = WindowMode.FIXED,
            window_size         = self.window_size,
            field_def           = FieldDefinition(self.field_def),
            cmi_regimes         = CMIRegimes(self.cmi_regimes),
            threshold_mode      = ThresholdMode(self.threshold_mode),
            sink_mode           = SinkMode(self.sink_mode),
            score_threshold_bot = 0.0,
            score_threshold_top = 0.0,
            cooldown            = 0,
        )


@dataclass
class Result:
    te_estimator:    str
    window_size:     int
    field_def:       str
    cmi_regimes:     int
    threshold_mode:  str
    sink_mode:       str
    thr_bot:         float
    thr_top:         float
    cooldown:        int
    max_posiciones:  int
    pnl_pct:         float
    portfolio_final: float
    bh_pnl:          float
    alpha_bh:        float
    n_compras:       int
    n_ventas:        int
    n_trades:        int
    n_ignorados:     int
    win_rate:        float
    pos_finales:     int
    valid:           bool

    def to_dict(self) -> dict:
        d = asdict(self)
        for k in ("pnl_pct", "portfolio_final", "bh_pnl", "alpha_bh"):
            d[k] = round(d[k], 2)
        d["win_rate"] = round(d["win_rate"], 1)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Result":
        return cls(
            te_estimator   = str(d["te_estimator"]),
            window_size    = int(d["window_size"]),
            field_def      = str(d["field_def"]),
            cmi_regimes    = int(d["cmi_regimes"]),
            threshold_mode = str(d["threshold_mode"]),
            sink_mode      = str(d["sink_mode"]),
            thr_bot        = float(d["thr_bot"]),
            thr_top        = float(d["thr_top"]),
            cooldown       = int(d["cooldown"]),
            max_posiciones = int(d["max_posiciones"]),
            pnl_pct        = float(d["pnl_pct"]),
            portfolio_final= float(d["portfolio_final"]),
            bh_pnl         = float(d["bh_pnl"]),
            alpha_bh       = float(d["alpha_bh"]),
            n_compras      = int(d["n_compras"]),
            n_ventas       = int(d["n_ventas"]),
            n_trades       = int(d["n_trades"]),
            n_ignorados    = int(d["n_ignorados"]),
            win_rate       = float(d["win_rate"]),
            pos_finales    = int(d["pos_finales"]),
            valid          = str(d.get("valid", "True")).lower() != "false",
        )


# ══════════════════════════════════════════════════════════════════════════════
# GENERADORES
# ══════════════════════════════════════════════════════════════════════════════

def all_compute_configs(include_knn: bool = True) -> List[ComputeConfig]:
    estimators = DEEP_TE_ESTIMATORS if include_knn else [
        e for e in DEEP_TE_ESTIMATORS if e != "knn"
    ]
    configs = []
    for te, win, fd, cr, tm, sm in product(
        estimators, DEEP_WINDOWS, DEEP_FIELD_DEFS,
        DEEP_CMI_REGIMES, DEEP_THRESHOLD_MODES, DEEP_SINK_MODES,
    ):
        if te == "knn" and win < KNN_MIN_WINDOW:
            continue
        configs.append(ComputeConfig(
            te_estimator=te, window_size=win, field_def=fd,
            cmi_regimes=cr, threshold_mode=tm, sink_mode=sm,
        ))
    return configs


def all_decision_combos() -> list:
    """432 combinaciones de parámetros de decisión."""
    return list(product(GRID_SCORE_BOT, GRID_SCORE_TOP,
                        GRID_COOLDOWNS, GRID_MAX_POS))


# ══════════════════════════════════════════════════════════════════════════════
# WORKER — variables globales por proceso
# ══════════════════════════════════════════════════════════════════════════════

_W_CANDLES:     list       = []
_W_BHPNL:       float      = 0.0
_W_USDT:        float      = 1000.0
_W_COMM:        float      = 0.1
_W_MINTR:       int        = 10
_W_LAST_PRICE:  float      = 0.0
_W_CLOSES:      np.ndarray = np.array([], dtype=np.float64)
_W_VOLUMES:     np.ndarray = np.array([], dtype=np.float64)
_W_TAKERS:      np.ndarray = np.array([], dtype=np.float64)
_W_PRICE_SLOPE: np.ndarray = np.array([], dtype=np.float64)
_W_VOL_ACCEL2:  np.ndarray = np.array([], dtype=np.float64)
_W_RSI:         np.ndarray = np.array([], dtype=np.float64)
_W_MA20:        np.ndarray = np.array([], dtype=np.float64)
_W_TS_INDEX:    dict       = {}


def _worker_init(db_path, db_table, fecha_ini, fecha_fin,
                 symbol, usdt, comm, min_trades):
    global _W_CANDLES, _W_BHPNL, _W_USDT, _W_COMM, _W_MINTR, _W_LAST_PRICE
    global _W_CLOSES, _W_VOLUMES, _W_TAKERS, _W_PRICE_SLOPE
    global _W_VOL_ACCEL2, _W_RSI, _W_MA20, _W_TS_INDEX

    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    import logging
    logging.disable(logging.WARNING)

    from actors.price_feed import SQLiteFeed
    feed           = SQLiteFeed(db_path=db_path, table=db_table)
    _W_CANDLES     = feed.get_candles(fecha_ini, fecha_fin, symbol)
    ini            = feed.get_candles(fecha_ini, fecha_ini, symbol)
    p_ini          = ini[0].close if ini else 1.0
    p_fin          = _W_CANDLES[-1].close if _W_CANDLES else 1.0
    _W_BHPNL       = (p_fin / p_ini - 1) * 100.0
    _W_USDT        = usdt
    _W_COMM        = comm
    _W_MINTR       = min_trades
    _W_LAST_PRICE  = float(p_fin)

    if not _W_CANDLES:
        return

    _W_CLOSES  = np.array([c.close  for c in _W_CANDLES], dtype=np.float64)
    _W_VOLUMES = np.array([c.volume for c in _W_CANDLES], dtype=np.float64)
    _W_TAKERS  = np.array([
        (c.taker_buy_base_vol / c.volume)
        if (c.taker_buy_base_vol is not None and c.volume > 1e-10) else 0.5
        for c in _W_CANDLES
    ], dtype=np.float64)
    _W_PRICE_SLOPE = np.diff(_W_CLOSES)
    _W_VOL_ACCEL2  = np.diff(_W_VOLUMES, 2)
    _W_TS_INDEX    = {c.ts: i for i, c in enumerate(_W_CANDLES)}
    _W_RSI         = _precompute_rsi_series(_W_CLOSES)
    _W_MA20        = _precompute_ma_series(_W_CLOSES)


# ══════════════════════════════════════════════════════════════════════════════
# _FastDivergenceField — usa arrays globales en todos los métodos de cómputo
# ══════════════════════════════════════════════════════════════════════════════

class _FastDivergenceField(DivergenceFieldStrategy):
    """
    Sobreescribe _compute_te/cmi/field/sink para operar sobre slices de
    arrays numpy globales en lugar de reconstruirlos desde Candle en cada vela.
    Fallback a super() si el timestamp no está en el índice global.
    """

    def _slice(self, window) -> tuple:
        e = _W_TS_INDEX.get(window[-1].ts)
        if e is None:
            return None, None
        s = e - len(window) + 1
        return (None, None) if s < 0 else (s, e)

    def _compute_te(self, window) -> float:
        s, e = self._slice(window)
        if s is None:
            return super()._compute_te(window)
        ps = _W_PRICE_SLOPE[s:e]
        ts = _W_TAKERS[s:e]
        if   self.cfg.te_estimator == TEEstimator.BINNING:
            return te_binning(ts, ps, k_bins=self.cfg.k_bins)
        elif self.cfg.te_estimator == TEEstimator.KDE:
            return te_kde(ts, ps)
        else:
            return te_knn(ts, ps, k=self.cfg.k_nn)

    def _compute_cmi(self, window) -> float:
        s, e = self._slice(window)
        if s is None:
            return super()._compute_cmi(window)
        n      = e - s + 1
        closes = _W_CLOSES[s:e+1]
        rsi    = _W_RSI[s:e+1]
        va     = np.zeros(n, dtype=np.float64)
        if n >= 3:
            ve = s + n - 2
            if ve <= len(_W_VOL_ACCEL2):
                va[2:] = _W_VOL_ACCEL2[s:ve]
        if min(20, n) == 20 and s >= 19:
            ma20 = _W_MA20[s:e+1]
        else:
            mp   = min(20, n)
            cum  = np.concatenate([[0.0], np.cumsum(closes)])
            ma20 = np.array([
                (cum[i+1] - cum[max(0,i-mp+1)]) / (i - max(0,i-mp+1) + 1)
                for i in range(n)
            ], dtype=np.float64)
        pvm = (closes - ma20) / (ma20 + 1e-10)
        return cmi_binning(rsi, va, pvm,
                           n_regimes=int(self.cfg.cmi_regimes),
                           k_bins=self.cfg.k_bins)

    def _compute_field(self, window):
        s, e = self._slice(window)
        if s is None:
            return super()._compute_field(window)
        c = _W_CLOSES[s:e+1]; v = _W_VOLUMES[s:e+1]; t = _W_TAKERS[s:e+1]
        if self.cfg.field_def == FieldDefinition.ANALOGICAL:
            return _field_analogical_arr(c, v, t)
        return _field_jacobian_arr(c, v, t)

    def _compute_sink(self, window) -> float:
        s, e = self._slice(window)
        if s is None:
            return super()._compute_sink(window)
        k    = self.cfg.sink_window
        n    = e - s + 1
        if n < k + 1:
            return 1.0
        vols    = _W_VOLUMES[s:e+1]
        vol_avg = vols.mean()
        return 1.0 if vol_avg < 1e-10 else float(vols[-k:].mean() / vol_avg)


# ══════════════════════════════════════════════════════════════════════════════
# FASE 1 — recolectar scores para una ComputeConfig
# ══════════════════════════════════════════════════════════════════════════════

def _collect_scores(cc: ComputeConfig) -> dict:
    """
    Backtest real con score_threshold=0.0 y cooldown=0 para capturar
    todos los scores posibles sin filtrar ninguna vela.
    Devuelve arrays de las velas donde algún score >= MIN_SCORE.

    global_idx: _candles_seen 1-based, igual que en on_candle().
    """
    from actors.wallet import MemoryWallet

    cfg      = cc.to_dfconfig()
    strategy = _FastDivergenceField(cfg)
    dummy    = MemoryWallet(usdt_inicial=_W_USDT, max_posiciones=9)
    strategy.on_start(dummy)

    gidxs  = []; tss    = []; prices = []
    sbots  = []; stops  = []

    for candle in _W_CANDLES:
        strategy._tick(candle, dummy)
        sb = strategy.last_score_bot
        st = strategy.last_score_top
        if sb >= MIN_SCORE or st >= MIN_SCORE:
            gidxs.append(strategy.candles_seen)
            tss.append(candle.ts)
            prices.append(candle.close)
            sbots.append(sb)
            stops.append(st)

    return {
        "global_idx": np.array(gidxs,  dtype=np.int32),
        "ts":         np.array(tss,    dtype=np.int64),
        "prices":     np.array(prices, dtype=np.float32),
        "score_bot":  np.array(sbots,  dtype=np.float32),
        "score_top":  np.array(stops,  dtype=np.float32),
    }


# ══════════════════════════════════════════════════════════════════════════════
# FASE 2 — replay con wallet + orderbook real
# ══════════════════════════════════════════════════════════════════════════════

def _replay_decision(sd: dict, thr_bot: float, thr_top: float,
                     cooldown: int, max_pos: int) -> dict:
    """
    Reproduce on_candle() usando los scores de Fase 1.
    Mismo SimulatedOrderBook + MemoryWallet que el backtest original.

    Garantías de fidelidad con on_candle():
      1. SELL prioridad sobre BUY (if / elif)
      2. last_*_idx actualizado ANTES de execute_with_guards (aunque rechace)
      3. global_idx 1-based = _candles_seen de la estrategia
    """
    from actors.wallet     import MemoryWallet
    from actors.order_book import SimulatedOrderBook, OrderSide

    wallet = MemoryWallet(usdt_inicial=_W_USDT, max_posiciones=max_pos)
    ob     = SimulatedOrderBook(commission_pct=_W_COMM, max_posiciones=max_pos)

    NEG          = -(10 ** 9)
    last_bot_idx = NEG
    last_top_idx = NEG
    n_buy = n_sell = n_ign = n_wins = 0

    gidxs  = sd["global_idx"]
    tss    = sd["ts"]
    prices = sd["prices"]
    sbots  = sd["score_bot"]
    stops  = sd["score_top"]

    # Prefiltar con numpy — minimiza iteraciones del loop Python
    candidates = np.where((sbots >= thr_bot) | (stops >= thr_top))[0]

    for k in candidates:
        gidx = int(gidxs[k])
        sb   = float(sbots[k])
        st   = float(stops[k])
        cd_ok_bot = cooldown == 0 or (gidx - last_bot_idx) >= cooldown
        cd_ok_top = cooldown == 0 or (gidx - last_top_idx) >= cooldown

        if st >= thr_top and cd_ok_top:
            last_top_idx = gidx  # antes de execute — igual que on_candle
            price     = float(prices[k])
            avg_entry = wallet.precio_promedio_posiciones()
            order = ob.execute_with_guards(OrderSide.SELL, price, wallet,
                                           candle_ts=int(tss[k]))
            if order.is_filled:
                n_sell += 1
                if avg_entry > 0 and price > avg_entry:
                    n_wins += 1
            else:
                n_ign += 1

        elif sb >= thr_bot and cd_ok_bot:
            last_bot_idx = gidx  # antes de execute — igual que on_candle
            order = ob.execute_with_guards(OrderSide.BUY, float(prices[k]),
                                           wallet, candle_ts=int(tss[k]))
            if order.is_filled:
                n_buy += 1
            else:
                n_ign += 1

    port     = wallet.portfolio_value(_W_LAST_PRICE)
    pnl      = (port / _W_USDT - 1) * 100.0
    n_trades = n_buy + n_sell
    wr       = (n_wins / n_sell * 100.0) if n_sell > 0 else 0.0

    return {
        "pnl_pct":         pnl,
        "portfolio_final": port,
        "bh_pnl":          _W_BHPNL,
        "alpha_bh":        pnl - _W_BHPNL,
        "n_compras":       n_buy,
        "n_ventas":        n_sell,
        "n_trades":        n_trades,
        "n_ignorados":     n_ign,
        "win_rate":        wr,
        "pos_finales":     wallet.positions_count,
        "valid":           n_trades >= _W_MINTR,
    }


# ══════════════════════════════════════════════════════════════════════════════
# WORKER — tarea completa: Fase 1 + todas las Fase 2 de una ComputeConfig
# ══════════════════════════════════════════════════════════════════════════════

def _run_compute_config_task(cc: ComputeConfig) -> List[Result]:
    """1 backtest real (Fase 1) + 432 replays (Fase 2). Retorna 432 Results."""
    try:
        sd = _collect_scores(cc)

        if len(sd["global_idx"]) == 0:
            return [
                Result(
                    te_estimator=cc.te_estimator, window_size=cc.window_size,
                    field_def=cc.field_def, cmi_regimes=cc.cmi_regimes,
                    threshold_mode=cc.threshold_mode, sink_mode=cc.sink_mode,
                    thr_bot=tb, thr_top=tt, cooldown=cd, max_posiciones=mp,
                    pnl_pct=0.0, portfolio_final=_W_USDT, bh_pnl=_W_BHPNL,
                    alpha_bh=-_W_BHPNL, n_compras=0, n_ventas=0, n_trades=0,
                    n_ignorados=0, win_rate=0.0, pos_finales=0, valid=False,
                )
                for tb, tt, cd, mp in all_decision_combos()
            ]

        results = []
        for tb, tt, cd, mp in all_decision_combos():
            replay = _replay_decision(sd, tb, tt, cd, mp)
            results.append(Result(
                te_estimator=cc.te_estimator, window_size=cc.window_size,
                field_def=cc.field_def, cmi_regimes=cc.cmi_regimes,
                threshold_mode=cc.threshold_mode, sink_mode=cc.sink_mode,
                thr_bot=tb, thr_top=tt, cooldown=cd, max_posiciones=mp,
                **replay,
            ))
        return results

    except Exception:
        import traceback; traceback.print_exc()
        return []


# ══════════════════════════════════════════════════════════════════════════════
# CHECKPOINT
# ══════════════════════════════════════════════════════════════════════════════

def save_checkpoint(results: List[Result]) -> None:
    with open(CHECKPOINT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in results:
            w.writerow(r.to_dict())


def load_checkpoint() -> Tuple[List[Result], set]:
    """Una ComputeConfig está completa si tiene los 432 decision combos."""
    if not Path(CHECKPOINT_CSV).exists():
        return [], set()
    results = []
    with open(CHECKPOINT_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                results.append(Result.from_dict(row))
            except Exception:
                pass
    expected = len(all_decision_combos())
    from collections import Counter
    counts    = Counter(
        f"{r.te_estimator}|{r.window_size}|{r.field_def}|"
        f"{r.cmi_regimes}|{r.threshold_mode}|{r.sink_mode}"
        for r in results
    )
    done_keys = {k for k, v in counts.items() if v >= expected}
    return results, done_keys


# ══════════════════════════════════════════════════════════════════════════════
# OUTPUT
# ══════════════════════════════════════════════════════════════════════════════

def print_table(results: List[Result], top_n: int, min_trades: int) -> None:
    valid = sorted([r for r in results if r.valid],
                   key=lambda r: r.pnl_pct, reverse=True)
    bh   = valid[0].bh_pnl if valid else 0.0
    sep  = "─" * 118
    sep2 = "═" * 118
    print(f"\n{sep2}")
    print(f"  RESULTADOS FINALES — {len(valid):,} configs válidas "
          f"(≥{min_trades} trades) de {len(results):,} total  |  B&H: {bh:+.2f}%")
    print(sep2)
    print(f"  {'#':>4}  {'TE':>7} {'W':>3} {'Field':>5} {'C':>2} "
          f"{'Th':>5} {'Sk':>5}  "
          f"{'thr_b':>5} {'thr_t':>5} {'cd':>4} {'mp':>3}  "
          f"{'PnL%':>9} {'α_BH':>8}  {'B/S':>9}  {'WR%':>5} {'Pos':>3}")
    print(sep)
    for i, r in enumerate(valid[:top_n], 1):
        pnl_s = f"{'+' if r.pnl_pct>=0 else ''}{r.pnl_pct:.2f}%"
        abh_s = f"{'+' if r.alpha_bh>=0 else ''}{r.alpha_bh:.2f}%"
        cd_s  = str(r.cooldown) if r.cooldown else "off"
        q     = ("✓" if r.win_rate >= 50 and r.n_trades >= 30 else
                 "⚠" if r.win_rate < 42 else "")
        print(f"  {i:>4}.  "
              f"{r.te_estimator:>7} {r.window_size:>3} "
              f"{r.field_def[:5]:>5} {r.cmi_regimes:>2} "
              f"{r.threshold_mode[:5]:>5} {r.sink_mode[:5]:>5}  "
              f"{r.thr_bot:>5.2f} {r.thr_top:>5.2f} {cd_s:>4} {r.max_posiciones:>3}  "
              f"{pnl_s:>9} {abh_s:>9}  "
              f"{r.n_compras}B/{r.n_ventas}S  "
              f"{r.win_rate:>5.1f}% {r.pos_finales:>3} {q}")
    print(sep)
    if len(valid) > top_n:
        print(f"  ... {len(valid)-top_n:,} configs más en {OUT_CSV}")
    print(f"\n  ✓ = WR≥50% y trades≥30  |  ⚠ = WR<42%")


def print_best(r: Result, usdt: float, comm: float) -> None:
    te_v = f"TEEstimator.{r.te_estimator.upper()}"
    fd_v = f"FieldDefinition.{'ANALOGICAL' if r.field_def=='analogical' else 'JACOBIAN'}"
    cr_v = f"CMIRegimes.{'BINARY' if r.cmi_regimes==2 else 'TERNARY'}"
    tm_v = ("ThresholdMode.ADAPTIVE_PERCENTILE"
            if "adap" in r.threshold_mode else "ThresholdMode.FIXED")
    sm_v = ("SinkMode.SCORE_COMPONENT"
            if "score" in r.sink_mode else "SinkMode.FILTER_AND")
    print(f"""
╔══════════════════════════════════════════════════════════╗
║  MEJOR CONFIG — pegar en backtest_divergence_field.py   ║
╚══════════════════════════════════════════════════════════╝
  PnL    : {r.pnl_pct:+.2f}%    α B&H: {r.alpha_bh:+.2f}%
  Trades : {r.n_trades}  ({r.n_compras}B / {r.n_ventas}S)   WR%: {r.win_rate:.1f}%

# ── backtest_divergence_field.py ─────────────────────────────────────────────
CONFIG = DFConfig(
    te_estimator        = {te_v},
    window_mode         = WindowMode.FIXED,
    window_size         = {r.window_size},
    field_def           = {fd_v},
    cmi_regimes         = {cr_v},
    threshold_mode      = {tm_v},
    sink_mode           = {sm_v},
    score_threshold_bot = {r.thr_bot},
    score_threshold_top = {r.thr_top},
    cooldown            = {r.cooldown},
)

# ── config_local.py ───────────────────────────────────────────────────────────
MAX_POSICIONES     = {r.max_posiciones}
SALDO_USDT_INICIAL = {usdt}
COMMISSION_PCT     = {comm}
""")


def save_final(results: List[Result], elapsed_s: float, min_trades: int) -> None:
    all_s = sorted(results, key=lambda r: r.pnl_pct, reverse=True)
    valid = [r for r in all_s if r.valid]
    bh    = valid[0].bh_pnl if valid else 0.0

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in all_s:
            w.writerow(r.to_dict())

    meta = {
        "arquitectura":  "bifasica_scores_precomputados_wallet_real",
        "n_total":       len(all_s),
        "n_valid":       len(valid),
        "min_trades":    min_trades,
        "bh_pnl":        round(bh, 2),
        "elapsed_min":   round(elapsed_s / 60, 1),
        "best_pnl":      round(valid[0].pnl_pct, 2) if valid else None,
        "best_wr":       round(valid[0].win_rate, 1) if valid else None,
        "grid_params": {
            "GRID_SCORE_BOT": GRID_SCORE_BOT, "GRID_SCORE_TOP": GRID_SCORE_TOP,
            "GRID_COOLDOWNS": GRID_COOLDOWNS, "DEEP_TE_ESTIMATORS": DEEP_TE_ESTIMATORS,
            "DEEP_WINDOWS": DEEP_WINDOWS, "DEEP_FIELD_DEFS": DEEP_FIELD_DEFS,
            "DEEP_CMI_REGIMES": DEEP_CMI_REGIMES,
            "DEEP_THRESHOLD_MODES": DEEP_THRESHOLD_MODES,
            "DEEP_SINK_MODES": DEEP_SINK_MODES, "GRID_MAX_POS": GRID_MAX_POS,
        },
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "top_50": [r.to_dict() for r in valid[:50]]},
                  f, indent=2, ensure_ascii=False)
    print(f"\n  ✓ CSV ({len(all_s):,} filas) → {OUT_CSV}")
    print(f"  ✓ JSON top-50   → {OUT_JSON}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Optimizador bifásico DivergenceField — resultados idénticos al backtest"
    )
    parser.add_argument("--no-knn",     action="store_true")
    parser.add_argument("--workers",    type=int, default=None)
    parser.add_argument("--resume",     action="store_true")
    parser.add_argument("--top",        type=int, default=20)
    parser.add_argument("--min-trades", type=int, default=10, dest="min_trades")
    args = parser.parse_args()

    import config_local as CL

    compute_configs = all_compute_configs(include_knn=not args.no_knn)
    n_cc      = len(compute_configs)
    n_dec     = len(all_decision_combos())   # 432
    n_workers = args.workers or max(1, multiprocessing.cpu_count() - 1)

    print("╔══════════════════════════════════════════════════════════╗")
    print("║   OPTIMIZADOR BIFÁSICO — DivergenceField  BTC/USDT      ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"  Dataset         : {CL.FECHA_INICIO} → {CL.FECHA_FIN}")
    print(f"  Capital         : ${CL.SALDO_USDT_INICIAL:,.2f}  Comisión: {CL.COMMISSION_PCT}%")
    print(f"  MAX_POS grid    : {GRID_MAX_POS}")
    print(f"  Configs cómputo : {n_cc}  ×  {n_dec} decisión  =  {n_cc*n_dec:,} resultados")
    print(f"  Workers         : {n_workers} / {multiprocessing.cpu_count()} cores")
    print(f"  KNN             : {'No' if args.no_knn else 'Sí'}")
    print(f"  Min trades      : {args.min_trades}")
    print(f"  Checkpoint      : {'continuar' if args.resume else 'desde cero'}")
    print("─" * 60)

    # Estimación: solo Fase 1 importa (~30–90s por config binning/kde)
    n_bin = sum(1 for cc in compute_configs if cc.te_estimator == "binning")
    n_kde = sum(1 for cc in compute_configs if cc.te_estimator == "kde")
    n_knn = sum(1 for cc in compute_configs if cc.te_estimator == "knn")
    est_s = (n_bin * 60 + n_kde * 120 + n_knn * 600) / n_workers
    h, m  = divmod(int(est_s / 60), 60)
    print(f"  Tiempo estimado : ~{h}h {m:02d}min\n")

    results:      List[Result] = []
    done_cc_keys: set          = set()
    if args.resume:
        results, done_cc_keys = load_checkpoint()
        if results:
            print(f"  ✓ Checkpoint: {len(results):,} resultados  "
                  f"({len(done_cc_keys)} configs completas)")

    pending = [cc for cc in compute_configs if cc.key() not in done_cc_keys]
    if done_cc_keys:
        print(f"  ✓ Pendientes  : {len(pending)} / {n_cc} configs\n")

    if not pending:
        print("  ✓ Todas las configs calculadas — mostrando resultados\n")
    else:
        from actors.price_feed import SQLiteFeed
        feed = SQLiteFeed(db_path=CL.DB_PATH, table=CL.DB_TABLE)
        ini  = feed.get_candles(CL.FECHA_INICIO, CL.FECHA_INICIO, CL.SYMBOL)
        fin  = feed.get_candles(CL.FECHA_FIN,    CL.FECHA_FIN,    CL.SYMBOL)
        p_i  = ini[0].close if ini else 1.0
        p_f  = (fin or feed.get_candles(CL.FECHA_INICIO, CL.FECHA_FIN, CL.SYMBOL))[-1].close
        print(f"  B&H: {(p_f/p_i-1)*100:+.2f}%  |  ${p_i:,.0f} → ${p_f:,.0f}")
        print(f"  Lanzando {n_workers} workers...\n  ", end="")

        init_args = (
            CL.DB_PATH, CL.DB_TABLE,
            CL.FECHA_INICIO, CL.FECHA_FIN, CL.SYMBOL,
            CL.SALDO_USDT_INICIAL, CL.COMMISSION_PCT, args.min_trades,
        )

        t0           = time.time()
        best_pnl     = max((r.pnl_pct for r in results if r.valid), default=-9999.0)
        done_configs = len(done_cc_keys)
        saved_since  = 0

        ctx  = multiprocessing.get_context("spawn")
        pool = ctx.Pool(processes=n_workers, initializer=_worker_init,
                        initargs=init_args)

        try:
            for batch in pool.imap_unordered(_run_compute_config_task,
                                              pending, chunksize=1):
                done_configs += 1
                saved_since  += 1
                if batch:
                    results.extend(batch)
                    for r in batch:
                        if r.valid and r.pnl_pct > best_pnl:
                            best_pnl = r.pnl_pct

                pct   = done_configs / n_cc
                bar   = "█"*int(28*pct) + "░"*(28-int(28*pct))
                el    = time.time() - t0
                eta   = (el/pct - el) if pct > 0.001 else 0
                eta_s = (f"{eta/3600:.1f}h" if eta>3600 else
                         f"{eta/60:.0f}m"   if eta>60   else f"{eta:.0f}s")
                bp_s  = (f"{'+' if best_pnl>=0 else ''}{best_pnl:.1f}%"
                         if best_pnl > -9999 else "—")
                print(f"\r  [{bar}] {pct*100:5.1f}%  "
                      f"{done_configs:>4}/{n_cc} configs  "
                      f"ETA:{eta_s:>7}  Best:{bp_s:>8} ",
                      end="", flush=True)

                if saved_since >= SAVE_EVERY:
                    save_checkpoint(results)
                    saved_since = 0

        except KeyboardInterrupt:
            print(f"\n\n  [CTRL+C] Guardando {len(results):,} resultados...")
            pool.terminate(); pool.join()
            save_checkpoint(results)
            print(f"  ✓ {CHECKPOINT_CSV}  |  --resume para continuar")
            _show_results(results, args.top, args.min_trades, CL)
            return
        else:
            pool.close(); pool.join()
            print()

        elapsed = time.time() - t0
        print(f"\n  ✓ {elapsed/60:.1f} min  |  "
              f"{len(results):,} resultados  |  "
              f"{sum(1 for r in results if r.valid):,} válidos")

    _show_results(results, args.top, args.min_trades, CL)

    if Path(CHECKPOINT_CSV).exists():
        try: os.remove(CHECKPOINT_CSV)
        except Exception: pass

    print("\n✓ Optimización completada.")


def _show_results(results: List[Result], top_n: int, min_trades: int, CL) -> None:
    if not results:
        print("  Sin resultados."); return
    print_table(results, top_n=top_n, min_trades=min_trades)
    save_final(results, 0, min_trades)
    valid = sorted([r for r in results if r.valid],
                   key=lambda r: r.pnl_pct, reverse=True)
    if valid:
        print_best(valid[0], CL.SALDO_USDT_INICIAL, CL.COMMISSION_PCT)
    else:
        print(f"\n  ⚠ Sin configs válidas con ≥{min_trades} trades.")
        print("  Intentar: python optimize_divergence_field.py --min-trades 5")


if __name__ == "__main__":
    main()
