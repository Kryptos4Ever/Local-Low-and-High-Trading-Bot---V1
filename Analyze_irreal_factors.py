"""
analyze_irreal_factors.py — Análisis de Factores Predictivos (Irreal)
═══════════════════════════════════════════════════════════════════════
Detecta los bottoms y tops del oráculo perfecto sobre la DB de precios
y analiza qué factores técnicos individuales —medidos sobre las velas
PREVIAS a cada evento— son los mejores predictores de esos eventos.

Cada factor se analiza de forma independiente (sin combinaciones).
Para cada factor × ventana (10..24) se calcula:
  · AUC Mann-Whitney  (separabilidad factor vs candles neutros)
  · Cohen's d         (tamaño del efecto)
  · Dirección         (HIGH = factor alto predice evento / LOW = factor bajo)
  · Estadísticas descriptivas por grupo (mean, median, std)

Salida:
  · factors_analysis.csv      — tabla completa factor × ventana
  · factors_analysis.json     — idem + resumen con mejores factores
  · factors_ranking.png       — heatmaps AUC + bar charts top-10
  · (consola) resumen ejecutivo

Uso:
    python analyze_irreal_factors.py

Dependencias:
    numpy, matplotlib  (recomendadas, el script degrada sin ellas)
    sqlite3            (stdlib)
"""

from __future__ import annotations

import csv
import json
import math
import random
import sqlite3
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── Importar config local ─────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
import config_local as CL

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import matplotlib.ticker as mticker
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ══════════════════════════════════════════════════════════════════════════════

VENTANA_ORACULO   = 10                   # debe coincidir con backtest_irreal.py
VENTANAS_ANALISIS = list(range(10, 25))  # [10, 11, ..., 24]

# Cuántos candles neutros muestrear (para equilibrar vs N eventos, evitar OOM)
MAX_NEUTROS = 600

OUTPUT_CSV  = "factors_analysis.csv"
OUTPUT_JSON = "factors_analysis.json"
OUTPUT_PNG  = "factors_ranking.png"

DARK_MODE = getattr(CL, "DARK_MODE", True)


# ══════════════════════════════════════════════════════════════════════════════
# ESTRUCTURAS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Candle:
    ts:     int
    open:   float
    high:   float
    low:    float
    close:  float
    volume: float


class EV:          # event types
    BOTTOM  = "BOTTOM"
    TOP     = "TOP"
    NEUTRAL = "NEUTRAL"


@dataclass
class Event:
    idx:        int
    ts:         int
    price:      float
    event_type: str


# ══════════════════════════════════════════════════════════════════════════════
# CARGA DE DATOS DESDE SQLITE
# ══════════════════════════════════════════════════════════════════════════════

def load_candles(db_path: str, table: str,
                 fecha_inicio: str, fecha_fin: str) -> List[Candle]:
    """Lee velas en el rango dado desde la DB SQLite."""
    conn = sqlite3.connect(db_path)
    cur  = conn.cursor()

    # Detectar columnas disponibles
    cur.execute(f"PRAGMA table_info({table})")
    cols     = [row[1].lower() for row in cur.fetchall()]
    vol_col  = "volume" if "volume" in cols else "0.0"
    ts_col   = "timestamp" if "timestamp" in cols else "ts"

    # Detectar si el timestamp está en ms o en s
    cur.execute(f"SELECT {ts_col} FROM {table} ORDER BY {ts_col} ASC LIMIT 1")
    sample_ts = cur.fetchone()[0]
    ts_divisor = 1000 if sample_ts > 1e12 else 1      # ms → s

    query = f"""
        SELECT {ts_col}, open, high, low, close, {vol_col}
        FROM   {table}
        WHERE  date(datetime({ts_col}/{ts_divisor}, 'unixepoch'))
               BETWEEN ? AND ?
        ORDER BY {ts_col} ASC
    """
    cur.execute(query, (fecha_inicio, fecha_fin))
    rows   = cur.fetchall()
    conn.close()

    return [
        Candle(ts=r[0], open=float(r[1]), high=float(r[2]),
               low=float(r[3]),  close=float(r[4]), volume=float(r[5]))
        for r in rows
    ]


# ══════════════════════════════════════════════════════════════════════════════
# DETECCIÓN DE EVENTOS (ORÁCULO PERFECTO)
# ══════════════════════════════════════════════════════════════════════════════

def detect_events(candles: List[Candle], ventana: int) -> List[Event]:
    """
    Replica la lógica de IrrealStrategy: detecta mínimos y máximos
    locales con ventana N a cada lado.
    """
    events: List[Event] = []
    n = len(candles)

    for i in range(ventana, n - ventana):
        centro   = candles[i]
        vecinos  = [candles[j] for j in range(i - ventana, i + ventana + 1) if j != i]

        es_min = all(centro.low  <= v.low  for v in vecinos)
        es_max = all(centro.high >= v.high for v in vecinos)

        # Prioridad BUY si ambos (vela plana extrema)
        if es_min:
            events.append(Event(
                idx=i, ts=centro.ts,
                price=centro.low, event_type=EV.BOTTOM,
            ))
        elif es_max:
            events.append(Event(
                idx=i, ts=centro.ts,
                price=centro.high, event_type=EV.TOP,
            ))

    return events


def get_neutral_indices(candles: List[Candle], events: List[Event],
                        ventana_max: int) -> List[int]:
    """
    Retorna índices de velas neutras (ni bottom ni top) con buffer
    suficiente a cada lado para que cualquier ventana de análisis quepa.
    """
    event_set = {e.idx for e in events}
    return [
        i for i in range(ventana_max, len(candles) - ventana_max)
        if i not in event_set
    ]


# ══════════════════════════════════════════════════════════════════════════════
# FACTORES TÉCNICOS
# ══════════════════════════════════════════════════════════════════════════════

def _rsi(closes: List[float], period: int = 14) -> float:
    """RSI clásico de Wilder sobre los últimos `period` deltas."""
    if len(closes) < period + 1:
        return 50.0
    deltas = [closes[i] - closes[i-1] for i in range(1, len(closes))]
    gains  = [max(d, 0) for d in deltas[-period:]]
    losses = [max(-d, 0) for d in deltas[-period:]]
    ag = sum(gains)  / period
    al = sum(losses) / period
    if al == 0:
        return 100.0
    return round(100 - 100 / (1 + ag / al), 4)


def _linreg_slope_norm(values: List[float]) -> float:
    """Pendiente de regresión lineal, normalizada por el valor medio (% por vela)."""
    n = len(values)
    if n < 2:
        return 0.0
    xm = (n - 1) / 2
    ym = sum(values) / n
    num = sum((i - xm) * (values[i] - ym) for i in range(n))
    den = sum((i - xm) ** 2 for i in range(n))
    if den == 0 or ym == 0:
        return 0.0
    return round((num / den) / ym * 100, 4)


def _streak(closes: List[float], opens: List[float], direction: str) -> int:
    """Velas consecutivas bajistas ('bear') o alcistas ('bull') al final."""
    count = 0
    for i in range(len(closes)-1, -1, -1):
        if direction == "bear" and closes[i] < opens[i]:
            count += 1
        elif direction == "bull" and closes[i] > opens[i]:
            count += 1
        else:
            break
    return count


def compute_factors(window: List[Candle]) -> Dict[str, float]:
    """
    15 factores técnicos sobre la ventana de velas dada.
    La ventana son las N velas PREVIAS al evento (sin incluir la vela del evento).
    """
    if not window:
        return {}

    closes  = [c.close  for c in window]
    highs   = [c.high   for c in window]
    lows    = [c.low    for c in window]
    opens   = [c.open   for c in window]
    volumes = [c.volume for c in window]
    n       = len(window)
    last    = closes[-1]
    mean_c  = sum(closes) / n

    # ── 1. RSI ───────────────────────────────────────────────────────────────
    rsi = _rsi(closes)

    # ── 2. ATR relativo al precio (%): volatilidad reciente ──────────────────
    true_ranges = [
        max(highs[i] - lows[i],
            abs(highs[i] - closes[i-1]),
            abs(lows[i]  - closes[i-1]))
        for i in range(1, n)
    ]
    atr     = sum(true_ranges) / len(true_ranges) if true_ranges else 0
    atr_rel = (atr / last * 100) if last else 0

    # ── 3. Momentum: % cambio total del close en la ventana ──────────────────
    momentum = (closes[-1] - closes[0]) / closes[0] * 100 if closes[0] else 0

    # ── 4. Volume ratio: volumen final vs media de la ventana ────────────────
    mean_vol  = sum(volumes) / n if n else 1
    vol_ratio = volumes[-1] / mean_vol if mean_vol else 1

    # ── 5. Pendiente del precio (regresión lineal normalizada) ───────────────
    price_slope = _linreg_slope_norm(closes)

    # ── 6. Distancia al SMA de la ventana (%) ────────────────────────────────
    dist_sma = (last - mean_c) / mean_c * 100 if mean_c else 0

    # ── 7. Bollinger Band %B: posición del close dentro de las bandas ────────
    variance = sum((c - mean_c)**2 for c in closes) / n
    std_c    = math.sqrt(variance) if variance > 0 else 0
    upper    = mean_c + 2 * std_c
    lower    = mean_c - 2 * std_c
    bb_width = upper - lower
    bb_pos   = (last - lower) / bb_width if bb_width > 0 else 0.5

    # ── 8. Drawdown desde máximo local de la ventana (%) ────────────────────
    max_high = max(highs)
    drawdown = (max_high - last) / max_high * 100 if max_high else 0

    # ── 9. Recovery desde mínimo local de la ventana (%) ────────────────────
    min_low  = min(lows)
    recovery = (last - min_low) / min_low * 100 if min_low else 0

    # ── 10. Body ratio medio: cuerpo / rango (solidez de velas) ─────────────
    body_ratios = [
        abs(closes[i] - opens[i]) / (highs[i] - lows[i])
        if (highs[i] - lows[i]) > 0 else 0
        for i in range(n)
    ]
    mean_body_ratio = sum(body_ratios) / n

    # ── 11. Pendiente del volumen (tendencia alcista/bajista del volumen) ────
    vol_slope = _linreg_slope_norm(volumes)

    # ── 12. HL spread medio (%): volatilidad intra-vela ─────────────────────
    hl_spreads = [
        (highs[i] - lows[i]) / closes[i] * 100
        for i in range(n) if closes[i] > 0
    ]
    mean_hl_spread = sum(hl_spreads) / len(hl_spreads) if hl_spreads else 0

    # ── 13. Bear streak: velas bajistas consecutivas al final ────────────────
    bear_streak = _streak(closes, opens, "bear")

    # ── 14. Bull streak: velas alcistas consecutivas al final ────────────────
    bull_streak = _streak(closes, opens, "bull")

    # ── 15. Close position: dónde está el último close dentro del rango total
    total_high  = max(highs)
    total_low   = min(lows)
    total_range = total_high - total_low
    close_pos   = (last - total_low) / total_range if total_range > 0 else 0.5

    return {
        "rsi":            round(rsi, 4),
        "atr_rel_pct":    round(atr_rel, 4),
        "momentum_pct":   round(momentum, 4),
        "volume_ratio":   round(vol_ratio, 4),
        "price_slope":    round(price_slope, 4),
        "dist_sma_pct":   round(dist_sma, 4),
        "bb_position":    round(bb_pos, 4),
        "drawdown_pct":   round(drawdown, 4),
        "recovery_pct":   round(recovery, 4),
        "body_ratio":     round(mean_body_ratio, 4),
        "volume_slope":   round(vol_slope, 4),
        "hl_spread_pct":  round(mean_hl_spread, 4),
        "bear_streak":    float(bear_streak),
        "bull_streak":    float(bull_streak),
        "close_position": round(close_pos, 4),
    }


# ══════════════════════════════════════════════════════════════════════════════
# ESTADÍSTICAS
# ══════════════════════════════════════════════════════════════════════════════

def _mean(vals: List[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0

def _std(vals: List[float]) -> float:
    if len(vals) < 2:
        return 0.0
    m = _mean(vals)
    return math.sqrt(sum((v - m)**2 for v in vals) / len(vals))

def _median(vals: List[float]) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    n = len(s)
    return (s[n // 2] + s[(n - 1) // 2]) / 2


def auc_mann_whitney(group: List[float], neutral: List[float]) -> Tuple[float, str]:
    """
    AUC basada en Mann-Whitney U (no paramétrica).
    Retorna (auc ≥ 0.5, direction).
    direction="HIGH" → factor ALTO antes del evento.
    direction="LOW"  → factor BAJO antes del evento.
    """
    if not group or not neutral:
        return 0.5, "N/A"

    n1, n2 = len(group), len(neutral)

    if HAS_NUMPY:
        # Vectorizado: más rápido para muestras grandes
        g = np.array(group)
        neu = np.array(neutral)
        u = float(np.sum(g[:, None] > neu[None, :]) +
                  0.5 * np.sum(g[:, None] == neu[None, :]))
    else:
        u = sum(
            1.0 if g > neu else 0.5 if g == neu else 0.0
            for g in group
            for neu in neutral
        )

    auc_raw = u / (n1 * n2)
    direction = "HIGH" if auc_raw >= 0.5 else "LOW"
    return round(max(auc_raw, 1 - auc_raw), 4), direction


def cohens_d(group: List[float], neutral: List[float]) -> float:
    """Effect size de Cohen's d (sin signo)."""
    if not group or not neutral:
        return 0.0
    m1, m2   = _mean(group), _mean(neutral)
    s1, s2   = _std(group),  _std(neutral)
    n1, n2   = len(group),   len(neutral)
    denom    = n1 + n2 - 2
    if denom <= 0:
        return 0.0
    pooled = math.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / denom)
    return round(abs(m1 - m2) / pooled if pooled > 0 else 0.0, 4)


# ══════════════════════════════════════════════════════════════════════════════
# ANÁLISIS PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

def build_factor_matrix(
    candles:      List[Candle],
    events:       List[Event],
    neutral_idxs: List[int],
    ventanas:     List[int],
) -> List[Dict]:
    """
    Para cada factor × ventana calcula estadísticas comparativas entre
    bottoms, tops y neutros.
    Retorna lista de registros (una fila por factor × ventana).
    """
    bottoms = [e for e in events if e.event_type == EV.BOTTOM]
    tops    = [e for e in events if e.event_type == EV.TOP]

    print(f"\n  Eventos detectados : {len(bottoms)} bottoms  |  {len(tops)} tops")
    print(f"  Candles neutros    : {len(neutral_idxs):,}  (muestrando {min(MAX_NEUTROS, len(neutral_idxs))})")

    random.seed(42)
    sample_neu = random.sample(neutral_idxs, min(MAX_NEUTROS, len(neutral_idxs)))

    all_records: List[Dict] = []

    for ventana in ventanas:
        print(f"  ventana={ventana:>2d} ... ", end="", flush=True)
        t0 = time.time()

        # Extraer factores para cada grupo
        bot_f: Dict[str, List[float]] = {}
        top_f: Dict[str, List[float]] = {}
        neu_f: Dict[str, List[float]] = {}

        for ev_list, store in [(bottoms, bot_f), (tops, top_f)]:
            for ev in ev_list:
                start = ev.idx - ventana
                if start < 0:
                    continue
                window = candles[start : ev.idx]   # velas PREVIAS, sin incluir la del evento
                for k, v in compute_factors(window).items():
                    store.setdefault(k, []).append(v)

        for idx in sample_neu:
            start = idx - ventana
            if start < 0:
                continue
            window = candles[start : idx]
            for k, v in compute_factors(window).items():
                neu_f.setdefault(k, []).append(v)

        # Estadísticas por factor
        factor_names = sorted(set(bot_f) | set(top_f))
        for factor in factor_names:
            bv = bot_f.get(factor, [])
            tv = top_f.get(factor, [])
            nv = neu_f.get(factor, [])

            auc_b, dir_b = auc_mann_whitney(bv, nv)
            auc_t, dir_t = auc_mann_whitney(tv, nv)
            cd_b         = cohens_d(bv, nv)
            cd_t         = cohens_d(tv, nv)

            # Score compuesto: (AUC - 0.5) * 2  ∈ [0,1]  ×  Cohen's d
            # Captura separabilidad y tamaño del efecto simultáneamente
            score_b = round((auc_b - 0.5) * 2 * cd_b, 4)
            score_t = round((auc_t - 0.5) * 2 * cd_t, 4)

            all_records.append({
                "factor":       factor,
                "ventana":      ventana,
                # BOTTOM
                "bot_n":        len(bv),
                "bot_mean":     round(_mean(bv), 4),
                "bot_median":   round(_median(bv), 4),
                "bot_std":      round(_std(bv), 4),
                "bot_auc":      auc_b,
                "bot_dir":      dir_b,
                "bot_cohens_d": cd_b,
                "bot_score":    score_b,
                # TOP
                "top_n":        len(tv),
                "top_mean":     round(_mean(tv), 4),
                "top_median":   round(_median(tv), 4),
                "top_std":      round(_std(tv), 4),
                "top_auc":      auc_t,
                "top_dir":      dir_t,
                "top_cohens_d": cd_t,
                "top_score":    score_t,
                # NEUTRO (referencia)
                "neu_n":        len(nv),
                "neu_mean":     round(_mean(nv), 4),
                "neu_median":   round(_median(nv), 4),
                "neu_std":      round(_std(nv), 4),
            })

        n_bot_ok = sum(1 for e in bottoms if e.idx - ventana >= 0)
        n_top_ok = sum(1 for e in tops    if e.idx - ventana >= 0)
        print(f"OK  bot={n_bot_ok}  top={n_top_ok}  ({time.time()-t0:.1f}s)")

    return all_records


# ══════════════════════════════════════════════════════════════════════════════
# SÍNTESIS: MEJOR VENTANA POR FACTOR
# ══════════════════════════════════════════════════════════════════════════════

def synthesize(records: List[Dict]) -> Dict:
    """
    Para cada factor, encuentra la ventana que maximiza el AUC
    para BOTTOM y TOP por separado.
    Retorna un dict de resúmenes ordenados por AUC descendente.
    """
    from collections import defaultdict

    # Agrupar por factor
    by_factor: Dict[str, List[Dict]] = defaultdict(list)
    for r in records:
        by_factor[r["factor"]].append(r)

    summary_bot = []
    summary_top = []

    for factor, rows in by_factor.items():
        # BOTTOM: mejor ventana por AUC
        best_b = max(rows, key=lambda r: r["bot_auc"])
        # Promedio AUC a través de ventanas
        avg_auc_b = round(sum(r["bot_auc"] for r in rows) / len(rows), 4)
        summary_bot.append({
            "factor":       factor,
            "avg_auc":      avg_auc_b,
            "best_ventana": best_b["ventana"],
            "best_auc":     best_b["bot_auc"],
            "best_dir":     best_b["bot_dir"],
            "best_cohens_d":best_b["bot_cohens_d"],
            "best_score":   best_b["bot_score"],
            "bot_mean_best":best_b["bot_mean"],
            "neu_mean_best":best_b["neu_mean"],
            "interpretacion": (
                f"Factor {'alto' if best_b['bot_dir']=='HIGH' else 'bajo'} "
                f"predice BOTTOM (ventana={best_b['ventana']}, AUC={best_b['bot_auc']})"
            ),
        })

        # TOP: mejor ventana por AUC
        best_t = max(rows, key=lambda r: r["top_auc"])
        avg_auc_t = round(sum(r["top_auc"] for r in rows) / len(rows), 4)
        summary_top.append({
            "factor":       factor,
            "avg_auc":      avg_auc_t,
            "best_ventana": best_t["ventana"],
            "best_auc":     best_t["top_auc"],
            "best_dir":     best_t["top_dir"],
            "best_cohens_d":best_t["top_cohens_d"],
            "best_score":   best_t["top_score"],
            "top_mean_best":best_t["top_mean"],
            "neu_mean_best":best_t["neu_mean"],
            "interpretacion": (
                f"Factor {'alto' if best_t['top_dir']=='HIGH' else 'bajo'} "
                f"predice TOP (ventana={best_t['ventana']}, AUC={best_t['top_auc']})"
            ),
        })

    summary_bot.sort(key=lambda x: -x["best_auc"])
    summary_top.sort(key=lambda x: -x["best_auc"])

    return {
        "top_predictores_BOTTOM": summary_bot,
        "top_predictores_TOP":    summary_top,
    }


# ══════════════════════════════════════════════════════════════════════════════
# SALIDAS
# ══════════════════════════════════════════════════════════════════════════════

def save_csv(records: List[Dict], path: str) -> None:
    if not records:
        return
    fieldnames = list(records[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(records)
    print(f"  ✓ CSV  → {path}  ({len(records):,} filas)")


def save_json(records: List[Dict], synthesis: Dict,
              events: List[Event], path: str) -> None:
    output = {
        "meta": {
            "fecha_inicio":    CL.FECHA_INICIO,
            "fecha_fin":       CL.FECHA_FIN,
            "ventana_oraculo": VENTANA_ORACULO,
            "ventanas_analisis": VENTANAS_ANALISIS,
            "n_bottoms": sum(1 for e in events if e.event_type == EV.BOTTOM),
            "n_tops":    sum(1 for e in events if e.event_type == EV.TOP),
        },
        "ranking": synthesis,
        "detalle": records,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"  ✓ JSON → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# VISUALIZACIÓN
# ══════════════════════════════════════════════════════════════════════════════

def plot_ranking(records: List[Dict], synthesis: Dict,
                 path: str, dark: bool = True) -> None:
    if not HAS_MPL:
        print("  ✗ matplotlib no disponible — saltando visualización")
        return

    style = "dark_background" if dark else "default"
    plt.style.use(style)
    bg   = "#0e1117" if dark else "#ffffff"
    fg   = "#e0e0e0" if dark else "#1a1a1a"
    acc  = "#00bfff" if dark else "#1f77b4"
    acc2 = "#ff6b6b" if dark else "#d62728"

    fig = plt.figure(figsize=(22, 18), facecolor=bg)
    fig.suptitle(
        f"Análisis de Factores Predictivos  ·  Irreal Oráculo  ·  "
        f"{CL.FECHA_INICIO} → {CL.FECHA_FIN}",
        fontsize=14, color=fg, fontweight="bold", y=0.98,
    )

    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.42, wspace=0.32)

    factor_names  = sorted({r["factor"] for r in records})
    ventana_vals  = sorted({r["ventana"] for r in records})

    # ── Helpers ───────────────────────────────────────────────────────────────
    def build_heatmap_data(event_key: str) -> List[List[float]]:
        """factor × ventana → AUC matrix"""
        lut = {(r["factor"], r["ventana"]): r[event_key] for r in records}
        return [
            [lut.get((f, v), 0.5) for v in ventana_vals]
            for f in factor_names
        ]

    # ── 1. Heatmap AUC BOTTOM ─────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    data_b = build_heatmap_data("bot_auc")
    im1 = ax1.imshow(data_b, aspect="auto", cmap="RdYlGn",
                     vmin=0.5, vmax=1.0, origin="upper")
    ax1.set_xticks(range(len(ventana_vals)))
    ax1.set_xticklabels(ventana_vals, fontsize=7, color=fg)
    ax1.set_yticks(range(len(factor_names)))
    ax1.set_yticklabels(factor_names, fontsize=8, color=fg)
    ax1.set_xlabel("Ventana (velas previas)", color=fg, fontsize=9)
    ax1.set_title("AUC por Factor × Ventana  —  BOTTOM", color=fg, fontsize=10, pad=8)
    plt.colorbar(im1, ax=ax1).ax.yaxis.set_tick_params(color=fg)
    for spine in ax1.spines.values():
        spine.set_edgecolor(fg)
    ax1.tick_params(colors=fg)

    # ── 2. Heatmap AUC TOP ────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    data_t = build_heatmap_data("top_auc")
    im2 = ax2.imshow(data_t, aspect="auto", cmap="RdYlGn",
                     vmin=0.5, vmax=1.0, origin="upper")
    ax2.set_xticks(range(len(ventana_vals)))
    ax2.set_xticklabels(ventana_vals, fontsize=7, color=fg)
    ax2.set_yticks(range(len(factor_names)))
    ax2.set_yticklabels(factor_names, fontsize=8, color=fg)
    ax2.set_xlabel("Ventana (velas previas)", color=fg, fontsize=9)
    ax2.set_title("AUC por Factor × Ventana  —  TOP", color=fg, fontsize=10, pad=8)
    plt.colorbar(im2, ax=ax2).ax.yaxis.set_tick_params(color=fg)
    for spine in ax2.spines.values():
        spine.set_edgecolor(fg)
    ax2.tick_params(colors=fg)

    # ── 3. Bar chart: top-10 factores para BOTTOM ─────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    top10_b = synthesis["top_predictores_BOTTOM"][:10]
    names_b = [r["factor"] for r in top10_b]
    aucs_b  = [r["best_auc"] for r in top10_b]
    dirs_b  = [r["best_dir"] for r in top10_b]
    cols_b  = [acc if d == "HIGH" else acc2 for d in dirs_b]
    bars = ax3.barh(range(len(names_b)), aucs_b, color=cols_b, alpha=0.85)
    ax3.axvline(0.5, color=fg, lw=0.8, ls="--", alpha=0.5)
    ax3.set_yticks(range(len(names_b)))
    ax3.set_yticklabels(names_b, fontsize=9, color=fg)
    ax3.set_xlim(0.48, 1.0)
    ax3.set_xlabel("AUC (mejor ventana)", color=fg, fontsize=9)
    ax3.set_title("Top-10 Factores → BOTTOM", color=fg, fontsize=10, pad=8)
    ax3.invert_yaxis()
    for i, (bar, d) in enumerate(zip(bars, dirs_b)):
        ax3.text(bar.get_width() + 0.003, i,
                 f"{'▲' if d=='HIGH' else '▼'} {aucs_b[i]:.3f}",
                 va="center", fontsize=8, color=fg)
    ax3.set_facecolor(bg)
    ax3.tick_params(colors=fg)
    for spine in ax3.spines.values():
        spine.set_edgecolor(fg)

    # ── 4. Bar chart: top-10 factores para TOP ────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    top10_t = synthesis["top_predictores_TOP"][:10]
    names_t = [r["factor"] for r in top10_t]
    aucs_t  = [r["best_auc"] for r in top10_t]
    dirs_t  = [r["best_dir"] for r in top10_t]
    cols_t  = [acc if d == "HIGH" else acc2 for d in dirs_t]
    bars2 = ax4.barh(range(len(names_t)), aucs_t, color=cols_t, alpha=0.85)
    ax4.axvline(0.5, color=fg, lw=0.8, ls="--", alpha=0.5)
    ax4.set_yticks(range(len(names_t)))
    ax4.set_yticklabels(names_t, fontsize=9, color=fg)
    ax4.set_xlim(0.48, 1.0)
    ax4.set_xlabel("AUC (mejor ventana)", color=fg, fontsize=9)
    ax4.set_title("Top-10 Factores → TOP", color=fg, fontsize=10, pad=8)
    ax4.invert_yaxis()
    for i, (bar, d) in enumerate(zip(bars2, dirs_t)):
        ax4.text(bar.get_width() + 0.003, i,
                 f"{'▲' if d=='HIGH' else '▼'} {aucs_t[i]:.3f}",
                 va="center", fontsize=8, color=fg)
    ax4.set_facecolor(bg)
    ax4.tick_params(colors=fg)
    for spine in ax4.spines.values():
        spine.set_edgecolor(fg)

    # ── 5. AUC vs ventana: top-5 factores BOTTOM ─────────────────────────────
    ax5 = fig.add_subplot(gs[2, 0])
    top5_b_names = [r["factor"] for r in synthesis["top_predictores_BOTTOM"][:5]]
    lut_b = {(r["factor"], r["ventana"]): r["bot_auc"] for r in records}
    cmap5 = plt.cm.get_cmap("tab10", 5)
    for j, fname in enumerate(top5_b_names):
        aucs_line = [lut_b.get((fname, v), 0.5) for v in ventana_vals]
        ax5.plot(ventana_vals, aucs_line, marker="o", ms=4, lw=1.5,
                 label=fname, color=cmap5(j))
    ax5.axhline(0.5, color=fg, lw=0.7, ls="--", alpha=0.4)
    ax5.set_xlabel("Ventana", color=fg, fontsize=9)
    ax5.set_ylabel("AUC", color=fg, fontsize=9)
    ax5.set_title("Sensibilidad a la Ventana — Top-5 BOTTOM", color=fg, fontsize=10, pad=8)
    ax5.legend(fontsize=7, facecolor=bg, labelcolor=fg, framealpha=0.6)
    ax5.set_facecolor(bg)
    ax5.tick_params(colors=fg)
    for spine in ax5.spines.values():
        spine.set_edgecolor(fg)

    # ── 6. AUC vs ventana: top-5 factores TOP ────────────────────────────────
    ax6 = fig.add_subplot(gs[2, 1])
    top5_t_names = [r["factor"] for r in synthesis["top_predictores_TOP"][:5]]
    lut_t = {(r["factor"], r["ventana"]): r["top_auc"] for r in records}
    for j, fname in enumerate(top5_t_names):
        aucs_line = [lut_t.get((fname, v), 0.5) for v in ventana_vals]
        ax6.plot(ventana_vals, aucs_line, marker="o", ms=4, lw=1.5,
                 label=fname, color=cmap5(j))
    ax6.axhline(0.5, color=fg, lw=0.7, ls="--", alpha=0.4)
    ax6.set_xlabel("Ventana", color=fg, fontsize=9)
    ax6.set_ylabel("AUC", color=fg, fontsize=9)
    ax6.set_title("Sensibilidad a la Ventana — Top-5 TOP", color=fg, fontsize=10, pad=8)
    ax6.legend(fontsize=7, facecolor=bg, labelcolor=fg, framealpha=0.6)
    ax6.set_facecolor(bg)
    ax6.tick_params(colors=fg)
    for spine in ax6.spines.values():
        spine.set_edgecolor(fg)

    # ── Leyenda global ────────────────────────────────────────────────────────
    fig.text(
        0.5, 0.003,
        "▲ = factor ALTO antes del evento  |  ▼ = factor BAJO antes del evento  |  "
        "AUC=0.5 → sin poder predictivo  |  AUC=1.0 → separación perfecta",
        ha="center", fontsize=8, color=fg, alpha=0.7,
    )

    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=bg)
    plt.close(fig)
    print(f"  ✓ PNG  → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# CONSOLA: RESUMEN EJECUTIVO
# ══════════════════════════════════════════════════════════════════════════════

def print_summary(synthesis: Dict, events: List[Event]) -> None:
    n_b = sum(1 for e in events if e.event_type == EV.BOTTOM)
    n_t = sum(1 for e in events if e.event_type == EV.TOP)

    sep  = "═" * 70
    sep2 = "─" * 70

    print(f"\n{sep}")
    print("  RANKING DE FACTORES — TOP PREDICTORES")
    print(sep)
    print(f"  Oráculo ventana={VENTANA_ORACULO}  |  "
          f"Bottoms={n_b}  Tops={n_t}  |  "
          f"Período: {CL.FECHA_INICIO} → {CL.FECHA_FIN}")

    def print_table(title: str, rows: List[Dict], col_mean: str, ev_label: str) -> None:
        print(f"\n  {'─'*66}")
        print(f"  {title}")
        print(f"  {'─'*66}")
        hdr = (f"  {'FACTOR':<18} {'VENT':>4} {'AUC':>6} {'DIR':>5} "
               f"{'Cd':>6} {'Score':>7} {'Mean_ev':>9} {'Mean_neu':>9}")
        print(hdr)
        print(f"  {sep2}")
        for i, r in enumerate(rows[:10], 1):
            print(
                f"  {i:>2}. {r['factor']:<16} "
                f"{r['best_ventana']:>4}  "
                f"{r['best_auc']:>6.4f}  "
                f"{'▲' if r['best_dir']=='HIGH' else '▼':>4}  "
                f"{r['best_cohens_d']:>6.3f}  "
                f"{r['best_score']:>7.4f}  "
                f"{r[col_mean]:>9.3f}  "
                f"{r['neu_mean_best']:>9.3f}"
            )

    print_table(
        "TOP-10 PREDICTORES DE BOTTOM",
        synthesis["top_predictores_BOTTOM"],
        col_mean="bot_mean_best",
        ev_label="bot",
    )
    print_table(
        "TOP-10 PREDICTORES DE TOP",
        synthesis["top_predictores_TOP"],
        col_mean="top_mean_best",
        ev_label="top",
    )

    print(f"\n{sep}")
    print("  INTERPRETACIÓN")
    print(sep)
    for r in synthesis["top_predictores_BOTTOM"][:5]:
        print(f"  → {r['interpretacion']}")
    for r in synthesis["top_predictores_TOP"][:5]:
        print(f"  → {r['interpretacion']}")
    print(sep)
    print("  AUC: probabilidad de que el factor identifique correctamente el evento.")
    print("  0.5 = azar | 0.6 = leve | 0.7 = moderado | 0.8+ = fuerte predictor")
    print(sep)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    t0 = time.time()

    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║   ANÁLISIS DE FACTORES PREDICTIVOS — ORÁCULO IRREAL BTC/USDT   ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print(f"  Rango         : {CL.FECHA_INICIO} → {CL.FECHA_FIN}")
    print(f"  Ventana oráculo: {VENTANA_ORACULO}")
    print(f"  Ventanas análisis: {VENTANAS_ANALISIS[0]}..{VENTANAS_ANALISIS[-1]}")
    print(f"  numpy         : {'✓' if HAS_NUMPY else '✗ (modo lento)'}")
    print(f"  matplotlib    : {'✓' if HAS_MPL   else '✗ (sin gráfico)'}")
    print("─" * 68)

    # ── 1. Cargar velas ──────────────────────────────────────────────────────
    print("\n[1/5] Cargando velas desde DB...")
    candles = load_candles(CL.DB_PATH, CL.DB_TABLE, CL.FECHA_INICIO, CL.FECHA_FIN)
    print(f"  ✓ {len(candles):,} velas cargadas")

    if len(candles) < 2 * VENTANA_ORACULO + 1:
        print("✗ Dataset demasiado pequeño.")
        return

    # ── 2. Detectar eventos ──────────────────────────────────────────────────
    print("\n[2/5] Detectando bottoms y tops (oráculo perfecto)...")
    events = detect_events(candles, VENTANA_ORACULO)

    neutral_idxs = get_neutral_indices(candles, events, max(VENTANAS_ANALISIS))
    print(f"  ✓ {len(events)} eventos  |  {len(neutral_idxs):,} neutros")

    # ── 3. Calcular factores ─────────────────────────────────────────────────
    print("\n[3/5] Calculando factores por ventana...")
    records = build_factor_matrix(candles, events, neutral_idxs, VENTANAS_ANALISIS)
    print(f"  ✓ {len(records):,} registros (factores × ventanas)")

    # ── 4. Síntesis ──────────────────────────────────────────────────────────
    print("\n[4/5] Sintetizando ranking...")
    synthesis = synthesize(records)

    # ── 5. Guardar salidas ───────────────────────────────────────────────────
    print("\n[5/5] Guardando salidas...")
    save_csv(records, OUTPUT_CSV)
    save_json(records, synthesis, events, OUTPUT_JSON)
    plot_ranking(records, synthesis, OUTPUT_PNG, dark=DARK_MODE)

    print_summary(synthesis, events)

    print(f"\n✓ Completado en {time.time()-t0:.1f}s")
    print(f"  factors_analysis.csv  |  factors_analysis.json  |  factors_ranking.png")


if __name__ == "__main__":
    main()