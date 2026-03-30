"""
analyze_turning_points.py — Análisis Estadístico de Patrones en Turning Points
═══════════════════════════════════════════════════════════════════════════════
Para cada turning point detectado, extrae features del segmento de velas
que lo precedió (desde el turning point opuesto anterior) y analiza
qué patrones consistentemente predijeron el giro.

Outputs:
  · turning_features.json  — features completas por turning point
  · turning_features.csv   — mismo contenido en CSV
  · turning_analysis.html  — reporte interactivo con gráficos Plotly
  · Resumen en consola

Features extraídas por segmento:
  · Duración, magnitud del movimiento
  · Perfil de volumen (media, tendencia, últimas velas)
  · Presión taker (delta ratio, tendencia, últimas velas)
  · RSI(14): valor al final, mín/máx durante el segmento, tendencia
  · MAs (20, 50): precio vs MA, pendiente de MA20
  · Estructura de velas (body ratio, mechas)
  · Momentum, aceleración de precio y volumen
  · Volatilidad del segmento

Uso:
    python analyze_turning_points.py

Configuración: editar sección CONFIG.
"""

from __future__ import annotations

import csv
import json
import math
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent))

# ═══════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════

try:
    import config_local as CL
    DB_PATH      = CL.DB_PATH
    DB_TABLE     = CL.DB_TABLE
    FECHA_INICIO = CL.FECHA_INICIO
    FECHA_FIN    = CL.FECHA_FIN
except ImportError:
    DB_PATH      = "btc_hourly.db"
    DB_TABLE     = "btc_hourly"
    FECHA_INICIO = "2021-11-10"
    FECHA_FIN    = "2022-11-22"

TP_DB_PATH   = "turning_points.db"     # generado por build_turning_points_db.py

# Selección del método/params a analizar (None = todos)
# Ejemplo:  ANALYZE_METHOD = "zigzag"   ANALYZE_PARAMS = {"min_move_pct": 5.0}
ANALYZE_METHOD = "zigzag"
ANALYZE_PARAMS = {"min_move_pct": 5.0}

# Indicadores técnicos
RSI_PERIOD   = 14
MA_SHORT     = 20
MA_LONG      = 50
WARMUP_DAYS  = 60       # días extra de velas antes de FECHA_INICIO para warmup

# Outputs
OUT_JSON     = "turning_features.json"
OUT_CSV      = "turning_features.csv"
OUT_HTML     = "turning_analysis.html"

# Cuántos "últimas N velas" para features finales del segmento
LAST_N_CANDLES = 5


# ═══════════════════════════════════════════════════════════════════════════
# TIPOS
# ═══════════════════════════════════════════════════════════════════════════

class RawCandle:
    __slots__ = ("idx","ts","open","high","low","close","volume",
                 "quote_volume","trades_count","taker_buy_base_vol","taker_buy_quote_vol")

    def __init__(self, idx, ts, o, h, l, c, v, qv, tc, tbbv, tbqv):
        self.idx   = idx
        self.ts    = ts
        self.open  = float(o)
        self.high  = float(h)
        self.low   = float(l)
        self.close = float(c)
        self.volume              = float(v)
        self.quote_volume        = float(qv)   if qv   is not None else None
        self.trades_count        = int(tc)      if tc   is not None else None
        self.taker_buy_base_vol  = float(tbbv)  if tbbv is not None else None
        self.taker_buy_quote_vol = float(tbqv)  if tbqv is not None else None

    @property
    def delta_ratio(self) -> Optional[float]:
        if self.taker_buy_base_vol is None or self.volume == 0:
            return None
        return self.taker_buy_base_vol / self.volume

    @property
    def iso(self) -> str:
        return datetime.fromtimestamp(self.ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ═══════════════════════════════════════════════════════════════════════════
# CARGA DE DATOS
# ═══════════════════════════════════════════════════════════════════════════

def date_to_epoch_ms(date_str: str) -> int:
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def date_add_days(date_str: str, days: int) -> str:
    from datetime import timedelta
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    return (dt + timedelta(days=days)).strftime("%Y-%m-%d")


def load_candles(db_path: str, table: str, start: str, end: str) -> List[RawCandle]:
    start_ms = date_to_epoch_ms(start)
    end_ms   = date_to_epoch_ms(end) + 86_399_000

    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute(f"""
            SELECT timestamp, open, high, low, close, volume,
                   quote_volume, trades_count,
                   taker_buy_base_volume, taker_buy_quote_volume
            FROM   {table}
            WHERE  timestamp >= ? AND timestamp <= ?
            ORDER  BY timestamp ASC
        """, (start_ms, end_ms))
        rows = cur.fetchall()
    finally:
        conn.close()

    candles = []
    for i, (ts_ms, o, h, l, c, v, qv, tc, tbbv, tbqv) in enumerate(rows):
        candles.append(RawCandle(i, ts_ms // 1000, o, h, l, c, v, qv, tc, tbbv, tbqv))
    return candles


def load_turning_points(tp_db: str, method: Optional[str], params: Optional[dict]) -> List[dict]:
    """Carga turning points con filtro opcional de método/params."""
    conn = sqlite3.connect(tp_db)
    try:
        query = "SELECT * FROM turning_points WHERE 1=1"
        args  = []
        if method:
            query += " AND method = ?"
            args.append(method)
        if params:
            query += " AND params = ?"
            args.append(json.dumps(params, sort_keys=True))
        query += " ORDER BY ts ASC"
        cur = conn.cursor()
        cur.execute(query, args)
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()
    finally:
        conn.close()

    return [dict(zip(cols, r)) for r in rows]


# ═══════════════════════════════════════════════════════════════════════════
# INDICADORES TÉCNICOS (implementación standalone, sin librerías externas)
# ═══════════════════════════════════════════════════════════════════════════

def compute_rsi(closes: List[float], period: int = 14) -> List[Optional[float]]:
    """RSI(period) para una serie de closes. Retorna lista con None en warmup."""
    n   = len(closes)
    rsi = [None] * n
    if n <= period:
        return rsi

    gains  = [max(closes[i] - closes[i-1], 0.0) for i in range(1, n)]
    losses = [max(closes[i-1] - closes[i], 0.0) for i in range(1, n)]

    avg_g = sum(gains[:period])  / period
    avg_l = sum(losses[:period]) / period

    def _rsi_from_avgs(ag, al):
        if al == 0:
            return 100.0
        return 100.0 - 100.0 / (1.0 + ag / al)

    rsi[period] = _rsi_from_avgs(avg_g, avg_l)

    for i in range(period + 1, n):
        avg_g = (avg_g * (period - 1) + gains[i - 1])  / period
        avg_l = (avg_l * (period - 1) + losses[i - 1]) / period
        rsi[i] = _rsi_from_avgs(avg_g, avg_l)

    return rsi


def compute_sma(values: List[float], period: int) -> List[Optional[float]]:
    n  = len(values)
    ma = [None] * n
    for i in range(period - 1, n):
        ma[i] = sum(values[i - period + 1 : i + 1]) / period
    return ma


def linear_slope_norm(values: List[float]) -> float:
    """Pendiente lineal normalizada por el valor medio (retorna %/vela)."""
    n = len(values)
    if n < 2:
        return 0.0
    x_mean = (n - 1) / 2.0
    y_mean = sum(values) / n
    if y_mean == 0:
        return 0.0
    numer = sum((i - x_mean) * (values[i] - y_mean) for i in range(n))
    denom = sum((i - x_mean) ** 2                    for i in range(n))
    if denom == 0:
        return 0.0
    return (numer / denom) / abs(y_mean)


def safe_mean(vals: List[Optional[float]]) -> Optional[float]:
    v = [x for x in vals if x is not None]
    return sum(v) / len(v) if v else None


# ═══════════════════════════════════════════════════════════════════════════
# EXTRACCIÓN DE FEATURES
# ═══════════════════════════════════════════════════════════════════════════

def extract_features(
    segment:       List[RawCandle],   # velas del segmento (prev_tp → this_tp)
    all_candles:   List[RawCandle],   # todas las velas (para RSI/MA)
    end_global_idx: int,              # índice global del turning point
    tp_type:       str,               # "BOTTOM" | "TOP"
    rsi_all:       List[Optional[float]],
    ma_short_all:  List[Optional[float]],
    ma_long_all:   List[Optional[float]],
) -> Dict[str, Any]:
    """
    Calcula todas las features del segmento que precedió al turning point.
    Solo usa información disponible antes/en el turning point (sin look-ahead).
    """
    n = len(segment)
    if n == 0:
        return {}

    closes  = [c.close  for c in segment]
    highs   = [c.high   for c in segment]
    lows    = [c.low    for c in segment]
    volumes = [c.volume for c in segment]
    dr_raw  = [c.delta_ratio for c in segment]
    dr_vals = [x for x in dr_raw if x is not None]

    last_k  = min(LAST_N_CANDLES, n)

    # ─── 1. Duración y magnitud ───────────────────────────────────────────
    feat: Dict[str, Any] = {}
    feat["duration_candles"] = n

    start_p = segment[0].high   if tp_type == "BOTTOM" else segment[0].low
    end_p   = segment[-1].low   if tp_type == "BOTTOM" else segment[-1].high
    feat["move_pct"] = round(abs(end_p - start_p) / start_p * 100, 4) if start_p else None

    # Máxima excursión adversa durante el segmento (antes de la última vela)
    if n > 1 and tp_type == "BOTTOM":
        interim_highs = [c.high for c in segment[:-1]]
        mae = (max(interim_highs) - end_p) / end_p * 100 if end_p else 0.0
    elif n > 1 and tp_type == "TOP":
        interim_lows = [c.low for c in segment[:-1]]
        mae = (end_p - min(interim_lows)) / min(interim_lows) * 100 if min(interim_lows) else 0.0
    else:
        mae = 0.0
    feat["max_adverse_excursion_pct"] = round(mae, 4)

    # ─── 2. Volumen ───────────────────────────────────────────────────────
    vol_mean = sum(volumes) / n
    feat["vol_mean"]     = round(vol_mean, 2)
    feat["vol_slope"]    = round(linear_slope_norm(volumes), 6)

    half = max(1, n // 2)
    vol_h1 = sum(volumes[:half])       / half
    vol_h2 = sum(volumes[half:])       / max(1, n - half)
    feat["vol_first_vs_second_half"] = round(vol_h1 / vol_h2, 4) if vol_h2 > 0 else None

    vol_last_mean = sum(volumes[-last_k:]) / last_k
    feat[f"vol_last{LAST_N_CANDLES}_vs_avg"] = round(vol_last_mean / vol_mean, 4) if vol_mean > 0 else None

    # Aceleración de volumen (segunda mitad vs primera mitad de slope)
    if n >= 6:
        mid = n // 2
        slope_h1 = linear_slope_norm(volumes[:mid])
        slope_h2 = linear_slope_norm(volumes[mid:])
        feat["vol_acceleration"] = round(slope_h2 - slope_h1, 6)
    else:
        feat["vol_acceleration"] = None

    # ─── 3. Taker ratio (presión compradora) ─────────────────────────────
    if dr_vals:
        dr_mean = sum(dr_vals) / len(dr_vals)
        feat["taker_ratio_mean"]  = round(dr_mean, 4)
        feat["taker_ratio_slope"] = round(linear_slope_norm(dr_vals), 6)

        dr_last = [x for x in dr_raw[-last_k:] if x is not None]
        dr_last_mean = sum(dr_last) / len(dr_last) if dr_last else None
        feat[f"taker_last{LAST_N_CANDLES}_vs_avg"] = round(dr_last_mean / dr_mean, 4) \
            if (dr_last_mean is not None and dr_mean > 0) else None

        # Aceleración taker
        if len(dr_vals) >= 6:
            mid = len(dr_vals) // 2
            feat["taker_acceleration"] = round(
                linear_slope_norm(dr_vals[mid:]) - linear_slope_norm(dr_vals[:mid]), 6
            )
        else:
            feat["taker_acceleration"] = None
    else:
        feat["taker_ratio_mean"]                    = None
        feat["taker_ratio_slope"]                   = None
        feat[f"taker_last{LAST_N_CANDLES}_vs_avg"]  = None
        feat["taker_acceleration"]                  = None

    # ─── 4. RSI ───────────────────────────────────────────────────────────
    rsi_end = rsi_all[end_global_idx] if end_global_idx < len(rsi_all) else None
    feat["rsi_at_end"] = round(rsi_end, 2) if rsi_end is not None else None

    seg_rsi = [rsi_all[c.idx] for c in segment if rsi_all[c.idx] is not None]
    feat["rsi_min_during"]  = round(min(seg_rsi), 2)   if seg_rsi else None
    feat["rsi_max_during"]  = round(max(seg_rsi), 2)   if seg_rsi else None
    feat["rsi_range"]       = round(max(seg_rsi) - min(seg_rsi), 2) if seg_rsi else None

    rsi_last = [rsi_all[c.idx] for c in segment[-last_k:] if rsi_all[c.idx] is not None]
    feat[f"rsi_slope_last{LAST_N_CANDLES}"] = round(linear_slope_norm(rsi_last), 6) \
        if len(rsi_last) >= 2 else None

    # ─── 5. Medias móviles ────────────────────────────────────────────────
    ma_s_end = ma_short_all[end_global_idx] if end_global_idx < len(ma_short_all) else None
    ma_l_end = ma_long_all[end_global_idx]  if end_global_idx < len(ma_long_all)  else None
    end_close = all_candles[end_global_idx].close

    feat[f"price_vs_ma{MA_SHORT}_pct"] = round(
        (end_close - ma_s_end) / ma_s_end * 100, 4) if ma_s_end else None
    feat[f"price_vs_ma{MA_LONG}_pct"]  = round(
        (end_close - ma_l_end) / ma_l_end * 100, 4) if ma_l_end else None

    # Pendiente de MA_SHORT en las últimas velas del segmento
    ma_s_seg = [ma_short_all[c.idx] for c in segment[-last_k:]
                if c.idx < len(ma_short_all) and ma_short_all[c.idx] is not None]
    feat[f"ma{MA_SHORT}_slope_last{LAST_N_CANDLES}"] = round(linear_slope_norm(ma_s_seg), 6) \
        if len(ma_s_seg) >= 2 else None

    # ¿Precio por debajo / encima de ambas MAs?
    # Ambas claves se emiten siempre para garantizar columnas consistentes en CSV.
    # below_both_mas es informativo principalmente para BOTTOM.
    # above_both_mas es informativo principalmente para TOP.
    if ma_s_end and ma_l_end:
        feat["below_both_mas"] = 1 if end_close < ma_s_end and end_close < ma_l_end else 0
        feat["above_both_mas"] = 1 if end_close > ma_s_end and end_close > ma_l_end else 0
    else:
        feat["below_both_mas"] = None
        feat["above_both_mas"] = None

    # ─── 6. Estructura de velas ───────────────────────────────────────────
    bodies, lower_wicks, upper_wicks = [], [], []
    bearish_count, bullish_count = 0, 0
    for c in segment:
        tr = c.high - c.low
        if tr > 0:
            body = abs(c.close - c.open)
            bot  = min(c.open, c.close)
            top  = max(c.open, c.close)
            bodies.append(body / tr)
            lower_wicks.append((bot - c.low)  / tr)
            upper_wicks.append((c.high - top) / tr)
        if c.close < c.open:
            bearish_count += 1
        else:
            bullish_count += 1

    feat["body_ratio_mean"]       = round(sum(bodies)      / len(bodies),      4) if bodies      else None
    feat["lower_wick_ratio_mean"] = round(sum(lower_wicks) / len(lower_wicks), 4) if lower_wicks else None
    feat["upper_wick_ratio_mean"] = round(sum(upper_wicks) / len(upper_wicks), 4) if upper_wicks else None
    feat["pct_bearish_candles"]   = round(bearish_count / n * 100, 2)
    feat["pct_bullish_candles"]   = round(bullish_count / n * 100, 2)

    # Velas consecutivas en la misma dirección al final
    consec = 0
    if n >= 1:
        last_dir = 1 if segment[-1].close >= segment[-1].open else -1
        for c in reversed(segment):
            d = 1 if c.close >= c.open else -1
            if d == last_dir:
                consec += 1
            else:
                break
    feat["consec_same_dir_end"] = consec

    # ─── 7. Momentum y volatilidad ────────────────────────────────────────
    feat["price_slope"] = round(linear_slope_norm(closes), 6)

    if len(closes) >= 2:
        rets = [(closes[i] - closes[i-1]) / closes[i-1] for i in range(1, len(closes))]
        r_mean = sum(rets) / len(rets)
        feat["volatility_pct"] = round(
            math.sqrt(sum((r - r_mean)**2 for r in rets) / len(rets)) * 100, 4)
    else:
        feat["volatility_pct"] = 0.0

    # Aceleración de precio: slope segunda mitad vs primera mitad
    if n >= 4:
        mid = n // 2
        feat["price_acceleration"] = round(
            linear_slope_norm(closes[mid:]) - linear_slope_norm(closes[:mid]), 6)
    else:
        feat["price_acceleration"] = None

    # Momentum de volumen vs precio (divergencia)
    if n >= 4:
        feat["vol_price_divergence"] = round(
            linear_slope_norm(volumes) - linear_slope_norm(closes), 6)
    else:
        feat["vol_price_divergence"] = None

    return feat


# ═══════════════════════════════════════════════════════════════════════════
# ESTADÍSTICAS
# ═══════════════════════════════════════════════════════════════════════════

def _num_stats(vals: List[float]) -> Dict[str, float]:
    """Media, mediana, std, cv, p10, p90 para una lista de floats."""
    v = sorted(x for x in vals if x is not None)
    if not v:
        return {}
    n    = len(v)
    mean = sum(v) / n
    std  = math.sqrt(sum((x - mean)**2 for x in v) / n) if n > 1 else 0.0
    cv   = std / abs(mean) if mean != 0 else 0.0
    return {
        "n":      n,
        "mean":   round(mean, 4),
        "median": round(v[n // 2], 4),
        "std":    round(std, 4),
        "cv":     round(cv, 4),    # coeficiente de variación — menor = más consistente
        "p10":    round(v[max(0, int(n * 0.10))], 4),
        "p90":    round(v[min(n - 1, int(n * 0.90))], 4),
        "min":    round(v[0], 4),
        "max":    round(v[-1], 4),
    }


def compute_statistics(records: List[Dict]) -> Dict:
    """Estadísticas por tipo (BOTTOM/TOP) para cada feature."""
    feature_names = [k for k in records[0].keys()
                     if k not in ("tp_id","method","params","type","datetime","ts",
                                  "prev_ts","prev_price","candle_idx","move_from_prev_pct",
                                  "candles_from_prev")]

    stats = {"BOTTOM": {}, "TOP": {}}
    for tp_type in ("BOTTOM", "TOP"):
        subset = [r for r in records if r.get("type") == tp_type]
        for fname in feature_names:
            vals = [r.get(fname) for r in subset]
            s    = _num_stats(vals)
            if s:
                stats[tp_type][fname] = s

    return stats


def compute_correlations(records: List[Dict]) -> Dict:
    """
    Correlación de Pearson de cada feature con move_from_prev_pct.
    Separada por tipo de turning point.
    """
    feature_names = [k for k in records[0].keys()
                     if k not in ("tp_id","method","params","type","datetime","ts",
                                  "prev_ts","prev_price","candle_idx","move_from_prev_pct",
                                  "candles_from_prev")]

    result = {}
    for tp_type in ("BOTTOM", "TOP"):
        subset = [r for r in records
                  if r.get("type") == tp_type
                  and r.get("move_from_prev_pct") is not None]
        corrs = {}
        y = [r["move_from_prev_pct"] for r in subset]
        for fname in feature_names:
            x = [r.get(fname) for r in subset]
            pairs = [(xi, yi) for xi, yi in zip(x, y) if xi is not None]
            if len(pairs) < 5:
                continue
            xs, ys = zip(*pairs)
            corrs[fname] = round(_pearson(list(xs), list(ys)), 4)
        result[tp_type] = corrs
    return result


def _pearson(x: List[float], y: List[float]) -> float:
    n = len(x)
    if n < 2:
        return 0.0
    mx, my = sum(x)/n, sum(y)/n
    num    = sum((x[i]-mx)*(y[i]-my) for i in range(n))
    dx     = math.sqrt(sum((v-mx)**2 for v in x))
    dy     = math.sqrt(sum((v-my)**2 for v in y))
    if dx * dy == 0:
        return 0.0
    return num / (dx * dy)


# ═══════════════════════════════════════════════════════════════════════════
# GENERACIÓN DE OUTPUTS
# ═══════════════════════════════════════════════════════════════════════════

def save_json(records: List[Dict], stats: Dict, corrs: Dict, path: str) -> None:
    payload = {
        "meta": {
            "fecha_inicio":   FECHA_INICIO,
            "fecha_fin":      FECHA_FIN,
            "analyze_method": ANALYZE_METHOD,
            "analyze_params": ANALYZE_PARAMS,
            "total_records":  len(records),
        },
        "statistics":    stats,
        "correlations":  corrs,
        "turning_points": records,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
    print(f"  ✓ JSON guardado: {path}")


def save_csv(records: List[Dict], path: str) -> None:
    if not records:
        return
    # Recoger la unión ordenada de TODAS las keys de TODOS los records.
    # Usar solo records[0].keys() causaría ValueError si un record posterior
    # tiene una clave nueva (como 'above_both_mas' cuando el primero es BOTTOM).
    seen: set = set()
    keys: List[str] = []
    for r in records:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, restval=None)
        w.writeheader()
        w.writerows(records)
    print(f"  ✓ CSV guardado:  {path}")


def generate_html(
    records:  List[Dict],
    stats:    Dict,
    corrs:    Dict,
    all_candles: List[RawCandle],
    candles_start_ts: int,
    path: str,
) -> None:
    """Genera reporte HTML interactivo con gráficos Plotly embebidos."""

    # Separar por tipo
    bottoms = [r for r in records if r["type"] == "BOTTOM"]
    tops    = [r for r in records if r["type"] == "TOP"]

    # Feature numéricas (excluir metadatos)
    meta_keys = {"tp_id","method","params","type","datetime","ts",
                 "prev_ts","prev_price","candle_idx","move_from_prev_pct","candles_from_prev"}
    # Unión ordenada de todas las keys de todos los records (excluye metadatos).
    # Usar solo records[0].keys() omitiría 'above_both_mas' si el primer record es BOTTOM.
    if records:
        seen_f: set = set()
        feat_names: List[str] = []
        for r in records:
            for k in r.keys():
                if k not in seen_f and k not in meta_keys:
                    seen_f.add(k)
                    feat_names.append(k)
    else:
        feat_names = []

    # ── Datos para gráfico de precio ──────────────────────────────────────
    price_ts     = [c.iso for c in all_candles if c.ts >= candles_start_ts]
    price_close  = [c.close for c in all_candles if c.ts >= candles_start_ts]

    # Filtrar turning points del rango de análisis
    tp_all_ts    = [r["datetime"] for r in records]
    tp_all_price = [r.get("tp_price", None) for r in records]
    tp_all_type  = [r["type"] for r in records]

    bottom_ts    = [r["datetime"] for r in bottoms]
    bottom_price = [r.get("tp_price", None) for r in bottoms]
    top_ts       = [r["datetime"] for r in tops]
    top_price    = [r.get("tp_price", None) for r in tops]

    # ── Datos para histogramas de features ────────────────────────────────
    hist_data = {}
    for fn in feat_names:
        hist_data[fn] = {
            "bottom": [r[fn] for r in bottoms if r.get(fn) is not None],
            "top":    [r[fn] for r in tops    if r.get(fn) is not None],
        }

    # ── Feature consistency ranking (CV menor = más consistente) ─────────
    cv_bot = {fn: stats.get("BOTTOM", {}).get(fn, {}).get("cv") for fn in feat_names}
    cv_top = {fn: stats.get("TOP",    {}).get(fn, {}).get("cv") for fn in feat_names}

    cv_bot_sorted = sorted([(fn, v) for fn, v in cv_bot.items() if v is not None], key=lambda x: x[1])
    cv_top_sorted = sorted([(fn, v) for fn, v in cv_top.items() if v is not None], key=lambda x: x[1])

    # ── Correlaciones con move_from_prev_pct ─────────────────────────────
    corr_bot = corrs.get("BOTTOM", {})
    corr_top = corrs.get("TOP", {})
    corr_bot_sorted = sorted(corr_bot.items(), key=lambda x: abs(x[1]), reverse=True)
    corr_top_sorted = sorted(corr_top.items(), key=lambda x: abs(x[1]), reverse=True)

    # ── Stats summary table ───────────────────────────────────────────────
    def stats_table_html(tp_type: str) -> str:
        rows_html = ""
        for fn in feat_names:
            s = stats.get(tp_type, {}).get(fn, {})
            if not s:
                continue
            corr_val = corrs.get(tp_type, {}).get(fn, "N/A")
            rows_html += f"""
            <tr>
              <td>{fn}</td>
              <td>{s.get('n','')}</td>
              <td>{s.get('mean','')}</td>
              <td>{s.get('median','')}</td>
              <td>{s.get('std','')}</td>
              <td style="{'color:#e74c3c;font-weight:bold' if isinstance(s.get('cv'),float) and s['cv']<0.3 else ''}">{s.get('cv','')}</td>
              <td>{s.get('p10','')}</td>
              <td>{s.get('p90','')}</td>
              <td style="{'color:#27ae60' if isinstance(corr_val,float) and corr_val>0.3 else 'color:#e74c3c' if isinstance(corr_val,float) and corr_val<-0.3 else ''}">{corr_val if corr_val != 'N/A' else 'N/A'}</td>
            </tr>"""
        return rows_html

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Turning Points Analysis — {FECHA_INICIO} → {FECHA_FIN}</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Segoe UI', Tahoma, sans-serif; background: #0f1117; color: #e0e0e0; padding: 20px; }}
  h1 {{ color: #f0b429; margin-bottom: 6px; font-size: 1.6em; }}
  h2 {{ color: #7ec8e3; margin: 28px 0 10px; font-size: 1.2em; border-left: 3px solid #7ec8e3; padding-left: 10px; }}
  h3 {{ color: #a0c4ff; margin: 18px 0 8px; font-size: 1em; }}
  .meta {{ color: #aaa; margin-bottom: 20px; font-size: 0.9em; }}
  .grid2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
  .card {{ background: #1a1d27; border-radius: 8px; padding: 16px; }}
  .stat-bar {{ display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 16px; }}
  .stat-pill {{ background: #252836; border-radius: 6px; padding: 8px 14px; font-size: 0.85em; }}
  .stat-pill span {{ color: #f0b429; font-weight: bold; font-size: 1.1em; }}
  .chart-full {{ width: 100%; height: 450px; }}
  .chart-half {{ width: 100%; height: 380px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.8em; margin-top: 10px; }}
  th {{ background: #252836; color: #7ec8e3; padding: 8px 6px; text-align: left; position: sticky; top: 0; }}
  td {{ padding: 6px; border-bottom: 1px solid #252836; }}
  tr:hover td {{ background: #1f2233; }}
  .scrollable {{ max-height: 400px; overflow-y: auto; }}
  .tag-bottom {{ background: #1a4a2e; color: #56d07e; border-radius: 4px; padding: 2px 7px; font-size:0.8em; }}
  .tag-top {{ background: #4a1a1a; color: #e07070; border-radius: 4px; padding: 2px 7px; font-size:0.8em; }}
</style>
</head>
<body>

<h1>📈 Turning Points Analysis</h1>
<div class="meta">
  Rango: <b>{FECHA_INICIO}</b> → <b>{FECHA_FIN}</b> &nbsp;|&nbsp;
  Método: <b>{ANALYZE_METHOD}</b> &nbsp;|&nbsp;
  Params: <b>{json.dumps(ANALYZE_PARAMS)}</b> &nbsp;|&nbsp;
  Generado: <b>{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}</b>
</div>

<div class="stat-bar">
  <div class="stat-pill">Total turning points <span>{len(records)}</span></div>
  <div class="stat-pill">BOTTOMs <span>{len(bottoms)}</span></div>
  <div class="stat-pill">TOPs <span>{len(tops)}</span></div>
  <div class="stat-pill">Features extraídas <span>{len(feat_names)}</span></div>
</div>

<!-- SECCIÓN 1: Gráfico de precio con turning points -->
<h2>1 · Precio BTC con Turning Points detectados</h2>
<div class="card">
  <div id="price_chart" class="chart-full"></div>
</div>

<!-- SECCIÓN 2: Distribución de duración y magnitud -->
<h2>2 · Duración y Magnitud de los Segmentos</h2>
<div class="grid2">
  <div class="card"><div id="hist_duration" class="chart-half"></div></div>
  <div class="card"><div id="hist_move" class="chart-half"></div></div>
</div>

<!-- SECCIÓN 3: RSI -->
<h2>3 · RSI al final del segmento</h2>
<div class="grid2">
  <div class="card"><div id="hist_rsi_end" class="chart-half"></div></div>
  <div class="card"><div id="hist_rsi_range" class="chart-half"></div></div>
</div>

<!-- SECCIÓN 4: Medias móviles -->
<h2>4 · Precio vs Medias Móviles</h2>
<div class="grid2">
  <div class="card"><div id="hist_vs_ma_short" class="chart-half"></div></div>
  <div class="card"><div id="hist_vs_ma_long" class="chart-half"></div></div>
</div>

<!-- SECCIÓN 5: Taker ratio -->
<h2>5 · Presión Compradora (Taker Ratio)</h2>
<div class="grid2">
  <div class="card"><div id="hist_taker_mean" class="chart-half"></div></div>
  <div class="card"><div id="hist_taker_slope" class="chart-half"></div></div>
</div>

<!-- SECCIÓN 6: Volumen -->
<h2>6 · Perfil de Volumen</h2>
<div class="grid2">
  <div class="card"><div id="hist_vol_slope" class="chart-half"></div></div>
  <div class="card"><div id="hist_vol_h1h2" class="chart-half"></div></div>
</div>

<!-- SECCIÓN 7: Consistencia de features (CV) -->
<h2>7 · Consistencia de Features (Coeficiente de Variación — menor = más consistente)</h2>
<div class="grid2">
  <div class="card">
    <h3 class="tag-bottom">BOTTOM — Features más consistentes</h3>
    <div id="cv_bottom" class="chart-half"></div>
  </div>
  <div class="card">
    <h3 class="tag-top">TOP — Features más consistentes</h3>
    <div id="cv_top" class="chart-half"></div>
  </div>
</div>

<!-- SECCIÓN 8: Correlación con magnitud de movimiento -->
<h2>8 · Correlación con Magnitud del Movimiento (move_from_prev_pct)</h2>
<div class="grid2">
  <div class="card">
    <h3 class="tag-bottom">BOTTOM</h3>
    <div id="corr_bottom" class="chart-half"></div>
  </div>
  <div class="card">
    <h3 class="tag-top">TOP</h3>
    <div id="corr_top" class="chart-half"></div>
  </div>
</div>

<!-- SECCIÓN 9: Tabla de estadísticas completa -->
<h2>9 · Tabla Estadística Completa — BOTTOM</h2>
<div class="card scrollable">
  <table>
    <tr><th>Feature</th><th>N</th><th>Mean</th><th>Median</th><th>Std</th>
        <th title="Coef. Variación — menor=más consistente">CV</th>
        <th>P10</th><th>P90</th><th>Corr (move%)</th></tr>
    {stats_table_html("BOTTOM")}
  </table>
</div>

<h2>10 · Tabla Estadística Completa — TOP</h2>
<div class="card scrollable" style="margin-bottom:40px;">
  <table>
    <tr><th>Feature</th><th>N</th><th>Mean</th><th>Median</th><th>Std</th>
        <th title="Coef. Variación — menor=más consistente">CV</th>
        <th>P10</th><th>P90</th><th>Corr (move%)</th></tr>
    {stats_table_html("TOP")}
  </table>
</div>

<script>
const PLOTLY_LAYOUT = {{
  paper_bgcolor: '#1a1d27',
  plot_bgcolor:  '#13151f',
  font:          {{ color: '#e0e0e0', size: 11 }},
  margin:        {{ t: 40, b: 40, l: 50, r: 20 }},
  legend:        {{ bgcolor: 'rgba(0,0,0,0)' }},
  xaxis:         {{ gridcolor: '#2a2d3a' }},
  yaxis:         {{ gridcolor: '#2a2d3a' }},
}};

// ── Gráfico de precio ────────────────────────────────────────────────────
Plotly.newPlot('price_chart', [
  {{
    type: 'scatter', mode: 'lines',
    x: {json.dumps(price_ts)},
    y: {json.dumps(price_close)},
    name: 'BTC close', line: {{ color: '#7ec8e3', width: 1 }}
  }},
  {{
    type: 'scatter', mode: 'markers',
    x: {json.dumps(bottom_ts)},
    y: {json.dumps(bottom_price)},
    name: 'BOTTOM', marker: {{ color: '#56d07e', size: 8, symbol: 'triangle-up' }}
  }},
  {{
    type: 'scatter', mode: 'markers',
    x: {json.dumps(top_ts)},
    y: {json.dumps(top_price)},
    name: 'TOP', marker: {{ color: '#e07070', size: 8, symbol: 'triangle-down' }}
  }},
], {{...PLOTLY_LAYOUT, title: 'BTC/USDT — Turning Points detectados'}});

// ── Helper histograma ────────────────────────────────────────────────────
function plotHist(divId, fname, title, xLabel) {{
  const bdata = {json.dumps({fn: hist_data[fn]['bottom'] for fn in feat_names if hist_data.get(fn)})};
  const tdata = {json.dumps({fn: hist_data[fn]['top']    for fn in feat_names if hist_data.get(fn)})};
  const bvals = bdata[fname] || [];
  const tvals = tdata[fname] || [];
  Plotly.newPlot(divId, [
    {{ type:'histogram', x:bvals, name:'BOTTOM', marker:{{color:'rgba(86,208,126,0.65)'}}, opacity:0.8, nbinsx:25 }},
    {{ type:'histogram', x:tvals, name:'TOP',    marker:{{color:'rgba(224,112,112,0.65)'}}, opacity:0.8, nbinsx:25 }},
  ], {{...PLOTLY_LAYOUT, title, barmode:'overlay', xaxis:{{...PLOTLY_LAYOUT.xaxis, title:xLabel}}}});
}}

plotHist('hist_duration',    'duration_candles',       'Duración del segmento',       'Velas');
plotHist('hist_move',        'move_pct',                'Magnitud del movimiento',      '%');
plotHist('hist_rsi_end',     'rsi_at_end',              'RSI al final del segmento',    'RSI');
plotHist('hist_rsi_range',   'rsi_range',               'Rango de RSI en el segmento',  'RSI range');
plotHist('hist_vs_ma_short', 'price_vs_ma{MA_SHORT}_pct', 'Precio vs MA{MA_SHORT}',      '%');
plotHist('hist_vs_ma_long',  'price_vs_ma{MA_LONG}_pct',  'Precio vs MA{MA_LONG}',       '%');
plotHist('hist_taker_mean',  'taker_ratio_mean',         'Taker ratio medio',            'ratio');
plotHist('hist_taker_slope', 'taker_ratio_slope',        'Tendencia del taker ratio',    'slope');
plotHist('hist_vol_slope',   'vol_slope',                'Tendencia del volumen',         'slope');
plotHist('hist_vol_h1h2',    'vol_first_vs_second_half', 'Vol primera mitad / segunda',  'ratio');

// ── CV charts ────────────────────────────────────────────────────────────
const cvBotData = {json.dumps([[x[0], x[1]] for x in cv_bot_sorted[:20]])};
const cvTopData = {json.dumps([[x[0], x[1]] for x in cv_top_sorted[:20]])};

Plotly.newPlot('cv_bottom', [{{
  type:'bar', orientation:'h',
  x: cvBotData.map(d=>d[1]), y: cvBotData.map(d=>d[0]),
  marker:{{color:'rgba(86,208,126,0.7)'}},
  name: 'CV BOTTOM',
}}], {{...PLOTLY_LAYOUT, title:'Consistencia por feature (CV bajo = consistente)',
      xaxis:{{...PLOTLY_LAYOUT.xaxis, title:'CV'}}}});

Plotly.newPlot('cv_top', [{{
  type:'bar', orientation:'h',
  x: cvTopData.map(d=>d[1]), y: cvTopData.map(d=>d[0]),
  marker:{{color:'rgba(224,112,112,0.7)'}},
  name: 'CV TOP',
}}], {{...PLOTLY_LAYOUT, title:'Consistencia por feature (CV bajo = consistente)',
      xaxis:{{...PLOTLY_LAYOUT.xaxis, title:'CV'}}}});

// ── Correlation charts ────────────────────────────────────────────────────
const corrBot = {json.dumps([[x[0], x[1]] for x in corr_bot_sorted[:20]])};
const corrTop = {json.dumps([[x[0], x[1]] for x in corr_top_sorted[:20]])};

function corrColors(vals) {{
  return vals.map(v => v >= 0 ? 'rgba(86,208,126,0.75)' : 'rgba(224,112,112,0.75)');
}}

Plotly.newPlot('corr_bottom', [{{
  type:'bar', orientation:'h',
  x: corrBot.map(d=>d[1]), y: corrBot.map(d=>d[0]),
  marker:{{color: corrColors(corrBot.map(d=>d[1]))}},
}}], {{...PLOTLY_LAYOUT, title:'Correlación con magnitud del movimiento — BOTTOM',
      xaxis:{{...PLOTLY_LAYOUT.xaxis, title:'Pearson r', range:[-1,1]}}}});

Plotly.newPlot('corr_top', [{{
  type:'bar', orientation:'h',
  x: corrTop.map(d=>d[1]), y: corrTop.map(d=>d[0]),
  marker:{{color: corrColors(corrTop.map(d=>d[1]))}},
}}], {{...PLOTLY_LAYOUT, title:'Correlación con magnitud del movimiento — TOP',
      xaxis:{{...PLOTLY_LAYOUT.xaxis, title:'Pearson r', range:[-1,1]}}}});
</script>
</body>
</html>"""

    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  ✓ HTML guardado: {path}")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    import time as _t
    t0 = _t.time()

    print("╔══════════════════════════════════════════════════════════╗")
    print("║     ANALYZE TURNING POINTS — Análisis de Patrones       ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"  DB velas     : {DB_PATH}")
    print(f"  DB turning   : {TP_DB_PATH}")
    print(f"  Método       : {ANALYZE_METHOD}  {ANALYZE_PARAMS}")
    print(f"  Rango        : {FECHA_INICIO} → {FECHA_FIN}")
    print(f"  RSI period   : {RSI_PERIOD}  |  MAs: {MA_SHORT}/{MA_LONG}")
    print("─" * 60)

    # 1. Cargar turning points
    print("Cargando turning points...", end=" ", flush=True)
    tps = load_turning_points(TP_DB_PATH, ANALYZE_METHOD, ANALYZE_PARAMS)
    print(f"OK  ({len(tps)} puntos)")
    if not tps:
        print("✗  No hay turning points para el método/params seleccionados.")
        return

    # 2. Cargar velas con warmup
    warmup_start = date_add_days(FECHA_INICIO, -WARMUP_DAYS)
    print(f"Cargando velas con warmup desde {warmup_start}...", end=" ", flush=True)
    all_candles = load_candles(DB_PATH, DB_TABLE, warmup_start, FECHA_FIN)
    print(f"OK  ({len(all_candles):,} velas)")

    # Índice ts → candle_idx
    ts_to_idx = {c.ts: c.idx for c in all_candles}
    warmup_start_ts = all_candles[0].ts

    # Encontrar el primer timestamp del rango real (sin warmup) para gráficos
    candles_start_ts = date_to_epoch_ms(FECHA_INICIO) // 1000

    # 3. Computar indicadores técnicos sobre TODO el array (incluyendo warmup)
    print("Computando indicadores (RSI, MAs)...", end=" ", flush=True)
    closes_all  = [c.close for c in all_candles]
    rsi_all     = compute_rsi(closes_all, RSI_PERIOD)
    ma_short_all= compute_sma(closes_all, MA_SHORT)
    ma_long_all = compute_sma(closes_all, MA_LONG)
    print("OK")

    # 4. Construir mapa de turning points por timestamp
    tp_map = {tp["ts"]: tp for tp in tps}

    # 5. Extraer features para cada turning point (con segmento desde el anterior)
    print("Extrayendo features...", end=" ", flush=True)
    records = []

    for i, tp in enumerate(tps):
        if i == 0:
            continue   # el primer punto no tiene prev de tipo opuesto

        prev_tp = tps[i - 1]

        # Verificar que son de tipos opuestos (deben serlo por construcción)
        if prev_tp["type"] == tp["type"]:
            continue

        # Segmento: desde prev_tp hasta tp (inclusive)
        prev_global = ts_to_idx.get(prev_tp["ts"])
        this_global = ts_to_idx.get(tp["ts"])

        if prev_global is None or this_global is None:
            continue
        if this_global <= prev_global:
            continue

        segment = all_candles[prev_global : this_global + 1]
        if len(segment) < 2:
            continue

        feats = extract_features(
            segment       = segment,
            all_candles   = all_candles,
            end_global_idx= this_global,
            tp_type       = tp["type"],
            rsi_all       = rsi_all,
            ma_short_all  = ma_short_all,
            ma_long_all   = ma_long_all,
        )

        record = {
            "tp_id":              tp["id"],
            "method":             tp["method"],
            "params":             tp["params"],
            "type":               tp["type"],
            "datetime":           tp["datetime"],
            "ts":                 tp["ts"],
            "tp_price":           tp["price"],
            "move_from_prev_pct": tp.get("move_from_prev_pct"),
            "candles_from_prev":  tp.get("candles_from_prev"),
            "prev_ts":            tp.get("prev_ts"),
            "prev_price":         tp.get("prev_price"),
            **feats,
        }
        records.append(record)

    print(f"OK  ({len(records)} segmentos)")

    if not records:
        print("✗  No se pudieron extraer features. Verificar datos.")
        return

    # 6. Estadísticas y correlaciones
    print("Computando estadísticas...", end=" ", flush=True)
    stats = compute_statistics(records)
    corrs = compute_correlations(records)
    print("OK")

    # 7. Guardar outputs
    print("\nGuardando resultados:")
    save_json(records, stats, corrs, OUT_JSON)
    save_csv(records, OUT_CSV)
    generate_html(records, stats, corrs, all_candles, candles_start_ts, OUT_HTML)

    # 8. Resumen en consola
    elapsed   = _t.time() - t0
    bottoms   = [r for r in records if r["type"] == "BOTTOM"]
    tops      = [r for r in records if r["type"] == "TOP"]

    print(f"\n{'═'*60}")
    print(f"  RESUMEN — {len(records)} segmentos analizados en {elapsed:.1f}s")
    print(f"{'═'*60}")
    print(f"\n  {'Feature':<38}  {'BOTTOM mean':>12}  {'TOP mean':>10}  {'CV Bot':>7}  {'CV Top':>7}  {'rBot':>6}  {'rTop':>6}")
    print(f"  {'─'*95}")

    feature_names = [k for k in records[0].keys()
                     if k not in ("tp_id","method","params","type","datetime","ts",
                                  "tp_price","prev_ts","prev_price",
                                  "candle_idx","move_from_prev_pct","candles_from_prev")]

    for fn in feature_names:
        sb = stats.get("BOTTOM", {}).get(fn, {})
        st = stats.get("TOP",    {}).get(fn, {})
        if not sb and not st:
            continue
        bm  = f"{sb.get('mean',''):>12}" if sb else f"{'N/A':>12}"
        tm  = f"{st.get('mean',''):>10}" if st else f"{'N/A':>10}"
        bcv = f"{sb.get('cv',''):>7}"    if sb else f"{'N/A':>7}"
        tcv = f"{st.get('cv',''):>7}"    if st else f"{'N/A':>7}"
        rb  = f"{corrs.get('BOTTOM',{}).get(fn,''):>6}"
        rt  = f"{corrs.get('TOP',   {}).get(fn,''):>6}"
        print(f"  {fn:<38}  {bm}  {tm}  {bcv}  {tcv}  {rb}  {rt}")

    print(f"\n  TOP 5 features más consistentes para BOTTOM (CV más bajo):")
    sorted_cv_bot = sorted(
        [(fn, stats.get("BOTTOM",{}).get(fn,{}).get("cv") or 9) for fn in feature_names],
        key=lambda x: x[1]
    )[:5]
    for fn_s, cv_s in sorted_cv_bot:
        s = stats.get("BOTTOM",{}).get(fn_s,{})
        print(f"    {fn_s:<38}  mean={s.get('mean',''):>8}  CV={cv_s}")

    print(f"\n  TOP 5 features más consistentes para TOP (CV más bajo):")
    sorted_cv_top = sorted(
        [(fn, stats.get("TOP",{}).get(fn,{}).get("cv") or 9) for fn in feature_names],
        key=lambda x: x[1]
    )[:5]
    for fn_s, cv_s in sorted_cv_top:
        s = stats.get("TOP",{}).get(fn_s,{})
        print(f"    {fn_s:<38}  mean={s.get('mean',''):>8}  CV={cv_s}")

    print(f"\n  TOP 5 features con mayor correlación con move% — BOTTOM:")
    for fn, r in sorted(corrs.get("BOTTOM",{}).items(), key=lambda x: abs(x[1]), reverse=True)[:5]:
        print(f"    {fn:<38}  r={r}")

    print(f"\n  TOP 5 features con mayor correlación con move% — TOP:")
    for fn, r in sorted(corrs.get("TOP",{}).items(), key=lambda x: abs(x[1]), reverse=True)[:5]:
        print(f"    {fn:<38}  r={r}")

    print(f"\n{'═'*60}")
    print(f"\n✓  Outputs generados:")
    print(f"     {OUT_JSON}")
    print(f"     {OUT_CSV}")
    print(f"     {OUT_HTML}  ← abrir en navegador")


if __name__ == "__main__":
    main()