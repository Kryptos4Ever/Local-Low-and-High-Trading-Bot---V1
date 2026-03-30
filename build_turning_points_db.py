"""
build_turning_points_db.py — Constructor de DB de Turning Points
═════════════════════════════════════════════════════════════════
Detecta mínimos y máximos locales significativos en el precio de BTC
usando múltiples métodos y thresholds configurables.

Métodos de detección:
  1. Window  — ventana simétrica (igual que IrrealStrategy)
  2. ZigZag  — alternancia top/bottom con movimiento mínimo configurable

Output: turning_points.db (SQLite)
  · tabla turning_points: un registro por punto detectado
  · tabla detection_runs:  metadatos de cada combinación método/params

Uso:
    python build_turning_points_db.py

Configuración: editar la sección CONFIG debajo.
"""

from __future__ import annotations

import json
import sqlite3
import sys
import time as _time
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).parent))

# ═══════════════════════════════════════════════════════════════════════════
# CONFIG — editar para cada experimento
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

OUTPUT_DB      = "turning_points.db"

# Ventanas simétricas para método Window (velas a cada lado)
WINDOW_SIZES   = [5, 10, 15, 20]

# Thresholds de movimiento mínimo %:
#   · En ZigZag  → confirmación del giro
#   · En Window  → filtro post-detección del movimiento previo/posterior
MIN_MOVE_PCTS  = [1.0, 2.0, 5.0, 10.0]


# ═══════════════════════════════════════════════════════════════════════════
# TIPOS
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class RawCandle:
    idx:   int    # posición en el array global de velas
    ts:    int    # epoch seconds UTC
    open:  float
    high:  float
    low:   float
    close: float
    volume: float
    quote_volume:        Optional[float]
    trades_count:        Optional[int]
    taker_buy_base_vol:  Optional[float]
    taker_buy_quote_vol: Optional[float]

    @property
    def delta_ratio(self) -> Optional[float]:
        if self.taker_buy_base_vol is None or self.volume == 0:
            return None
        return self.taker_buy_base_vol / self.volume

    @property
    def iso(self) -> str:
        return datetime.fromtimestamp(self.ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class TurningPoint:
    method:   str
    params:   dict
    ts:       int
    price:    float           # low para BOTTOM, high para TOP
    tp_type:  str             # "TOP" | "BOTTOM"
    candle_idx: int
    move_from_prev_pct: Optional[float] = None
    candles_from_prev:  Optional[int]   = None
    prev_ts:            Optional[int]   = None
    prev_price:         Optional[float] = None


# ═══════════════════════════════════════════════════════════════════════════
# UTILIDADES DE TIEMPO
# ═══════════════════════════════════════════════════════════════════════════

def date_to_epoch_ms(date_str: str) -> int:
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def epoch_to_iso(ts_s: int) -> str:
    return datetime.fromtimestamp(ts_s, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ═══════════════════════════════════════════════════════════════════════════
# CARGA DE DATOS
# ═══════════════════════════════════════════════════════════════════════════

def load_candles(db_path: str, table: str, start: str, end: str) -> List[RawCandle]:
    """Carga velas desde SQLite para el rango [start, end] inclusive."""
    start_ms = date_to_epoch_ms(start)
    end_ms   = date_to_epoch_ms(end) + 86_399_000  # incluir todo el día final

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
        candles.append(RawCandle(
            idx  = i,
            ts   = ts_ms // 1000,
            open = float(o),
            high = float(h),
            low  = float(l),
            close= float(c),
            volume               = float(v),
            quote_volume         = float(qv)   if qv   is not None else None,
            trades_count         = int(tc)      if tc   is not None else None,
            taker_buy_base_vol   = float(tbbv)  if tbbv is not None else None,
            taker_buy_quote_vol  = float(tbqv)  if tbqv is not None else None,
        ))
    return candles


# ═══════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _calc_moves(points: List[TurningPoint]) -> List[TurningPoint]:
    """Calcula move % y distancia en velas entre puntos consecutivos."""
    for i, p in enumerate(points):
        if i == 0:
            continue
        prev = points[i - 1]
        if prev.price and prev.price != 0:
            p.move_from_prev_pct = round(abs(p.price - prev.price) / prev.price * 100, 4)
        p.candles_from_prev = p.candle_idx - prev.candle_idx
        p.prev_ts           = prev.ts
        p.prev_price        = prev.price
    return points


def _dedup_consecutive(points: List[TurningPoint]) -> List[TurningPoint]:
    """
    Elimina puntos consecutivos del mismo tipo, conservando el más extremo.
    Garantiza alternancia TOP/BOTTOM.
    """
    if not points:
        return points
    result = [points[0]]
    for p in points[1:]:
        if p.tp_type == result[-1].tp_type:
            last = result[-1]
            if p.tp_type == "BOTTOM" and p.price < last.price:
                result[-1] = p
            elif p.tp_type == "TOP" and p.price > last.price:
                result[-1] = p
        else:
            result.append(p)
    return result


# ═══════════════════════════════════════════════════════════════════════════
# MÉTODO 1: WINDOW
# ═══════════════════════════════════════════════════════════════════════════

def detect_window(
    candles: List[RawCandle],
    window: int,
    min_move_pct: Optional[float] = None,
) -> List[TurningPoint]:
    """
    Detecta extremos locales simétricos (igual que IrrealStrategy).
    La vela central de una ventana de (2·W + 1) es:
      BOTTOM: su low  <= todos los lows  vecinos
      TOP:    su high >= todos los highs vecinos
    Si min_move_pct está definido, descarta puntos donde el movimiento
    desde el punto anterior < threshold.
    """
    n      = len(candles)
    params = {"window": window, "min_move_pct": min_move_pct}
    raw: List[TurningPoint] = []

    for i in range(window, n - window):
        c   = candles[i]
        nb  = candles[i - window : i] + candles[i + 1 : i + window + 1]

        is_bottom = all(c.low  <= v.low  for v in nb)
        is_top    = all(c.high >= v.high for v in nb)

        if is_bottom:
            raw.append(TurningPoint("window", params, c.ts, c.low,  "BOTTOM", i))
        elif is_top:
            raw.append(TurningPoint("window", params, c.ts, c.high, "TOP",    i))

    raw = _dedup_consecutive(raw)
    raw = _calc_moves(raw)

    if min_move_pct is not None:
        raw = [p for p in raw
               if p.move_from_prev_pct is None
               or p.move_from_prev_pct >= min_move_pct]
        # Tras el filtro pueden quedar dos puntos consecutivos del mismo tipo.
        # Re-deduplicar antes de recalcular moves garantiza la alternancia.
        raw = _dedup_consecutive(raw)
        raw = _calc_moves(raw)   # recalcular tras el filtro

    return raw


# ═══════════════════════════════════════════════════════════════════════════
# MÉTODO 2: ZIGZAG
# ═══════════════════════════════════════════════════════════════════════════

def detect_zigzag(
    candles: List[RawCandle],
    min_move_pct: float,
) -> List[TurningPoint]:
    """
    Detecta turning points alternados usando ZigZag.
    Confirma un TOP cuando el precio cae >= min_move_pct desde el máximo,
    y un BOTTOM cuando sube >= min_move_pct desde el mínimo.
    Usa high para tops y low para bottoms.
    """
    n      = len(candles)
    params = {"min_move_pct": min_move_pct}
    result: List[TurningPoint] = []

    if n < 2:
        return result

    # Detectar dirección inicial
    direction: Optional[int] = None  # 1 = up, -1 = down
    ext_idx   = 0
    ext_price = 0.0

    for i in range(1, min(n, 500)):   # máximo 500 velas para encontrar la primera dirección
        up_move   = (candles[i].high - candles[0].low)  / candles[0].low  * 100
        down_move = (candles[0].high - candles[i].low)  / candles[0].high * 100

        if up_move >= min_move_pct and direction is None:
            result.append(TurningPoint("zigzag", params, candles[0].ts, candles[0].low, "BOTTOM", 0))
            direction = 1
            ext_idx   = i
            ext_price = candles[i].high
            break

        if down_move >= min_move_pct and direction is None:
            result.append(TurningPoint("zigzag", params, candles[0].ts, candles[0].high, "TOP", 0))
            direction = -1
            ext_idx   = i
            ext_price = candles[i].low
            break

    if direction is None:
        return result

    # Loop principal
    for i in range(1, n):
        c = candles[i]

        if direction == 1:   # siguiendo subida, buscando TOP
            if c.high >= ext_price:
                ext_price = c.high
                ext_idx   = i
            else:
                retrace = (ext_price - c.low) / ext_price * 100
                if retrace >= min_move_pct:
                    result.append(TurningPoint("zigzag", params,
                                               candles[ext_idx].ts, ext_price, "TOP", ext_idx))
                    direction = -1
                    ext_price = c.low
                    ext_idx   = i

        else:                # direction == -1, siguiendo bajada, buscando BOTTOM
            if c.low <= ext_price:
                ext_price = c.low
                ext_idx   = i
            else:
                bounce = (c.high - ext_price) / ext_price * 100
                if bounce >= min_move_pct:
                    result.append(TurningPoint("zigzag", params,
                                               candles[ext_idx].ts, ext_price, "BOTTOM", ext_idx))
                    direction = 1
                    ext_price = c.high
                    ext_idx   = i

    return _calc_moves(result)


# ═══════════════════════════════════════════════════════════════════════════
# PERSISTENCIA SQLITE
# ═══════════════════════════════════════════════════════════════════════════

def init_db(db_path: str) -> None:
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS turning_points (
                id                 INTEGER PRIMARY KEY AUTOINCREMENT,
                method             TEXT    NOT NULL,
                params             TEXT    NOT NULL,
                ts                 INTEGER NOT NULL,
                datetime           TEXT    NOT NULL,
                price              REAL    NOT NULL,
                type               TEXT    NOT NULL,
                candle_idx         INTEGER NOT NULL,
                move_from_prev_pct REAL,
                candles_from_prev  INTEGER,
                prev_ts            INTEGER,
                prev_price         REAL
            );

            CREATE TABLE IF NOT EXISTS detection_runs (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                method        TEXT NOT NULL,
                params        TEXT NOT NULL,
                run_at        TEXT NOT NULL,
                fecha_inicio  TEXT NOT NULL,
                fecha_fin     TEXT NOT NULL,
                total_points  INTEGER,
                total_bottoms INTEGER,
                total_tops    INTEGER
            );

            CREATE INDEX IF NOT EXISTS idx_tp_method ON turning_points(method, params);
            CREATE INDEX IF NOT EXISTS idx_tp_ts     ON turning_points(ts);
            CREATE INDEX IF NOT EXISTS idx_tp_type   ON turning_points(type);
        """)
        conn.commit()
    finally:
        conn.close()


def clear_db(db_path: str) -> None:
    conn = sqlite3.connect(db_path)
    conn.execute("DELETE FROM turning_points")
    conn.execute("DELETE FROM detection_runs")
    conn.commit()
    conn.close()


def save_points(
    db_path: str,
    points:  List[TurningPoint],
    fecha_inicio: str,
    fecha_fin:    str,
) -> None:
    if not points:
        return
    params_str = json.dumps(points[0].params, sort_keys=True)
    bottoms    = sum(1 for p in points if p.tp_type == "BOTTOM")
    tops       = sum(1 for p in points if p.tp_type == "TOP")

    conn = sqlite3.connect(db_path)
    try:
        conn.executemany("""
            INSERT INTO turning_points
            (method, params, ts, datetime, price, type, candle_idx,
             move_from_prev_pct, candles_from_prev, prev_ts, prev_price)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [(
            p.method, params_str, p.ts, epoch_to_iso(p.ts),
            p.price, p.tp_type, p.candle_idx,
            p.move_from_prev_pct, p.candles_from_prev,
            p.prev_ts, p.prev_price,
        ) for p in points])

        conn.execute("""
            INSERT INTO detection_runs
            (method, params, run_at, fecha_inicio, fecha_fin,
             total_points, total_bottoms, total_tops)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            points[0].method, params_str,
            datetime.now(timezone.utc).isoformat(),
            fecha_inicio, fecha_fin,
            len(points), bottoms, tops,
        ))
        conn.commit()
    finally:
        conn.close()


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    t0 = _time.time()

    print("╔══════════════════════════════════════════════════════════╗")
    print("║    BUILD TURNING POINTS DB — Detección Multi-Método     ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"  DB fuente      : {DB_PATH}")
    print(f"  Rango          : {FECHA_INICIO} → {FECHA_FIN}")
    print(f"  Ventanas Window: {WINDOW_SIZES}")
    print(f"  Thresholds %   : {MIN_MOVE_PCTS}")
    print(f"  Output DB      : {OUTPUT_DB}")
    print("─" * 60)

    # 1. Cargar velas
    print("Cargando velas desde SQLite...", end=" ", flush=True)
    candles = load_candles(DB_PATH, DB_TABLE, FECHA_INICIO, FECHA_FIN)
    print(f"OK  ({len(candles):,} velas)")

    if not candles:
        print("✗  No se encontraron velas. Verificar DB_PATH y fechas.")
        return

    # 2. Inicializar y limpiar DB
    init_db(OUTPUT_DB)
    clear_db(OUTPUT_DB)

    total_runs = 0

    # ── Método Window ────────────────────────────────────────────────────
    print("\n[Método: Window]")
    for window in WINDOW_SIZES:
        # Sin filtro de movimiento (solo ventana)
        pts = detect_window(candles, window, min_move_pct=None)
        save_points(OUTPUT_DB, pts, FECHA_INICIO, FECHA_FIN)
        b = sum(1 for p in pts if p.tp_type == "BOTTOM")
        t = sum(1 for p in pts if p.tp_type == "TOP")
        label = f"window={window}, min_move=None"
        print(f"  {label:<42} → {len(pts):>4} pts  ({b}↓ {t}↑)")
        total_runs += 1

        # Con filtro de movimiento mínimo
        for pct in MIN_MOVE_PCTS:
            pts = detect_window(candles, window, min_move_pct=pct)
            save_points(OUTPUT_DB, pts, FECHA_INICIO, FECHA_FIN)
            b = sum(1 for p in pts if p.tp_type == "BOTTOM")
            t = sum(1 for p in pts if p.tp_type == "TOP")
            label = f"window={window}, min_move={pct}%"
            print(f"  {label:<42} → {len(pts):>4} pts  ({b}↓ {t}↑)")
            total_runs += 1

    # ── Método ZigZag ────────────────────────────────────────────────────
    print("\n[Método: ZigZag]")
    for pct in MIN_MOVE_PCTS:
        pts = detect_zigzag(candles, pct)
        save_points(OUTPUT_DB, pts, FECHA_INICIO, FECHA_FIN)
        b = sum(1 for p in pts if p.tp_type == "BOTTOM")
        t = sum(1 for p in pts if p.tp_type == "TOP")
        label = f"min_move={pct}%"
        print(f"  {label:<42} → {len(pts):>4} pts  ({b}↓ {t}↑)")
        total_runs += 1

    # ── Resumen ──────────────────────────────────────────────────────────
    elapsed = _time.time() - t0
    conn = sqlite3.connect(OUTPUT_DB)
    total_rows = conn.execute("SELECT COUNT(*) FROM turning_points").fetchone()[0]
    conn.close()

    print(f"\n{'═'*60}")
    print(f"  {total_runs} combinaciones procesadas en {elapsed:.1f}s")
    print(f"  {total_rows:,} registros guardados en {OUTPUT_DB}")
    print(f"{'═'*60}")
    print(f"\n✓  Listo. Ejecutar analyze_turning_points.py para el análisis.")


if __name__ == "__main__":
    main()