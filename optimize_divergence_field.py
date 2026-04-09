"""
optimize_divergence_field.py — Optimizador Flat de DivergenceFieldStrategy
═══════════════════════════════════════════════════════════════════════════
Prueba TODAS las combinaciones posibles en un solo barrido plano.
Sin fases, sin shortcuts, cada combo corre el backtest real completo.

TOTAL DE COMBINACIONES
───────────────────────
  Sin KNN: 69,120   (~30-90 min con 4-8 cores)
  Con KNN: 82,944   (~90-250 min con 4-8 cores)

GARANTÍA DE REPRODUCIBILIDAD
──────────────────────────────
  Pegar CONFIG + MAX_POSICIONES del resultado en backtest_divergence_field.py
  produce exactamente el mismo PnL y WR%.


MULTIPROCESSING + CHECKPOINT
─────────────────────────────
  Las velas se cargan una vez por worker process.
  Los resultados se guardan cada 200 combos para no perder progreso.
  Si se interrumpe: python optimize_divergence_field.py --resume

USO
───
  python optimize_divergence_field.py              # todo
  python optimize_divergence_field.py --no-knn     # excluye KNN (3x más rápido)
  python optimize_divergence_field.py --workers 6  # forzar N procesos
  python optimize_divergence_field.py --resume     # continuar tras interrupción
  python optimize_divergence_field.py --top 30     # mostrar top-30
  python optimize_divergence_field.py --min-trades 20
"""

from __future__ import annotations

import argparse
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

sys.path.insert(0, str(Path(__file__).parent))


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

KNN_MIN_WINDOW       = 15     # KNN inestable con ventana < 15

OUT_CSV              = "opt_divfield_results.csv"
OUT_JSON             = "opt_divfield_results.json"
CHECKPOINT_CSV       = "opt_divfield_checkpoint.csv"
SAVE_EVERY           = 200


# ══════════════════════════════════════════════════════════════════════════════
# TIPOS
# ══════════════════════════════════════════════════════════════════════════════

CSV_FIELDS = [
    "te_estimator","window_size","field_def","cmi_regimes",
    "threshold_mode","sink_mode","thr_bot","thr_top",
    "cooldown","max_posiciones",
    "pnl_pct","portfolio_final","bh_pnl","alpha_bh",
    "n_compras","n_ventas","n_trades","n_ignorados",
    "win_rate","pos_finales","valid",
]


@dataclass
class Combo:
    te_estimator:   str
    window_size:    int
    field_def:      str
    cmi_regimes:    int
    threshold_mode: str
    sink_mode:      str
    thr_bot:        float
    thr_top:        float
    cooldown:       int
    max_posiciones: int

    def key(self) -> str:
        return (f"{self.te_estimator}|{self.window_size}|{self.field_def}|"
                f"{self.cmi_regimes}|{self.threshold_mode}|{self.sink_mode}|"
                f"{self.thr_bot}|{self.thr_top}|{self.cooldown}|{self.max_posiciones}")


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
        for k in ("pnl_pct","portfolio_final","bh_pnl","alpha_bh"):
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
            valid          = str(d.get("valid","True")).lower() != "false",
        )


# ══════════════════════════════════════════════════════════════════════════════
# GENERADOR DE COMBINACIONES
# ══════════════════════════════════════════════════════════════════════════════

def all_combos(include_knn: bool = True) -> List[Combo]:
    estimators = DEEP_TE_ESTIMATORS if include_knn else [
        e for e in DEEP_TE_ESTIMATORS if e != "knn"
    ]
    combos = []
    for (te, win, fd, cr, tm, sm, tb, tt, cd, mp) in product(
        estimators, DEEP_WINDOWS, DEEP_FIELD_DEFS,
        DEEP_CMI_REGIMES, DEEP_THRESHOLD_MODES, DEEP_SINK_MODES,
        GRID_SCORE_BOT, GRID_SCORE_TOP, GRID_COOLDOWNS, GRID_MAX_POS,
    ):
        if te == "knn" and win < KNN_MIN_WINDOW:
            continue
        combos.append(Combo(
            te_estimator=te, window_size=win, field_def=fd,
            cmi_regimes=cr, threshold_mode=tm, sink_mode=sm,
            thr_bot=tb, thr_top=tt, cooldown=cd, max_posiciones=mp,
        ))
    return combos


# ══════════════════════════════════════════════════════════════════════════════
# WORKER
# ══════════════════════════════════════════════════════════════════════════════

_W_CANDLES: list  = []
_W_BHPNL:   float = 0.0
_W_USDT:    float = 1000.0
_W_COMM:    float = 0.1
_W_MINTR:   int   = 10


def _worker_init(db_path, db_table, fecha_ini, fecha_fin,
                 symbol, usdt, comm, min_trades):
    global _W_CANDLES, _W_BHPNL, _W_USDT, _W_COMM, _W_MINTR
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    import logging; logging.disable(logging.WARNING)

    from actors.price_feed import SQLiteFeed
    feed        = SQLiteFeed(db_path=db_path, table=db_table)
    _W_CANDLES  = feed.get_candles(fecha_ini, fecha_fin, symbol)
    ini         = feed.get_candles(fecha_ini, fecha_ini, symbol)
    p_ini       = ini[0].close if ini else 1.0
    p_fin       = _W_CANDLES[-1].close if _W_CANDLES else 1.0
    _W_BHPNL    = (p_fin / p_ini - 1) * 100.0
    _W_USDT     = usdt
    _W_COMM     = comm
    _W_MINTR    = min_trades


def _run_combo(combo: Combo) -> Optional[Result]:
    """
    Backtest completo para una Combo usando MemoryWallet.
    WR%: comparar precio_venta vs precio_promedio_posiciones() ANTES del update.
    """
    try:
        from actors.wallet     import MemoryWallet
        from actors.order_book import SimulatedOrderBook, OrderSide
        from strategies.divergence_field_strategy import (
            DivergenceFieldStrategy, DFConfig,
            TEEstimator, WindowMode, FieldDefinition,
            CMIRegimes, ThresholdMode, SinkMode,
        )

        cfg = DFConfig(
            te_estimator        = TEEstimator(combo.te_estimator),
            window_mode         = WindowMode.FIXED,
            window_size         = combo.window_size,
            field_def           = FieldDefinition(combo.field_def),
            cmi_regimes         = CMIRegimes(combo.cmi_regimes),
            threshold_mode      = ThresholdMode(combo.threshold_mode),
            sink_mode           = SinkMode(combo.sink_mode),
            score_threshold_bot = combo.thr_bot,
            score_threshold_top = combo.thr_top,
            cooldown            = combo.cooldown,
        )

        wallet   = MemoryWallet(usdt_inicial=_W_USDT,
                                max_posiciones=combo.max_posiciones)
        ob       = SimulatedOrderBook(commission_pct=_W_COMM,
                                      max_posiciones=combo.max_posiciones)
        strategy = DivergenceFieldStrategy(cfg)
        strategy.on_start(wallet)

        n_buy = n_sell = n_ign = n_wins = 0

        for candle in _W_CANDLES:
            sig = strategy._tick(candle, wallet)
            if not sig.is_actionable:
                continue

            side = sig.to_order_side()

            # Capturar precio promedio ANTES del execute_with_guards/update
            avg_entry = (wallet.precio_promedio_posiciones()
                         if side == OrderSide.SELL else 0.0)

            order = ob.execute_with_guards(side, sig.price, wallet,
                                           candle_ts=candle.ts)

            if order.is_filled:
                if side == OrderSide.BUY:
                    n_buy += 1
                else:
                    n_sell += 1
                    # Ganador: precio venta > precio promedio de entrada
                    if avg_entry > 0 and sig.price > avg_entry:
                        n_wins += 1
            else:
                n_ign += 1

        strategy.on_stop(wallet)

        last_price = _W_CANDLES[-1].close
        port       = wallet.portfolio_value(last_price)
        pnl        = (port / _W_USDT - 1) * 100.0
        wr         = (n_wins / n_sell * 100.0) if n_sell > 0 else 0.0
        n_trades   = n_buy + n_sell

        return Result(
            te_estimator   = combo.te_estimator,
            window_size    = combo.window_size,
            field_def      = combo.field_def,
            cmi_regimes    = combo.cmi_regimes,
            threshold_mode = combo.threshold_mode,
            sink_mode      = combo.sink_mode,
            thr_bot        = combo.thr_bot,
            thr_top        = combo.thr_top,
            cooldown       = combo.cooldown,
            max_posiciones = combo.max_posiciones,
            pnl_pct        = pnl,
            portfolio_final= port,
            bh_pnl         = _W_BHPNL,
            alpha_bh       = pnl - _W_BHPNL,
            n_compras      = n_buy,
            n_ventas       = n_sell,
            n_trades       = n_trades,
            n_ignorados    = n_ign,
            win_rate       = wr,
            pos_finales    = wallet.positions_count,
            valid          = n_trades >= _W_MINTR,
        )
    except Exception:
        return None


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
    if not Path(CHECKPOINT_CSV).exists():
        return [], set()
    results = []
    with open(CHECKPOINT_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                results.append(Result.from_dict(row))
            except Exception:
                pass
    done_keys = {
        Combo(
            te_estimator=r.te_estimator, window_size=r.window_size,
            field_def=r.field_def, cmi_regimes=r.cmi_regimes,
            threshold_mode=r.threshold_mode, sink_mode=r.sink_mode,
            thr_bot=r.thr_bot, thr_top=r.thr_top,
            cooldown=r.cooldown, max_posiciones=r.max_posiciones,
        ).key()
        for r in results
    }
    return results, done_keys


# ══════════════════════════════════════════════════════════════════════════════
# OUTPUT
# ══════════════════════════════════════════════════════════════════════════════

def print_table(results: List[Result], top_n: int, min_trades: int) -> None:
    valid = sorted(
        [r for r in results if r.valid],
        key=lambda r: r.pnl_pct, reverse=True
    )
    bh = valid[0].bh_pnl if valid else 0.0

    sep  = "─" * 118
    sep2 = "═" * 118
    print(f"\n{sep2}")
    print(f"  RESULTADOS FINALES — {len(valid):,} configs válidas "
          f"(≥{min_trades} trades) de {len(results):,} total  |  B&H: {bh:+.2f}%")
    print(sep2)
    print(f"  {'#':>4}  {'TE':>7} {'W':>3} {'Field':>5} {'C':>2} "
          f"{'Th':>5} {'Sk':>5}  "
          f"{'thr_b':>5} {'thr_t':>5} {'cd':>4} {'mp':>3}  "
          f"{'PnL%':>9} {'α_BH':>8}  "
          f"{'B/S':>9}  {'WR%':>5} {'Pos':>3}")
    print(sep)

    for i, r in enumerate(valid[:top_n], 1):
        pnl_s = f"{'+' if r.pnl_pct>=0 else ''}{r.pnl_pct:.2f}%"
        abh_s = f"{'+' if r.alpha_bh>=0 else ''}{r.alpha_bh:.2f}%"
        cd_s  = str(r.cooldown) if r.cooldown else "off"
        bs_s  = f"{r.n_compras}B/{r.n_ventas}S"
        q     = ("✓" if r.win_rate >= 50 and r.n_trades >= 30 else
                 "⚠" if r.win_rate < 42 else "")
        print(f"  {i:>4}.  "
              f"{r.te_estimator:>7} {r.window_size:>3} "
              f"{r.field_def[:5]:>5} {r.cmi_regimes:>2} "
              f"{r.threshold_mode[:5]:>5} {r.sink_mode[:5]:>5}  "
              f"{r.thr_bot:>5.2f} {r.thr_top:>5.2f} {cd_s:>4} {r.max_posiciones:>3}  "
              f"{pnl_s:>9} {abh_s:>9}  "
              f"{bs_s:>9}  {r.win_rate:>5.1f}% {r.pos_finales:>3} {q}")

    print(sep)
    if len(valid) > top_n:
        print(f"  ... {len(valid)-top_n:,} configs más en {OUT_CSV}")
    print(f"\n  ✓ = WR≥50% y trades≥30  |  ⚠ = WR<42%  |  vacío = intermedio")


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


def save_final(results: List[Result], total: int,
               elapsed_s: float, min_trades: int) -> None:
    all_s = sorted(results, key=lambda r: r.pnl_pct, reverse=True)
    valid = [r for r in all_s if r.valid]

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in all_s:
            w.writerow(r.to_dict())

    bh = valid[0].bh_pnl if valid else 0.0
    meta = {
        "n_total": total, "n_calculadas": len(results),
        "n_valid": len(valid), "min_trades": min_trades,
        "bh_pnl": round(bh, 2),
        "elapsed_min": round(elapsed_s / 60, 1),
        "best_pnl": round(valid[0].pnl_pct,  2) if valid else None,
        "best_wr":  round(valid[0].win_rate,  1) if valid else None,
        "grid_params": {
            "GRID_SCORE_BOT": GRID_SCORE_BOT, "GRID_SCORE_TOP": GRID_SCORE_TOP,
            "GRID_COOLDOWNS": GRID_COOLDOWNS,  "DEEP_TE_ESTIMATORS": DEEP_TE_ESTIMATORS,
            "DEEP_WINDOWS":   DEEP_WINDOWS,    "DEEP_FIELD_DEFS": DEEP_FIELD_DEFS,
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
        description="Optimizador flat — todas las combinaciones DivergenceField",
    )
    parser.add_argument("--no-knn",     action="store_true")
    parser.add_argument("--workers",    type=int, default=None)
    parser.add_argument("--resume",     action="store_true")
    parser.add_argument("--top",        type=int, default=20)
    parser.add_argument("--min-trades", type=int, default=10,
                        dest="min_trades")
    args = parser.parse_args()

    import config_local as CL

    combos_all = all_combos(include_knn=not args.no_knn)
    total      = len(combos_all)
    n_workers  = args.workers or max(1, multiprocessing.cpu_count() - 1)

    print("╔══════════════════════════════════════════════════════════╗")
    print("║   OPTIMIZADOR FLAT — DivergenceField  BTC/USDT          ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"  Dataset       : {CL.FECHA_INICIO} → {CL.FECHA_FIN}")
    print(f"  Capital       : ${CL.SALDO_USDT_INICIAL:,.2f}  Comisión: {CL.COMMISSION_PCT}%")
    print(f"  MAX_POS grid  : {GRID_MAX_POS}")
    print(f"  Combinaciones : {total:,}")
    print(f"  Workers       : {n_workers} / {multiprocessing.cpu_count()} cores")
    print(f"  KNN           : {'No' if args.no_knn else 'Sí'}")
    print(f"  Min trades    : {args.min_trades}")
    print(f"  Checkpoint    : {'continuar' if args.resume else 'desde cero'}")
    print("─" * 60)

    # Tiempo estimado
    spd = {"binning": 0.8, "kde": 1.5, "knn": 6.0}
    cnt: Dict[str,int] = {}
    for c in combos_all:
        cnt[c.te_estimator] = cnt.get(c.te_estimator, 0) + 1
    est_s = sum(cnt.get(te,0)*s/n_workers for te,s in spd.items())
    print(f"  Tiempo estimado: ~{est_s/60:.0f} min\n")

    # Resume
    results:   List[Result] = []
    done_keys: set           = set()
    if args.resume:
        results, done_keys = load_checkpoint()
        if results:
            print(f"  ✓ Checkpoint: {len(results):,} resultados previos")
    combos_pending = [c for c in combos_all if c.key() not in done_keys]
    if done_keys:
        print(f"  ✓ Pendientes: {len(combos_pending):,} combinaciones\n")

    if not combos_pending:
        print("  ✓ Todas las combinaciones calculadas — mostrando resultados\n")
    else:
        # B&H
        from actors.price_feed import SQLiteFeed
        feed = SQLiteFeed(db_path=CL.DB_PATH, table=CL.DB_TABLE)
        ini  = feed.get_candles(CL.FECHA_INICIO, CL.FECHA_INICIO, CL.SYMBOL)
        fin  = feed.get_candles(CL.FECHA_FIN,    CL.FECHA_FIN,    CL.SYMBOL)
        p_i  = ini[0].close  if ini else 1.0
        p_f  = (fin or feed.get_candles(CL.FECHA_INICIO, CL.FECHA_FIN, CL.SYMBOL))[-1].close
        bh   = (p_f / p_i - 1) * 100.0
        print(f"  B&H: {bh:+.2f}%  |  Precio: ${p_i:,.0f} → ${p_f:,.0f}\n")
        print("  Iniciando workers...\n  ", end="")

        init_args = (
            CL.DB_PATH, CL.DB_TABLE,
            CL.FECHA_INICIO, CL.FECHA_FIN, CL.SYMBOL,
            CL.SALDO_USDT_INICIAL, CL.COMMISSION_PCT,
            args.min_trades,
        )

        t0       = time.time()
        best_pnl = max((r.pnl_pct for r in results if r.valid), default=-9999.0)
        done     = len(results)

        ctx  = multiprocessing.get_context("spawn")
        pool = ctx.Pool(
            processes   = n_workers,
            initializer = _worker_init,
            initargs    = init_args,
        )

        try:
            for result in pool.imap_unordered(_run_combo, combos_pending,
                                               chunksize=4):
                done += 1
                if result is not None:
                    results.append(result)
                    if result.valid and result.pnl_pct > best_pnl:
                        best_pnl = result.pnl_pct

                # Barra de progreso inline
                pct   = done / total
                blen  = 28
                filled= int(blen * pct)
                bar   = "█"*filled + "░"*(blen-filled)
                el    = time.time() - t0
                eta   = (el/pct - el) if pct > 0.001 else 0
                eta_s = f"{eta/60:.0f}m" if eta>60 else f"{eta:.0f}s"
                bp_s  = f"{'+' if best_pnl>=0 else ''}{best_pnl:.1f}%" if best_pnl>-9999 else "—"
                print(f"\r  [{bar}] {pct*100:5.1f}%  "
                      f"{done:>6,}/{total:,}  ETA:{eta_s:>5}  Best:{bp_s:>8} ",
                      end="", flush=True)

                if len(results) % SAVE_EVERY == 0:
                    save_checkpoint(results)

        except KeyboardInterrupt:
            print(f"\n\n  [CTRL+C] Guardando {len(results):,} resultados...")
            pool.terminate()
            pool.join()
            save_checkpoint(results)
            print(f"  ✓ Guardado en {CHECKPOINT_CSV}")
            print("  Para continuar: python optimize_divergence_field.py --resume")
            _show_partial(results, args.top, args.min_trades, CL)
            return
        else:
            pool.close()
            pool.join()
            print()

        elapsed = time.time() - t0
        print(f"\n  ✓ Completado en {elapsed/60:.1f} min  |  "
              f"{len(results):,} resultados  |  "
              f"{len([r for r in results if r.valid]):,} válidos")

    _show_partial(results, args.top, args.min_trades, CL)

    # Limpiar checkpoint
    if Path(CHECKPOINT_CSV).exists():
        try:
            os.remove(CHECKPOINT_CSV)
        except Exception:
            pass

    print("\n✓ Optimización completada.")


def _show_partial(results: List[Result], top_n: int, min_trades: int, CL) -> None:
    if not results:
        print("  Sin resultados.")
        return

    print_table(results, top_n=top_n, min_trades=min_trades)

    t0_dummy = time.time()
    save_final(results, len(results), 0, min_trades)

    valid = sorted([r for r in results if r.valid],
                   key=lambda r: r.pnl_pct, reverse=True)
    if valid:
        print_best(valid[0], CL.SALDO_USDT_INICIAL, CL.COMMISSION_PCT)
    else:
        print(f"\n  ⚠ No hay configs válidas con ≥{min_trades} trades.")
        print("  Considerar: python optimize_divergence_field.py --min-trades 5")


if __name__ == "__main__":
    main()
