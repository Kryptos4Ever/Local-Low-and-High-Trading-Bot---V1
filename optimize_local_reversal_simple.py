"""
optimize_local_reversal.py — Parameter sweep for LocalReversalStrategy
════════════════════════════════════════════════════════════════════
Performs a grid search over THR_B and THR_T to optimize final PNL.
Keeps only the top 10 results in a SQLite database to avoid large storage.

Usage:
    python optimize_local_reversal.py
"""

import sqlite3
import time
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

import config_local as CL
from actors.price_feed         import SQLiteFeed
from actors.wallet             import JSONWallet, TradeRecord
from actors.order_book         import SimulatedOrderBook, OrderSide
from actors.clock              import LocalClock
from risk.risk_manager         import RiskManager, RiskConfig
from state.state_manager       import MemoryStateManager, Checkpoint
from strategies.local_reversal import LocalReversalStrategy
from strategies.base_strategy  import SignalSide
from support.logger            import get_logger

log = get_logger("optimize_local_reversal")

# Parameter ranges for grid search (as per documentation)
THR_B_RANGE = np.linspace(0.45, 0.65, 20)  # 20 points from 0.45 to 0.65
THR_T_RANGE = np.linspace(0.40, 0.60, 20)  # 20 points from 0.40 to 0.60

# Database for top 10 results
DB_PATH = "top10_results.db"
CACHE_DIR = ".cache_local_reversal"


def init_db():
    """Initialize SQLite database for top 10 results."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            thr_b REAL,
            thr_t REAL,
            pnl_pct REAL,
            summary TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()


def insert_result(thr_b, thr_t, pnl_pct, summary):
    """Insert a result into the database, maintaining only top 10 by PNL."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Insert new result
    cursor.execute('''
        INSERT INTO results (thr_b, thr_t, pnl_pct, summary)
        VALUES (?, ?, ?, ?)
    ''', (thr_b, thr_t, pnl_pct, str(summary)))
    
    # Keep only top 10 results by PNL (descending)
    cursor.execute('''
        DELETE FROM results
        WHERE id NOT IN (
            SELECT id FROM results
            ORDER BY pnl_pct DESC
            LIMIT 10
        )
    ''')
    conn.commit()
    conn.close()


def get_top10():
    """Retrieve top 10 results from database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT thr_b, thr_t, pnl_pct, summary
        FROM results
        ORDER BY pnl_pct DESC
    ''')
    rows = cursor.fetchall()
    conn.close()
    return rows


def run_backtest(thr_b, thr_t, use_cache=True, clear_cache=False):
    """
    Run a single backtest with given thresholds.
    Returns the summary dictionary from the wallet.
    """
    # Handle cache
    if clear_cache and Path(CACHE_DIR).exists():
        import shutil
        shutil.rmtree(CACHE_DIR)
        print(f"✓ Cache borrado: {CACHE_DIR}")

    # ── 1. Construir actores ───────────────────────────────────────
    feed   = SQLiteFeed(db_path=CL.DB_PATH, table=CL.DB_TABLE)
    wallet = JSONWallet(
        usdt_inicial   = CL.SALDO_USDT_INICIAL,
        max_posiciones = CL.MAX_POSICIONES,
        json_path      = CL.RESULTS_JSON,  # Will be overwritten each run
    )
    ob     = SimulatedOrderBook(
        commission_pct = CL.COMMISSION_PCT,
        max_posiciones = CL.MAX_POSICIONES,
    )
    # Backtest: sin límites de riesgo
    risk   = RiskManager(config=RiskConfig.permissive(),
                         usdt_inicial=CL.SALDO_USDT_INICIAL)
    state  = MemoryStateManager()

    strategy = LocalReversalStrategy(
        thr_b           = thr_b,
        thr_t           = thr_t,
        cache_dir       = CACHE_DIR,
        force_recompute = not use_cache,
    )

    # ── 2. on_start: entrena o carga modelos sobre dataset completo ──
    print(f"Inicializando estrategia (THR_B={thr_b:.3f}, THR_T={thr_t:.3f})...")
    strategy.on_start(
        wallet = wallet,
        feed   = feed,
        start  = "2017-01-01",
        end    = "2030-01-01",
        symbol = CL.SYMBOL,
    )

    # ── 3. Clock solo para el rango de backtest ───────────────────
    clock = LocalClock(
        feed   = feed,
        start  = CL.FECHA_INICIO,
        end    = CL.FECHA_FIN,
        symbol = CL.SYMBOL,
    )

    # ── 4. Contadores ─────────────────────────────────────────────
    n_compras    = 0
    n_ventas     = 0
    n_ignorados  = 0
    ign_motivos: dict[str, int] = {}
    precio_min_comprado = float("inf")
    precio_max_vendido  = float("-inf")
    last_candle  = None

    # ── 5. Loop principal ─────────────────────────────────────────
    print("Procesando velas...", end=" ", flush=True)

    for candle in clock:
        last_candle = candle
        signal      = strategy._tick(candle, wallet)

        if not signal.is_actionable:
            continue

        order_side = signal.to_order_side()

        # Risk check
        risk_reason = risk.check(order_side, signal.price, wallet, candle)
        if risk_reason:
            n_ignorados += 1
            ign_motivos[risk_reason] = ign_motivos.get(risk_reason, 0) + 1
            wallet.update(TradeRecord(
                ts            = candle.ts,
                side          = order_side.value,
                price         = signal.price,
                ignored       = True,
                ignore_reason = risk_reason,
            ))
            continue

        # Ejecutar con guardias del OrderBook
        order = ob.execute_with_guards(
            order_side, signal.price, wallet, candle_ts=candle.ts
        )

        if order.is_filled:
            if order_side == OrderSide.BUY:
                n_compras += 1
                precio_min_comprado = min(precio_min_comprado, signal.price)
            else:
                n_ventas += 1
                precio_max_vendido = max(precio_max_vendido, signal.price)
        else:
            n_ignorados += 1
            motivo = order.reject_reason or "desconocido"
            ign_motivos[motivo] = ign_motivos.get(motivo, 0) + 1

        # Enriquecer el último trade con los scores del modelo
        if wallet.get_trade_log():
            last_trade = wallet.get_trade_log()[-1]
            last_trade["score_bot"] = (
                signal.score if signal.side == SignalSide.BUY  else 0.0
            )
            last_trade["score_top"] = (
                signal.score if signal.side == SignalSide.SELL else 0.0
            )

        risk.update_peak(wallet.portfolio_value(candle.close))
        state.save(Checkpoint.from_wallet(
            wallet, candle.ts, candle.close,
            metadata={"estrategia": strategy.name},
        ))

    print(f"OK  ({clock.total_candles:,} velas)")

    # ── 6. Métricas finales ───────────────────────────────────────
    strategy.on_stop(wallet)

    if last_candle is None:
        print("✗ No se encontraron velas en el rango indicado.")
        return None

    precio_final  = last_candle.close
    port_final    = wallet.portfolio_value(precio_final)
    pnl_pct       = (port_final / CL.SALDO_USDT_INICIAL - 1) * 100

    first_candles  = feed.get_candles(CL.FECHA_INICIO, CL.FECHA_INICIO)
    precio_inicial = first_candles[0].close if first_candles else precio_final
    bh_pnl         = (precio_final / precio_inicial - 1) * 100

    all_candles = feed.get_candles(CL.FECHA_INICIO, CL.FECHA_FIN)
    atl = min(c.low  for c in all_candles) if all_candles else 0
    ath = max(c.high for c in all_candles) if all_candles else 0

    # ── 7. Armar summary ──────────────────────────────────────────
    summary = {
        "estrategia"              : strategy.name,
        "fecha_inicio"            : CL.FECHA_INICIO,
        "fecha_fin"               : CL.FECHA_FIN,
        "saldo_inicial_usdt"      : CL.SALDO_USDT_INICIAL,
        "usdt_balance_final"      : round(wallet.get_usdt_balance(), 8),
        "btc_balance_final"       : round(wallet.get_btc_balance(), 10),
        "btc_acumulado_total"     : round(wallet.get_btc_acumulado(), 10),
        "btc_en_posiciones_final" : round(wallet.btc_en_posiciones(), 10),
        "precio_promedio_final"   : round(wallet.precio_promedio_posiciones(), 8),
        "portfolio_value_final"   : round(port_final, 4),
        "pnl_pct"                 : round(pnl_pct, 4),
        "buy_hold_pnl_pct"        : round(bh_pnl, 4),
        "alpha_vs_bh"             : round(pnl_pct - bh_pnl, 4),
        "precio_min_comprado"     : round(precio_min_comprado, 4) if n_compras else None,
        "precio_max_vendido"      : round(precio_max_vendido,  4) if n_ventas  else None,
        "atl_final"               : round(atl, 4),
        "ath_proyectado_final"    : round(ath, 4),
        "total_trades_ejecutados" : n_compras + n_ventas,
        "total_compras"           : n_compras,
        "total_ventas"            : n_ventas,
        "total_ignorados"         : n_ignorados,
        "ordenes_canceladas"      : 0,
        "ignorados_por_motivo"    : ign_motivos,
        "positions_count_final"   : wallet.positions_count,
        "usdt_reserva_aplicada"   : 0.0,
        "umbral_filtro"           : thr_b,
        "parametros"              : strategy.describe(),
    }

    # ── 8. Guardar JSON (overwrites previous, but we only care about summary) ─────
    wallet.flush(summary)

    return summary


def main():
    """Main parameter sweep function."""
    print("╔═════════════════════════════════════════════════════════╗")
    print("║   OPTIMIZACIÓN DE PARÁMETROS PARA LOCAL REVERSAL       ║")
    print("╚═════════════════════════════════════════════════════════╝")
    print(f"  Rango THR_B: {THR_B_RANGE[0]:.3f} → {THR_B_RANGE[-1]:.3f} ({len(THR_B_RANGE)} puntos)")
    print(f"  Rango THR_T: {THR_T_RANGE[0]:.3f} → {THR_T_RANGE[-1]:.3f} ({len(THR_T_RANGE)} puntos)")
    print(f"  Total combinaciones: {len(THR_B_RANGE) * len(THR_T_RANGE)}")
    print(f"  Base de datos top 10: {DB_PATH}")
    print("─" * 60)

    # Initialize database
    init_db()
    
    start_time = time.time()
    total_combinations = len(THR_B_RANGE) * len(THR_T_RANGE)
    current = 0

    # Grid search
    for thr_b in THR_B_RANGE:
        for thr_t in THR_T_RANGE:
            current += 1
            print(f"\n[{current}/{total_combinations}] Probando THR_B={thr_b:.3f}, THR_T={thr_t:.3f}...")
            
            try:
                summary = run_backtest(thr_b, thr_t, use_cache=True, clear_cache=False)
                if summary is None:
                    print("  → Backtest falló, saltando...")
                    continue
                    
                pnl_pct = summary["pnl_pct"]
                print(f"  → PNL: {pnl_pct:+.2f}%")
                
                # Insert result into database (keeps only top 10)
                insert_result(thr_b, thr_t, pnl_pct, summary)
                                
                # Show current top 10
                top10 = get_top10()
                print(f"  → Top 10 actual (mejor PNL: {top10[0][2]:+.2f}%):")
                for i, (tb, tt, pnl, _) in enumerate(top10[:5], 1):  # Show top 5 for brevity
                    print(f"     {i}. THR_B={tb:.3f}, THR_T={tt:.3f} → PNL={pnl:+.2f}%")
                    
            except Exception as e:
                print(f"  → Error en backtest: {e}")
                continue
                
    elapsed = time.time() - start_time
    print("\n" + "═" * 60)
    print("  BÚSQUEDA COMPLETADA")
    print("═" * 60)
    print(f"  Tiempo total: {elapsed:.1f}s")
    print(f"  Combinaciones evaluadas: {current}")
    
    # Final top 10
    top10 = get_top10()
    print(f"\n  TOP 10 RESULTADOS (ordenados por PNL):")
    print("─" * 60)
    for i, (thr_b, thr_t, pnl_pct, summary_str) in enumerate(top10, 1):
        print(f"{i:2d}. THR_B={thr_b:.3f}, THR_T={thr_t:.3f} → PNL={pnl_pct:+.2f}%")
    
    print(f"\n✓ Resultados guardados en: {DB_PATH}")
    print("  Puede consultar la base de datos con:")
    print(f"    sqlite3 {DB_PATH} \"SELECT * FROM results ORDER BY pnl_pct DESC LIMIT 10;\"")
    

if __name__ == "__main__":
    main()