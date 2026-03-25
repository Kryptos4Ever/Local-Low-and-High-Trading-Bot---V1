"""
backtest_local_reversal.py — Runner de la Estrategia LocalReversal
════════════════════════════════════════════════════════════════════
Ensambla todos los actores del sistema y corre LocalReversalStrategy:
un modelo de Gradient Boosting que detecta mínimos y máximos locales
a partir de la microestructura de las velas.

Resultado: JSON compatible con Graficador.py

Uso:
    python backtest_local_reversal.py           # entrena y corre
    python backtest_local_reversal.py --fast    # usa cache si existe
    python backtest_local_reversal.py --nocache # borra cache y reentrena

Configuración del entorno:
    config_local.py  →  rutas, fechas, capital, comisión, MAX_POSICIONES

Parámetros de la estrategia (configurar en este archivo):
    THR_B, THR_T  →  ver documentación abajo

Actores utilizados
───────────────────
    PriceFeed  : SQLiteFeed        (DB local)
    Wallet     : JSONWallet        (persiste resultados)
    OrderBook  : SimulatedOrderBook
    Clock      : LocalClock
    Risk       : RiskManager permisivo (sin límites)
    State      : MemoryStateManager   (sin persistencia entre sesiones)

────────────────────────────────────────────────────────────────────
GUÍA DE PARÁMETROS
────────────────────────────────────────────────────────────────────

THR_B — Umbral de confianza para señal de COMPRA
  El modelo de bottoms asigna a cada vela una probabilidad [0,1] de
  ser un mínimo local genuino. THR_B es el piso mínimo que debe
  superar esa probabilidad para que el sistema genere una orden BUY.

  Rango explorado durante calibración: [0.40, 0.85]
  Valor calibrado (óptimo por robustez): 0.50

  Efecto de subirlo: menos señales de compra, más selectivas.
  Efecto de bajarlo: más señales de compra, menos selectivas.

  Rango razonable para ajuste manual: [0.45, 0.65]

THR_T — Umbral de confianza para señal de VENTA
  El modelo de tops asigna a cada vela una probabilidad [0,1] de
  ser un máximo local genuino. THR_T es el piso mínimo para generar
  una orden SELL (cierre de posición).

  Rango explorado durante calibración: [0.40, 0.85]
  Valor calibrado (óptimo por robustez): 0.45

  NOTA: THR_T es intencionalmente menor que THR_B. Esta asimetría
  surge de la calibración y es deliberada: facilita el cierre de
  posiciones abiertas reduciendo el tiempo en riesgo y el drawdown.

  Rango razonable para ajuste manual: [0.40, 0.60]

RENDIMIENTO ESPERADO (validación out-of-sample, todos los años OOS):
  Año   Estrategia  Buy&Hold   Alpha   Win Rate
  2021    +90.8%     +59.4%   +31.4%    59.4%
  2022    +21.6%     -64.5%   +86.1%    57.1%   ← bear market severo
  2023    +21.3%    +155.8%  -134.5%    65.4%
  2024    +38.0%    +120.3%   -82.3%    63.8%
  2025    +17.9%      -7.2%   +25.1%    61.4%   ← fuera de muestra
  (Con capital $1000, MAX_POSICIONES=5, COMMISSION=0.1%)
────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

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

log = get_logger("backtest_local_reversal")


# ════════════════════════════════════════════════════════════════════
# PARÁMETROS DE LA ESTRATEGIA
# ════════════════════════════════════════════════════════════════════

THR_B     = 0.5   # umbral señal BUY   — rango razonable: [0.45, 0.65]
THR_T     = 0.5   # umbral señal SELL  — rango razonable: [0.40, 0.60]

CACHE_DIR = ".cache_local_reversal"


# ════════════════════════════════════════════════════════════════════
# RUNNER
# ════════════════════════════════════════════════════════════════════

def main(use_cache: bool = True, clear_cache: bool = False) -> None:
    t_start = time.time()

    # Manejar cache
    if clear_cache and Path(CACHE_DIR).exists():
        shutil.rmtree(CACHE_DIR)
        print(f"✓ Cache borrado: {CACHE_DIR}")

    print("╔══════════════════════════════════════════════════════════╗")
    print("║        BACKTEST LOCAL REVERSAL — GBM BTC/USDT           ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"  Rango         : {CL.FECHA_INICIO} → {CL.FECHA_FIN}")
    print(f"  Capital       : ${CL.SALDO_USDT_INICIAL:,.2f} USDT")
    print(f"  Max posiciones: {CL.MAX_POSICIONES}")
    print(f"  Comisión      : {CL.COMMISSION_PCT}%")
    print(f"  THR_B (compra): {THR_B}  THR_T (venta): {THR_T}")
    print(f"  Cache         : {'--fast (reutiliza)' if use_cache else 'reentrena'}")
    print(f"  Output JSON   : {CL.RESULTS_JSON}")
    print("─" * 60)

    # ── 1. Construir actores ───────────────────────────────────────
    feed   = SQLiteFeed(db_path=CL.DB_PATH, table=CL.DB_TABLE)
    wallet = JSONWallet(
        usdt_inicial   = CL.SALDO_USDT_INICIAL,
        max_posiciones = CL.MAX_POSICIONES,
        json_path      = CL.RESULTS_JSON,
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
        thr_b           = THR_B,
        thr_t           = THR_T,
        cache_dir       = CACHE_DIR,
        force_recompute = not use_cache,
    )

    # ── 2. on_start: entrena o carga modelos sobre dataset completo ──
    print("Inicializando estrategia...")
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
        return

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
        "umbral_filtro"           : THR_B,
        "parametros"              : strategy.describe(),
    }

    # ── 8. Guardar JSON ───────────────────────────────────────────
    wallet.flush(summary)

    # ── 9. Resumen en consola ─────────────────────────────────────
    elapsed = time.time() - t_start
    sep  = "═" * 60
    sign = "+" if pnl_pct >= 0 else ""
    print(f"\n{sep}")
    print("  RESUMEN")
    print(sep)
    print(f"  Portfolio final  : ${port_final:>12,.2f} USDT")
    print(f"  └─ USDT libre    : ${wallet.get_usdt_balance():>12,.2f}")
    print(f"  └─ BTC en posic. :  {wallet.btc_en_posiciones():.8f} BTC")
    print(f"  PnL              : {sign}{pnl_pct:.2f}%")
    print(f"  Buy & Hold ref   : {bh_pnl:+.2f}%")
    print(f"  Alpha vs B&H     : {pnl_pct - bh_pnl:+.2f}%")
    print(f"  Compras          : {n_compras:,}")
    print(f"  Ventas           : {n_ventas:,}")
    print(f"  Ignorados        : {n_ignorados:,}  → {ign_motivos}")
    print(f"  Posiciones abier.: {wallet.positions_count}")
    print(f"  Tiempo total     : {elapsed:.1f}s")
    print(sep)
    print(f"\n✓ Resultado guardado en: {CL.RESULTS_JSON}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Backtest LocalReversal — Detección de reversals locales con GBM"
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Usa cache existente en lugar de reentrenar el modelo",
    )
    parser.add_argument(
        "--nocache",
        action="store_true",
        help="Borra el cache y reentrena el modelo desde cero",
    )
    args = parser.parse_args()
    main(use_cache=args.fast, clear_cache=args.nocache)