"""
backtest_irreal.py — Runner del Benchmark Irreal
═════════════════════════════════════════════════
Ensambla todos los actores del sistema y corre la IrrealStrategy
(oráculo perfecto) para producir el techo teórico de rendimiento.

Resultado: JSON compatible con Graficador.py

Uso:
    python backtest_irreal.py

Configuración:
    · config_local.py  →  rutas, fechas, capital, MAX_POSICIONES, comisión
    · strategies/irreal.py → parámetros de la estrategia (VENTANA, precios)

Actores utilizados
───────────────────
    PriceFeed  : SQLiteFeed        (DB local)
    Wallet     : JSONWallet        (persiste resultados)
    OrderBook  : SimulatedOrderBook
    Clock      : LocalClock
    Risk       : RiskManager permisivo (sin límites)
    State      : MemoryStateManager   (sin persistencia entre sesiones)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import config_local as CL

from actors.price_feed    import SQLiteFeed
from actors.wallet        import JSONWallet, TradeRecord
from actors.order_book    import SimulatedOrderBook, OrderSide
from actors.clock         import LocalClock
from risk.risk_manager    import RiskManager, RiskConfig
from state.state_manager  import MemoryStateManager, Checkpoint
from strategies.irreal    import IrrealStrategy
from strategies.base_strategy import SignalSide
from support.logger       import get_logger

log = get_logger("backtest_irreal")


# ══════════════════════════════════════════════════════════════════════════════
# PARÁMETROS DE LA ESTRATEGIA
# ══════════════════════════════════════════════════════════════════════════════
VENTANA_LOCAL  = 10       # velas a cada lado para confirmar extremo
PRECIO_COMPRA  = "low"    # "low" | "close" | "open"
PRECIO_VENTA   = "high"   # "high" | "close" | "open"


# ══════════════════════════════════════════════════════════════════════════════
# RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    t_start = time.time()

    print("╔══════════════════════════════════════════════════════════╗")
    print("║       BACKTEST IRREAL — Oráculo Perfecto BTC/USDT       ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"  Rango         : {CL.FECHA_INICIO} → {CL.FECHA_FIN}")
    print(f"  Capital       : ${CL.SALDO_USDT_INICIAL:,.2f} USDT")
    print(f"  Max posiciones: {CL.MAX_POSICIONES}  "
          f"(slot inicial = ${CL.SALDO_USDT_INICIAL / CL.MAX_POSICIONES:,.2f})")
    print(f"  Ventana local : {VENTANA_LOCAL} velas a cada lado")
    print(f"  Precio compra : {PRECIO_COMPRA}   |  Precio venta: {PRECIO_VENTA}")
    print(f"  Comisión      : {CL.COMMISSION_PCT}%")
    print(f"  Output JSON   : {CL.RESULTS_JSON}")
    print("─" * 60)

    # ── 1. Construir actores ───────────────────────────────────────────────────
    feed  = SQLiteFeed(db_path=CL.DB_PATH, table=CL.DB_TABLE)
    clock = LocalClock(feed, start=CL.FECHA_INICIO, end=CL.FECHA_FIN,
                       symbol=CL.SYMBOL)
    wallet = JSONWallet(
        usdt_inicial   = CL.SALDO_USDT_INICIAL,
        max_posiciones = CL.MAX_POSICIONES,
        json_path      = CL.RESULTS_JSON,
    )
    ob    = SimulatedOrderBook(
        commission_pct = CL.COMMISSION_PCT,
        max_posiciones = CL.MAX_POSICIONES,
    )
    # Backtest: sin límites de riesgo
    risk  = RiskManager(config=RiskConfig.permissive(),
                        usdt_inicial=CL.SALDO_USDT_INICIAL)
    state = MemoryStateManager()

    strategy = IrrealStrategy(
        ventana       = VENTANA_LOCAL,
        precio_compra = PRECIO_COMPRA,
        precio_venta  = PRECIO_VENTA,
    )

    # ── 2. Iniciar estrategia ─────────────────────────────────────────────────
    strategy.on_start(wallet)

    # ── 3. Contadores ─────────────────────────────────────────────────────────
    n_compras    = 0
    n_ventas     = 0
    n_ignorados  = 0
    ign_motivos: dict[str, int] = {}
    precio_min_comprado  = float("inf")
    precio_max_vendido   = float("-inf")
    last_candle  = None

    # ── 4. Loop principal ─────────────────────────────────────────────────────
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
                ts=candle.ts, side=order_side.value, price=signal.price,
                ignored=True, ignore_reason=risk_reason,
            ))
            continue

        # Ejecutar con guardias del OrderBook
        order = ob.execute_with_guards(order_side, signal.price, wallet,
                                       candle_ts=candle.ts)

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

        risk.update_peak(wallet.portfolio_value(candle.close))
        state.save(Checkpoint.from_wallet(
            wallet, candle.ts, candle.close,
            metadata={"estrategia": strategy.name},
        ))

    print(f"OK  ({clock.total_candles:,} velas)")

    # ── 5. Métricas finales ───────────────────────────────────────────────────
    strategy.on_stop(wallet)

    if last_candle is None:
        print("✗ No se encontraron velas en el rango indicado.")
        return

    precio_final  = last_candle.close
    port_final    = wallet.portfolio_value(precio_final)
    pnl_pct       = (port_final / CL.SALDO_USDT_INICIAL - 1) * 100

    first_candles  = feed.get_candles(CL.FECHA_INICIO, CL.FECHA_INICIO)
    precio_inicial = first_candles[0].close if first_candles else last_candle.close
    bh_pnl         = (precio_final / precio_inicial - 1) * 100

    all_candles = feed.get_candles(CL.FECHA_INICIO, CL.FECHA_FIN)
    atl = min(c.low  for c in all_candles) if all_candles else 0
    ath = max(c.high for c in all_candles) if all_candles else 0

    # ── 6. Armar summary ──────────────────────────────────────────────────────
    summary = {
        "estrategia":               strategy.name,
        "fecha_inicio":             CL.FECHA_INICIO,
        "fecha_fin":                CL.FECHA_FIN,
        "saldo_inicial_usdt":       CL.SALDO_USDT_INICIAL,
        "usdt_balance_final":       round(wallet.get_usdt_balance(), 8),
        "btc_balance_final":        round(wallet.get_btc_balance(), 10),
        "btc_acumulado_total":      round(wallet.get_btc_acumulado(), 10),
        "btc_en_posiciones_final":  round(wallet.btc_en_posiciones(), 10),
        "precio_promedio_final":    round(wallet.precio_promedio_posiciones(), 8),
        "portfolio_value_final":    round(port_final, 4),
        "pnl_pct":                  round(pnl_pct, 4),
        "buy_hold_pnl_pct":         round(bh_pnl, 4),
        "alpha_vs_bh":              round(pnl_pct - bh_pnl, 4),
        "precio_min_comprado":      round(precio_min_comprado, 4) if n_compras else None,
        "precio_max_vendido":       round(precio_max_vendido, 4)  if n_ventas  else None,
        "atl_final":                round(atl, 4),
        "ath_proyectado_final":     round(ath, 4),
        "total_trades_ejecutados":  n_compras + n_ventas,
        "total_compras":            n_compras,
        "total_ventas":             n_ventas,
        "total_ignorados":          n_ignorados,
        "ordenes_canceladas":       0,
        "ignorados_por_motivo":     ign_motivos,
        "positions_count_final":    wallet.positions_count,
        "usdt_reserva_aplicada":    0.0,
        "umbral_filtro":            None,
        "parametros": {
            **strategy.describe(),
            "max_posiciones":    CL.MAX_POSICIONES,
            "commission_pct":    CL.COMMISSION_PCT,
            "slot_usdt_final":   round(wallet.get_slot_usdt(), 4),
            "guardia_compra":    True,
            "guardia_venta":     True,
            "rsi_length":        "N/A",
            "ath_caida_maxima":  "N/A",
            "atl_subida_maxima": "N/A",
            "factor_caida":      "N/A",
            "factor_subida":     "N/A",
            "N":                 "N/A",
        },
    }

    # ── 7. Guardar JSON ───────────────────────────────────────────────────────
    wallet.flush(summary)

    # ── 8. Resumen en consola ─────────────────────────────────────────────────
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
    print(f"  Tiempo           : {elapsed:.1f}s")
    print(sep)
    print(f"\n✓ Resultado guardado en: {CL.RESULTS_JSON}")


if __name__ == "__main__":
    main()