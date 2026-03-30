"""
backtest_mean_reversion.py — Runner de la Estrategia Mean Reversion
════════════════════════════════════════════════════════════════════
Ensambla todos los actores del sistema y corre MeanReversionStrategy:
una estrategia de reversión a la media basada en indicadores técnicos
(RSI, Bollinger, caída reciente, mechas de vela, cruce EMA).

No requiere entrenamiento ni cache — todos los cálculos son en línea.

Resultado: JSON compatible con Graficador.py

Uso:
    python backtest_mean_reversion.py           # run normal
    python backtest_mean_reversion.py --grid    # grid search de parámetros

Configuración del entorno:
    config_local.py  →  rutas, fechas, capital, comisión, MAX_POSICIONES

Parámetros de la estrategia (configurar en este archivo):
    Ver sección PARÁMETROS DE LA ESTRATEGIA más abajo.

────────────────────────────────────────────────────────────────────
GUÍA DE PARÁMETROS
────────────────────────────────────────────────────────────────────
El sistema de puntos permite calibrar la selectividad:

  puntos_buy_min = 4   → equilibrio señales/calidad (recomendado)
  puntos_buy_min = 3   → más operaciones, menor win rate
  puntos_buy_min = 5   → menos operaciones, mayor win rate

  stop_loss_pct = 0.035  → corta pérdidas en -3.5% por posición
  stop_loss_pct = 0.0    → sin stop-loss (igual al local_reversal original)

  rsi_buy_strong = 28    → más exigente (menos señales de sobreventa fuerte)
  rsi_buy_weak   = 35    → umbral de entrada moderado

Para grid search rápido usar --grid:
  Testa combinaciones de puntos_buy_min y stop_loss_pct automáticamente.
────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import config_local as CL

from actors.price_feed           import SQLiteFeed
from actors.wallet               import JSONWallet, MemoryWallet, TradeRecord
from actors.order_book           import SimulatedOrderBook, OrderSide
from actors.clock                import LocalClock
from risk.risk_manager           import RiskManager, RiskConfig
from state.state_manager         import MemoryStateManager, Checkpoint
from strategies.mean_reversion   import MeanReversionStrategy
from strategies.base_strategy    import SignalSide
from support.logger              import get_logger

log = get_logger("backtest_mean_reversion")


# ════════════════════════════════════════════════════════════════════
# PARÁMETROS DE LA ESTRATEGIA
# ════════════════════════════════════════════════════════════════════

# Indicadores
RSI_PERIOD       = 14
BB_PERIOD        = 20
BB_STD           = 2.0
EMA_FAST         = 3
EMA_SLOW         = 8
DROP_WINDOW      = 24     # velas para calcular caída/subida reciente

# Umbrales RSI
RSI_BUY_STRONG   = 25.0   # sobreventa fuerte  → 3 puntos
RSI_BUY_WEAK     = 35.0   # sobreventa moderada → 2 puntos
RSI_SELL_STRONG  = 75.0   # sobrecompra fuerte  → 3 puntos
RSI_SELL_WEAK    = 70.0   # sobrecompra moderada → 2 puntos

# Umbrales de caída/subida reciente
DROP_STRONG      = -0.050  # -3.0% → 2 puntos (cubre 64% de bottoms del irreal)
DROP_WEAK        = -0.018  # -1.8% → 1 punto  (cubre 82% de bottoms del irreal)
RISE_STRONG      =  0.050  # +3.0% → 2 puntos
RISE_WEAK        =  0.018  # +1.8% → 1 punto

# Umbrales de vela
LOW_REJ_STRONG   = 0.60    # mecha inferior fuerte → 2 puntos
LOW_REJ_WEAK     = 0.40    # mecha inferior moderada → 1 punto
HIGH_REJ_STRONG  = 0.60    # mecha superior fuerte → 2 puntos
HIGH_REJ_WEAK    = 0.40    # mecha superior moderada → 1 punto

# Puntaje mínimo para operar
PUNTOS_BUY_MIN   = 8.5       # recomendado: 4 (rango razonable: 3-5)
PUNTOS_SELL_MIN  = 7.5       # recomendado: 4 (rango razonable: 3-5)

# Gestión de riesgo
STOP_LOSS_PCT    = 0.0   # 3.5% stop-loss por posición (0.0 = desactivado)
TAKE_PROFIT_PCT  = 0.2   # +2.5% añade 1 punto extra a la señal de venta

# Warmup
WARMUP_VELAS     = 50      # velas antes de comenzar a operar


# ════════════════════════════════════════════════════════════════════
# RUNNER
# ════════════════════════════════════════════════════════════════════

def run_backtest(
    puntos_buy_min:  int   = PUNTOS_BUY_MIN,
    puntos_sell_min: int   = PUNTOS_SELL_MIN,
    stop_loss_pct:   float = STOP_LOSS_PCT,
    results_json:    str   = None,
    silent:          bool  = False,
) -> dict:
    """
    Ejecuta un backtest completo y retorna el dict de summary.
    Separado en función para poder ser llamado desde el grid search.
    """
    t_start = time.time()
    json_path = results_json or CL.RESULTS_JSON

    if not silent:
        print("╔══════════════════════════════════════════════════════════╗")
        print("║      BACKTEST MEAN REVERSION — Técnico BTC/USDT         ║")
        print("╚══════════════════════════════════════════════════════════╝")
        print(f"  Rango         : {CL.FECHA_INICIO} → {CL.FECHA_FIN}")
        print(f"  Capital       : ${CL.SALDO_USDT_INICIAL:,.2f} USDT")
        print(f"  Max posiciones: {CL.MAX_POSICIONES}")
        print(f"  Comisión      : {CL.COMMISSION_PCT}%")
        print(f"  Puntos BUY/SELL: {puntos_buy_min}/{puntos_sell_min}")
        print(f"  Stop-loss     : {stop_loss_pct*100:.1f}%")
        print(f"  Output JSON   : {json_path}")
        print("─" * 60)

    # ── 1. Construir actores ───────────────────────────────────────
    feed   = SQLiteFeed(db_path=CL.DB_PATH, table=CL.DB_TABLE)
    wallet = JSONWallet(
        usdt_inicial   = CL.SALDO_USDT_INICIAL,
        max_posiciones = CL.MAX_POSICIONES,
        json_path      = json_path,
    )
    ob     = SimulatedOrderBook(
        commission_pct = CL.COMMISSION_PCT,
        max_posiciones = CL.MAX_POSICIONES,
    )
    risk   = RiskManager(config=RiskConfig.permissive(),
                         usdt_inicial=CL.SALDO_USDT_INICIAL)
    state  = MemoryStateManager()

    strategy = MeanReversionStrategy(
        rsi_period      = RSI_PERIOD,
        bb_period       = BB_PERIOD,
        bb_std          = BB_STD,
        ema_fast        = EMA_FAST,
        ema_slow        = EMA_SLOW,
        drop_window     = DROP_WINDOW,
        rsi_buy_strong  = RSI_BUY_STRONG,
        rsi_buy_weak    = RSI_BUY_WEAK,
        rsi_sell_strong = RSI_SELL_STRONG,
        rsi_sell_weak   = RSI_SELL_WEAK,
        drop_strong     = DROP_STRONG,
        drop_weak       = DROP_WEAK,
        rise_strong     = RISE_STRONG,
        rise_weak       = RISE_WEAK,
        low_rej_strong  = LOW_REJ_STRONG,
        low_rej_weak    = LOW_REJ_WEAK,
        high_rej_strong = HIGH_REJ_STRONG,
        high_rej_weak   = HIGH_REJ_WEAK,
        puntos_buy_min  = puntos_buy_min,
        puntos_sell_min = puntos_sell_min,
        stop_loss_pct   = stop_loss_pct,
        take_profit_pct = TAKE_PROFIT_PCT,
        warmup_velas    = WARMUP_VELAS,
    )

    # ── 2. Iniciar estrategia ──────────────────────────────────────
    strategy.on_start(wallet)

    # ── 3. Clock ──────────────────────────────────────────────────
    clock = LocalClock(
        feed   = feed,
        start  = CL.FECHA_INICIO,
        end    = CL.FECHA_FIN,
        symbol = CL.SYMBOL,
    )

    # ── 4. Contadores ─────────────────────────────────────────────
    n_compras   = 0
    n_ventas    = 0
    n_ignorados = 0
    n_stoploss  = 0
    ign_motivos: dict[str, int] = {}
    precio_min_comprado = float("inf")
    precio_max_vendido  = float("-inf")
    last_candle = None

    # ── 5. Loop principal ─────────────────────────────────────────
    if not silent:
        print("Procesando velas...", end=" ", flush=True)

    for candle in clock:
        last_candle = candle
        signal      = strategy._tick(candle, wallet)

        if not signal.is_actionable:
            continue

        order_side = signal.to_order_side()

        risk_reason = risk.check(order_side, signal.price, wallet, candle)
        if risk_reason:
            n_ignorados += 1
            ign_motivos[risk_reason] = ign_motivos.get(risk_reason, 0) + 1
            wallet.update(TradeRecord(
                ts=candle.ts, side=order_side.value, price=signal.price,
                ignored=True, ignore_reason=risk_reason,
            ))
            continue

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
                if signal.score == 0.0:   # stop-loss
                    n_stoploss += 1
        else:
            n_ignorados += 1
            motivo = order.reject_reason or "desconocido"
            ign_motivos[motivo] = ign_motivos.get(motivo, 0) + 1

        # Enriquecer trade con scores
        if wallet.get_trade_log():
            last_trade = wallet.get_trade_log()[-1]
            last_trade["score_bot"] = signal.score if signal.side == SignalSide.BUY  else 0.0
            last_trade["score_top"] = signal.score if signal.side == SignalSide.SELL else 0.0

        risk.update_peak(wallet.portfolio_value(candle.close))
        state.save(Checkpoint.from_wallet(
            wallet, candle.ts, candle.close,
            metadata={"estrategia": strategy.name},
        ))

    if not silent:
        print(f"OK  ({clock.total_candles:,} velas)")

    # ── 6. Métricas finales ───────────────────────────────────────
    strategy.on_stop(wallet)

    if last_candle is None:
        print("✗ No se encontraron velas en el rango indicado.")
        return {}

    precio_final = last_candle.close
    port_final   = wallet.portfolio_value(precio_final)
    pnl_pct      = (port_final / CL.SALDO_USDT_INICIAL - 1) * 100

    first_c      = feed.get_candles(CL.FECHA_INICIO, CL.FECHA_INICIO)
    precio_i     = first_c[0].close if first_c else precio_final
    bh_pnl       = (precio_final / precio_i - 1) * 100

    all_range = feed.get_candles(CL.FECHA_INICIO, CL.FECHA_FIN)
    atl = min(c.low  for c in all_range) if all_range else 0
    ath = max(c.high for c in all_range) if all_range else 0

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
        "total_stoploss"          : n_stoploss,
        "total_ignorados"         : n_ignorados,
        "ordenes_canceladas"      : 0,
        "ignorados_por_motivo"    : ign_motivos,
        "positions_count_final"   : wallet.positions_count,
        "usdt_reserva_aplicada"   : 0.0,
        "umbral_filtro"           : puntos_buy_min,
        "parametros"              : strategy.describe(),
    }

    wallet.flush(summary)

    elapsed = time.time() - t_start
    sign = "+" if pnl_pct >= 0 else ""

    if not silent:
        sep = "═" * 60
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
        print(f"  Ventas           : {n_ventas:,}  (stop-loss: {n_stoploss})")
        print(f"  Ignorados        : {n_ignorados:,}")
        print(f"  Posiciones abier.: {wallet.positions_count}")
        print(f"  Tiempo total     : {elapsed:.1f}s")
        print(sep)
        print(f"\n✓ Resultado guardado en: {json_path}")

    return summary


# ════════════════════════════════════════════════════════════════════
# GRID SEARCH
# ════════════════════════════════════════════════════════════════════

def run_grid_search() -> None:
    """
    Grid search sobre puntos_buy_min, puntos_sell_min y stop_loss_pct.
    Usa MemoryWallet para velocidad (sin escritura JSON por run).
    """
    print("╔══════════════════════════════════════════════════════════╗")
    print("║         GRID SEARCH — MEAN REVERSION                    ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"  Período: {CL.FECHA_INICIO} → {CL.FECHA_FIN}")
    print("─" * 60)

    import itertools

    grid = {
        "puntos_buy_min":  [3, 4, 5, 6, 7, 8, 9],
        "puntos_sell_min": [3, 4, 5, 6, 7, 8, 9],
        "stop_loss_pct":   [0.0],
    }

    combinaciones = list(itertools.product(*grid.values()))
    print(f"  {len(combinaciones)} combinaciones a testear\n")

    resultados = []
    for i, (pmin_b, pmin_s, sl) in enumerate(combinaciones, 1):
        res = run_backtest(
            puntos_buy_min  = pmin_b,
            puntos_sell_min = pmin_s,
            stop_loss_pct   = sl,
            results_json    = "/tmp/mr_grid_tmp.json",
            silent          = True,
        )
        if res:
            resultados.append({
                "buy_min": pmin_b, "sell_min": pmin_s, "sl": sl,
                "pnl":     res["pnl_pct"],
                "bh":      res["buy_hold_pnl_pct"],
                "alpha":   res["alpha_vs_bh"],
                "compras": res["total_compras"],
                "ventas":  res["total_ventas"],
            })
            if i % 10 == 0:
                print(f"  Progreso: {i}/{len(combinaciones)}", flush=True)

    resultados.sort(key=lambda r: r["pnl"], reverse=True)

    print(f"\n{'─'*70}")
    print(f"  {'B_min':>5}  {'S_min':>5}  {'SL%':>5}  {'PnL%':>8}  {'BH%':>8}  "
          f"{'Alpha':>8}  {'B':>5}  {'V':>5}")
    print(f"{'─'*70}")
    for r in resultados[:15]:
        sl_str = f"{r['sl']*100:.1f}%" if r['sl'] > 0 else "  —  "
        print(f"  {r['buy_min']:>5}  {r['sell_min']:>5}  {sl_str:>5}  "
              f"{r['pnl']:>+8.2f}%  {r['bh']:>+8.2f}%  {r['alpha']:>+8.2f}%  "
              f"{r['compras']:>5}  {r['ventas']:>5}")
    print(f"{'─'*70}")
    print(f"\nTop 1: buy_min={resultados[0]['buy_min']}  "
          f"sell_min={resultados[0]['sell_min']}  "
          f"sl={resultados[0]['sl']*100:.1f}%  "
          f"→ PnL={resultados[0]['pnl']:+.2f}%  "
          f"alpha={resultados[0]['alpha']:+.2f}%")


# ════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Backtest Mean Reversion — Indicadores técnicos BTC/USDT"
    )
    parser.add_argument(
        "--grid",
        action="store_true",
        help="Ejecutar grid search de parámetros",
    )
    args = parser.parse_args()

    if args.grid:
        run_grid_search()
    else:
        run_backtest()
