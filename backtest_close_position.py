"""
backtest_close_position.py — Runner de ClosePositionStrategy
══════════════════════════════════════════════════════════════
Corre ClosePositionStrategy (factor único close_position, ventana=6)
y produce un JSON compatible con Graficador.py.

Uso:
    python backtest_close_position.py

Modo grid (barre umbrales y cooldowns sin re-correr el backtest):
    python backtest_close_position.py --grid
    python backtest_close_position.py --grid --json otro_archivo.json

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GUÍA DE PARÁMETROS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
VENTANA       velas hacia atrás para el rango high-low       (optimal=6)
UMBRAL_BOT    close_position ≤ UMBRAL_BOT  → BUY             (0, 0.5)
UMBRAL_TOP    close_position ≥ UMBRAL_TOP  → SELL            (0.5, 1.0)
COOLDOWN      velas mínimas entre señales del mismo tipo      0=off

Umbrales de referencia (factors_analysis.json, ventana=6, 2021-2025):
  Bottoms reales: media=0.226  mediana=0.177  → umbral_bot ≈ 0.15-0.25
  Tops reales:    media=0.800  mediana=0.848  → umbral_top ≈ 0.75-0.85
  Neutros:        media=0.520  (distribución casi uniforme)

Cobertura estimada por umbral (regla empírica):
  umbral_bot=0.177 (mediana) → captura ~50% de los bottoms reales
  umbral_bot=0.226 (media)   → captura ~55% de los bottoms reales
  umbral_top=0.848 (mediana) → captura ~50% de los tops reales
  umbral_top=0.800 (media)   → captura ~55% de los tops reales
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import config_local as CL

from actors.price_feed    import SQLiteFeed
from actors.wallet        import JSONWallet, MemoryWallet, TradeRecord
from actors.order_book    import SimulatedOrderBook, OrderSide
from actors.clock         import LocalClock
from risk.risk_manager    import RiskManager, RiskConfig
from state.state_manager  import MemoryStateManager, Checkpoint
from strategies.close_position_strategy import ClosePositionStrategy
from strategies.base_strategy           import SignalSide
from support.logger import get_logger

log = get_logger("backtest_close_position")


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG — Editar aquí antes de cada ejecución
# ══════════════════════════════════════════════════════════════════════════════

VENTANA    = 6       # óptimo por AUC (no cambiar sin análisis previo)
UMBRAL_BOT = 0.15    # punto de partida calibrado (mediana bottoms ≈ 0.177)
UMBRAL_TOP = 0.90    # punto de partida calibrado (media tops ≈ 0.800)
COOLDOWN   = 48      # velas entre señales del mismo tipo (0=desactivado)

RESULTS_JSON = CL.RESULTS_JSON

# ── Parámetros del grid ───────────────────────────────────────────────────────
GRID_UMBRAL_BOT = [0.01, 0.10, 0.15, 0.20, 0.25, 0.30]
GRID_UMBRAL_TOP = [0.70, 0.75, 0.80, 0.85, 0.90, 0.99]
GRID_COOLDOWNS  = [0, 24, 48, 72, 96, 144]
GRID_CSV        = "grid_close_position_results.csv"
GRID_JSON_OUT   = "grid_close_position_results.json"


# ══════════════════════════════════════════════════════════════════════════════
# HELPER: enriquecer trade_log con el valor del factor
# ══════════════════════════════════════════════════════════════════════════════

def _enrich(wallet: JSONWallet, strategy: ClosePositionStrategy) -> None:
    """Añade close_position al último trade del log para el modo --grid."""
    entries = wallet.get_trade_log()
    if not entries:
        return
    entries[-1]["close_position"] = strategy.last_close_position
    entries[-1]["score_bot"] = (
        round(1.0 - strategy.last_close_position, 4)
        if strategy.last_close_position is not None else None
    )
    entries[-1]["score_top"] = strategy.last_close_position


# ══════════════════════════════════════════════════════════════════════════════
# RUNNER PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    t_start = time.time()

    print("╔══════════════════════════════════════════════════════════╗")
    print("║     BACKTEST CLOSE POSITION — Factor Único BTC/USDT     ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"  Rango          : {CL.FECHA_INICIO} → {CL.FECHA_FIN}")
    print(f"  Capital        : ${CL.SALDO_USDT_INICIAL:,.2f} USDT")
    print(f"  Max posiciones : {CL.MAX_POSICIONES}")
    print(f"  Comisión       : {CL.COMMISSION_PCT}%")
    print(f"  Ventana        : {VENTANA} velas  (AUC ref: BOT=0.8247 TOP=0.8223)")
    print(f"  Umbral BUY     : close_position ≤ {UMBRAL_BOT}")
    print(f"  Umbral SELL    : close_position ≥ {UMBRAL_TOP}")
    print(f"  Cooldown       : {COOLDOWN} velas" if COOLDOWN else
          f"  Cooldown       : desactivado")
    print(f"  Output JSON    : {RESULTS_JSON}")
    print("─" * 60)

    feed   = SQLiteFeed(db_path=CL.DB_PATH, table=CL.DB_TABLE)
    clock  = LocalClock(feed=feed, start=CL.FECHA_INICIO, end=CL.FECHA_FIN,
                        symbol=CL.SYMBOL)
    wallet = JSONWallet(
        usdt_inicial   = CL.SALDO_USDT_INICIAL,
        max_posiciones = CL.MAX_POSICIONES,
        json_path      = RESULTS_JSON,
    )
    ob    = SimulatedOrderBook(
        commission_pct = CL.COMMISSION_PCT,
        max_posiciones = CL.MAX_POSICIONES,
    )
    risk  = RiskManager(config=RiskConfig.permissive(),
                        usdt_inicial=CL.SALDO_USDT_INICIAL)
    state = MemoryStateManager()

    strategy = ClosePositionStrategy(
        ventana    = VENTANA,
        umbral_bot = UMBRAL_BOT,
        umbral_top = UMBRAL_TOP,
        cooldown   = COOLDOWN,
    )
    strategy.on_start(wallet)

    n_compras = n_ventas = n_ignorados = 0
    ign_motivos: dict[str, int] = {}
    precio_min_comprado = float("inf")
    precio_max_vendido  = float("-inf")
    last_candle         = None

    print("Procesando velas...", end=" ", flush=True)

    for candle in clock:
        last_candle = candle
        signal      = strategy._tick(candle, wallet)

        if not signal.is_actionable:
            continue

        order_side  = signal.to_order_side()
        risk_reason = risk.check(order_side, signal.price, wallet, candle)

        if risk_reason:
            n_ignorados += 1
            ign_motivos[risk_reason] = ign_motivos.get(risk_reason, 0) + 1
            wallet.update(TradeRecord(
                ts=candle.ts, side=order_side.value, price=signal.price,
                ignored=True, ignore_reason=risk_reason,
            ))
            _enrich(wallet, strategy)
            continue

        order = ob.execute_with_guards(
            order_side, signal.price, wallet, candle_ts=candle.ts
        )
        _enrich(wallet, strategy)

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

    strategy.on_stop(wallet)

    if last_candle is None:
        print("✗ No se encontraron velas en el rango indicado.")
        return

    precio_final   = last_candle.close
    port_final     = wallet.portfolio_value(precio_final)
    pnl_pct        = (port_final / CL.SALDO_USDT_INICIAL - 1) * 100
    first_candles  = feed.get_candles(CL.FECHA_INICIO, CL.FECHA_INICIO)
    precio_inicial = first_candles[0].close if first_candles else precio_final
    bh_pnl         = (precio_final / precio_inicial - 1) * 100
    all_candles    = feed.get_candles(CL.FECHA_INICIO, CL.FECHA_FIN)
    atl = min(c.low  for c in all_candles) if all_candles else 0
    ath = max(c.high for c in all_candles) if all_candles else 0

    desc = strategy.describe()
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
        "precio_max_vendido":       round(precio_max_vendido,  4) if n_ventas  else None,
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
        "umbral_filtro":            UMBRAL_BOT,
        "parametros": {
            **desc,
            "max_posiciones":  CL.MAX_POSICIONES,
            "commission_pct":  CL.COMMISSION_PCT,
            "slot_usdt_final": round(wallet.get_slot_usdt(), 4),
        },
    }

    wallet.flush(summary)

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
    print(f"\n✓ Resultado guardado en: {RESULTS_JSON}")
    print(f"\n  Flujo recomendado para calibrar umbrales:")
    print(f"  1. python backtest_close_position.py        (corre con defaults)")
    print(f"  2. python backtest_close_position.py --grid (barre umbrales y cooldowns)")
    print(f"  3. Ajustar UMBRAL_BOT, UMBRAL_TOP, COOLDOWN con el óptimo del grid")


# ══════════════════════════════════════════════════════════════════════════════
# MODO --grid
# ══════════════════════════════════════════════════════════════════════════════

def _simulate_combo(
    trades:     list,
    thr_bot:    float,
    thr_top:    float,
    cooldown:   int,
    usdt_ini:   float,
    max_pos:    int,
    commission: float,
    bh_pnl:     float,
) -> dict:
    """
    Re-simula la wallet con umbrales y cooldown distintos usando los
    close_position guardados en el trade_history del backtest inicial.

    Solo requiere que el JSON haya sido generado con umbral muy bajo
    (ej. 0.40/0.60) para capturar el rango completo del factor.
    """
    wallet = MemoryWallet(usdt_ini, max_pos)
    n_buy = n_sell = n_ign = 0
    ganancias: list[float] = []

    _NEG_INF     = -(10 ** 9)
    last_bot_idx = _NEG_INF
    last_top_idx = _NEG_INF

    for idx, t in enumerate(trades):
        price = t.get("price", 0.0)
        ts    = t.get("ts", 0)
        cp    = t.get("close_position")
        if price <= 0 or cp is None:
            continue

        cd_ok_bot = (cooldown == 0 or (idx - last_bot_idx) >= cooldown)
        cd_ok_top = (cooldown == 0 or (idx - last_top_idx) >= cooldown)

        # SELL prioridad
        if cp >= thr_top and cd_ok_top:
            last_top_idx = idx
            if wallet.positions_count == 0:
                n_ign += 1
            else:
                bpv = wallet.get_btc_por_venta()
                if bpv <= 0:
                    n_ign += 1
                else:
                    usdt_bruto = bpv * price
                    comm       = usdt_bruto * commission / 100.0
                    usdt_neto  = usdt_bruto - comm
                    ganancia   = usdt_neto - wallet.get_slot_usdt()
                    wallet.update(TradeRecord(
                        ts=ts, side="SELL", price=price,
                        btc_sold=bpv, usdt_received=usdt_neto,
                        commission=comm, ganancia_usdt=ganancia,
                    ))
                    ganancias.append(ganancia)
                    n_sell += 1
            continue

        if cp <= thr_bot and cd_ok_bot:
            last_bot_idx = idx
            if wallet.positions_count >= max_pos:
                n_ign += 1
            else:
                slot = wallet.get_slot_usdt()
                if slot > wallet.get_usdt_balance() + 1e-9:
                    n_ign += 1
                else:
                    comm = slot * commission / 100.0
                    btc  = (slot - comm) / price
                    wallet.update(TradeRecord(
                        ts=ts, side="BUY", price=price,
                        usdt_spent=slot, btc_bought=btc, commission=comm,
                    ))
                    n_buy += 1

    last_price = trades[-1].get("price", 0.0) if trades else 0.0
    port  = wallet.portfolio_value(last_price)
    pnl   = (port / usdt_ini - 1) * 100.0
    wr    = (sum(1 for g in ganancias if g > 0) / len(ganancias) * 100.0
             if ganancias else 0.0)

    return {
        "umbral_bot": thr_bot,
        "umbral_top": thr_top,
        "cooldown":   cooldown,
        "pnl_pct":    round(pnl, 2),
        "portfolio":  round(port, 2),
        "alpha_bh":   round(pnl - bh_pnl, 2),
        "n_buy":      n_buy,
        "n_sell":     n_sell,
        "n_trades":   n_buy + n_sell,
        "n_ign":      n_ign,
        "win_rate":   round(wr, 1),
    }


def grid_analysis(json_path: str) -> None:
    import csv as csv_mod

    print("╔══════════════════════════════════════════════════════════╗")
    print("║    GRID — close_position: umbrales × cooldowns          ║")
    print("╚══════════════════════════════════════════════════════════╝")

    json_file = Path(json_path)
    if not json_file.exists():
        print(f"\n✗ No se encontró: {json_path}")
        print("  Ejecutar primero: python backtest_close_position.py")
        return

    with open(json_file, encoding="utf-8") as f:
        data = json.load(f)

    trades_raw = data.get("trade_history", [])
    summary_in = data.get("summary", {})
    orig_pnl   = summary_in.get("pnl_pct", 0.0)
    bh_pnl     = summary_in.get("buy_hold_pnl_pct", 0.0)

    # Filtrar trades con close_position disponible
    trades = [t for t in trades_raw
              if t.get("type") in ("BUY", "SELL") and t.get("close_position") is not None]
    missing = len([t for t in trades_raw if t.get("type") in ("BUY", "SELL")]) - len(trades)

    print(f"  JSON           : {json_path}")
    print(f"  Período        : {summary_in.get('fecha_inicio')} → {summary_in.get('fecha_fin')}")
    print(f"  PnL original   : {orig_pnl:+.2f}%   B&H: {bh_pnl:+.2f}%")
    print(f"  Trades con cp  : {len(trades):,}  (sin cp: {missing})")

    if not trades:
        print("\n✗ No hay trades con close_position. Ejecutar backtest con umbrales amplios primero.")
        print("  Sugerencia: UMBRAL_BOT=0.40, UMBRAL_TOP=0.60, COOLDOWN=0")
        return

    usdt_ini   = summary_in.get("saldo_inicial_usdt", CL.SALDO_USDT_INICIAL)
    max_pos    = (summary_in.get("parametros", {}).get("max_posiciones", CL.MAX_POSICIONES))
    commission = CL.COMMISSION_PCT

    total = len(GRID_UMBRAL_BOT) * len(GRID_UMBRAL_TOP) * len(GRID_COOLDOWNS)
    print(f"\n  Barriendo {len(GRID_UMBRAL_BOT)} umbrales_bot × "
          f"{len(GRID_UMBRAL_TOP)} umbrales_top × "
          f"{len(GRID_COOLDOWNS)} cooldowns = {total} combinaciones")
    print(f"  Umbrales BOT : {GRID_UMBRAL_BOT}")
    print(f"  Umbrales TOP : {GRID_UMBRAL_TOP}")
    print(f"  Cooldowns    : {GRID_COOLDOWNS} (velas; 0=off)")

    t0      = time.time()
    results = []

    for thr_b in GRID_UMBRAL_BOT:
        for thr_t in GRID_UMBRAL_TOP:
            if thr_t - thr_b < 0.30:
                continue   # zona muerta < 30% → no tiene sentido
            for cd in GRID_COOLDOWNS:
                r = _simulate_combo(
                    trades, thr_b, thr_t, cd,
                    usdt_ini, max_pos, commission, bh_pnl
                )
                r["alpha_orig"] = round(r["pnl_pct"] - orig_pnl, 2)
                results.append(r)

    results.sort(key=lambda x: -x["pnl_pct"])
    elapsed = time.time() - t0

    # ── Tabla de resultados ───────────────────────────────────────────────────
    sep  = "─" * 86
    sep2 = "═" * 86
    print(f"\n{sep2}")
    print(f"  TOP-{min(20, len(results))} COMBINACIONES — ordenadas por PnL")
    print(sep2)
    print(f"  {'#':>2}  {'thr_bot':>7} {'thr_top':>7} {'cd':>5}  "
          f"{'PnL%':>8} {'α_BH':>8} {'α_orig':>8}  {'B/S':>9}  {'WR%':>5}")
    print(sep)
    for i, r in enumerate(results[:20], 1):
        pnl_s  = f"{'+' if r['pnl_pct']>=0 else ''}{r['pnl_pct']:.2f}%"
        abh_s  = f"{'+' if r['alpha_bh']>=0 else ''}{r['alpha_bh']:.2f}%"
        aor_s  = f"{'+' if r['alpha_orig']>=0 else ''}{r['alpha_orig']:.2f}%"
        cd_s   = str(r["cooldown"]) if r["cooldown"] else "off"
        bs_s   = f"{r['n_buy']}B/{r['n_sell']}S"
        print(f"  {i:>2}.  {r['umbral_bot']:>7.2f} {r['umbral_top']:>7.2f} {cd_s:>5}  "
              f"{pnl_s:>9} {abh_s:>9} {aor_s:>9}  {bs_s:>9}  {r['win_rate']:>5.1f}%")
    print(sep)
    print(f"  B&H: {bh_pnl:+.2f}%   Original: {orig_pnl:+.2f}%   "
          f"Tiempo: {elapsed:.1f}s")

    # Guardar CSV
    if results:
        fieldnames = list(results[0].keys())
        with open(GRID_CSV, "w", newline="", encoding="utf-8") as f:
            w = csv_mod.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(results)

    # Guardar JSON
    output = {
        "meta": {
            "fecha_inicio":  summary_in.get("fecha_inicio"),
            "fecha_fin":     summary_in.get("fecha_fin"),
            "pnl_original":  orig_pnl,
            "bh":            bh_pnl,
            "grid_bot":      GRID_UMBRAL_BOT,
            "grid_top":      GRID_UMBRAL_TOP,
            "grid_cooldown": GRID_COOLDOWNS,
            "n_combos":      len(results),
            "elapsed_s":     round(elapsed, 2),
        },
        "top_20": results[:20],
        "all":    results,
    }
    with open(GRID_JSON_OUT, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Grid completado  →  {GRID_CSV}  |  {GRID_JSON_OUT}")
    print(f"\n  Siguiente paso: ajustar UMBRAL_BOT/TOP/COOLDOWN con el top-1 del grid")
    print(f"  y re-correr: python backtest_close_position.py")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Backtest ClosePositionStrategy — factor único close_position",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Flujo de calibración recomendado:
  1. UMBRAL_BOT=0.40, UMBRAL_TOP=0.60, COOLDOWN=0  (captura el rango completo)
     python backtest_close_position.py
  2. python backtest_close_position.py --grid       (barre combinaciones)
  3. Ajustar UMBRAL_BOT, UMBRAL_TOP, COOLDOWN con el top-1
  4. python backtest_close_position.py              (backtest definitivo)
""",
    )
    parser.add_argument(
        "--grid", action="store_true",
        help="Barre umbrales y cooldowns sobre el trade_history existente",
    )
    parser.add_argument(
        "--json", default=RESULTS_JSON, metavar="PATH",
        help=f"JSON para el modo --grid (default: {RESULTS_JSON})",
    )
    args = parser.parse_args()

    if args.grid:
        grid_analysis(args.json)
    else:
        main()
