"""
order_book.py — Actor 3: Libro de órdenes
══════════════════════════════════════════
Responsabilidad única: abrir y cerrar posiciones, calcular comisiones,
retornar TradeRecord para que la Wallet actualice su estado.

SEPARACIÓN INTENCIÓN / EJECUCIÓN
─────────────────────────────────
  1. create_order(side, ...)  →  Order   (intención)
  2. submit(order)            →  Order   (envío / ejecución)
  3. check(order_id)          →  Order   (confirmación)

En simulación los tres pasos colapsan en uno (ejecución instantánea).
En producción con Binance son llamadas REST separadas con posible latencia.

Parámetro candle_ts
────────────────────
Todos los métodos de ejecución reciben candle_ts (epoch s de la vela que
generó la señal). Se aplica ANTES de submit() para que el TradeRecord
tenga la fecha real de la vela, no el timestamp de ejecución del proceso.

Implementaciones
─────────────────
  SimulatedOrderBook  →  ejecución instantánea, lógica de slots del Irreal
  BinanceOrderBook    →  órdenes reales (actors/binance_order_book.py)

Factory
────────
  build_order_book(mode, ...)  →  OrderBook
    mode="local"  →  SimulatedOrderBook  (default, backtest)
    mode="live"   →  BinanceOrderBook    (producción/testnet)

  No lee mode_config — el runner decide el modo explícitamente.
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from actors.wallet      import Wallet, TradeRecord
from support.logger     import get_logger
from support.time_utils import now_epoch_s

log = get_logger("order_book")


# ══════════════════════════════════════════════════════════════════════════════
# TIPOS
# ══════════════════════════════════════════════════════════════════════════════

class OrderSide(str, Enum):
    BUY  = "BUY"
    SELL = "SELL"


class OrderStatus(str, Enum):
    PENDING   = "PENDING"
    SUBMITTED = "SUBMITTED"
    FILLED    = "FILLED"
    REJECTED  = "REJECTED"
    IGNORED   = "IGNORED"


@dataclass
class Order:
    """Ciclo de vida de una orden: PENDING → SUBMITTED → FILLED | REJECTED | IGNORED."""
    order_id:      str
    side:          OrderSide
    price:         float
    ts:            int                   # epoch s UTC — sobreescrito con candle_ts

    usdt_amount:   Optional[float]       = None
    btc_amount:    Optional[float]       = None
    status:        OrderStatus           = OrderStatus.PENDING
    reject_reason: Optional[str]         = None
    trade:         Optional[TradeRecord] = None

    @property
    def is_filled(self)   -> bool: return self.status == OrderStatus.FILLED
    @property
    def is_rejected(self) -> bool: return self.status == OrderStatus.REJECTED
    @property
    def is_ignored(self)  -> bool: return self.status == OrderStatus.IGNORED


# ══════════════════════════════════════════════════════════════════════════════
# INTERFAZ ABSTRACTA
# ══════════════════════════════════════════════════════════════════════════════

class OrderBook(ABC):

    @abstractmethod
    def create_order(
        self,
        side:        OrderSide,
        price:       float,
        usdt_amount: Optional[float] = None,
        btc_amount:  Optional[float] = None,
    ) -> Order: ...

    @abstractmethod
    def submit(self, order: Order) -> Order: ...

    @abstractmethod
    def check(self, order_id: str) -> Order: ...

    def execute(
        self,
        side:      OrderSide,
        price:     float,
        wallet:    Wallet,
        candle_ts: int = 0,
    ) -> Order:
        """
        Encadena create → (fijar ts) → submit → check → notificar wallet.
        candle_ts se fija ANTES de submit para que el TradeRecord tenga
        la fecha real de la vela.
        """
        if side == OrderSide.BUY:
            order = self.create_order(side, price,
                                      usdt_amount=wallet.get_slot_usdt())
        else:
            order = self.create_order(side, price,
                                      btc_amount=wallet.get_btc_por_venta())

        if candle_ts:
            order.ts = candle_ts

        order = self.submit(order)
        order = self.check(order.order_id)

        if order.trade:
            wallet.update(order.trade)

        return order


# ══════════════════════════════════════════════════════════════════════════════
# IMPLEMENTACIÓN LOCAL: SimulatedOrderBook
# ══════════════════════════════════════════════════════════════════════════════

class SimulatedOrderBook(OrderBook):
    """
    Ejecución instantánea al precio dado.
    Lógica de slots idéntica al Backtest_irreal.py (benchmark canónico).
    """

    def __init__(self, commission_pct: float, max_posiciones: int) -> None:
        self._commission_pct = commission_pct
        self._max_posiciones = max_posiciones
        self._orders: dict[str, Order] = {}
        log.info("SimulatedOrderBook inicializado",
                 commission=f"{commission_pct}%", max_pos=max_posiciones)

    # ── Interfaz ──────────────────────────────────────────────────────────────

    def create_order(
        self,
        side:        OrderSide,
        price:       float,
        usdt_amount: Optional[float] = None,
        btc_amount:  Optional[float] = None,
    ) -> Order:
        order = Order(
            order_id    = str(uuid.uuid4())[:8],
            side        = side,
            price       = price,
            ts          = now_epoch_s(),
            usdt_amount = usdt_amount,
            btc_amount  = btc_amount,
        )
        self._orders[order.order_id] = order
        return order

    def submit(self, order: Order) -> Order:
        if order.side == OrderSide.BUY:
            self._execute_buy(order)
        else:
            self._execute_sell(order)
        if order.trade:
            order.trade.ts = order.ts
        return order

    def check(self, order_id: str) -> Order:
        return self._orders.get(order_id, Order(
            order_id="unknown", side=OrderSide.BUY, price=0, ts=0,
            status=OrderStatus.REJECTED,
            reject_reason="order_id no encontrado",
        ))

    # ── Ejecución interna ─────────────────────────────────────────────────────

    def _execute_buy(self, order: Order) -> None:
        usdt_a_gastar = order.usdt_amount or 0.0
        if usdt_a_gastar < 1.0:
            self._reject(order, "usdt_insuficiente(slot<1)")
            return
        commission   = round(usdt_a_gastar * self._commission_pct / 100.0, 8)
        btc_comprado = round((usdt_a_gastar - commission) / order.price, 10)
        order.status = OrderStatus.FILLED
        order.trade  = TradeRecord(
            ts         = order.ts,
            side       = "BUY",
            price      = order.price,
            usdt_spent = round(usdt_a_gastar, 8),
            btc_bought = btc_comprado,
            commission = commission,
        )
        log.debug("BUY ejecutado", price=order.price,
                  usdt=usdt_a_gastar, btc=btc_comprado)

    def _execute_sell(self, order: Order) -> None:
        btc_a_vender = order.btc_amount or 0.0
        if btc_a_vender <= 0:
            self._reject(order, "sin_btc")
            return
        usdt_bruto   = round(btc_a_vender * order.price, 8)
        commission   = round(usdt_bruto * self._commission_pct / 100.0, 8)
        usdt_neto    = round(usdt_bruto - commission, 8)
        order.status = OrderStatus.FILLED
        order.trade  = TradeRecord(
            ts            = order.ts,
            side          = "SELL",
            price         = order.price,
            btc_sold      = round(btc_a_vender, 10),
            usdt_received = usdt_neto,
            commission    = commission,
            ganancia_usdt = None,
        )
        log.debug("SELL ejecutado", price=order.price,
                  btc=btc_a_vender, usdt_rec=usdt_neto)

    def _reject(self, order: Order, reason: str) -> None:
        order.status        = OrderStatus.REJECTED
        order.reject_reason = reason
        order.trade = TradeRecord(
            ts=order.ts, side=order.side.value, price=order.price,
            ignored=True, ignore_reason=reason,
        )
        log.debug("orden rechazada", reason=reason, side=order.side)

    # ── Guardias ──────────────────────────────────────────────────────────────

    def check_buy_guards(self, wallet: Wallet) -> Optional[str]:
        if wallet.positions_count >= self._max_posiciones:
            return f"max_posiciones({self._max_posiciones})"
        slot = wallet.get_slot_usdt()
        if slot > wallet.get_usdt_balance() + 1e-9:
            return (f"usdt_insuficiente("
                    f"slot={slot:.2f}>balance={wallet.get_usdt_balance():.2f})")
        if slot < 1.0:
            return "slot_menor_a_1_usdt"
        return None

    def check_sell_guards(self, wallet: Wallet) -> Optional[str]:
        if wallet.positions_count == 0:
            return "sin_posiciones"
        if wallet.get_btc_por_venta() <= 0:
            return "btc_por_venta_cero"
        return None

    def execute_with_guards(
        self,
        side:      OrderSide,
        price:     float,
        wallet:    Wallet,
        candle_ts: int = 0,
    ) -> Order:
        """Verifica guardias y delega a execute() pasando candle_ts."""
        reason = (self.check_buy_guards(wallet)
                  if side == OrderSide.BUY
                  else self.check_sell_guards(wallet))

        if reason:
            ts    = candle_ts if candle_ts else now_epoch_s()
            order = self.create_order(
                side        = side,
                price       = price,
                usdt_amount = wallet.get_slot_usdt()     if side == OrderSide.BUY  else None,
                btc_amount  = wallet.get_btc_por_venta() if side == OrderSide.SELL else None,
            )
            order.ts            = ts
            order.status        = OrderStatus.IGNORED
            order.reject_reason = reason
            order.trade = TradeRecord(
                ts=ts, side=side.value, price=price,
                ignored=True, ignore_reason=reason,
            )
            wallet.update(order.trade)
            return order

        return self.execute(side, price, wallet, candle_ts=candle_ts)


# ══════════════════════════════════════════════════════════════════════════════
# FACTORY
# ══════════════════════════════════════════════════════════════════════════════

def build_order_book(
    mode:           str   = "local",
    commission_pct: float = None,
    max_posiciones: int   = None,
) -> OrderBook:
    """
    Helper para construir un OrderBook sin instanciar manualmente.
    No lee mode_config — el runner decide el modo explícitamente.

    mode="local"  →  SimulatedOrderBook  (default, backtest)
    mode="live"   →  BinanceOrderBook    (producción/testnet)

    Los parámetros con None se leen desde config_local como fallback.

    Para control fino, instanciar directamente:
        ob = SimulatedOrderBook(commission_pct=0.1, max_posiciones=5)
        ob = BinanceOrderBook(max_posiciones=5, commission_pct=0.1)
    """
    try:
        import config_local as CL
        _commission = commission_pct or getattr(CL, "COMMISSION_PCT", 0.1)
        _maxpos     = max_posiciones or getattr(CL, "MAX_POSICIONES",  5)
    except ImportError:
        _commission = commission_pct or 0.1
        _maxpos     = max_posiciones or 5

    if mode == "live":
        from actors.binance_order_book import BinanceOrderBook
        log.info("OrderBook → BinanceOrderBook",
                 commission=f"{_commission}%", max_pos=_maxpos)
        return BinanceOrderBook(
            max_posiciones = _maxpos,
            commission_pct = _commission,
        )

    log.info("OrderBook → SimulatedOrderBook",
             commission=f"{_commission}%", max_pos=_maxpos)
    return SimulatedOrderBook(commission_pct=_commission, max_posiciones=_maxpos)