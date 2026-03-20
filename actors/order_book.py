"""
order_book.py — Actor 3: Libro de órdenes
══════════════════════════════════════════
Responsabilidad única: abrir y cerrar posiciones, calcular comisiones,
retornar TradeRecord para que la Wallet actualice su estado.

SEPARACIÓN INTENCIÓN / EJECUCIÓN
─────────────────────────────────
El OrderBook separa tres momentos distintos:

  1. create_order(side, ...)  →  Order   (intención: qué queremos hacer)
  2. submit(order)            →  Order   (envío: en Binance coloca la orden)
  3. check(order_id)          →  Order   (confirmación: ¿se ejecutó?)

En simulación los tres pasos colapsan en uno (ejecución instantánea).
En producción con Binance son llamadas REST separadas con posible latencia.

Interfaz abstracta OrderBook
─────────────────────────────
  create_order(side, price, usdt_amount?, btc_amount?) → Order
  submit(order)                                         → Order
  check(order_id)                                       → Order
  execute(side, price, wallet)                          → TradeRecord
      [método de conveniencia que encadena los 3 pasos]

Implementaciones
─────────────────
  SimulatedOrderBook  →  ejecución instantánea al precio dado
                          lógica de slots del Backtest_irreal.py
  BinanceOrderBook    →  stub para producción
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from actors.wallet      import Wallet, TradeRecord
from support.logger     import get_logger
from support.time_utils import now_epoch_s

log = get_logger("order_book")


# ══════════════════════════════════════════════════════════════════════════════
# TIPOS INTERNOS
# ══════════════════════════════════════════════════════════════════════════════

class OrderSide(str, Enum):
    BUY  = "BUY"
    SELL = "SELL"


class OrderStatus(str, Enum):
    PENDING   = "PENDING"    # creada, no enviada
    SUBMITTED = "SUBMITTED"  # enviada al exchange
    FILLED    = "FILLED"     # ejecutada completamente
    REJECTED  = "REJECTED"   # rechazada (fondos insuficientes, etc.)
    IGNORED   = "IGNORED"    # descartada por guardia (max_pos, sin_btc, etc.)


@dataclass
class Order:
    """
    Representa una orden en cualquier punto de su ciclo de vida.
    Se crea en create_order(), se actualiza en submit() y check().
    """
    order_id:      str
    side:          OrderSide
    price:         float
    ts:            int              # epoch s UTC de creación

    # Montos (uno de los dos se rellena según el lado)
    usdt_amount:   Optional[float] = None   # BUY:  USDT a gastar
    btc_amount:    Optional[float] = None   # SELL: BTC a vender

    # Estado
    status:        OrderStatus = OrderStatus.PENDING
    reject_reason: Optional[str] = None

    # Resultado (se rellena tras FILLED)
    trade: Optional[TradeRecord] = None

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
    """
    Contrato que deben cumplir todas las implementaciones del libro de órdenes.
    """

    @abstractmethod
    def create_order(
        self,
        side:        OrderSide,
        price:       float,
        usdt_amount: Optional[float] = None,
        btc_amount:  Optional[float] = None,
    ) -> Order:
        """
        Crea una Order con estado PENDING.
        No envía nada al exchange todavía.
        """

    @abstractmethod
    def submit(self, order: Order) -> Order:
        """
        Envía la orden al exchange.
        En simulación: ejecuta instantáneamente → FILLED o REJECTED.
        En Binance:    coloca la orden → SUBMITTED (puede quedar pendiente).
        """

    @abstractmethod
    def check(self, order_id: str) -> Order:
        """
        Consulta el estado de una orden enviada.
        En simulación: siempre retorna el estado final (ya conocido).
        En Binance:    consulta via REST si la orden fue ejecutada.
        """

    def execute(
        self,
        side:   OrderSide,
        price:  float,
        wallet: Wallet,
    ) -> Order:
        """
        Método de conveniencia: encadena create → submit → check.
        Usa la Wallet para determinar montos según la lógica de slots.
        Retorna la Order final con su TradeRecord adjunto.
        """
        # Determinar monto según el lado y la lógica de slots de la Wallet
        if side == OrderSide.BUY:
            usdt_amount = wallet.get_slot_usdt()
            order = self.create_order(side, price, usdt_amount=usdt_amount)
        else:
            btc_amount = wallet.get_btc_por_venta()
            order = self.create_order(side, price, btc_amount=btc_amount)

        order = self.submit(order)
        order = self.check(order.order_id)

        # Notificar a la Wallet del resultado
        if order.trade:
            wallet.update(order.trade)

        return order


# ══════════════════════════════════════════════════════════════════════════════
# IMPLEMENTACIÓN LOCAL: SimulatedOrderBook
# ══════════════════════════════════════════════════════════════════════════════

class SimulatedOrderBook(OrderBook):
    """
    Libro de órdenes simulado para backtesting.

    Ejecuta órdenes instantáneamente al precio dado.
    Implementa exactamente la lógica de Backtest_irreal.py:

    BUY:
      · Usa slot_usdt de la Wallet como monto fijo por operación
      · Descuenta comisión del USDT antes de calcular BTC comprado
      · Rechaza si slot > usdt_disponible o si hay max_posiciones abiertas

    SELL:
      · Vende btc_por_venta (calculado por la Wallet tras cada BUY)
      · Calcula ganancia contra costo FIFO de las posiciones
      · Rechaza si no hay posiciones abiertas
    """

    def __init__(
        self,
        commission_pct: float,
        max_posiciones: int,
    ) -> None:
        self._commission_pct = commission_pct   # ej: 0.1 (= 0.1%)
        self._max_posiciones = max_posiciones
        self._orders: dict[str, Order] = {}
        log.info(
            "SimulatedOrderBook inicializado",
            commission=f"{commission_pct}%",
            max_pos=max_posiciones,
        )

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
            status      = OrderStatus.PENDING,
        )
        self._orders[order.order_id] = order
        return order

    def submit(self, order: Order) -> Order:
        """En simulación: ejecuta inmediatamente y marca como FILLED o REJECTED."""
        if order.side == OrderSide.BUY:
            self._execute_buy(order)
        else:
            self._execute_sell(order)
        return order

    def check(self, order_id: str) -> Order:
        """En simulación: el estado ya es final tras submit()."""
        return self._orders.get(order_id, Order(
            order_id="unknown", side=OrderSide.BUY,
            price=0, ts=0, status=OrderStatus.REJECTED,
            reject_reason="order_id no encontrado",
        ))

    # ── Lógica de ejecución ───────────────────────────────────────────────────

    def _execute_buy(self, order: Order) -> None:
        usdt_a_gastar = order.usdt_amount or 0.0

        if usdt_a_gastar < 1.0:
            self._reject(order, "usdt_insuficiente(slot<1)")
            return

        commission   = round(usdt_a_gastar * self._commission_pct / 100.0, 8)
        usdt_neto    = usdt_a_gastar - commission
        btc_comprado = round(usdt_neto / order.price, 10)

        order.status = OrderStatus.FILLED
        order.trade  = TradeRecord(
            ts           = order.ts,
            side         = "BUY",
            price        = order.price,
            usdt_spent   = round(usdt_a_gastar, 8),
            btc_bought   = btc_comprado,
            commission   = commission,
        )
        log.debug(
            "BUY ejecutado",
            price=order.price,
            usdt=usdt_a_gastar,
            btc=btc_comprado,
        )

    def _execute_sell(self, order: Order) -> None:
        btc_a_vender = order.btc_amount or 0.0

        if btc_a_vender <= 0:
            self._reject(order, "sin_btc")
            return

        usdt_bruto  = round(btc_a_vender * order.price, 8)
        commission  = round(usdt_bruto * self._commission_pct / 100.0, 8)
        usdt_neto   = round(usdt_bruto - commission, 8)

        # La ganancia real se calcula en la Wallet (que conoce el costo FIFO)
        # Aquí solo registramos los montos de la operación
        order.status = OrderStatus.FILLED
        order.trade  = TradeRecord(
            ts            = order.ts,
            side          = "SELL",
            price         = order.price,
            btc_sold      = round(btc_a_vender, 10),
            usdt_received = usdt_neto,
            commission    = commission,
            ganancia_usdt = None,   # la Wallet calcula ganancia FIFO
        )
        log.debug(
            "SELL ejecutado",
            price=order.price,
            btc=btc_a_vender,
            usdt_rec=usdt_neto,
        )

    def _reject(self, order: Order, reason: str) -> None:
        order.status        = OrderStatus.REJECTED
        order.reject_reason = reason
        order.trade = TradeRecord(
            ts            = order.ts,
            side          = order.side.value,
            price         = order.price,
            ignored       = True,
            ignore_reason = reason,
        )
        log.debug("orden rechazada", reason=reason, side=order.side)

    # ── Guardias pre-ejecución ────────────────────────────────────────────────

    def check_buy_guards(self, wallet: Wallet) -> Optional[str]:
        """
        Verifica condiciones previas a un BUY.
        Retorna None si puede proceder, o el motivo de rechazo.
        Llamar antes de execute() desde la estrategia o el runner.
        """
        if wallet.positions_count >= self._max_posiciones:
            return f"max_posiciones({self._max_posiciones})"
        slot = wallet.get_slot_usdt()
        if slot > wallet.get_usdt_balance() + 1e-9:
            return f"usdt_insuficiente(slot={slot:.2f}>balance={wallet.get_usdt_balance():.2f})"
        if slot < 1.0:
            return "slot_menor_a_1_usdt"
        return None

    def check_sell_guards(self, wallet: Wallet) -> Optional[str]:
        """
        Verifica condiciones previas a un SELL.
        Retorna None si puede proceder, o el motivo de rechazo.
        """
        if wallet.positions_count == 0:
            return "sin_posiciones"
        if wallet.get_btc_por_venta() <= 0:
            return "btc_por_venta_cero"
        return None

    def execute_with_guards(
        self,
        side:   OrderSide,
        price:  float,
        wallet: Wallet,
    ) -> Order:
        """
        Como execute() pero verifica guardias antes de operar.
        Versión recomendada para usar desde las estrategias.
        Retorna Order con status=IGNORED si hay guardia activa.
        """
        if side == OrderSide.BUY:
            reason = self.check_buy_guards(wallet)
        else:
            reason = self.check_sell_guards(wallet)

        if reason:
            order = self.create_order(
                side  = side,
                price = price,
                usdt_amount = wallet.get_slot_usdt()      if side == OrderSide.BUY  else None,
                btc_amount  = wallet.get_btc_por_venta()  if side == OrderSide.SELL else None,
            )
            order.status        = OrderStatus.IGNORED
            order.reject_reason = reason
            order.trade = TradeRecord(
                ts=order.ts, side=side.value, price=price,
                ignored=True, ignore_reason=reason,
            )
            wallet.update(order.trade)   # registra ignorado en la Wallet
            return order

        return self.execute(side, price, wallet, candle_ts=candle_ts)


# ══════════════════════════════════════════════════════════════════════════════
# STUB producción
# ══════════════════════════════════════════════════════════════════════════════

class BinanceOrderBook(OrderBook):
    """
    Coloca órdenes reales en Binance via REST API.
    Implementación completa en etapa de producción.
    """

    def create_order(self, side, price, usdt_amount=None, btc_amount=None):
        raise NotImplementedError("BinanceOrderBook pendiente de implementación.")

    def submit(self, order):
        raise NotImplementedError("BinanceOrderBook pendiente de implementación.")

    def check(self, order_id):
        raise NotImplementedError("BinanceOrderBook pendiente de implementación.")


# ══════════════════════════════════════════════════════════════════════════════
# FACTORY
# ══════════════════════════════════════════════════════════════════════════════

def build_order_book() -> OrderBook:
    """
    Construye la implementación correcta según mode_config y config_local.

    Modos (mode_config.py):
        USE_LIVE_ORDERBOOK = False  →  SimulatedOrderBook
        USE_LIVE_ORDERBOOK = True   →  BinanceOrderBook
    """
    try:
        import mode_config as MC
        use_live = getattr(MC, "USE_LIVE_ORDERBOOK", False)
    except ImportError:
        use_live = False

    try:
        import config_local as CL
        commission = getattr(CL, "COMMISSION_PCT",  0.1)
        max_pos    = getattr(CL, "MAX_POSICIONES",  5)
    except ImportError:
        commission = 0.1
        max_pos    = 5

    if use_live:
        log.info("OrderBook modo LIVE → BinanceOrderBook")
        return BinanceOrderBook()

    log.info("OrderBook modo LOCAL → SimulatedOrderBook",
             commission=f"{commission}%", max_pos=max_pos)
    return SimulatedOrderBook(commission_pct=commission, max_posiciones=max_pos)
