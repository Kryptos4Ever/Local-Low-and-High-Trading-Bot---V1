"""
actors/binance_order_book.py — OrderBook para Binance Testnet/Producción
═════════════════════════════════════════════════════════════════════════
Coloca órdenes MARKET reales contra la API REST de Binance (o testnet).

Flujo de una orden
───────────────────
  create_order()  →  Order(PENDING)
  submit()        →  POST /api/v3/order → Order(SUBMITTED)
  check()         →  GET  /api/v3/order → Order(FILLED) o Order(REJECTED)

El método execute_with_guards() (heredado de OrderBook via la interfaz) se
sobreescribe aquí para mantener los mismos guardias que SimulatedOrderBook
pero ejecutando en Binance real.

Firma HMAC-SHA256
──────────────────
  Todos los endpoints autenticados de Binance requieren:
    · Header "X-MBX-APIKEY": tu API key
    · Parámetro "signature": HMAC-SHA256(query_string, secret)
    · Parámetro "timestamp": epoch ms actual
    · Parámetro "recvWindow": ventana de validez (default 5000ms)

  La firma se calcula sobre el query string completo incluyendo timestamp.
  Este módulo implementa el signing internamente sin depender de SDK externo.

Limitaciones testnet
─────────────────────
  · El precio ejecutado puede diferir del solicitado (slippage de mercado).
  · Binance testnet tiene liquidez artificial — los fills son más rápidos
    y a veces a precios ligeramente distintos a producción.
  · El mínimo de orden en testnet puede diferir. Si Binance rechaza la orden
    por notional demasiado bajo, se loggea y se trata como IGNORED.
"""

from __future__ import annotations

import hashlib
import hmac
import time
import urllib.parse
import uuid
from typing import Optional

import requests

from actors.order_book  import (
    OrderBook, OrderSide, OrderStatus, Order,
)
from actors.wallet      import Wallet, TradeRecord
from support.logger     import get_logger
from support.secrets    import secrets
from support.time_utils import now_epoch_s

log = get_logger("binance_order_book")


def _ts_ms(base_url: str) -> int:
    """Timestamp en ms compensado por desfase de reloj vs Binance."""
    try:
        from support.time_sync import TimeSync
        return TimeSync.get(base_url).now_ms()
    except Exception:
        import time
        return int(time.time() * 1000)


def _get_config() -> dict:
    try:
        import config_world as CW
        return {
            "base_url":   CW.BINANCE_TESTNET_URL if CW.USE_TESTNET else CW.BINANCE_BASE_URL,
            "timeout":    CW.REQUEST_TIMEOUT_S,
            "max_retries":CW.MAX_RETRIES,
            "recv_window":CW.RECV_WINDOW_MS,
            "symbol":     CW.SYMBOL,
        }
    except ImportError:
        return {
            "base_url":   "https://testnet.binance.vision",
            "timeout":    10,
            "max_retries":3,
            "recv_window":5000,
            "symbol":     "BTCUSDT",
        }


class BinanceOrderBook(OrderBook):
    """
    Coloca órdenes MARKET reales en Binance (testnet o producción).

    Extiende la interfaz OrderBook con la misma separación
    create → submit → check que SimulatedOrderBook, pero haciendo
    llamadas REST reales y firmando con HMAC-SHA256.
    """

    def __init__(self, max_posiciones: int = 5, commission_pct: float = 0.1) -> None:
        cfg = _get_config()
        self._base_url      = cfg["base_url"]
        self._timeout       = cfg["timeout"]
        self._max_retries   = cfg["max_retries"]
        self._recv_window   = cfg["recv_window"]
        self._symbol        = cfg["symbol"]
        self._max_posiciones = max_posiciones
        self._commission_pct = commission_pct
        self._orders: dict[str, Order] = {}

        # Cargar credenciales del .env via SecretsManager
        # Testnet usa keys de testnet.binance.vision, no las de producción
        self._api_key = secrets.get("BINANCE_TESTNET_API_KEY")
        self._secret  = secrets.get("BINANCE_TESTNET_SECRET")

        log.info(
            "BinanceOrderBook inicializado",
            base=self._base_url,
            api_key=self._api_key[:8] + "...",
            max_pos=max_posiciones,
        )

    # ── Interfaz OrderBook ────────────────────────────────────────────────────

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
        """Envía la orden a Binance via REST POST /api/v3/order."""
        if order.side == OrderSide.BUY:
            return self._submit_buy(order)
        else:
            return self._submit_sell(order)

    def check(self, order_id: str) -> Order:
        """
        Retorna el estado actual de la orden.
        En el flujo actual submit() ya espera el fill, así que
        check() simplemente retorna el Order del dict local.
        """
        return self._orders.get(order_id, Order(
            order_id="unknown", side=OrderSide.BUY, price=0, ts=0,
            status=OrderStatus.REJECTED,
            reject_reason="order_id no encontrado",
        ))

    # ── Guardias (idénticas a SimulatedOrderBook) ─────────────────────────────

    def check_buy_guards(self, wallet: Wallet) -> Optional[str]:
        if wallet.positions_count >= self._max_posiciones:
            return f"max_posiciones({self._max_posiciones})"
        slot = wallet.get_slot_usdt()
        if slot > wallet.get_usdt_balance() + 1e-9:
            return (f"usdt_insuficiente("
                    f"slot={slot:.2f}>balance={wallet.get_usdt_balance():.2f})")
        if slot < 10.0:   # mínimo notional de Binance suele ser ~10 USDT
            return "slot_menor_a_minimo_binance(10 USDT)"
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
        """
        Idéntico a SimulatedOrderBook.execute_with_guards():
        verifica guardias y delega a execute() con candle_ts.
        """
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
            log.info("orden ignorada por guardia", reason=reason, side=side.value)
            return order

        return self.execute(side, price, wallet, candle_ts=candle_ts)

    # ── Ejecución privada ─────────────────────────────────────────────────────

    def _submit_buy(self, order: Order) -> Order:
        """
        Coloca orden MARKET BUY en Binance.
        Binance acepta el parámetro 'quoteOrderQty' para especificar
        cuánto USDT gastar — exactamente lo que necesitamos.
        """
        usdt = order.usdt_amount or 0.0
        if usdt < 10.0:
            self._local_reject(order, "notional_insuficiente(<10 USDT)")
            return order

        params = {
            "symbol":        self._symbol,
            "side":          "BUY",
            "type":          "MARKET",
            "quoteOrderQty": f"{usdt:.8f}",   # USDT a gastar
        }

        result = self._signed_post("/api/v3/order", params)
        if result is None:
            self._local_reject(order, "error_api_binance")
            return order

        return self._parse_fill(order, result, "BUY")

    def _submit_sell(self, order: Order) -> Order:
        """
        Coloca orden MARKET SELL en Binance.
        Usa 'quantity' (BTC a vender).
        """
        btc = order.btc_amount or 0.0
        if btc <= 0:
            self._local_reject(order, "btc_cero")
            return order

        # Binance requiere precisión de 5 decimales para BTC en spot
        btc_str = f"{btc:.5f}"

        params = {
            "symbol":   self._symbol,
            "side":     "SELL",
            "type":     "MARKET",
            "quantity": btc_str,
        }

        result = self._signed_post("/api/v3/order", params)
        if result is None:
            self._local_reject(order, "error_api_binance")
            return order

        return self._parse_fill(order, result, "SELL")

    def _parse_fill(self, order: Order, result: dict, side: str) -> Order:
        """
        Extrae precio ejecutado, comisión y cantidades del response de Binance.
        Binance retorna los fills detallados en result['fills'].
        """
        try:
            status = result.get("status", "")
            if status not in ("FILLED", "PARTIALLY_FILLED"):
                self._local_reject(order, f"binance_status:{status}")
                return order

            fills = result.get("fills", [])
            if not fills:
                # Si no hay fills detallados usar los campos sumarios
                fills = [{
                    "price":           result.get("price", str(order.price)),
                    "qty":             result.get("executedQty", "0"),
                    "commission":      "0",
                    "commissionAsset": "BNB",
                }]

            # Precio promedio ponderado de todos los fills
            total_qty   = sum(float(f["qty"]) for f in fills)
            avg_price   = (
                sum(float(f["price"]) * float(f["qty"]) for f in fills) / total_qty
                if total_qty > 0 else order.price
            )
            # Comisión total en USDT (si está en BNB se convierte aproximado)
            commission_usdt = sum(
                float(f["commission"])
                * (avg_price if f.get("commissionAsset") != "USDT" else 1.0)
                for f in fills
            )

            order.status = OrderStatus.FILLED

            if side == "BUY":
                usdt_gastado = float(result.get("cummulativeQuoteQty", order.usdt_amount or 0))
                btc_comprado = total_qty
                order.trade = TradeRecord(
                    ts         = order.ts,
                    side       = "BUY",
                    price      = round(avg_price, 8),
                    usdt_spent = round(usdt_gastado, 8),
                    btc_bought = round(btc_comprado, 10),
                    commission = round(commission_usdt, 8),
                )
                log.info(
                    "BUY ejecutado en Binance",
                    price=f"{avg_price:.2f}", btc=f"{btc_comprado:.6f}",
                    usdt=f"{usdt_gastado:.2f}", commission=f"{commission_usdt:.4f}",
                )

            else:   # SELL
                usdt_recibido = float(result.get("cummulativeQuoteQty", 0)) - commission_usdt
                btc_vendido   = total_qty
                order.trade = TradeRecord(
                    ts            = order.ts,
                    side          = "SELL",
                    price         = round(avg_price, 8),
                    btc_sold      = round(btc_vendido, 10),
                    usdt_received = round(usdt_recibido, 8),
                    commission    = round(commission_usdt, 8),
                )
                log.info(
                    "SELL ejecutado en Binance",
                    price=f"{avg_price:.2f}", btc=f"{btc_vendido:.6f}",
                    usdt_rec=f"{usdt_recibido:.2f}", commission=f"{commission_usdt:.4f}",
                )

            # Garantizar consistencia de ts
            if order.trade:
                order.trade.ts = order.ts

            return order

        except (KeyError, ValueError, ZeroDivisionError) as e:
            log.error("error parseando fill de Binance", error=str(e), result=str(result))
            self._local_reject(order, f"parse_error:{e}")
            return order

    def _local_reject(self, order: Order, reason: str) -> None:
        """Marca la orden como rechazada localmente (sin llamada a Binance)."""
        order.status        = OrderStatus.REJECTED
        order.reject_reason = reason
        order.trade = TradeRecord(
            ts=order.ts, side=order.side.value, price=order.price,
            ignored=True, ignore_reason=reason,
        )
        log.warning("orden rechazada", reason=reason, side=order.side.value)

    # ── HTTP firmado ──────────────────────────────────────────────────────────

    def _signed_post(self, endpoint: str, params: dict) -> Optional[dict]:
        """
        POST autenticado con firma HMAC-SHA256.
        Reintenta hasta self._max_retries veces ante errores de red.
        Retorna el dict de respuesta o None ante error persistente.
        """
        params["timestamp"]  = _ts_ms(self._base_url)
        params["recvWindow"] = self._recv_window

        query_string = urllib.parse.urlencode(params)
        signature    = hmac.new(
            self._secret.encode("utf-8"),
            query_string.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        params["signature"] = signature

        url     = f"{self._base_url}{endpoint}"
        headers = {"X-MBX-APIKEY": self._api_key}

        for attempt in range(1, self._max_retries + 1):
            try:
                resp = requests.post(
                    url, params=params,
                    headers=headers, timeout=self._timeout,
                )
                # Errores 4xx de Binance (orden rechazada, saldo insuficiente, etc.)
                if resp.status_code == 400:
                    err = resp.json()
                    log.error(
                        "Binance rechazó la orden",
                        code=err.get("code"), msg=err.get("msg"),
                    )
                    return None

                resp.raise_for_status()
                return resp.json()

            except requests.RequestException as e:
                wait = 2 ** attempt
                log.warning(
                    "error en POST orden",
                    attempt=attempt, error=str(e), retry_in=f"{wait}s",
                )
                if attempt < self._max_retries:
                    time.sleep(wait)

        log.error("fallo persistente al colocar orden", endpoint=endpoint)
        return None

    def _signed_get(self, endpoint: str, params: dict) -> Optional[dict]:
        """GET autenticado — usado en check() si se necesita polling."""
        params["timestamp"]  = int(time.time() * 1000)
        params["recvWindow"] = self._recv_window
        query_string         = urllib.parse.urlencode(params)
        signature            = hmac.new(
            self._secret.encode("utf-8"),
            query_string.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        params["signature"] = signature

        url     = f"{self._base_url}{endpoint}"
        headers = {"X-MBX-APIKEY": self._api_key}

        try:
            resp = requests.get(url, params=params,
                                headers=headers, timeout=self._timeout)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            log.error("error en GET firmado", error=str(e))
            return None
