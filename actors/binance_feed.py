"""
actors/binance_feed.py — Implementaciones live de PriceFeed
════════════════════════════════════════════════════════════
BinanceRESTFeed  → descarga velas históricas via REST (warmup del modelo)
BinanceWSFeed    → stream de velas cerradas via WebSocket (loop principal)

Flujo de uso en el live trader
───────────────────────────────
  feed = BinanceWSFeed()

  # 1. Warmup: cargar ~250 velas históricas para el modelo
  candles_hist = feed.get_candles("2024-01-01", "now")

  # 2. Stream: callback se llama al cierre de cada vela horaria
  feed.subscribe(callback=mi_funcion, symbol="BTCUSDT")

Notas de implementación
────────────────────────
  · BinanceRESTFeed.get_candles() pagina automáticamente si el rango
    supera el límite de 1000 velas por request de Binance.
  · BinanceWSFeed.subscribe() corre el loop async en un thread daemon
    para no bloquear el hilo principal del trader.
  · El stream reconnecta automáticamente con backoff exponencial ante
    desconexiones transitorias (cortes de red, mantenimiento de Binance).
  · Toda la lógica de tiempo usa time_utils — ningún timestamp hardcodeado.
  · Las credenciales se cargan desde support.secrets (lee el .env).
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from typing import Callable, List, Optional

import requests

from actors.price_feed  import Candle, PriceFeed
from support.logger     import get_logger
from support.time_utils import to_epoch_s, to_epoch_ms, to_iso, now_epoch_s, TimeInput

log = get_logger("binance_feed")


# ─── Configuración ────────────────────────────────────────────────────────────

def _get_config():
    """Lee URLs y parámetros desde config_world.py con fallbacks."""
    try:
        import config_world as CW
        return {
            "base_url":       CW.BINANCE_TESTNET_URL if CW.USE_TESTNET else CW.BINANCE_BASE_URL,
            "ws_url":         CW.BINANCE_WS_TESTNET  if CW.USE_TESTNET else CW.BINANCE_WS_URL,
            "timeout":        CW.REQUEST_TIMEOUT_S,
            "max_retries":    CW.MAX_RETRIES,
            "reconnect_delay":CW.WS_RECONNECT_DELAY_S,
            "symbol":         CW.SYMBOL,
            "interval":       CW.KLINE_INTERVAL,
        }
    except ImportError:
        return {
            "base_url":        "https://testnet.binance.vision",
            "ws_url":          "wss://stream.testnet.binance.vision/ws",
            "timeout":         10,
            "max_retries":     3,
            "reconnect_delay": 5,
            "symbol":          "BTCUSDT",
            "interval":        "1h",
        }


# ══════════════════════════════════════════════════════════════════════════════
# BinanceRESTFeed — histórico via REST
# ══════════════════════════════════════════════════════════════════════════════

class BinanceRESTFeed(PriceFeed):
    """
    Descarga velas históricas desde la API REST de Binance (o testnet).
    Pagina automáticamente rangos que superan 1000 velas.

    Usado por BinanceWSFeed para el warmup inicial del modelo y para
    completar la ventana de features antes de que el stream arranque.
    """

    _KLINES_LIMIT = 1000   # máximo de velas por request en Binance

    def __init__(self) -> None:
        cfg = _get_config()
        self._base_url  = cfg["base_url"]
        self._timeout   = cfg["timeout"]
        self._max_retry = cfg["max_retries"]
        self._interval  = cfg["interval"]
        log.info("BinanceRESTFeed inicializado", base=self._base_url)

    # ── Interfaz pública ──────────────────────────────────────────────────────

    def get_candles(
        self,
        start:  TimeInput,
        end:    TimeInput,
        symbol: str = "BTCUSDT",
    ) -> List[Candle]:
        """
        Descarga todas las velas del rango [start, end].
        Pagina automáticamente si el rango supera 1000 velas.
        Retorna lista ordenada cronológicamente.
        """
        start_ms = to_epoch_ms(start)
        end_ms   = to_epoch_ms(end) if str(end).lower() != "now" \
                   else now_epoch_s() * 1000

        candles: List[Candle] = []
        cursor  = start_ms

        log.info(
            "descargando velas históricas",
            symbol=symbol, interval=self._interval,
            start=to_iso(start), end=to_iso(end_ms // 1000),
        )

        while cursor < end_ms:
            batch = self._fetch_page(
                symbol   = symbol,
                interval = self._interval,
                start_ms = cursor,
                end_ms   = end_ms,
                limit    = self._KLINES_LIMIT,
            )
            if not batch:
                break
            candles.extend(batch)
            # Avanzar cursor a la siguiente vela después de la última recibida
            cursor = batch[-1].ts * 1000 + 1
            log.debug("página recibida", count=len(batch), last=batch[-1].iso())

        log.info("velas descargadas", total=len(candles), symbol=symbol)
        return candles

    def subscribe(
        self,
        callback: Callable[[Candle], None],
        symbol:   str = "BTCUSDT",
    ) -> None:
        raise NotImplementedError(
            "BinanceRESTFeed no soporta stream en tiempo real. "
            "Usá BinanceWSFeed.subscribe() para el loop de producción."
        )

    # ── HTTP helpers ──────────────────────────────────────────────────────────

    def _fetch_page(
        self,
        symbol:   str,
        interval: str,
        start_ms: int,
        end_ms:   int,
        limit:    int,
    ) -> List[Candle]:
        """
        Hace un GET /api/v3/klines con reintentos exponenciales.
        Retorna lista de Candle o lista vacía ante error persistente.
        """
        url    = f"{self._base_url}/api/v3/klines"
        params = {
            "symbol":    symbol,
            "interval":  interval,
            "startTime": start_ms,
            "endTime":   end_ms,
            "limit":     limit,
        }

        for attempt in range(1, self._max_retry + 1):
            try:
                resp = requests.get(url, params=params,
                                    timeout=self._timeout)
                resp.raise_for_status()
                return [self._row_to_candle(r) for r in resp.json()]

            except requests.RequestException as e:
                wait = 2 ** attempt
                log.warning(
                    "error en REST klines",
                    attempt=attempt, error=str(e), retry_in=f"{wait}s",
                )
                if attempt < self._max_retry:
                    time.sleep(wait)

        log.error("fallo persistente en GET klines", symbol=symbol, start_ms=start_ms)
        return []

    @staticmethod
    def _row_to_candle(row: list) -> Candle:
        """
        Convierte una fila de la respuesta de /api/v3/klines a Candle.
        Formato Binance: [open_time, open, high, low, close, volume,
                          close_time, quote_vol, trades, taker_base,
                          taker_quote, ignore]
        """
        return Candle(
            ts                  = int(row[0]) // 1000,   # ms → s
            open                = float(row[1]),
            high                = float(row[2]),
            low                 = float(row[3]),
            close               = float(row[4]),
            volume              = float(row[5]),
            quote_volume        = float(row[7]),
            trades_count        = int(row[8]),
            taker_buy_base_vol  = float(row[9]),
            taker_buy_quote_vol = float(row[10]),
        )


# ══════════════════════════════════════════════════════════════════════════════
# BinanceWSFeed — stream en tiempo real via WebSocket
# ══════════════════════════════════════════════════════════════════════════════

class BinanceWSFeed(PriceFeed):
    """
    Stream de velas cerradas en tiempo real via WebSocket de Binance.

    get_candles()  →  delega a BinanceRESTFeed (warmup histórico)
    subscribe()    →  abre el stream WS y llama al callback en cada cierre

    El WebSocket corre en un thread daemon separado para no bloquear
    el hilo principal del trader. El callback se llama en ese thread,
    lo que significa que debe ser thread-safe (o usar una Queue).

    Reconnection
    ─────────────
    El loop reconnecta automáticamente con backoff exponencial:
      intento 1 → 5s, intento 2 → 10s, intento 3 → 20s, ...
    Se loggea cada reconexión. No hay límite de reintentos — el stream
    se mantiene vivo mientras el proceso esté corriendo.
    """

    def __init__(self) -> None:
        cfg = _get_config()
        self._ws_url         = cfg["ws_url"]
        self._reconnect_delay= cfg["reconnect_delay"]
        self._rest_feed      = BinanceRESTFeed()
        self._stop_event     = threading.Event()
        self._stream_thread: Optional[threading.Thread] = None
        log.info("BinanceWSFeed inicializado", ws=self._ws_url)

    # ── Interfaz pública ──────────────────────────────────────────────────────

    def get_candles(
        self,
        start:  TimeInput,
        end:    TimeInput,
        symbol: str = "BTCUSDT",
    ) -> List[Candle]:
        """
        Descarga historial via REST — usado para el warmup del modelo.
        Delega completamente a BinanceRESTFeed.
        """
        return self._rest_feed.get_candles(start, end, symbol)

    def subscribe(
        self,
        callback: Callable[[Candle], None],
        symbol:   str = "BTCUSDT",
    ) -> None:
        """
        Abre el stream WebSocket y llama a callback(candle) por cada vela
        que cierra. Corre en un thread daemon — no bloquea.

        El callback recibe una Candle con todos los campos de microestructura
        (taker_buy_base_vol, quote_volume, trades_count) idénticos a los
        que entrega SQLiteFeed, garantizando que las features del modelo
        se calculan igual en backtest y en producción.

        Llamar stop() para cerrar el stream limpiamente.
        """
        if self._stream_thread and self._stream_thread.is_alive():
            log.warning("subscribe() llamado dos veces — ignorando")
            return

        self._stop_event.clear()
        self._stream_thread = threading.Thread(
            target  = self._run_loop,
            args    = (callback, symbol),
            daemon  = True,
            name    = f"ws-feed-{symbol}",
        )
        self._stream_thread.start()
        log.info("stream WebSocket iniciado", symbol=symbol)

    def stop(self) -> None:
        """Cierra el stream WebSocket limpiamente."""
        self._stop_event.set()
        if self._stream_thread:
            self._stream_thread.join(timeout=10)
        log.info("stream WebSocket detenido")

    # ── Loop async privado ────────────────────────────────────────────────────

    def _run_loop(
        self,
        callback: Callable[[Candle], None],
        symbol:   str,
    ) -> None:
        """Entry point del thread — crea un event loop y corre el stream."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(
                self._stream_with_reconnect(callback, symbol)
            )
        finally:
            loop.close()

    async def _stream_with_reconnect(
        self,
        callback: Callable[[Candle], None],
        symbol:   str,
    ) -> None:
        """
        Loop de reconnexión con backoff exponencial.
        Se detiene solo cuando self._stop_event está seteado.
        """
        attempt = 0
        while not self._stop_event.is_set():
            try:
                await self._stream_once(callback, symbol)
                attempt = 0   # reset backoff tras conexión exitosa
            except Exception as e:
                if self._stop_event.is_set():
                    break
                attempt += 1
                delay = min(self._reconnect_delay * (2 ** (attempt - 1)), 300)
                log.warning(
                    "WebSocket desconectado — reconnectando",
                    error=str(e), attempt=attempt, delay_s=delay,
                )
                await asyncio.sleep(delay)

    async def _stream_once(
        self,
        callback: Callable[[Candle], None],
        symbol:   str,
    ) -> None:
        """
        Conecta al stream, escucha mensajes y llama al callback
        por cada vela que cierra (is_closed == True).

        El stream es: wss://.../{symbol_lower}@kline_{interval}
        Binance envía un mensaje por cada tick de la vela (cada segundo
        aproximadamente). Solo procesamos los mensajes donde kline.x == true
        que indican que la vela acaba de cerrar definitivamente.
        """
        import websockets

        stream_name = f"{symbol.lower()}@kline_1h"
        url         = f"{self._ws_url}/{stream_name}"

        log.info("conectando WebSocket", url=url)

        async with websockets.connect(
            url,
            ping_interval   = 20,
            ping_timeout    = 10,
            close_timeout   = 5,
        ) as ws:
            log.info("WebSocket conectado", stream=stream_name)

            async for raw_msg in ws:
                if self._stop_event.is_set():
                    break

                try:
                    msg = json.loads(raw_msg)
                except json.JSONDecodeError:
                    continue

                kline = msg.get("k", {})

                # Solo procesar cuando la vela cierra definitivamente
                if not kline.get("x", False):
                    continue

                candle = self._kline_to_candle(kline)
                log.info(
                    "vela cerrada",
                    ts=candle.iso(),
                    close=candle.close,
                    delta_ratio=f"{candle.delta_ratio:.3f}" if candle.delta_ratio else "N/A",
                )

                try:
                    callback(candle)
                except Exception as e:
                    log.error("error en callback de vela", error=str(e))

    @staticmethod
    def _kline_to_candle(k: dict) -> Candle:
        """
        Convierte el payload 'k' de un mensaje WebSocket de Binance a Candle.

        Campos del payload kline de Binance:
          t: open_time (ms), T: close_time (ms)
          o: open, h: high, l: low, c: close
          v: volume (base), q: quote_volume
          n: trades_count
          V: taker_buy_base_volume
          Q: taker_buy_quote_volume
          x: is_closed (bool)
        """
        return Candle(
            ts                  = int(k["t"]) // 1000,   # ms → s
            open                = float(k["o"]),
            high                = float(k["h"]),
            low                 = float(k["l"]),
            close               = float(k["c"]),
            volume              = float(k["v"]),
            quote_volume        = float(k["q"]),
            trades_count        = int(k["n"]),
            taker_buy_base_vol  = float(k["V"]),
            taker_buy_quote_vol = float(k["Q"]),
        )
