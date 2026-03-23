"""
actors/live_clock.py — Implementación de LiveClock
════════════════════════════════════════════════════
Director de ciclos para el modo live/testnet.

En vez de iterar velas de una DB, LiveClock bloquea en tick() hasta
que BinanceWSFeed entrega una vela cerrada, luego retorna inmediatamente.

Arquitectura
─────────────
  BinanceWSFeed.subscribe()  →  callback → Queue(maxsize=1)
  LiveClock.tick()           →  Queue.get(block=True) → Candle

La Queue desacopla el thread del WebSocket del hilo principal del trader.
La capacidad 1 es intencional: si el trader es más lento que el exchange
(poco probable a 1h/vela) la vela vieja se descarta y se procesa la nueva.

Uso en live_local_reversal.py
───────────────────────────────
  feed  = BinanceWSFeed()
  clock = LiveClock(feed, symbol="BTCUSDT")

  # Bloquea hasta el cierre de la primera vela horaria
  for candle in clock:
      signal = strategy._tick(candle, wallet)
      ...
"""

from __future__ import annotations

import queue
import threading
from typing import Callable, Optional

from actors.clock       import Clock
from actors.price_feed  import Candle, PriceFeed
from support.logger     import get_logger
from support.time_utils import to_iso

log = get_logger("live_clock")


class LiveClock(Clock):
    """
    Director de ciclos en tiempo real.

    tick() bloquea hasta que cierra la próxima vela horaria,
    luego retorna inmediatamente con los datos definitivos.

    Nunca retorna None excepto cuando stop() fue llamado explícitamente.
    """

    def __init__(
        self,
        feed:       PriceFeed,
        symbol:     str   = "BTCUSDT",
        queue_size: int   = 1,
    ) -> None:
        """
        feed:       instancia de BinanceWSFeed (o cualquier PriceFeed con subscribe())
        symbol:     par de trading, debe coincidir con config_world.SYMBOL
        queue_size: capacidad de la cola interna. 1 = procesar solo la vela más reciente.
        """
        self._feed       = feed
        self._symbol     = symbol
        self._queue:     queue.Queue[Optional[Candle]] = queue.Queue(maxsize=queue_size)
        self._running    = False
        self._stop_event = threading.Event()
        log.info("LiveClock inicializado", symbol=symbol)

    # ── Interfaz Clock ────────────────────────────────────────────────────────

    @property
    def is_live(self) -> bool:
        return True

    def tick(self) -> Optional[Candle]:
        """
        Bloquea hasta que llega la próxima vela cerrada.
        Retorna None solo si stop() fue llamado.

        El timeout interno de 1 segundo permite chequear stop_event
        sin bloquear indefinidamente (útil para Ctrl+C limpio).
        """
        if not self._running:
            self._start()

        while not self._stop_event.is_set():
            try:
                candle = self._queue.get(block=True, timeout=1.0)
                if candle is None:
                    return None   # señal de parada
                log.debug(
                    "tick recibido",
                    ts=candle.iso(),
                    close=candle.close,
                )
                return candle
            except queue.Empty:
                continue   # timeout — volver a esperar

        return None

    def reset(self) -> None:
        """No tiene efecto en producción — incluido por contrato de interfaz."""
        pass

    def stop(self) -> None:
        """
        Para el clock limpiamente.
        Hace que tick() retorne None en la próxima llamada.
        """
        log.info("LiveClock detenido")
        self._stop_event.set()
        self._running = False
        # Desbloquear tick() si está esperando en la Queue
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        # Detener el feed WebSocket
        if hasattr(self._feed, "stop"):
            self._feed.stop()

    # ── Privado ───────────────────────────────────────────────────────────────

    def _start(self) -> None:
        """Inicia el stream WebSocket al primer tick()."""
        self._running = True
        log.info("iniciando stream WebSocket", symbol=self._symbol)
        self._feed.subscribe(
            callback = self._on_candle,
            symbol   = self._symbol,
        )

    def _on_candle(self, candle: Candle) -> None:
        """
        Callback llamado por BinanceWSFeed cuando cierra una vela.
        Corre en el thread del WebSocket — solo encola, no procesa.

        Si la queue está llena (trader lento) descarta la vela vieja
        y enola la nueva para evitar acumulación de lag.
        """
        if self._stop_event.is_set():
            return

        # Vaciar la queue si está llena antes de encolar la nueva vela
        if self._queue.full():
            try:
                old = self._queue.get_nowait()
                log.warning(
                    "queue llena — descartando vela antigua",
                    descartada=old.iso() if old else "None",
                    nueva=candle.iso(),
                )
            except queue.Empty:
                pass

        try:
            self._queue.put_nowait(candle)
        except queue.Full:
            pass   # race condition extremadamente improbable — ignorar
