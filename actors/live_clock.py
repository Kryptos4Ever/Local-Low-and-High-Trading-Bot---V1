"""
actors/live_clock.py — Implementación de LiveClock
════════════════════════════════════════════════════
Director de ciclos para el modo live/testnet.

En vez de iterar velas de una DB, LiveClock bloquea en tick() hasta
que BinanceWSFeed entrega una vela cerrada, luego retorna inmediatamente.

Arquitectura
─────────────
  BinanceWSFeed.subscribe()  →  callback → Queue(maxsize=2)
  LiveClock.tick()           →  Queue.get(block=True) → Candle

La Queue desacopla el thread del WebSocket del hilo principal del trader.
La capacidad 2 garantiza que nunca se descarta una vela a menos que el
trader tarde más de 2 horas en procesar un tick (imposible en condiciones
normales con velas de 1h).

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
from typing import Optional

from actors.clock       import Clock
from actors.price_feed  import Candle, PriceFeed
from support.logger     import get_logger

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
        queue_size: int   = 2,
    ) -> None:
        """
        feed:       instancia de BinanceWSFeed (o cualquier PriceFeed con subscribe())
        symbol:     par de trading, debe coincidir con config_world.SYMBOL
        queue_size: capacidad de la cola interna. 2 = tolera hasta 2 velas
                    sin procesar antes de descartar (con velas de 1h esto
                    nunca ocurre en condiciones normales de operación).
        """
        self._feed        = feed
        self._symbol      = symbol
        self._queue:      queue.Queue[Optional[Candle]] = queue.Queue(maxsize=queue_size)
        self._stop_event  = threading.Event()
        self._start_lock  = threading.Lock()
        self._started     = False
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

        FIX: _start() ahora usa un Lock para garantizar que subscribe()
        se completa exactamente una vez sin race condition entre el
        primer tick() y posibles velas que lleguen antes de que el
        flag de inicialización anterior estuviera seteado.
        """
        self._ensure_started()

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
        # Desbloquear tick() si está esperando en la Queue
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        # Detener el feed WebSocket
        if hasattr(self._feed, "stop"):
            self._feed.stop()

    # ── Privado ───────────────────────────────────────────────────────────────

    def _ensure_started(self) -> None:
        """
        Inicia el stream WebSocket exactamente una vez, de forma thread-safe.

        FIX: la versión anterior usaba un flag booleano `_running` que se
        seteaba a True antes de llamar a subscribe(). Si el WebSocket
        entregaba una vela muy rápido (en tests o en condiciones de red
        favorable), _on_candle podía ser llamado antes de que _started
        fuera True, y un segundo tick() llamaba subscribe() de nuevo.

        Ahora se usa un Lock + flag para garantizar que subscribe() se
        llama exactamente una vez, independientemente de la velocidad
        del WebSocket.
        """
        if self._started:
            return
        with self._start_lock:
            if self._started:   # double-checked locking
                return
            log.info("iniciando stream WebSocket", symbol=self._symbol)
            self._feed.subscribe(
                callback = self._on_candle,
                symbol   = self._symbol,
            )
            self._started = True

    def _on_candle(self, candle: Candle) -> None:
        """
        Callback llamado por BinanceWSFeed cuando cierra una vela.
        Corre en el thread del WebSocket — solo encola, no procesa.

        Si la queue está llena significa que el trader lleva más de
        queue_size horas sin procesar un tick — situación anómala que
        se registra como error. Se descarta la vela más antigua para
        evitar acumulación de lag indefinida.
        """
        if self._stop_event.is_set():
            return

        if self._queue.full():
            try:
                old = self._queue.get_nowait()
                log.error(
                    "VELA DESCARTADA — el trader no procesó la vela anterior a tiempo",
                    descartada=old.iso() if old else "None",
                    nueva=candle.iso(),
                    accion="revisar latencia del modelo o reducir complejidad del tick",
                )
            except queue.Empty:
                pass

        try:
            self._queue.put_nowait(candle)
        except queue.Full:
            log.error(
                "no se pudo encolar vela — queue llena tras vaciado",
                ts=candle.iso(),
            )