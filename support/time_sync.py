"""
support/time_sync.py — Compensación de desfase de reloj local
══════════════════════════════════════════════════════════════
Binance rechaza requests cuyo timestamp difiere más de recvWindow (5s)
del tiempo real del servidor. Si el reloj local está desfasado, todas
las firmas HMAC fallan con error -1021.

Solución correcta: sincronizar el reloj del sistema con NTP.
  Windows (PowerShell Admin): w32tm /resync /force
  Linux/Mac:                  sudo ntpdate pool.ntp.org

Solución en código (este módulo): medir el desfase contra el servidor
de Binance al arrancar y compensarlo en cada timestamp generado.

Uso:
    from support.time_sync import TimeSync
    ts = TimeSync.get()       # mide el desfase una vez al arrancar
    ms = ts.now_ms()          # timestamp corregido en ms para firmar
"""

from __future__ import annotations

import time
from typing import Optional

import requests

from support.logger import get_logger

log = get_logger("time_sync")

_BASE_URL = "https://testnet.binance.vision"


class TimeSync:
    """
    Singleton que mide el desfase de reloj una sola vez al arrancar
    y lo aplica a todos los timestamps generados para Binance.
    """

    _instance:  Optional["TimeSync"] = None
    _offset_ms: int = 0   # desfase: server_time - local_time (en ms)

    def __init__(self, offset_ms: int = 0) -> None:
        self._offset_ms = offset_ms

    @classmethod
    def get(cls, base_url: str = _BASE_URL) -> "TimeSync":
        """
        Retorna la instancia singleton. Si no existe, mide el desfase
        contra el servidor de Binance y crea la instancia.
        Llamar una vez al arrancar el trader, antes de cualquier firma.
        """
        if cls._instance is not None:
            return cls._instance

        offset_ms = cls._measure_offset(base_url)
        cls._instance = cls(offset_ms)
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Fuerza una nueva medición en la próxima llamada a get()."""
        cls._instance = None

    def now_ms(self) -> int:
        """Timestamp en milisegundos corregido por el desfase del reloj."""
        return int(time.time() * 1000) + self._offset_ms

    @property
    def offset_s(self) -> float:
        """Desfase en segundos (positivo = reloj local atrasado)."""
        return self._offset_ms / 1000

    @staticmethod
    def _measure_offset(base_url: str) -> int:
        """
        Hace 3 requests a /api/v3/time y toma la mediana para reducir
        el efecto de latencia variable. Estima el tiempo local en el
        punto medio del RTT (round-trip time) de cada request.
        """
        offsets = []
        for _ in range(3):
            try:
                t_before = int(time.time() * 1000)
                resp     = requests.get(f"{base_url}/api/v3/time", timeout=5)
                t_after  = int(time.time() * 1000)
                server   = resp.json()["serverTime"]
                t_local  = (t_before + t_after) // 2   # punto medio del RTT
                offsets.append(server - t_local)
            except Exception:
                pass

        if not offsets:
            log.warning("no se pudo medir el desfase — usando 0")
            return 0

        offsets.sort()
        median = offsets[len(offsets) // 2]

        if abs(median) > 1000:
            log.warning(
                "desfase de reloj detectado",
                desfase_s=f"{median/1000:.1f}s",
                accion="compensando automáticamente",
            )
        else:
            log.info("reloj OK", desfase_ms=median)

        return median
