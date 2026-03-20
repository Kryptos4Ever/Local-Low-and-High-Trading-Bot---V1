"""
time_utils.py — Normalización universal de tiempo
══════════════════════════════════════════════════
Regla única del sistema: todo timestamp interno es UTC epoch en segundos (int).
Cada actor convierte en sus propios bordes usando estas funciones.

Tipos soportados en entrada:
    · int / float  → asume epoch ms si > 1e10, epoch s si menor
    · str          → ISO 8601 ("2024-01-15 08:00:00" o "2024-01-15T08:00:00")
    · datetime     → naive (asume UTC) o aware
    · pd.Timestamp → idem
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Union

# Tipo aceptado como timestamp en cualquier punto de entrada del sistema
TimeInput = Union[int, float, str, datetime]

# Umbral para distinguir epoch-ms de epoch-s
_MS_THRESHOLD = 1_000_000_000_000   # 10^12  →  año ~2001 en ms


def to_epoch_s(value: TimeInput) -> int:
    """
    Convierte cualquier representación de tiempo a UTC epoch seconds (int).
    Es la función central del módulo — todos los actores la usan en sus bordes.
    """
    if isinstance(value, (int, float)):
        v = int(value)
        return v // 1000 if v > _MS_THRESHOLD else v

    if isinstance(value, str):
        return _parse_str(value)

    if isinstance(value, datetime):
        return _from_datetime(value)

    # pandas Timestamp (importación diferida para no requerir pandas en soporte)
    try:
        import pandas as pd
        if isinstance(value, pd.Timestamp):
            return _from_datetime(value.to_pydatetime())
    except ImportError:
        pass

    raise TypeError(f"Tipo no soportado para conversión de tiempo: {type(value)}")


def to_epoch_ms(value: TimeInput) -> int:
    """Epoch milisegundos — usado solo en los bordes con APIs externas."""
    return to_epoch_s(value) * 1000


def to_datetime(value: TimeInput) -> datetime:
    """Convierte a datetime UTC aware."""
    return datetime.fromtimestamp(to_epoch_s(value), tz=timezone.utc)


def to_iso(value: TimeInput) -> str:
    """Convierte a string ISO 8601 con sufijo Z."""
    return to_datetime(value).strftime("%Y-%m-%dT%H:%M:%SZ")


def to_date_str(value: TimeInput) -> str:
    """Convierte a 'YYYY-MM-DD' (para nombres de archivo, logs, etc.)."""
    return to_datetime(value).strftime("%Y-%m-%d")


def now_epoch_s() -> int:
    """Timestamp actual en UTC epoch seconds."""
    return int(datetime.now(tz=timezone.utc).timestamp())


def epoch_s_from_date_str(date_str: str) -> int:
    """
    Convierte 'YYYY-MM-DD' al epoch s del inicio del día UTC.
    Usado en config_local para parsear FECHA_INICIO / FECHA_FIN.
    """
    return _parse_str(date_str + "T00:00:00")


# ── Helpers privados ──────────────────────────────────────────────────────────

_ISO_PATTERN = re.compile(
    r"^(\d{4})-(\d{2})-(\d{2})"        # YYYY-MM-DD
    r"(?:[T ](\d{2}):(\d{2}):(\d{2}))?"  # opcional HH:MM:SS
    r"(?:Z|[+-]\d{2}:\d{2})?$"          # opcional timezone suffix
)


def _parse_str(s: str) -> int:
    m = _ISO_PATTERN.match(s.strip())
    if not m:
        raise ValueError(f"Formato de fecha no reconocido: '{s}'")
    year, month, day = int(m.group(1)), int(m.group(2)), int(m.group(3))
    hour   = int(m.group(4)) if m.group(4) else 0
    minute = int(m.group(5)) if m.group(5) else 0
    second = int(m.group(6)) if m.group(6) else 0
    dt = datetime(year, month, day, hour, minute, second, tzinfo=timezone.utc)
    return int(dt.timestamp())


def _from_datetime(dt: datetime) -> int:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp())
