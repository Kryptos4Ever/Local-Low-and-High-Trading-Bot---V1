"""
logger.py — Logging estructurado del sistema
══════════════════════════════════════════════
Provee un logger con:
    · Salida en consola (legible por humanos)
    · Salida en archivo JSON línea-a-línea (legible por máquinas)
    · Niveles: DEBUG, INFO, WARNING, ERROR, CRITICAL
    · Timestamp UTC en cada entrada
    · Contexto adicional via kwargs (símbolo, estrategia, precio, etc.)

Uso:
    from support.logger import get_logger
    log = get_logger("price_feed")
    log.info("vela recibida", symbol="BTCUSDT", close=65000.0)
    log.warning("señal ignorada", motivo="max_posiciones")
    log.error("orden rechazada", order_id="abc123", detalle=str(e))

Configuración (via mode_config.py):
    LOG_TO_FILE  = True        # False → solo consola
    LOG_LEVEL    = "INFO"      # DEBUG | INFO | WARNING | ERROR | CRITICAL
    LOG_DIR      = "logs/"     # directorio donde se crean los archivos
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ── Configuración por defecto (puede ser sobreescrita por mode_config) ────────
_LOG_TO_FILE: bool = False
_LOG_LEVEL:   str  = "INFO"
_LOG_DIR:     str  = "logs"

# Intenta leer desde mode_config si existe
try:
    import mode_config as MC
    _LOG_TO_FILE = getattr(MC, "LOG_TO_FILE", _LOG_TO_FILE)
    _LOG_LEVEL   = getattr(MC, "LOG_LEVEL",   _LOG_LEVEL)
    _LOG_DIR     = getattr(MC, "LOG_DIR",      _LOG_DIR)
except ImportError:
    pass


# ── Formatter legible para consola ────────────────────────────────────────────

class _ConsoleFormatter(logging.Formatter):
    LEVEL_COLORS = {
        "DEBUG":    "\033[36m",   # cyan
        "INFO":     "\033[32m",   # green
        "WARNING":  "\033[33m",   # yellow
        "ERROR":    "\033[31m",   # red
        "CRITICAL": "\033[35m",   # magenta
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        ts    = datetime.fromtimestamp(record.created, tz=timezone.utc).strftime("%H:%M:%S")
        color = self.LEVEL_COLORS.get(record.levelname, "")
        level = f"{color}{record.levelname:<8}{self.RESET}"
        name  = f"\033[90m[{record.name}]{self.RESET}"
        msg   = record.getMessage()

        # Contexto extra (kwargs pasados al logger)
        extra = getattr(record, "_context", {})
        ctx   = "  " + "  ".join(f"{k}={v}" for k, v in extra.items()) if extra else ""

        return f"{ts}  {level} {name}  {msg}{ctx}"


# ── Formatter JSON para archivo ───────────────────────────────────────────────

class _JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, Any] = {
            "ts":      datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level":   record.levelname,
            "logger":  record.name,
            "msg":     record.getMessage(),
        }
        extra = getattr(record, "_context", {})
        if extra:
            entry.update(extra)
        if record.exc_info:
            entry["exc"] = self.formatException(record.exc_info)
        return json.dumps(entry, default=str, ensure_ascii=False)


# ── Logger con soporte para contexto extra ────────────────────────────────────

class _ContextLogger:
    """
    Wrapper liviano sobre logging.Logger que acepta kwargs como contexto
    estructurado en cada llamada de log.
    """

    def __init__(self, logger: logging.Logger) -> None:
        self._logger = logger

    def _log(self, level: int, msg: str, **context: Any) -> None:
        if not self._logger.isEnabledFor(level):
            return
        record = self._logger.makeRecord(
            name=self._logger.name,
            level=level,
            fn="",
            lno=0,
            msg=msg,
            args=(),
            exc_info=None,
        )
        record._context = context      # type: ignore[attr-defined]
        self._logger.handle(record)

    def debug(self,    msg: str, **ctx: Any) -> None: self._log(logging.DEBUG,    msg, **ctx)
    def info(self,     msg: str, **ctx: Any) -> None: self._log(logging.INFO,     msg, **ctx)
    def warning(self,  msg: str, **ctx: Any) -> None: self._log(logging.WARNING,  msg, **ctx)
    def error(self,    msg: str, **ctx: Any) -> None: self._log(logging.ERROR,    msg, **ctx)
    def critical(self, msg: str, **ctx: Any) -> None: self._log(logging.CRITICAL, msg, **ctx)


# ── Registro de loggers creados (evita duplicar handlers) ─────────────────────
_loggers: dict[str, _ContextLogger] = {}


def get_logger(name: str) -> _ContextLogger:
    """
    Retorna (o crea) un logger nombrado con handlers configurados.
    Llamar múltiples veces con el mismo nombre retorna la misma instancia.
    """
    if name in _loggers:
        return _loggers[name]

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, _LOG_LEVEL.upper(), logging.INFO))
    logger.propagate = False

    # Handler de consola
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(_ConsoleFormatter())
        logger.addHandler(ch)

    # Handler de archivo (JSON, append-only)
    if _LOG_TO_FILE:
        log_dir = Path(_LOG_DIR)
        log_dir.mkdir(parents=True, exist_ok=True)
        date_str  = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
        log_path  = log_dir / f"{date_str}_{name}.jsonl"
        fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        fh.setFormatter(_JSONFormatter())
        logger.addHandler(fh)

    ctx_logger = _ContextLogger(logger)
    _loggers[name] = ctx_logger
    return ctx_logger
