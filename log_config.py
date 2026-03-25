"""
log_config.py — Configuración de Logging
══════════════════════════════════════════════════
Este archivo SOLO gobierna el logging del sistema.

`support/logger.py` lo importa (si existe) y espera estas constantes:
  - LOG_TO_FILE: si escribe logs en JSONL (append-only)
  - LOG_LEVEL  : DEBUG | INFO | WARNING | ERROR | CRITICAL
  - LOG_DIR    : carpeta donde se escriben los JSONL (si LOG_TO_FILE=True)
"""

# ── Logger ────────────────────────────────────────────────────────────────────
LOG_TO_FILE = False  # True → escribe logs en logs/YYYY-MM-DD_<modulo>.jsonl
LOG_LEVEL = "INFO"   # DEBUG | INFO | WARNING | ERROR | CRITICAL
LOG_DIR = "logs"
