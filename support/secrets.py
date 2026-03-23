"""
secrets.py — Carga de credenciales sensibles
═════════════════════════════════════════════
Lee exclusivamente desde variables de entorno o archivo .env local.
NUNCA desde config.py ni ningún archivo que pueda commitearse.

Uso:
    from support.secrets import secrets
    key    = secrets.binance_api_key
    secret = secrets.binance_secret

Archivo .env esperado (en la raíz del proyecto, jamás commiteado):
    BINANCE_API_KEY=xxxxxxxxxxxxxxxxxxxx
    BINANCE_SECRET=yyyyyyyyyyyyyyyyyyyy

Si una credencial no se encuentra, SecretsManager lanza CredentialNotFound
con un mensaje claro que indica qué variable de entorno falta.
"""

from __future__ import annotations

import os
from pathlib import Path


class CredentialNotFound(Exception):
    """Se lanza cuando una credencial requerida no está en el entorno."""


class SecretsManager:
    """
    Carga credenciales desde variables de entorno.
    Si existe un archivo .env en la raíz del proyecto, lo carga primero
    (sin requerir python-dotenv — implementación mínima incluida).
    """

    def __init__(self, env_file: str = ".env"):
        self._loaded = False
        self._env_file = env_file
        self._ensure_loaded()

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        env_path = self._find_env_file()
        if env_path:
            self._load_env_file(env_path)
        self._loaded = True

    def _find_env_file(self) -> Path | None:
        """Busca .env desde el directorio actual hacia arriba (max 3 niveles)."""
        current = Path.cwd()
        for _ in range(4):
            candidate = current / self._env_file
            if candidate.exists():
                return candidate
            current = current.parent
        return None

    @staticmethod
    def _load_env_file(path: Path) -> None:
        """
        Parser mínimo de .env: KEY=value, ignora comentarios y líneas vacías.
        No sobreescribe variables ya presentes en el entorno del SO.
        """
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key   = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value

    def get(self, key: str, required: bool = True) -> str | None:
        """
        Retorna el valor de la variable de entorno `key`.
        Si `required=True` y no existe, lanza CredentialNotFound.
        """
        value = os.environ.get(key)
        if value is None and required:
            raise CredentialNotFound(
                f"Credencial '{key}' no encontrada. "
                f"Agregá '{key}=<valor>' en tu archivo .env o como variable de entorno."
            )
        return value

    # ── Propiedades nombradas para las credenciales conocidas ─────────────────

    @property
    def binance_api_key(self) -> str:
        return self.get("BINANCE_API_KEY")
    @property
    def binance_testnet_api_key(self) -> str:
        return self.get("BINANCE_TESTNET_API_KEY")

    @property
    def binance_secret(self) -> str:
        return self.get("BINANCE_SECRET")
    @property
    def binance_testnet_secret(self) -> str:
        return self.get("BINANCE_TESTNET_SECRET")

    @property
    def has_binance_credentials(self) -> bool:
        """True si ambas credenciales de Binance están disponibles."""
        return (
            self.get("BINANCE_API_KEY", required=False) is not None and
            self.get("BINANCE_SECRET",  required=False) is not None
        )


# Instancia singleton — importar directamente en los actores
secrets = SecretsManager()
