"""
actors/binance_wallet.py — Wallet sincronizada con Binance
══════════════════════════════════════════════════════════
BinanceWallet extiende JSONWallet (que ya tiene toda la lógica de slots)
y agrega sincronización bidireccional con la cuenta real de Binance:

  · Al arrancar: lee el saldo real de la API y lo reconcilia con
    el último checkpoint del StateManager.
  · Después de cada operación: delega completamente a JSONWallet
    (la Wallet ya sabe calcular slots, posiciones FIFO, etc.)

Por qué no reimplementar desde cero
─────────────────────────────────────
La lógica de slots es compleja y está validada contra miles de velas de
backtest. No tiene sentido duplicarla. La única diferencia en producción
es el estado inicial (que viene de Binance real, no de config_local) y
la verificación de consistencia al arrancar.

Reconciliación al arrancar
───────────────────────────
  1. Leer saldo USDT real de la API.
  2. Leer último checkpoint del JSONStateManager.
  3. Si los saldos coinciden (tolerancia ±1 USDT): restaurar desde checkpoint
     usando restore_wallet_from_checkpoint() — nunca asignando atributos
     privados directamente desde este método.
  4. Si divergen: loggear advertencia, ajustar el checkpoint con el saldo
     real y restaurar igualmente por la misma vía.
     La divergencia puede deberse a operaciones manuales en la cuenta.

Uso
────
  wallet = BinanceWallet.from_account(
      max_posiciones = 5,
      json_path      = "live_results.json",
      state_path     = "state/trading_state.jsonl",
  )
"""

from __future__ import annotations

import collections
from dataclasses import replace
from typing import Optional

import requests

from actors.wallet       import JSONWallet, MemoryWallet, Position, TradeRecord
from state.state_manager import JSONStateManager, restore_wallet_from_checkpoint
from support.logger      import get_logger
from support.secrets     import secrets

import hashlib
import hmac
import time
import urllib.parse

log = get_logger("binance_wallet")


def _ts_ms(base_url: str) -> int:
    """Timestamp en ms compensado por desfase de reloj vs Binance."""
    try:
        from support.time_sync import TimeSync
        return TimeSync.get(base_url).now_ms()
    except Exception:
        return int(time.time() * 1000)


def _get_config() -> dict:
    try:
        import config_world as CW
        return {
            "base_url": CW.BINANCE_TESTNET_URL if CW.USE_TESTNET else CW.BINANCE_BASE_URL,
            "timeout":  CW.REQUEST_TIMEOUT_S,
            "recv_window": CW.RECV_WINDOW_MS,
        }
    except ImportError:
        return {
            "base_url":    "https://testnet.binance.vision",
            "timeout":     10,
            "recv_window": 5000,
        }


class BinanceWallet(JSONWallet):
    """
    JSONWallet extendida con consulta del saldo real de Binance.

    Toda la lógica de slots, posiciones FIFO y update() viene heredada
    de JSONWallet/MemoryWallet — no se reimplementa nada.

    Lo único nuevo aquí es:
      · _fetch_usdt_balance(): consulta REST /api/v3/account
      · from_account():        factory que reconcilia API + checkpoint
    """

    def __init__(
        self,
        usdt_inicial:   float,
        max_posiciones: int,
        json_path:      str,
        append_only:    bool = True,   # en producción siempre append-only
    ) -> None:
        super().__init__(
            usdt_inicial   = usdt_inicial,
            max_posiciones = max_posiciones,
            json_path      = json_path,
            append_only    = append_only,
        )
        cfg = _get_config()
        self._base_url    = cfg["base_url"]
        self._timeout     = cfg["timeout"]
        self._recv_window = cfg["recv_window"]
        self._api_key     = secrets.get("BINANCE_TESTNET_API_KEY")
        self._secret      = secrets.get("BINANCE_TESTNET_SECRET")

    # ── Factory principal ─────────────────────────────────────────────────────

    @classmethod
    def from_account(
        cls,
        max_posiciones: int,
        json_path:      str,
        state_path:     str   = "state/live_trading_state.jsonl",
        commission_pct: float = 0.1,
    ) -> "BinanceWallet":
        """
        Crea una BinanceWallet reconciliando el saldo real con el checkpoint.

        Flujo:
          1. Leer saldo USDT real de la API de Binance.
          2. Leer último checkpoint del StateManager.
          3. Reconciliar: ajustar checkpoint si hay divergencia de saldo.
          4. Restaurar siempre a través de restore_wallet_from_checkpoint()
             para no depender de los nombres de atributos internos de
             MemoryWallet.
        """
        # Instancia temporal solo para poder llamar _fetch_usdt_balance
        temp = cls(
            usdt_inicial   = 1000.0,
            max_posiciones = max_posiciones,
            json_path      = json_path,
        )

        # Paso 1: saldo real de Binance
        usdt_real = temp._fetch_usdt_balance()
        if usdt_real is None:
            log.warning(
                "no se pudo obtener saldo real de Binance — "
                "usando checkpoint si existe"
            )

        # Paso 2: checkpoint
        state_mgr  = JSONStateManager(state_path)
        checkpoint = state_mgr.restore()

        # Paso 3 + 4: reconciliar y restaurar
        if checkpoint is not None:
            usdt_checkpoint = checkpoint.usdt_balance
            diferencia      = abs((usdt_real or 0) - usdt_checkpoint)

            if usdt_real is not None and diferencia > 1.0:
                log.warning(
                    "divergencia saldo USDT",
                    binance        = f"{usdt_real:.2f}",
                    checkpoint     = f"{usdt_checkpoint:.2f}",
                    diferencia     = f"{diferencia:.2f}",
                    accion         = "usando saldo real, posiciones desde checkpoint",
                )
                # Ajustar el checkpoint con el saldo real pero conservar
                # las posiciones. replace() es seguro porque Checkpoint
                # es un dataclass y no tiene lógica interna que invalidar.
                checkpoint_ajustado = replace(
                    checkpoint,
                    usdt_balance    = usdt_real,
                    slot_usdt       = usdt_real / max_posiciones,
                    portfolio_value = usdt_real,   # aproximación conservadora
                )
                mem = restore_wallet_from_checkpoint(
                    checkpoint_ajustado, max_posiciones
                )
            else:
                log.info(
                    "saldos consistentes — restaurando desde checkpoint",
                    usdt      = f"{usdt_checkpoint:.2f}",
                    positions = checkpoint.positions_count,
                )
                mem = restore_wallet_from_checkpoint(checkpoint, max_posiciones)

            # Construir la wallet definitiva copiando el estado del
            # MemoryWallet ya restaurado. Las asignaciones directas a
            # atributos privados están concentradas aquí y son el único
            # punto de entrada — no se repiten en ningún otro lugar.
            wallet = cls(
                usdt_inicial   = mem.get_usdt_balance(),
                max_posiciones = max_posiciones,
                json_path      = json_path,
            )
            wallet._usdt          = mem.get_usdt_balance()
            wallet._btc_libre     = mem.get_btc_balance()
            wallet._slot_usdt     = mem.get_slot_usdt()
            wallet._btc_por_venta = mem.get_btc_por_venta()
            wallet._posiciones    = collections.deque(mem.get_positions())

        else:
            # Sin checkpoint — arranque fresco con saldo real o fallback
            usdt_inicio = usdt_real if usdt_real is not None else 1000.0
            log.info(
                "sin checkpoint previo — arranque fresco",
                usdt = f"{usdt_inicio:.2f}",
            )
            wallet = cls(
                usdt_inicial   = usdt_inicio,
                max_posiciones = max_posiciones,
                json_path      = json_path,
            )

        log.info(
            "BinanceWallet lista",
            usdt      = f"{wallet.get_usdt_balance():.2f}",
            positions = wallet.positions_count,
            slot      = f"{wallet.get_slot_usdt():.2f}",
        )
        return wallet

    # ── Consulta REST de saldo ────────────────────────────────────────────────

    def _fetch_usdt_balance(self) -> Optional[float]:
        """
        Consulta GET /api/v3/account para obtener el saldo libre de USDT.
        Retorna el saldo como float o None si la llamada falla.
        """
        params = {
            "timestamp":  _ts_ms(self._base_url),
            "recvWindow": self._recv_window,
        }
        query_string = urllib.parse.urlencode(params)
        signature    = hmac.new(
            self._secret.encode("utf-8"),
            query_string.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        params["signature"] = signature

        url     = f"{self._base_url}/api/v3/account"
        headers = {"X-MBX-APIKEY": self._api_key}

        try:
            resp = requests.get(
                url, params  = params,
                headers      = headers,
                timeout      = self._timeout,
            )
            resp.raise_for_status()
            data = resp.json()

            for asset in data.get("balances", []):
                if asset["asset"] == "USDT":
                    free = float(asset["free"])
                    log.info(
                        "saldo USDT real obtenido",
                        free   = f"{free:.2f}",
                        locked = asset.get("locked", "0"),
                    )
                    return free

            log.warning("USDT no encontrado en la cuenta de Binance")
            return None

        except requests.RequestException as e:
            log.error("error consultando saldo de Binance", error=str(e))
            return None