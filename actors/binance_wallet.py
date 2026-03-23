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
  3. Si los saldos coinciden (tolerancia ±1 USDT): usar el checkpoint.
  4. Si divergen: loggear advertencia y usar el saldo real como base.
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

from typing import Optional

import requests

from actors.wallet      import JSONWallet, MemoryWallet, Position, TradeRecord
from state.state_manager import JSONStateManager, restore_wallet_from_checkpoint
from support.logger     import get_logger
from support.secrets    import secrets

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
        import time
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
        self._base_url   = cfg["base_url"]
        self._timeout    = cfg["timeout"]
        self._recv_window = cfg["recv_window"]
        self._api_key    = secrets.get("BINANCE_TESTNET_API_KEY")
        self._secret     = secrets.get("BINANCE_TESTNET_SECRET")

    # ── Factory principal ─────────────────────────────────────────────────────

    @classmethod
    def from_account(
        cls,
        max_posiciones: int,
        json_path:      str,
        state_path:     str = "state/trading_state.jsonl",
        commission_pct: float = 0.1,
    ) -> "BinanceWallet":
        """
        Crea una BinanceWallet reconciliando el saldo real con el checkpoint.

        Flujo:
          1. Leer saldo USDT real de la API de Binance.
          2. Leer último checkpoint del StateManager.
          3. Reconciliar y construir la wallet con el estado correcto.
        """
        # Crear instancia temporal para poder llamar _fetch_usdt_balance
        temp = cls(
            usdt_inicial   = 1000.0,   # placeholder — se sobreescribe
            max_posiciones = max_posiciones,
            json_path      = json_path,
        )

        # Paso 1: saldo real de Binance
        usdt_real = temp._fetch_usdt_balance()
        if usdt_real is None:
            log.warning(
                "no se pudo obtener saldo real — usando checkpoint si existe"
            )

        # Paso 2: checkpoint
        state_mgr  = JSONStateManager(state_path)
        checkpoint = state_mgr.restore()

        # Paso 3: reconciliar
        if checkpoint is not None:
            usdt_checkpoint = checkpoint.usdt_balance
            diferencia      = abs((usdt_real or 0) - usdt_checkpoint)

            if usdt_real is not None and diferencia > 1.0:
                log.warning(
                    "divergencia saldo USDT",
                    binance=f"{usdt_real:.2f}",
                    checkpoint=f"{usdt_checkpoint:.2f}",
                    diferencia=f"{diferencia:.2f}",
                    accion="usando saldo real como base",
                )
                # Usar saldo real pero mantener posiciones del checkpoint
                wallet = cls(
                    usdt_inicial   = usdt_real,
                    max_posiciones = max_posiciones,
                    json_path      = json_path,
                )
                # Restaurar posiciones desde el checkpoint
                from collections import deque
                wallet._usdt          = usdt_real
                wallet._btc_libre     = checkpoint.btc_libre
                wallet._slot_usdt     = usdt_real / max_posiciones
                wallet._btc_por_venta = checkpoint.btc_por_venta
                wallet._posiciones    = deque(
                    Position(
                        entry_price = p["entry_price"],
                        btc         = p["btc"],
                        opened_at   = p["opened_at"],
                    )
                    for p in checkpoint.positions
                )
            else:
                # Saldos consistentes — restaurar desde checkpoint completo
                log.info(
                    "saldos consistentes — restaurando desde checkpoint",
                    usdt=f"{usdt_checkpoint:.2f}",
                    positions=checkpoint.positions_count,
                )
                mem = restore_wallet_from_checkpoint(checkpoint, max_posiciones)
                wallet = cls(
                    usdt_inicial   = checkpoint.usdt_balance,
                    max_posiciones = max_posiciones,
                    json_path      = json_path,
                )
                # Copiar estado del MemoryWallet restaurado
                from collections import deque
                wallet._usdt          = mem.get_usdt_balance()
                wallet._btc_libre     = mem.get_btc_balance()
                wallet._slot_usdt     = mem.get_slot_usdt()
                wallet._btc_por_venta = mem.get_btc_por_venta()
                wallet._posiciones    = deque(mem.get_positions())

        else:
            # Sin checkpoint — arranque fresco
            usdt_inicio = usdt_real if usdt_real is not None else 1000.0
            log.info(
                "sin checkpoint previo — arranque fresco",
                usdt=f"{usdt_inicio:.2f}",
            )
            wallet = cls(
                usdt_inicial   = usdt_inicio,
                max_posiciones = max_posiciones,
                json_path      = json_path,
            )

        log.info(
            "BinanceWallet lista",
            usdt=f"{wallet.get_usdt_balance():.2f}",
            positions=wallet.positions_count,
            slot=f"{wallet.get_slot_usdt():.2f}",
        )
        return wallet

    # ── Consulta REST de saldo ─────────────────────────────────────────────────

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
                url, params=params,
                headers=headers, timeout=self._timeout,
            )
            resp.raise_for_status()
            data = resp.json()

            # Buscar USDT en la lista de balances
            for asset in data.get("balances", []):
                if asset["asset"] == "USDT":
                    free = float(asset["free"])
                    log.info(
                        "saldo USDT real obtenido",
                        free=f"{free:.2f}",
                        locked=asset.get("locked", "0"),
                    )
                    return free

            log.warning("USDT no encontrado en la cuenta de Binance")
            return None

        except requests.RequestException as e:
            log.error("error consultando saldo de Binance", error=str(e))
            return None
