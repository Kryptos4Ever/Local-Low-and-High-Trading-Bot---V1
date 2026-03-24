"""
wallet.py — Actor 2: Billetera
══════════════════════════════
Responsabilidad única: custodiar y reportar el estado del capital.

La Wallet sabe cuánto USDT y BTC tiene el sistema en cada momento,
qué posiciones están abiertas y cuál es el slot de capital actual.
NO toma decisiones — solo registra y responde consultas.

Interfaz abstracta Wallet
──────────────────────────
  get_usdt_balance()   →  float
  get_btc_balance()    →  float          # BTC libre acumulado (no en posiciones)
  get_btc_acumulado()  →  float          # BTC total vendido históricamente
  get_positions()      →  List[Position]
  get_slot_usdt()      →  float          # tamaño del slot actual
  get_btc_por_venta()  →  float          # BTC a vender en la próxima señal SELL
  update(trade)        →  None           # actualiza estado tras cada operación

Lógica de slots (del Backtest_irreal.py — benchmark canónico)
──────────────────────────────────────────────────────────────
  Al llegar a 0 posiciones:
      slot_usdt = usdt_balance / MAX_POSICIONES

  El slot permanece INMUTABLE mientras haya posiciones abiertas.
  Sólo se recalcula al volver a 0 posiciones.

  Después de cada BUY:
      btc_por_venta = btc_en_posiciones / positions_count

  Este valor queda fijo hasta la siguiente compra.

Implementaciones
─────────────────
  JSONWallet     →  persiste en archivo .json (backtest con registro completo)
  MemoryWallet   →  solo RAM (grid search, sin I/O)
  BinanceWallet  →  stub para producción
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional

from support.logger     import get_logger
from support.time_utils import to_iso, now_epoch_s

log = get_logger("wallet")


# ══════════════════════════════════════════════════════════════════════════════
# TIPOS INTERNOS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(slots=True)
class Position:
    """Una posición abierta: una compra de BTC no vendida aún."""
    entry_price: float
    btc:         float
    opened_at:   int    # epoch s UTC

    @property
    def value_at(self, price: float) -> float:
        return self.btc * price


@dataclass(slots=True)
class TradeRecord:
    """
    Resultado de una operación ejecutada por el OrderBook.
    La Wallet recibe un TradeRecord en update() para actualizar su estado.
    """
    ts:             int       # epoch s UTC
    side:           str       # "BUY" | "SELL"
    price:          float
    # BUY fields
    usdt_spent:     Optional[float] = None
    btc_bought:     Optional[float] = None
    commission:     Optional[float] = None
    # SELL fields
    btc_sold:       Optional[float] = None
    usdt_received:  Optional[float] = None
    ganancia_usdt:  Optional[float] = None
    # Metadata
    ignored:        bool   = False
    ignore_reason:  Optional[str] = None


# ══════════════════════════════════════════════════════════════════════════════
# INTERFAZ ABSTRACTA
# ══════════════════════════════════════════════════════════════════════════════

class Wallet(ABC):
    """
    Contrato que deben cumplir todas las implementaciones de billetera.
    Los actores y estrategias solo conocen esta interfaz.
    """

    @abstractmethod
    def get_usdt_balance(self) -> float:
        """USDT libre disponible para operar."""

    @abstractmethod
    def get_btc_balance(self) -> float:
        """BTC libre acumulado (no está en posiciones abiertas)."""

    @abstractmethod
    def get_btc_acumulado(self) -> float:
        """
        BTC total vendido históricamente durante la sesión.
        Usado en el summary del JSON de resultados.
        FIX: declarado explícitamente en la interfaz para que cualquier
        implementación alternativa de Wallet esté obligada a proveerlo.
        """

    @abstractmethod
    def get_positions(self) -> List[Position]:
        """Lista de posiciones abiertas (BTC comprado y no vendido)."""

    @abstractmethod
    def get_slot_usdt(self) -> float:
        """
        Tamaño del slot actual en USDT.
        Se recalcula solo cuando positions_count llega a 0.
        """

    @abstractmethod
    def get_btc_por_venta(self) -> float:
        """
        BTC a vender en la próxima operación SELL.
        Se recalcula después de cada BUY.
        """

    @abstractmethod
    def update(self, trade: TradeRecord) -> None:
        """
        Actualiza el estado de la billetera después de una operación.
        Llamado por el OrderBook tras ejecutar una orden.
        """

    @abstractmethod
    def snapshot(self, current_price: float) -> dict:
        """
        Retorna el estado completo de la billetera en un momento dado.
        Usado por el StateManager y por el JSON de resultados.
        """

    # ── Propiedades calculadas (no abstractas — disponibles en todas) ─────────

    @property
    def positions_count(self) -> int:
        return len(self.get_positions())

    def btc_en_posiciones(self) -> float:
        return sum(p.btc for p in self.get_positions())

    def precio_promedio_posiciones(self) -> float:
        positions = self.get_positions()
        total_btc = sum(p.btc for p in positions)
        if total_btc == 0:
            return 0.0
        return sum(p.entry_price * p.btc for p in positions) / total_btc

    def portfolio_value(self, current_price: float) -> float:
        """USDT libre + valor de mercado del BTC en posiciones."""
        return self.get_usdt_balance() + self.btc_en_posiciones() * current_price


# ══════════════════════════════════════════════════════════════════════════════
# IMPLEMENTACIÓN: MemoryWallet
# ══════════════════════════════════════════════════════════════════════════════

class MemoryWallet(Wallet):
    """
    Billetera en memoria pura — sin I/O.
    Ideal para grid search y backtesting rápido donde no se necesita JSON.

    Implementa exactamente la lógica de slots del Backtest_irreal.py.
    """

    def __init__(self, usdt_inicial: float, max_posiciones: int) -> None:
        self._usdt:                float          = usdt_inicial
        self._btc_libre:           float          = 0.0
        self._btc_acumulado_total: float          = 0.0
        self._posiciones:          deque[Position] = deque()
        self._max_pos:             int            = max_posiciones
        self._slot_usdt:           float          = usdt_inicial / max_posiciones
        self._btc_por_venta:       float          = 0.0
        self._usdt_inicial:        float          = usdt_inicial

    # ── Interfaz ──────────────────────────────────────────────────────────────

    def get_usdt_balance(self)   -> float:           return self._usdt
    def get_btc_balance(self)    -> float:           return self._btc_libre
    def get_btc_acumulado(self)  -> float:           return self._btc_acumulado_total
    def get_positions(self)      -> List[Position]:  return list(self._posiciones)
    def get_slot_usdt(self)      -> float:           return self._slot_usdt
    def get_btc_por_venta(self)  -> float:           return self._btc_por_venta

    def update(self, trade: TradeRecord) -> None:
        if trade.ignored:
            return

        if trade.side == "BUY":
            self._usdt -= (trade.usdt_spent or 0.0)
            self._posiciones.append(Position(
                entry_price = trade.price,
                btc         = trade.btc_bought or 0.0,
                opened_at   = trade.ts,
            ))
            self._recalcular_btc_por_venta()

        elif trade.side == "SELL":
            self._usdt += (trade.usdt_received or 0.0)
            btc_vendido = trade.btc_sold or 0.0
            self._reducir_posiciones_fifo(btc_vendido)
            self._btc_acumulado_total += btc_vendido
            if self.positions_count == 0:
                self._recalcular_slot()

    def snapshot(self, current_price: float) -> dict:
        return {
            "usdt_balance":               round(self._usdt, 8),
            "btc_libre":                  round(self._btc_libre, 10),
            "btc_acumulado_total":        round(self._btc_acumulado_total, 10),
            "btc_en_posiciones":          round(self.btc_en_posiciones(), 10),
            "positions_count":            self.positions_count,
            "precio_promedio_posiciones": round(self.precio_promedio_posiciones(), 8),
            "slot_usdt":                  round(self._slot_usdt, 4),
            "btc_por_venta":              round(self._btc_por_venta, 10),
            "portfolio_value":            round(self.portfolio_value(current_price), 4),
            "pnl_pct":                    round(
                (self.portfolio_value(current_price) - self._usdt_inicial)
                / self._usdt_inicial * 100, 4
            ),
        }

    def reset(self) -> None:
        """Reinicia la billetera al estado inicial — útil entre runs del grid."""
        self._usdt                = self._usdt_inicial
        self._btc_libre           = 0.0
        self._btc_acumulado_total = 0.0
        self._posiciones          = deque()
        self._slot_usdt           = self._usdt_inicial / self._max_pos
        self._btc_por_venta       = 0.0

    # ── Helpers privados ──────────────────────────────────────────────────────

    def _recalcular_slot(self) -> None:
        """Recalcula slot_usdt cuando positions_count llega a 0."""
        self._slot_usdt     = self._usdt / self._max_pos
        self._btc_por_venta = 0.0

    def _recalcular_btc_por_venta(self) -> None:
        """Actualiza btc_por_venta después de cada BUY."""
        n = self.positions_count
        self._btc_por_venta = self.btc_en_posiciones() / n if n > 0 else 0.0

    def _reducir_posiciones_fifo(self, btc_a_reducir: float) -> None:
        """Reduce posiciones FIFO por el monto vendido."""
        restante = btc_a_reducir
        while restante > 1e-10 and self._posiciones:
            pos = self._posiciones[0]
            if pos.btc <= restante + 1e-10:
                restante -= pos.btc
                self._posiciones.popleft()
            else:
                pos.btc  -= restante
                restante  = 0.0


# ══════════════════════════════════════════════════════════════════════════════
# IMPLEMENTACIÓN: JSONWallet
# ══════════════════════════════════════════════════════════════════════════════

class JSONWallet(MemoryWallet):
    """
    Extiende MemoryWallet agregando persistencia en archivo JSON.

    En cada update() el estado se puede guardar (modo append-only en producción,
    sobreescritura en backtest para compatibilidad con Graficador.py).

    El JSON producido es compatible con el esquema esperado por Graficador.py.
    """

    def __init__(
        self,
        usdt_inicial:   float,
        max_posiciones: int,
        json_path:      str,
        append_only:    bool = False,
    ) -> None:
        super().__init__(usdt_inicial, max_posiciones)
        self._json_path   = Path(json_path)
        self._append_only = append_only
        self._trade_log:  list[dict] = []
        log.info("JSONWallet inicializado", path=json_path, append=append_only)

    def update(self, trade: TradeRecord) -> None:
        """Actualiza estado en memoria y registra el trade en el log."""
        super().update(trade)
        self._trade_log.append(self._trade_to_dict(trade))

    def flush(self, summary: dict) -> None:
        """
        Escribe el JSON final al disco.
        Llamar al finalizar el backtest desde el runner.
        Compatible con el esquema de Graficador.py.
        """
        payload = {
            "summary":       summary,
            "trade_history": self._trade_log,
        }
        self._json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
        log.info("JSON guardado", path=str(self._json_path), trades=len(self._trade_log))

    def get_trade_log(self) -> list[dict]:
        """Retorna una copia del log de trades para el runner."""
        return list(self._trade_log)

    # ── Conversión TradeRecord → dict (esquema Graficador) ────────────────────

    def _trade_to_dict(self, t: TradeRecord) -> dict:
        snap = self.snapshot(t.price)
        return {
            "datetime":                   to_iso(t.ts),
            "type":                       t.side,
            "price":                      round(t.price, 8),
            "score_bot":                  None,
            "score_top":                  None,
            "usdt_balance":               snap["usdt_balance"],
            "btc_balance":                snap["btc_libre"],
            "btc_en_posiciones":          snap["btc_en_posiciones"],
            "positions_count":            snap["positions_count"],
            "precio_promedio_posiciones": snap["precio_promedio_posiciones"],
            "ignorado":                   t.ignored,
            "motivo_ignorado":            t.ignore_reason,
            "usdt_spent":                 round(t.usdt_spent,    8) if t.usdt_spent    else None,
            "btc_bought":                 round(t.btc_bought,   10) if t.btc_bought    else None,
            "commission_usdt":            round(t.commission,    8) if t.commission    else None,
            "btc_sold":                   round(t.btc_sold,     10) if t.btc_sold      else None,
            "btc_accumulated":            round(self._btc_acumulado_total, 10),  # FIX: era siempre 0.0
            "usdt_received":              round(t.usdt_received, 8) if t.usdt_received else None,
            "ganancia_usdt":              round(t.ganancia_usdt, 8) if t.ganancia_usdt else None,
            "pct_capital_usado":          None,
        }


# ══════════════════════════════════════════════════════════════════════════════
# STUB producción
# ══════════════════════════════════════════════════════════════════════════════

class BinanceWallet(Wallet):
    """
    Consulta balances reales via REST API de Binance.
    Implementación completa en actors/binance_wallet.py.
    """

    def get_usdt_balance(self)  -> float:          raise NotImplementedError
    def get_btc_balance(self)   -> float:          raise NotImplementedError
    def get_btc_acumulado(self) -> float:          raise NotImplementedError
    def get_positions(self)     -> List[Position]: raise NotImplementedError
    def get_slot_usdt(self)     -> float:          raise NotImplementedError
    def get_btc_por_venta(self) -> float:          raise NotImplementedError
    def update(self, trade)     -> None:           raise NotImplementedError
    def snapshot(self, price)   -> dict:           raise NotImplementedError


# ══════════════════════════════════════════════════════════════════════════════
# FACTORY
# ══════════════════════════════════════════════════════════════════════════════

def build_wallet(mode: str = "auto") -> Wallet:
    """
    Construye la implementación correcta según mode_config y config_local.

    Modos (mode_config.py):
        USE_LIVE_WALLET  = False  →  JSONWallet  (backtest)
        USE_LIVE_WALLET  = False  +  USE_MEMORY_WALLET = True  →  MemoryWallet (grid)
        USE_LIVE_WALLET  = True   →  BinanceWallet (producción)
    """
    try:
        import mode_config as MC
        use_live   = getattr(MC, "USE_LIVE_WALLET",   False)
        use_memory = getattr(MC, "USE_MEMORY_WALLET", False)
    except ImportError:
        use_live = use_memory = False

    try:
        import config_local as CL
        usdt_ini  = getattr(CL, "SALDO_USDT_INICIAL", 1000.0)
        max_pos   = getattr(CL, "MAX_POSICIONES",      5)
        json_path = getattr(CL, "RESULTS_JSON",        "backtest_results.json")
    except ImportError:
        usdt_ini  = 1000.0
        max_pos   = 5
        json_path = "backtest_results.json"

    if use_live:
        log.info("Wallet modo LIVE → BinanceWallet")
        return BinanceWallet()

    if use_memory:
        log.info("Wallet modo MEMORY → MemoryWallet", usdt=usdt_ini, max_pos=max_pos)
        return MemoryWallet(usdt_ini, max_pos)

    log.info("Wallet modo LOCAL → JSONWallet", path=json_path)
    return JSONWallet(usdt_ini, max_pos, json_path)