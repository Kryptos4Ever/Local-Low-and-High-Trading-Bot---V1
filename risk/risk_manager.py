"""
risk_manager.py — Capa de gestión de riesgo
════════════════════════════════════════════
Responsabilidad única: aprobar o rechazar cada orden ANTES de que llegue
al OrderBook. Es el guardián que protege el capital de bugs y condiciones
de mercado extremas.

En backtest: puede configurarse para aprobar todo (modo permisivo, default)
             o para simular límites reales de riesgo.
En producción: es la primera línea de defensa antes de tocar dinero real.

Reglas implementadas
─────────────────────
  1. MAX_DRAWDOWN_PCT      → detiene operaciones si el portfolio cayó X% desde el pico
  2. MAX_DAILY_LOSS_USDT   → detiene si la pérdida del día supera Y USDT
  3. MAX_ORDER_USDT        → rechaza órdenes individuales mayores a Z USDT
  4. MIN_ORDER_USDT        → rechaza órdenes menores al mínimo operativo
  5. DEDUP_WINDOW_S        → evita órdenes duplicadas en ventana de tiempo

Factory
────────
  build_risk_manager(usdt_inicial, mode, ...)  →  RiskManager
    mode="permissive"   →  sin límites          (default, backtest)
    mode="conservative" →  límites predefinidos (producción)
    mode="custom"       →  parámetros explícitos

  No lee mode_config — el runner decide el modo explícitamente.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from actors.price_feed  import Candle
from actors.wallet      import Wallet
from actors.order_book  import OrderSide
from support.logger     import get_logger
from support.time_utils import now_epoch_s

log = get_logger("risk_manager")


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class RiskConfig:
    """
    Parámetros de riesgo. Valores None = regla desactivada.
    """
    max_drawdown_pct:    Optional[float] = None
    max_daily_loss_usdt: Optional[float] = None
    max_order_usdt:      Optional[float] = None
    min_order_usdt:      float           = 1.0
    dedup_window_s:      int             = 0

    @classmethod
    def permissive(cls) -> "RiskConfig":
        """Sin límites — para backtesting sin restricciones de riesgo."""
        return cls()

    @classmethod
    def conservative(cls) -> "RiskConfig":
        """Límites razonables para producción."""
        return cls(
            max_drawdown_pct    = 20.0,
            max_daily_loss_usdt = 100.0,
            max_order_usdt      = 300.0,
            min_order_usdt      = 5.0,
            dedup_window_s      = 60,
        )


# ══════════════════════════════════════════════════════════════════════════════
# RISK MANAGER
# ══════════════════════════════════════════════════════════════════════════════

class RiskManager:
    """
    Evalúa cada orden contra las reglas de riesgo configuradas.
    Retorna None si la orden puede proceder, o el motivo de rechazo.

    Es stateful: mantiene el pico del portfolio y las pérdidas del día.
    """

    def __init__(self, config: RiskConfig, usdt_inicial: float) -> None:
        self._cfg              = config
        self._usdt_inicial     = usdt_inicial
        self._peak_value:      float = usdt_inicial
        self._day_start_value: float = usdt_inicial
        self._day_key:         str   = ""
        self._last_orders:     Dict[str, int] = {}

        log.info(
            "RiskManager inicializado",
            max_dd    = f"{config.max_drawdown_pct}%"    if config.max_drawdown_pct    else "desactivado",
            max_daily = f"${config.max_daily_loss_usdt}" if config.max_daily_loss_usdt else "desactivado",
            max_order = f"${config.max_order_usdt}"      if config.max_order_usdt      else "desactivado",
        )

    # ── Interfaz pública ──────────────────────────────────────────────────────

    def check(
        self,
        side:   OrderSide,
        price:  float,
        wallet: Wallet,
        candle: Optional[Candle] = None,
    ) -> Optional[str]:
        """
        Evalúa la orden contra todas las reglas activas.
        Retorna None si aprobada, o string con el motivo de rechazo.
        """
        current_value = wallet.portfolio_value(price)
        self._update_state(current_value, candle)

        for reason in [
            self._check_drawdown(current_value),
            self._check_daily_loss(current_value),
            self._check_order_size(side, wallet),
            self._check_dedup(side),
        ]:
            if reason:
                log.warning("orden bloqueada por risk", reason=reason,
                            side=side.value)
                return reason

        self._last_orders[side.value] = now_epoch_s()
        return None

    def update_peak(self, portfolio_value: float) -> None:
        """Actualiza el pico del portfolio. Llamar al final de cada vela."""
        if portfolio_value > self._peak_value:
            self._peak_value = portfolio_value

    # ── Reglas individuales ───────────────────────────────────────────────────

    def _check_drawdown(self, current_value: float) -> Optional[str]:
        if self._cfg.max_drawdown_pct is None or self._peak_value <= 0:
            return None
        drawdown = (self._peak_value - current_value) / self._peak_value * 100.0
        if drawdown >= self._cfg.max_drawdown_pct:
            return (f"max_drawdown_superado("
                    f"dd={drawdown:.1f}%>={self._cfg.max_drawdown_pct}%)")
        return None

    def _check_daily_loss(self, current_value: float) -> Optional[str]:
        if self._cfg.max_daily_loss_usdt is None:
            return None
        daily_loss = self._day_start_value - current_value
        if daily_loss >= self._cfg.max_daily_loss_usdt:
            return (f"max_daily_loss_superado("
                    f"loss=${daily_loss:.2f}>=${self._cfg.max_daily_loss_usdt})")
        return None

    def _check_order_size(self, side: OrderSide, wallet: Wallet) -> Optional[str]:
        if side == OrderSide.BUY:
            amount = wallet.get_slot_usdt()
            if amount < self._cfg.min_order_usdt:
                return f"orden_menor_minimo(${amount:.2f}<${self._cfg.min_order_usdt})"
            if self._cfg.max_order_usdt and amount > self._cfg.max_order_usdt:
                return f"orden_supera_maximo(${amount:.2f}>${self._cfg.max_order_usdt})"
        return None

    def _check_dedup(self, side: OrderSide) -> Optional[str]:
        if self._cfg.dedup_window_s <= 0:
            return None
        last_ts = self._last_orders.get(side.value, 0)
        elapsed = now_epoch_s() - last_ts
        if elapsed < self._cfg.dedup_window_s:
            return (f"dedup_activo("
                    f"elapsed={elapsed}s<{self._cfg.dedup_window_s}s)")
        return None

    def _update_state(
        self,
        current_value: float,
        candle:        Optional[Candle],
    ) -> None:
        self.update_peak(current_value)
        if candle:
            from support.time_utils import to_datetime
            day_str = to_datetime(candle.ts).strftime("%Y-%m-%d")
        else:
            from datetime import datetime, timezone
            day_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if day_str != self._day_key:
            self._day_key         = day_str
            self._day_start_value = current_value


# ══════════════════════════════════════════════════════════════════════════════
# FACTORY
# ══════════════════════════════════════════════════════════════════════════════

def build_risk_manager(
    usdt_inicial:        float,
    mode:                str            = "permissive",
    max_drawdown_pct:    Optional[float] = None,
    max_daily_loss_usdt: Optional[float] = None,
    max_order_usdt:      Optional[float] = None,
    min_order_usdt:      float           = 1.0,
    dedup_window_s:      int             = 0,
) -> RiskManager:
    """
    Helper para construir un RiskManager sin instanciar manualmente.
    No lee mode_config — el runner decide el modo explícitamente.

    mode="permissive"   →  sin límites (default, backtest)
    mode="conservative" →  límites predefinidos para producción
    mode="custom"       →  usar los parámetros explícitos pasados

    Para producción live se recomienda mode="conservative" o mode="custom"
    con los límites apropiados al capital real disponible.

    Ejemplo backtest (sin restricciones):
        risk = build_risk_manager(usdt_inicial=1000.0)

    Ejemplo live conservador:
        risk = build_risk_manager(usdt_inicial=usdt_real, mode="conservative")

    Ejemplo custom:
        risk = build_risk_manager(
            usdt_inicial=5000.0, mode="custom",
            max_drawdown_pct=15.0, max_order_usdt=500.0,
        )
    """
    if mode == "conservative":
        config = RiskConfig.conservative()
        log.info("RiskManager modo CONSERVADOR")
    elif mode == "custom":
        config = RiskConfig(
            max_drawdown_pct    = max_drawdown_pct,
            max_daily_loss_usdt = max_daily_loss_usdt,
            max_order_usdt      = max_order_usdt,
            min_order_usdt      = min_order_usdt,
            dedup_window_s      = dedup_window_s,
        )
        log.info("RiskManager modo CUSTOM",
                 max_dd=max_drawdown_pct, max_daily=max_daily_loss_usdt)
    else:
        config = RiskConfig.permissive()
        log.info("RiskManager modo PERMISIVO (sin límites)")

    return RiskManager(config=config, usdt_inicial=usdt_inicial)