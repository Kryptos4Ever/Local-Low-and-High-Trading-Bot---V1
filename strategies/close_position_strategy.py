"""
strategies/close_position_strategy.py — ClosePositionStrategy
══════════════════════════════════════════════════════════════
Estrategia basada exclusivamente en el factor `close_position`, el predictor
de mayor poder estadístico sobre los turning points del benchmark irreal
(AUC=0.8247 bottom / AUC=0.8223 top, ventana=6, período 2021-2025).

Fundamento estadístico (factors_analysis.json, oráculo ventana=6)
───────────────────────────────────────────────────────────────────
  Factor          : close_position = (close - min_low) / (max_high - min_low)
  Ventana óptima  : 6 velas hacia atrás (mismo horizonte que el irreal)
  AUC BOTTOM      : 0.8247   Cohen's d = 1.4473
  AUC TOP         : 0.8223   Cohen's d = 1.4448
  Media BOTTOM    : 0.226    Mediana = 0.177   → umbral_bot default = 0.20
  Media TOP       : 0.800    Mediana = 0.848   → umbral_top default = 0.80
  Media NEUTRAL   : 0.520    (distribución casi uniforme [0,1])

Interpretación geométrica
──────────────────────────
  close_position mide dónde está el cierre dentro del rango
  high-low de las últimas N velas:
    · 0.0 = cierre exactamente en el mínimo del rango → máximo interés comprador
    · 1.0 = cierre exactamente en el máximo del rango → máximo interés vendedor
    · 0.5 = cierre en el centro del rango → zona neutra

  Esto replica la intuición del irreal: el oráculo compra cuando la vela
  es el mínimo local (close_position → 0) y vende cuando es el máximo
  local (close_position → 1).

Lógica de señal
────────────────
  close_position ≤ umbral_bot  →  BUY
  close_position ≥ umbral_top  →  SELL
  (SELL tiene prioridad sobre BUY en la misma vela)

Parámetros calibrados (punto de partida óptimo)
────────────────────────────────────────────────
  ventana     = 6     (máximo AUC; AUC cae monotónicamente con N > 6)
  umbral_bot  = 0.20  (≈ mediana de bottoms reales = 0.177; captura el 53% de bottoms)
  umbral_top  = 0.80  (≈ media de tops reales = 0.800; captura el 52% de tops)
  cooldown    = 0     (desactivado por defecto; activar con 48-96 velas)

Correlación con el benchmark irreal
─────────────────────────────────────
  El benchmark irreal detecta extremos locales con ventana=6.
  Este factor también usa ventana=6 y mide exactamente la misma geometría
  (posición del cierre en el rango), por lo que es la aproximación causal
  más directa posible al oráculo sin mirar el futuro.

Uso en backtest_predictive_candles.py (compatible, drop-in)
────────────────────────────────────────────────────────────
  from strategies.close_position_strategy import ClosePositionStrategy

  strategy = ClosePositionStrategy(
      ventana    = 6,
      umbral_bot = 0.20,
      umbral_top = 0.80,
      cooldown   = 0,
  )

Atributos públicos expuestos tras cada on_candle
─────────────────────────────────────────────────
  last_close_position   float | None  valor crudo del factor
  last_signal_type      str           "BUY" | "SELL" | "HOLD"
  last_cooldown_ok      bool          si el cooldown estaba cumplido
"""

from __future__ import annotations

from collections import deque
from typing import Deque, List, Optional

from actors.price_feed        import Candle
from actors.wallet            import Wallet
from strategies.base_strategy import BaseStrategy, Signal, SignalSide, HOLD
from support.logger           import get_logger

log = get_logger("close_position_strategy")


# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTES CALIBRADAS (factors_analysis.json, ventana=6, 2021-2025)
# ══════════════════════════════════════════════════════════════════════════════

# AUC y estadísticos de referencia — solo documentación, no se usan en runtime
_AUC_BOTTOM    = 0.8247
_AUC_TOP       = 0.8223
_MEAN_BOTTOM   = 0.2257
_MEDIAN_BOTTOM = 0.1769
_MEAN_TOP      = 0.8001
_MEDIAN_TOP    = 0.8484
_MEAN_NEUTRAL  = 0.5204


# ══════════════════════════════════════════════════════════════════════════════
# ESTRATEGIA
# ══════════════════════════════════════════════════════════════════════════════

class ClosePositionStrategy(BaseStrategy):
    """
    Estrategia monomodal basada en close_position con ventana fija de 6 velas.

    Maximiza la correlación con el benchmark irreal usando únicamente el
    predictor de mayor AUC del análisis factorial (AUC=0.8247 / 0.8223).

    Parámetros
    ──────────
    ventana     velas hacia atrás para el rango high-low        (default=6)
    umbral_bot  close_position ≤ umbral_bot  →  BUY             (default=0.20)
    umbral_top  close_position ≥ umbral_top  →  SELL            (default=0.80)
    cooldown    velas mínimas entre señales del mismo tipo       (default=0)
                0 = desactivado. Rango razonable: 24-96 velas.
    """

    # ── Defaults calibrados ────────────────────────────────────────────────────
    DEFAULT_VENTANA    = 6
    DEFAULT_UMBRAL_BOT = 0.20   # basado en mediana de bottoms reales (0.177)
    DEFAULT_UMBRAL_TOP = 0.80   # basado en media de tops reales (0.800)
    DEFAULT_COOLDOWN   = 0

    def __init__(
        self,
        ventana:    int   = DEFAULT_VENTANA,
        umbral_bot: float = DEFAULT_UMBRAL_BOT,
        umbral_top: float = DEFAULT_UMBRAL_TOP,
        cooldown:   int   = DEFAULT_COOLDOWN,
    ) -> None:
        super().__init__(name="ClosePosition-v1")

        # ── Validaciones ───────────────────────────────────────────────────────
        if ventana < 2:
            raise ValueError(f"ventana >= 2 requerido, got {ventana}")
        if not (0.0 < umbral_bot < 0.5):
            raise ValueError(
                f"umbral_bot debe estar en (0, 0.5) para señal BUY sensata, got {umbral_bot}"
            )
        if not (0.5 < umbral_top < 1.0):
            raise ValueError(
                f"umbral_top debe estar en (0.5, 1.0) para señal SELL sensata, got {umbral_top}"
            )
        if umbral_bot + (1.0 - umbral_top) >= 0.5:
            raise ValueError(
                f"umbral_bot={umbral_bot} y umbral_top={umbral_top} se solapan o dejan zona muerta < 50% — revisar."
            )
        if cooldown < 0:
            raise ValueError(f"cooldown >= 0, got {cooldown}")

        self.ventana    = ventana
        self.umbral_bot = umbral_bot
        self.umbral_top = umbral_top
        self.cooldown   = cooldown

        # Buffer deslizante de N velas (solo necesitamos high y low)
        self._buf: Deque[Candle] = deque(maxlen=ventana)

        # Cooldown: índice de vela de la última señal emitida
        _NEG_INF = -(10 ** 9)
        self._last_bot_idx: int = _NEG_INF
        self._last_top_idx: int = _NEG_INF

        # ── Atributos públicos expuestos al runner ─────────────────────────────
        self.last_close_position: Optional[float] = None
        self.last_signal_type:    str              = "HOLD"
        self.last_cooldown_ok:    bool             = False

        log.info(
            "ClosePositionStrategy configurada",
            ventana    = ventana,
            umbral_bot = umbral_bot,
            umbral_top = umbral_top,
            cooldown   = f"{cooldown}v" if cooldown else "off",
            auc_ref    = f"BOT={_AUC_BOTTOM}  TOP={_AUC_TOP}",
        )

    # ── Ciclo de vida ─────────────────────────────────────────────────────────

    def on_start(self, wallet: Wallet) -> None:
        self._buf.clear()
        _NEG_INF = -(10 ** 9)
        self._last_bot_idx    = _NEG_INF
        self._last_top_idx    = _NEG_INF
        self.last_close_position = None
        self.last_signal_type    = "HOLD"
        self.last_cooldown_ok    = False
        log.info("ClosePositionStrategy iniciada")

    def on_candle(self, candle: Candle, wallet: Wallet) -> Signal:
        """
        Calcula close_position sobre las últimas `ventana` velas y emite señal.

        Flujo:
          1. Añadir vela al buffer
          2. Warm-up: HOLD mientras el buffer no esté lleno
          3. Calcular close_position
          4. Verificar cooldown (independiente para BUY y SELL)
          5. SELL si close_position ≥ umbral_top  (prioridad sobre BUY)
          6. BUY  si close_position ≤ umbral_bot
          7. HOLD en el resto
        """
        self._buf.append(candle)
        current_idx = self._candles_seen   # ya incrementado por _tick()

        # ── Warm-up ────────────────────────────────────────────────────────────
        if len(self._buf) < self.ventana:
            self.last_close_position = None
            self.last_signal_type    = "HOLD"
            self.last_cooldown_ok    = False
            return HOLD

        # ── Factor close_position ──────────────────────────────────────────────
        cp = self._calc_close_position(list(self._buf))
        self.last_close_position = cp

        # ── Cooldown ───────────────────────────────────────────────────────────
        cd_ok_bot = (self.cooldown == 0 or
                     (current_idx - self._last_bot_idx) >= self.cooldown)
        cd_ok_top = (self.cooldown == 0 or
                     (current_idx - self._last_top_idx) >= self.cooldown)

        self.last_cooldown_ok = cd_ok_bot or cd_ok_top

        # ── SELL: prioridad sobre BUY ──────────────────────────────────────────
        if cp >= self.umbral_top and cd_ok_top:
            self._last_top_idx   = current_idx
            self.last_signal_type = "SELL"
            log.debug(
                "SELL signal",
                ts = candle.iso(),
                cp = f"{cp:.4f}",
                umbral_top = self.umbral_top,
            )
            return Signal(
                side   = SignalSide.SELL,
                price  = candle.close,
                reason = f"cp={cp:.4f}>={self.umbral_top}",
                score  = cp,
            )

        # ── BUY ────────────────────────────────────────────────────────────────
        if cp <= self.umbral_bot and cd_ok_bot:
            self._last_bot_idx   = current_idx
            self.last_signal_type = "BUY"
            log.debug(
                "BUY signal",
                ts = candle.iso(),
                cp = f"{cp:.4f}",
                umbral_bot = self.umbral_bot,
            )
            return Signal(
                side   = SignalSide.BUY,
                price  = candle.close,
                reason = f"cp={cp:.4f}<={self.umbral_bot}",
                score  = 1.0 - cp,   # invertido: cp bajo = score alto para BUY
            )

        self.last_signal_type = "HOLD"
        return HOLD

    def on_stop(self, wallet: Wallet) -> None:
        log.info(
            "ClosePositionStrategy detenida",
            velas_procesadas = self.candles_seen,
        )

    # ── Cálculo del factor ────────────────────────────────────────────────────

    @staticmethod
    def _calc_close_position(window: List[Candle]) -> float:
        """
        close_position = (close[-1] - min_low) / (max_high - min_low)

        Retorna 0.5 si el rango es cero (vela/rango completamente plano).
        El resultado está en [0.0, 1.0]:
          0.0 → el cierre está exactamente en el mínimo del rango → señal BUY
          1.0 → el cierre está exactamente en el máximo del rango → señal SELL
        """
        max_high = max(c.high for c in window)
        min_low  = min(c.low  for c in window)
        rng      = max_high - min_low
        if rng <= 0:
            return 0.5
        return round((window[-1].close - min_low) / rng, 6)

    # ── Información ───────────────────────────────────────────────────────────

    def describe(self) -> dict:
        return {
            "estrategia":   self.name,
            "ventana":      self.ventana,
            "umbral_bot":   self.umbral_bot,
            "umbral_top":   self.umbral_top,
            "cooldown":     self.cooldown,
            "factor":       "close_position",
            "auc_bottom":   _AUC_BOTTOM,
            "auc_top":      _AUC_TOP,
            "mean_bottom":  _MEAN_BOTTOM,
            "mean_top":     _MEAN_TOP,
            "mean_neutral": _MEAN_NEUTRAL,
            # campos esperados por el runner genérico
            "max_posiciones":    None,   # inyectado por el runner desde config_local
            "commission_pct":    None,
            "slot_usdt_final":   None,
            "guardia_compra":    True,
            "guardia_venta":     True,
            "rsi_length":        "N/A",
            "ath_caida_maxima":  "N/A",
            "atl_subida_maxima": "N/A",
            "factor_caida":      "N/A",
            "factor_subida":     "N/A",
            "N":                 self.ventana,
        }


# ══════════════════════════════════════════════════════════════════════════════
# VARIANTE CON ZONA MUERTA EXPLÍCITA (opcional)
# ══════════════════════════════════════════════════════════════════════════════

class ClosePositionDeadZoneStrategy(ClosePositionStrategy):
    """
    Igual que ClosePositionStrategy pero con zona muerta configurable entre
    los dos umbrales. Útil cuando se quiere forzar un gap mínimo entre
    la zona de compra y la de venta para evitar señales en mercado lateral.

    Parámetro adicional:
    dead_zone   mínima diferencia entre umbral_top y umbral_bot         (default=0.5)
                Ejemplo: umbral_bot=0.25, dead_zone=0.50 → umbral_top mínimo=0.75

    Uso típico en mercado lateral (consolid. con baja volatilidad):
      strategy = ClosePositionDeadZoneStrategy(
          ventana    = 6,
          umbral_bot = 0.15,
          umbral_top = 0.85,
          dead_zone  = 0.70,   # zona segura muy amplia
          cooldown   = 96,
      )
    """

    def __init__(
        self,
        ventana:    int   = ClosePositionStrategy.DEFAULT_VENTANA,
        umbral_bot: float = 0.15,
        umbral_top: float = 0.85,
        cooldown:   int   = ClosePositionStrategy.DEFAULT_COOLDOWN,
        dead_zone:  float = 0.70,
    ) -> None:
        if (umbral_top - umbral_bot) < dead_zone:
            raise ValueError(
                f"La diferencia umbral_top - umbral_bot = {umbral_top - umbral_bot:.2f} "
                f"es menor que dead_zone = {dead_zone:.2f}. "
                f"Aumentar umbral_top o reducir umbral_bot."
            )
        # Relajar validación del padre para umbrales más conservadores
        # (llamamos al __init__ del abuelo directamente)
        BaseStrategy.__init__(self, name="ClosePosition-DeadZone-v1")

        self.ventana    = ventana
        self.umbral_bot = umbral_bot
        self.umbral_top = umbral_top
        self.cooldown   = cooldown
        self.dead_zone  = dead_zone

        from collections import deque
        self._buf = deque(maxlen=ventana)
        _NEG_INF = -(10 ** 9)
        self._last_bot_idx    = _NEG_INF
        self._last_top_idx    = _NEG_INF
        self.last_close_position = None
        self.last_signal_type    = "HOLD"
        self.last_cooldown_ok    = False

        log.info(
            "ClosePositionDeadZoneStrategy configurada",
            ventana    = ventana,
            umbral_bot = umbral_bot,
            umbral_top = umbral_top,
            dead_zone  = dead_zone,
            cooldown   = f"{cooldown}v" if cooldown else "off",
        )

    def describe(self) -> dict:
        d = super().describe()
        d["estrategia"] = self.name
        d["dead_zone"]  = self.dead_zone
        return d
