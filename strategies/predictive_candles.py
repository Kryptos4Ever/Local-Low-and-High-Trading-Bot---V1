"""
strategies/predictive_candles.py — PredictiveCandles Strategy
══════════════════════════════════════════════════════════════
Score ponderado por AUC sobre los factores técnicos con mayor poder
predictivo sobre turning points reales (análisis factorial sobre oráculo
perfecto IrrealStrategy, período 2018-2025, 2059 bottoms / 2025 tops).

Predictores calibrados
───────────────────────
  BOTTOM → señal BUY:
    close_position   AUC=0.854  ▼  (close - min_low) / (max_high - min_low)
    bb_position      AUC=0.833  ▼  Bollinger %B: (close - lower) / (upper - lower)
    recovery_pct     AUC=0.823  ▼  (close - min_low) / min_low × 100

  TOP → señal SELL:
    close_position   AUC=0.839  ▲  mismo cálculo, dirección opuesta
    drawdown_pct     AUC=0.826  ▼  (max_high - close) / max_high × 100
    bb_position      AUC=0.821  ▲  mismo cálculo, dirección opuesta

Score ponderado por AUC
─────────────────────────
  peso_i = AUC_i / Σ AUC_j  (solo predictores activos, renormalizado)
  contrib_i = (1 - norm_i) si dirección LOW, norm_i si dirección HIGH
  score = Σ peso_i × contrib_i

Cooldown
─────────
  cooldown_bot / cooldown_top: velas mínimas entre señales del mismo tipo.
  0 = desactivado. El contador se reinicia cuando el score supera el umbral
  y se emite la señal, independientemente de si el runner la ejecuta o no.

Atributos expuestos tras cada on_candle (para el runner y el --grid):
  last_pred_values       dict con los 4 valores crudos (None si no calculado)
  last_score_bot         score BOT de la última vela procesada
  last_score_top         score TOP de la última vela procesada
  last_cooldown_ok_bot   bool: cooldown BOT cumplido en la última vela
  last_cooldown_ok_top   bool: cooldown TOP cumplido en la última vela
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Dict, FrozenSet, List, Optional

from actors.price_feed        import Candle
from actors.wallet            import Wallet
from strategies.base_strategy import BaseStrategy, Signal, SignalSide, HOLD
from support.logger           import get_logger

log = get_logger("predictive_candles")


# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTES DEL ANÁLISIS FACTORIAL
# ══════════════════════════════════════════════════════════════════════════════

AUC_BOT: Dict[str, float] = {
    "close_position": 0.8541,
    "bb_position":    0.8326,
    "recovery_pct":   0.8233,
}

AUC_TOP: Dict[str, float] = {
    "close_position": 0.8388,
    "drawdown_pct":   0.8257,
    "bb_position":    0.8212,
}

DIR_BOT: Dict[str, str] = {
    "close_position": "LOW",
    "bb_position":    "LOW",
    "recovery_pct":   "LOW",
}

DIR_TOP: Dict[str, str] = {
    "close_position": "HIGH",
    "drawdown_pct":   "LOW",
    "bb_position":    "HIGH",
}

ALL_BOT_PREDICTORS: List[str] = list(AUC_BOT.keys())
ALL_TOP_PREDICTORS: List[str] = list(AUC_TOP.keys())


# ══════════════════════════════════════════════════════════════════════════════
# FUNCIONES AUXILIARES (exportadas para el --grid)
# ══════════════════════════════════════════════════════════════════════════════

def norm_weights(active: List[str], auc: Dict[str, float]) -> Dict[str, float]:
    """Pesos normalizados por AUC. Garantiza Σ = 1.0 siempre."""
    if not active:
        return {}
    total = sum(auc[k] for k in active)
    return ({k: auc[k] / total for k in active}
            if total > 0 else {k: 1.0 / len(active) for k in active})


def pct_rank(value: float, history: List[float]) -> float:
    """Rango percentil de value dentro de history. Retorna [0, 1]."""
    if not history:
        return 0.5
    return sum(1 for v in history if v < value) / len(history)


# ══════════════════════════════════════════════════════════════════════════════
# ESTRATEGIA
# ══════════════════════════════════════════════════════════════════════════════

class PredictiveCandlesStrategy(BaseStrategy):
    """
    Score ponderado por AUC sobre predictores de BOTTOM y TOP con cooldown.

    Parámetros
    ──────────
    ventana                 velas hacia atrás para calcular factores     (default=10)
    umbral_bot              score_bot mínimo para emitir BUY             (default=0.5)
    umbral_top              score_top mínimo para emitir SELL            (default=0.5)
    use_bot_close_position  activa close_position para BOTTOM
    use_bot_bb_position     activa bb_position para BOTTOM
    use_bot_recovery_pct    activa recovery_pct para BOTTOM
    use_top_close_position  activa close_position para TOP
    use_top_drawdown_pct    activa drawdown_pct para TOP
    use_top_bb_position     activa bb_position para TOP
    cooldown_bot            velas mínimas entre señales BUY  (0=desactivado)
    cooldown_top            velas mínimas entre señales SELL (0=desactivado)
    n_norm                  historia rolling para percentil recovery/drawdown
    """

    def __init__(
        self,
        ventana:                int   = 10,
        umbral_bot:             float = 0.5,
        umbral_top:             float = 0.5,
        use_bot_close_position: bool  = True,
        use_bot_bb_position:    bool  = True,
        use_bot_recovery_pct:   bool  = True,
        use_top_close_position: bool  = True,
        use_top_drawdown_pct:   bool  = True,
        use_top_bb_position:    bool  = True,
        cooldown_bot:           int   = 0,
        cooldown_top:           int   = 0,
        n_norm:                 int   = 200,
    ) -> None:
        super().__init__(name="PredictiveCandles")

        if ventana < 2:
            raise ValueError(f"ventana >= 2 requerido, got {ventana}")
        if not (0.0 < umbral_bot <= 1.0):
            raise ValueError(f"umbral_bot en (0,1], got {umbral_bot}")
        if not (0.0 < umbral_top <= 1.0):
            raise ValueError(f"umbral_top en (0,1], got {umbral_top}")
        if cooldown_bot < 0:
            raise ValueError(f"cooldown_bot >= 0, got {cooldown_bot}")
        if cooldown_top < 0:
            raise ValueError(f"cooldown_top >= 0, got {cooldown_top}")

        self.ventana      = ventana
        self.umbral_bot   = umbral_bot
        self.umbral_top   = umbral_top
        self.cooldown_bot = cooldown_bot
        self.cooldown_top = cooldown_top
        self.n_norm       = n_norm

        self._active_bot: List[str] = [
            k for k, flag in {
                "close_position": use_bot_close_position,
                "bb_position":    use_bot_bb_position,
                "recovery_pct":   use_bot_recovery_pct,
            }.items() if flag
        ]
        self._active_top: List[str] = [
            k for k, flag in {
                "close_position": use_top_close_position,
                "drawdown_pct":   use_top_drawdown_pct,
                "bb_position":    use_top_bb_position,
            }.items() if flag
        ]

        if not self._active_bot and not self._active_top:
            raise ValueError("Al menos un predictor debe estar activo")

        self._factors_needed: FrozenSet[str] = frozenset(
            self._active_bot + self._active_top
        )
        self._w_bot = norm_weights(self._active_bot, AUC_BOT)
        self._w_top = norm_weights(self._active_top, AUC_TOP)

        self._buf:   Deque[Candle] = deque(maxlen=ventana)
        self._h_rec: List[float]   = []
        self._h_dra: List[float]   = []

        # índice de vela de la última señal emitida (−∞ para que el primer
        # tick siempre pase el cooldown)
        _NEG_INF = -(10 ** 9)
        self._last_bot_idx: int = _NEG_INF
        self._last_top_idx: int = _NEG_INF

        # Atributos públicos para el runner
        self.last_pred_values:     Dict[str, Optional[float]] = {
            "close_position": None,
            "bb_position":    None,
            "recovery_pct":   None,
            "drawdown_pct":   None,
        }
        self.last_score_bot:       float = 0.0
        self.last_score_top:       float = 0.0
        self.last_cooldown_ok_bot: bool  = False
        self.last_cooldown_ok_top: bool  = False

        log.info(
            "PredictiveCandlesStrategy configurada",
            ventana=ventana,
            umbral_bot=umbral_bot, umbral_top=umbral_top,
            cooldown_bot=f"{cooldown_bot}v" if cooldown_bot else "off",
            cooldown_top=f"{cooldown_top}v" if cooldown_top else "off",
            active_bot=self._active_bot,
            active_top=self._active_top,
        )

    # ── Interfaz BaseStrategy ─────────────────────────────────────────────────

    def on_start(self, wallet: Wallet, **kwargs) -> None:
        self._buf.clear()
        self._h_rec.clear()
        self._h_dra.clear()
        _NEG_INF = -(10 ** 9)
        self._last_bot_idx = _NEG_INF
        self._last_top_idx = _NEG_INF
        self.last_pred_values = {k: None for k in self.last_pred_values}
        self.last_score_bot = self.last_score_top = 0.0
        self.last_cooldown_ok_bot = self.last_cooldown_ok_top = False
        log.info("PredictiveCandlesStrategy iniciada",
                 active_bot=self._active_bot, active_top=self._active_top)

    def on_candle(self, candle: Candle, wallet: Wallet) -> Signal:
        self._buf.append(candle)
        # _candles_seen ya fue incrementado por _tick() antes de llamar on_candle
        current_idx = self._candles_seen

        if len(self._buf) < self.ventana:
            self.last_pred_values = {k: None for k in self.last_pred_values}
            self.last_score_bot = self.last_score_top = 0.0
            self.last_cooldown_ok_bot = self.last_cooldown_ok_top = False
            return HOLD

        # ── Factores crudos ───────────────────────────────────────────────────
        pv = self._raw_factors(list(self._buf))

        # ── Actualizar historias de normalización ─────────────────────────────
        if pv["recovery_pct"] is not None:
            self._h_rec.append(pv["recovery_pct"])
            if len(self._h_rec) > self.n_norm:
                self._h_rec.pop(0)
        if pv["drawdown_pct"] is not None:
            self._h_dra.append(pv["drawdown_pct"])
            if len(self._h_dra) > self.n_norm:
                self._h_dra.pop(0)

        # ── Scores ────────────────────────────────────────────────────────────
        sb = (self._event_score(
                  self._active_bot, self._w_bot, DIR_BOT,
                  pv, self._h_rec, self._h_dra)
              if self._active_bot else 0.0)
        st = (self._event_score(
                  self._active_top, self._w_top, DIR_TOP,
                  pv, self._h_rec, self._h_dra)
              if self._active_top else 0.0)

        # ── Cooldown ──────────────────────────────────────────────────────────
        cd_ok_bot = (self.cooldown_bot == 0 or
                     (current_idx - self._last_bot_idx) >= self.cooldown_bot)
        cd_ok_top = (self.cooldown_top == 0 or
                     (current_idx - self._last_top_idx) >= self.cooldown_top)

        # ── Exponer atributos públicos ────────────────────────────────────────
        self.last_pred_values     = pv
        self.last_score_bot       = sb
        self.last_score_top       = st
        self.last_cooldown_ok_bot = cd_ok_bot
        self.last_cooldown_ok_top = cd_ok_top

        # ── Señal (SELL prioridad sobre BUY) ──────────────────────────────────
        if self._active_top and st >= self.umbral_top and cd_ok_top:
            self._last_top_idx = current_idx
            return Signal(
                side   = SignalSide.SELL,
                price  = candle.close,
                reason = (f"score_top={st:.3f}>={self.umbral_top}"
                          + (f" cd={self.cooldown_top}v" if self.cooldown_top else "")),
                score  = st,
            )

        if self._active_bot and sb >= self.umbral_bot and cd_ok_bot:
            self._last_bot_idx = current_idx
            return Signal(
                side   = SignalSide.BUY,
                price  = candle.close,
                reason = (f"score_bot={sb:.3f}>={self.umbral_bot}"
                          + (f" cd={self.cooldown_bot}v" if self.cooldown_bot else "")),
                score  = sb,
            )

        return HOLD

    def on_stop(self, wallet: Wallet) -> None:
        log.info("PredictiveCandlesStrategy detenida",
                 velas_procesadas=self.candles_seen)

    def describe(self) -> dict:
        return {
            "estrategia":        self.name,
            "ventana":           self.ventana,
            "umbral_bot":        self.umbral_bot,
            "umbral_top":        self.umbral_top,
            "cooldown_bot":      self.cooldown_bot,
            "cooldown_top":      self.cooldown_top,
            "active_bot":        self._active_bot,
            "active_top":        self._active_top,
            "weights_bot":       {k: round(v, 4) for k, v in self._w_bot.items()},
            "weights_top":       {k: round(v, 4) for k, v in self._w_top.items()},
            "n_norm":            self.n_norm,
            "rsi_length":        "N/A",
            "ath_caida_maxima":  "N/A",
            "atl_subida_maxima": "N/A",
            "factor_caida":      "N/A",
            "factor_subida":     "N/A",
            "N":                 self.ventana,
            "guardia_compra":    True,
            "guardia_venta":     True,
        }

    # ── Factores ──────────────────────────────────────────────────────────────

    def _raw_factors(self, window: List[Candle]) -> Dict[str, Optional[float]]:
        """Calcula solo los factores requeridos por los predictores activos."""
        closes = [c.close for c in window]
        highs  = [c.high  for c in window]
        lows   = [c.low   for c in window]
        last   = closes[-1]
        max_h  = max(highs)
        min_l  = min(lows)
        rng    = max_h - min_l

        pv: Dict[str, Optional[float]] = {
            "close_position": None,
            "bb_position":    None,
            "recovery_pct":   None,
            "drawdown_pct":   None,
        }

        if "close_position" in self._factors_needed:
            pv["close_position"] = round(
                (last - min_l) / rng if rng > 0 else 0.5, 4)

        if "bb_position" in self._factors_needed:
            n = len(closes); mc = sum(closes) / n
            var = sum((c - mc) ** 2 for c in closes) / n
            std = var ** 0.5 if var > 0 else 0.0
            bw  = 4.0 * std
            pv["bb_position"] = round(
                max(0.0, min(1.0, (last - (mc - 2.0 * std)) / bw)) if bw > 0 else 0.5, 4)

        if "recovery_pct" in self._factors_needed:
            pv["recovery_pct"] = round(
                (last - min_l) / min_l * 100.0 if min_l > 0 else 0.0, 4)

        if "drawdown_pct" in self._factors_needed:
            pv["drawdown_pct"] = round(
                (max_h - last) / max_h * 100.0 if max_h > 0 else 0.0, 4)

        return pv

    @staticmethod
    def _event_score(
        active:  List[str],
        weights: Dict[str, float],
        dirs:    Dict[str, str],
        pv:      Dict[str, Optional[float]],
        h_rec:   List[float],
        h_dra:   List[float],
    ) -> float:
        """Score ponderado por AUC con renormalización si falta algún predictor."""
        score = w_total = 0.0
        for pred in active:
            raw = pv.get(pred)
            if raw is None:
                continue
            w = weights.get(pred, 0.0)
            if pred in ("close_position", "bb_position"):
                norm = raw
            elif pred == "recovery_pct":
                norm = pct_rank(raw, h_rec) if h_rec else 0.5
            else:
                norm = pct_rank(raw, h_dra) if h_dra else 0.5
            score   += w * ((1.0 - norm) if dirs[pred] == "LOW" else norm)
            w_total += w
        if w_total <= 0:
            return 0.0
        return round(score / w_total if abs(w_total - 1.0) > 1e-9 else score, 4)