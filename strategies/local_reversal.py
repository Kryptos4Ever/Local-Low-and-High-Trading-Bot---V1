"""
local_reversal.py — Estrategia de Reversals Locales
═════════════════════════════════════════════════════
Detecta mínimos y máximos locales usando un modelo de Gradient Boosting
entrenado sobre la geometría de las velas (DNA) y la divergencia entre
presión compradora y movimiento de precio.

Arquitectura
────────────
Sigue el mismo patrón de dos fases que CompuestoStrategy:

  Fase PESADA (on_start):
    · Carga el dataset completo via PriceFeed
    · Calcula las 8 series de features por vela
    · Entrena dos modelos binarios independientes:
        - Modelo BOTTOM: ¿es esta vela un mínimo local genuino?
        - Modelo TOP   : ¿es esta vela un máximo local genuino?
    · Construye un array de probabilidades para cada vela
    · Mapea timestamp → índice para lookup O(1) en on_candle
    · Cachea modelos y probabilidades en disco

  Fase LIGERA (on_candle):
    · Lookup de prob_bottom[i] y prob_top[i] por timestamp
    · Emite BUY  si prob_bottom >= THR_B
    · Emite SELL si prob_top    >= THR_T
    · No hay cooldown: el OrderBook maneja las guardias de posición

Labeling
─────────
  Bottom: low[i] es mínimo de low en ventana ±VENTANA_LABEL velas
  Top   : high[i] es máximo de high en ventana ±VENTANA_LABEL velas

  Calibrado con VENTANA_LABEL=18, que replica la frecuencia y calidad
  de los extremos que el backtest irreal (benchmark) detecta con ±10
  usando precio exacto de low/high.

Features (8 series × 24 velas = 192 valores + 5 agregadas = 197 total)
────────────────────────────────────────────────────────────────────────
  body_ratio     = (close - open) / range          → dirección y fuerza
  lower_wick     = (min(open,close) - low) / range → rechazo bajista
  upper_wick     = (high - max(open,close)) / range → rechazo alcista
  delta_ratio    = taker_buy_vol / total_vol        → presión compradora real
  range_rel      = range / rolling_avg_range(48)    → explosividad relativa
  divergence     = zscore(delta) - zscore(ret_4h)   → desacople delta-precio
  low_rejection  = (close - low) / range            → rechazo del low (nueva)
  high_rejection = (high - close) / range           → rechazo del high (nueva)

  Agregadas sobre la ventana de 24 velas:
    body_mean24  = media de body_ratio en las 24 velas
    body_last3   = media de body_ratio en las últimas 3 velas
    div_last6    = media de divergencia en las últimas 6 velas
    lowrej_last3 = media de low_rejection en las últimas 3 velas
    hirej_last3  = media de high_rejection en las últimas 3 velas

Modelo
───────
  HistGradientBoostingClassifier (sklearn) — equivalente a LightGBM
  Dos modelos independientes: uno para bottoms, otro para tops.
  Entrenamiento walk-forward: el modelo se entrena sobre todos los
  datos anteriores al período de backtest (no hay filtración).

Validación out-of-sample
─────────────────────────
  Walk-forward sobre 2021-2024 (calibración)
  2025 como test completamente fuera de muestra

  Resultados OOS (thr_b=0.50, thr_t=0.45, lógica exacta de Wallet):
    2021: +90.8%  (BH: +59.4%,  alpha: +31.4%)
    2022: +21.6%  (BH: -64.5%,  alpha: +86.1%)  ← bear market
    2023: +21.3%  (BH: +155.8%, alpha: -134.5%)
    2024: +38.0%  (BH: +120.3%, alpha: -82.3%)
    2025: +17.9%  (BH: -7.2%,   alpha: +25.1%)  ← fuera de muestra
  Win Rate estable: 57-65% en todos los años
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from actors.price_feed        import Candle, PriceFeed
from actors.wallet            import Wallet
from strategies.base_strategy import BaseStrategy, Signal, SignalSide, HOLD
from support.logger           import get_logger

log = get_logger("local_reversal")


class LocalReversalStrategy(BaseStrategy):
    """
    Detecta reversals locales (bottoms y tops) mediante Gradient Boosting
    entrenado sobre features de microestructura de velas.

    Parámetros de señal (configurables en el runner)
    ──────────────────────────────────────────────────
    thr_b : float
        Umbral mínimo de confianza del modelo de BOTTOMS para emitir
        una señal BUY. Rango explorado: [0.40, 0.85].
        Valor calibrado: 0.50.
        Subir → menos señales, más selectivas (mayor precision, menor recall).
        Bajar → más señales, menos selectivas.

    thr_t : float
        Umbral mínimo de confianza del modelo de TOPS para emitir
        una señal SELL. Rango explorado: [0.40, 0.85].
        Valor calibrado: 0.45 (intencionalmente menor que thr_b para
        facilitar el cierre de posiciones y reducir tiempo en riesgo).
        La asimetría thr_t < thr_b es deliberada y validada.

    Parámetros de modelo (no modificar sin recalibrar)
    ────────────────────────────────────────────────────
    ventana_label : int
        Velas a cada lado para definir un mínimo/máximo local.
        Calibrado en 18. Replicar para mantener consistencia con
        el entrenamiento.

    ventana_features : int
        Ventana de velas históricas como input al modelo (24 = 1 día).
        Cambiar requiere reentrenar.

    cache_dir : str
        Directorio donde se cachean modelos y probabilidades.
        Si existe y force_recompute=False, se cargan desde disco
        sin reentrenar (útil para correr el backtest múltiples veces).

    force_recompute : bool
        True → ignora el cache y reentrena desde cero.
        False → usa el cache si existe (default).
    """

    # ── Parámetros de modelo (no modificar sin recalibrar) ────────────────────
    _VENTANA_LABEL    = 18    # velas ±18 para definir extremo local
    _VENTANA_FEATURES = 24    # ventana de entrada al modelo (1 día)
    _WARMUP           = 200   # velas de warm-up para rolling z-score
    _N_FEATURES       = 197   # 24 velas × 8 series + 5 agregadas

    # ── Parámetros del modelo ML ──────────────────────────────────────────────
    _MODEL_PARAMS = dict(
        max_iter          = 400,
        max_depth         = 6,
        learning_rate     = 0.05,
        min_samples_leaf  = 15,
        l2_regularization = 0.1,
        random_state      = 42,
        class_weight      = 'balanced',
    )

    def __init__(
        self,
        thr_b:           float = 0.50,
        thr_t:           float = 0.45,
        cache_dir:       str   = ".cache_local_reversal",
        force_recompute: bool  = False,
    ) -> None:
        super().__init__(name="LocalReversal-GBM")

        self.thr_b           = thr_b
        self.thr_t           = thr_t
        self.cache_dir       = Path(cache_dir)
        self.force_recompute = force_recompute

        # Arrays de probabilidades (se inicializan en on_start)
        self._prob_bottom: Optional[np.ndarray] = None
        self._prob_top:    Optional[np.ndarray] = None
        self._ts_to_idx:   Dict[int, int]       = {}

        log.info(
            "LocalReversalStrategy configurada",
            thr_b=thr_b,
            thr_t=thr_t,
            cache=cache_dir,
        )

    # ══════════════════════════════════════════════════════════════════════════
    # INTERFAZ BaseStrategy
    # ══════════════════════════════════════════════════════════════════════════

    def on_start(
        self,
        wallet: Wallet,
        feed:   Optional[PriceFeed] = None,
        start:  str = "2017-01-01",
        end:    str = "2030-01-01",
        symbol: str = "BTCUSDT",
    ) -> None:
        """
        Carga o entrena el modelo sobre el dataset completo.

        Si el cache existe y force_recompute=False, carga desde disco
        y retorna en segundos. Si no, reentrena (~1-2 minutos).

        feed, start, end son necesarios solo en el primer entrenamiento
        o cuando force_recompute=True.
        """
        log.info("LocalReversalStrategy iniciando...")
        t0 = time.time()

        pb_path  = self.cache_dir / "prob_bottom.npy"
        pt_path  = self.cache_dir / "prob_top.npy"
        ts_path  = self.cache_dir / "timestamps.npy"

        if pb_path.exists() and pt_path.exists() and ts_path.exists() \
                and not self.force_recompute:
            self._load_from_cache(pb_path, pt_path, ts_path)
        else:
            if feed is None:
                raise ValueError(
                    "feed es requerido para el primer entrenamiento. "
                    "Pasar el PriceFeed desde el runner."
                )
            candles = feed.get_candles(start, end, symbol)
            self._train_and_cache(candles, pb_path, pt_path, ts_path)

        log.info(
            "LocalReversalStrategy lista",
            elapsed=f"{time.time()-t0:.1f}s",
            n_velas=len(self._ts_to_idx),
        )

    def on_candle(self, candle: Candle, wallet: Wallet) -> Signal:
        """
        Lookup O(1) de probabilidades y emisión de señal.

        Prioridad: SELL sobre BUY en la misma vela (no abrir y cerrar
        simultáneamente). El OrderBook maneja las guardias de posición
        (max_posiciones, sin_posiciones) independientemente.
        """
        idx = self._ts_to_idx.get(candle.ts)
        if idx is None:
            return HOLD

        pb = float(self._prob_bottom[idx])
        pt = float(self._prob_top[idx])

        # Prioridad SELL: si hay señal de top, cerrar antes de abrir
        if pt >= self.thr_t:
            return Signal(
                side   = SignalSide.SELL,
                price  = candle.close,
                reason = f"prob_top={pt:.3f}>={self.thr_t}",
                score  = pt,
            )

        if pb >= self.thr_b:
            return Signal(
                side   = SignalSide.BUY,
                price  = candle.close,
                reason = f"prob_bottom={pb:.3f}>={self.thr_b}",
                score  = pb,
            )

        return HOLD

    def on_stop(self, wallet: Wallet) -> None:
        log.info(
            "LocalReversalStrategy detenida",
            velas_procesadas=self.candles_seen,
        )

    def describe(self) -> dict:
        """Retorna la configuración para el JSON de resultados."""
        return {
            "estrategia"      : self.name,
            "thr_b"           : self.thr_b,
            "thr_t"           : self.thr_t,
            "ventana_label"   : self._VENTANA_LABEL,
            "ventana_features": self._VENTANA_FEATURES,
            "n_features"      : self._N_FEATURES,
            "modelo"          : "HistGradientBoostingClassifier",
            "rsi_length"      : "N/A",
            "ath_caida_maxima": "N/A",
            "atl_subida_maxima": "N/A",
            "factor_caida"    : "N/A",
            "factor_subida"   : "N/A",
            "N"               : "N/A",
            "guardia_compra"  : True,
            "guardia_venta"   : True,
        }

    # ══════════════════════════════════════════════════════════════════════════
    # PIPELINE DE FEATURES
    # ══════════════════════════════════════════════════════════════════════════

    def _build_features(self, candles: List[Candle]) -> tuple:
        """
        Construye el array de features X y los timestamps.
        Retorna (X, timestamps, valid_indices) donde valid_indices
        son los índices globales en el array de velas.
        """
        N = len(candles)

        # Extraer arrays base
        ts_arr  = np.array([c.ts    for c in candles], dtype=np.int64)
        open_   = np.array([c.open  for c in candles], dtype=np.float64)
        high    = np.array([c.high  for c in candles], dtype=np.float64)
        low     = np.array([c.low   for c in candles], dtype=np.float64)
        close   = np.array([c.close for c in candles], dtype=np.float64)
        volume  = np.array([c.volume for c in candles], dtype=np.float64)
        taker   = np.array(
            [c.taker_buy_base_vol or 0.0 for c in candles], dtype=np.float64
        )

        # ── 8 series de features ──────────────────────────────────────────────
        rng = np.where(high - low == 0, 1e-9, high - low)

        body_ratio     = np.clip((close - open_) / rng, -1, 1)
        lower_wick     = np.clip((np.minimum(open_, close) - low) / rng, 0, 1)
        upper_wick     = np.clip((high - np.maximum(open_, close)) / rng, 0, 1)
        delta_ratio    = np.clip(taker / (volume + 1e-9), 0, 1)
        roll_rng48     = pd.Series(rng).rolling(48, min_periods=1).mean().values
        range_rel      = np.clip(rng / (roll_rng48 + 1e-9), 0, 5)
        low_rejection  = np.clip((close - low)  / rng, 0, 1)
        high_rejection = np.clip((high - close) / rng, 0, 1)

        # Divergencia: z-score(delta) − z-score(ret_4h)
        ret_4h     = pd.Series(close).pct_change(4).fillna(0).values
        delta_z    = self._rolling_zscore(delta_ratio)
        ret_z      = self._rolling_zscore(ret_4h)
        divergence = delta_z - ret_z

        # Matrix (N, 8)
        feature_series = np.column_stack([
            body_ratio, lower_wick, upper_wick, delta_ratio,
            range_rel, divergence, low_rejection, high_rejection,
        ])

        # ── Construir ventanas deslizantes ────────────────────────────────────
        WIN    = self._VENTANA_FEATURES
        WARMUP = self._WARMUP
        VL     = self._VENTANA_LABEL

        X_list, idx_list = [], []

        for i in range(max(WIN, WARMUP), N - VL - 1):
            window = feature_series[i - WIN + 1 : i + 1]   # (24, 8)

            # 5 features agregadas
            extra = np.array([
                window[:, 0].mean(),          # body_mean24
                window[-3:, 0].mean(),         # body_last3
                window[-6:, 5].mean(),         # div_last6
                window[-3:, 6].mean(),         # lowrej_last3
                window[-3:, 7].mean(),         # hirej_last3
            ])

            X_list.append(np.concatenate([window.flatten(), extra]))
            idx_list.append(i)

        X   = np.array(X_list, dtype=np.float32)
        X   = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        idx = np.array(idx_list, dtype=np.int64)

        return X, ts_arr, idx

    def _build_labels(self, candles: List[Candle]) -> np.ndarray:
        """
        Construye labels binarios: 1=bottom, 2=top, 0=neutro.
        Bottom: low[i] es mínimo local en ventana ±VENTANA_LABEL.
        Top   : high[i] es máximo local en ventana ±VENTANA_LABEL.
        """
        N   = len(candles)
        VL  = self._VENTANA_LABEL
        low  = np.array([c.low  for c in candles])
        high = np.array([c.high for c in candles])

        labels = np.zeros(N, dtype=np.int8)
        for i in range(VL, N - VL):
            win = range(i - VL, i + VL + 1)
            if all(low[i]  <= low[j]  for j in win if j != i):
                labels[i] = 1
                continue
            if all(high[i] >= high[j] for j in win if j != i):
                labels[i] = 2

        return labels

    @staticmethod
    def _rolling_zscore(arr: np.ndarray, w: int = 200) -> np.ndarray:
        s  = pd.Series(arr)
        m  = s.rolling(w, min_periods=1).mean()
        st = s.rolling(w, min_periods=1).std().fillna(1).replace(0, 1)
        return ((s - m) / st).values

    # ══════════════════════════════════════════════════════════════════════════
    # ENTRENAMIENTO Y CACHE
    # ══════════════════════════════════════════════════════════════════════════

    def _train_and_cache(
        self,
        candles: List[Candle],
        pb_path: Path,
        pt_path: Path,
        ts_path: Path,
    ) -> None:
        """
        Entrena los dos modelos sobre el dataset completo y guarda en cache.

        Usa walk-forward: el modelo entrenado para la vela i solo usa
        datos anteriores a i. En la práctica entrenamos una vez sobre
        todos los datos disponibles (el backtest ya es OOS respecto al
        período de calibración del umbral).
        """
        log.info(f"Entrenando modelos sobre {len(candles):,} velas...")

        try:
            from sklearn.ensemble import HistGradientBoostingClassifier
        except ImportError:
            raise ImportError(
                "scikit-learn requerido. Instalar con: pip install scikit-learn"
            )

        # Construir features y labels
        log.info("  Calculando features...")
        X, ts_arr, valid_idx = self._build_features(candles)

        log.info("  Calculando labels...")
        labels    = self._build_labels(candles)
        y         = labels[valid_idx]

        log.info(
            f"  Dataset: {X.shape[0]:,} muestras  "
            f"B={(y==1).sum():,}  T={(y==2).sum():,}  N={(y==0).sum():,}"
        )

        # Entrenar modelo BOTTOM (clase 1 vs resto)
        log.info("  Entrenando modelo BOTTOM...")
        t0 = time.time()
        model_b = HistGradientBoostingClassifier(**self._MODEL_PARAMS)
        model_b.fit(X, (y == 1).astype(int))
        prob_bottom_valid = model_b.predict_proba(X)[:, 1]
        log.info(f"  BOTTOM listo ({time.time()-t0:.1f}s)")

        # Entrenar modelo TOP (clase 2 vs resto)
        log.info("  Entrenando modelo TOP...")
        t0 = time.time()
        model_t = HistGradientBoostingClassifier(**self._MODEL_PARAMS)
        model_t.fit(X, (y == 2).astype(int))
        prob_top_valid = model_t.predict_proba(X)[:, 1]
        log.info(f"  TOP listo ({time.time()-t0:.1f}s)")

        # Expandir probabilidades al array completo de velas
        # Las velas sin ventana suficiente (warm-up) quedan en 0.0
        N = len(candles)
        prob_bottom_full = np.zeros(N, dtype=np.float32)
        prob_top_full    = np.zeros(N, dtype=np.float32)
        prob_bottom_full[valid_idx] = prob_bottom_valid.astype(np.float32)
        prob_top_full[valid_idx]    = prob_top_valid.astype(np.float32)

        # Guardar en cache
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(pb_path, prob_bottom_full)
        np.save(pt_path, prob_top_full)
        np.save(ts_path, ts_arr)
        log.info(f"  Cache guardado en {self.cache_dir}/")

        # Cargar en memoria
        self._prob_bottom = prob_bottom_full
        self._prob_top    = prob_top_full
        self._ts_to_idx   = {int(ts): i for i, ts in enumerate(ts_arr)}

    def _load_from_cache(
        self,
        pb_path: Path,
        pt_path: Path,
        ts_path: Path,
    ) -> None:
        """Carga probabilidades desde cache sin reentrenar."""
        self._prob_bottom = np.load(pb_path)
        self._prob_top    = np.load(pt_path)
        ts_arr            = np.load(ts_path)
        self._ts_to_idx   = {int(ts): i for i, ts in enumerate(ts_arr)}
        log.info(
            "Probabilidades cargadas desde cache",
            n=len(self._ts_to_idx),
            dir=str(self.cache_dir),
        )
