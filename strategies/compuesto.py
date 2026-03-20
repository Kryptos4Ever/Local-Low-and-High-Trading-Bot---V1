"""
compuesto.py — Estrategia de Señal Compuesta
═════════════════════════════════════════════
DNA de velas + Lyapunov + PE + Delta → Score 0-100 adaptativo.

Arquitectura en dos fases
──────────────────────────
  Fase PESADA (on_start):
    · Carga el dataset completo via PriceFeed
    · Calcula o carga desde cache: DNA, Lyapunov, HFD, PE, TE, RF, Score
    · Construye dos arrays: score_bot[N] y score_top[N]
    · Mapea timestamp → índice para lookup O(1) en on_candle

  Fase LIGERA (on_candle):
    · Lookup del score por timestamp
    · Emite BUY si score_bot[i] >= THR_BOT
    · Emite SELL si score_top[i] >= THR_TOP
    · Respeta cooldown entre señales del mismo tipo

Parámetros propios de la estrategia
──────────────────────────────────────
Viven dentro de esta clase.

Dependencias opcionales
────────────────────────
  numpy, pandas, scipy, scikit-learn — requeridos para el cómputo.
  Si no están disponibles, on_start() lanza ImportError con mensaje claro.
"""

from __future__ import annotations

import os
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from actors.price_feed        import Candle, PriceFeed
from actors.wallet            import Wallet
from strategies.base_strategy import BaseStrategy, Signal, SignalSide, HOLD
from support.logger           import get_logger

log = get_logger("compuesto")


class CompuestoStrategy(BaseStrategy):
    """
    Estrategia de señal compuesta: DNA + Lyapunov + Entropía de Permutación
    + Delta ratio → score adaptativo 0-100.

    Todos los parámetros tienen valores por defecto validados.
    Sobreescribir al instanciar:
        strat = CompuestoStrategy(thr_bot=70, cooldown=20)
    """

    # ── Parámetros de señal ────────────────────────────────────────────────────
    DEFAULT_THR_BOT          = 75.0
    DEFAULT_THR_TOP          = 75.0
    DEFAULT_COOLDOWN         = 16       # horas mínimas entre señales del mismo tipo
    DEFAULT_SUAVIZADO        = 6        # rolling mean sobre el score antes del dedup
    DEFAULT_VENTANA_SCORE    = 500      # ventana del percentil adaptativo

    # ── Parámetros DNA ─────────────────────────────────────────────────────────
    DEFAULT_VENTANA_DNA      = 48

    # ── Parámetros Lyapunov ────────────────────────────────────────────────────
    DEFAULT_TAU              = 4
    DEFAULT_DIM              = 5
    DEFAULT_W_LYAPUNOV       = 8
    DEFAULT_K_VECINOS        = 5
    DEFAULT_WIN_HFD          = 64
    DEFAULT_KMAX_HFD         = 8
    DEFAULT_WIN_LYAP_NORM    = 500

    # ── Parámetros PE ──────────────────────────────────────────────────────────
    DEFAULT_PE_ORDER         = 4
    DEFAULT_PE_DELAY         = 1
    DEFAULT_PE_VENTANA       = 64
    DEFAULT_WIN_PE_NORM      = 500
    DEFAULT_PE_PESOS         = (0.40, 0.30, 0.15, 0.15)  # close, delta, lwk, trade

    # ── Parámetros RF ──────────────────────────────────────────────────────────
    DEFAULT_RF_ESTIMATORS    = 300
    DEFAULT_RF_DEPTH         = 12
    DEFAULT_RF_MIN_SAMPLES   = 10
    DEFAULT_RF_CV_SPLITS     = 5
    DEFAULT_LABEL_ORDERS     = (6, 12, 24, 48, 96)
    DEFAULT_LABEL_MIN_SWING  = 0.015
    DEFAULT_NEUTROS_RATIO    = 3
    DEFAULT_LR_C             = 1.0

    def __init__(
        self,
        # Señal
        thr_bot:         float = DEFAULT_THR_BOT,
        thr_top:         float = DEFAULT_THR_TOP,
        cooldown:        int   = DEFAULT_COOLDOWN,
        suavizado:       int   = DEFAULT_SUAVIZADO,
        ventana_score:   int   = DEFAULT_VENTANA_SCORE,
        # Cache
        cache_dir:       str   = ".cache_compuesto",
        force_recompute: bool  = False,
    ) -> None:
        super().__init__(name="Compuesto-DNA+Lyapunov+PE+Delta")

        self.thr_bot       = thr_bot
        self.thr_top       = thr_top
        self.cooldown      = cooldown
        self.suavizado     = suavizado
        self.ventana_score = ventana_score
        self.cache_dir     = Path(cache_dir)
        self.force_recompute = force_recompute

        # Estado del score (se inicializa en on_start)
        self._score_bot:  Optional[np.ndarray] = None
        self._score_top:  Optional[np.ndarray] = None
        self._ts_to_idx:  Dict[int, int]       = {}   # timestamp → índice array

        # Control de cooldown
        self._last_bot_ts: int = 0
        self._last_top_ts: int = 0

        log.info(
            "CompuestoStrategy configurada",
            thr_bot=thr_bot, thr_top=thr_top,
            cooldown=cooldown, suavizado=suavizado,
            ventana_score=ventana_score,
        )

    # ══════════════════════════════════════════════════════════════════════════
    # INTERFAZ BaseStrategy
    # ══════════════════════════════════════════════════════════════════════════

    def on_start(self, wallet: Wallet, feed: PriceFeed = None,
                 start: str = None, end: str = None,
                 symbol: str = "BTCUSDT") -> None:
        """
        Carga o calcula el score compuesto sobre el dataset completo.
        Requiere acceso al PriceFeed para cargar todas las velas.

        feed, start, end son opcionales — si no se pasan se intenta
        cargar desde cache. Si no hay cache lanza ValueError.
        """
        log.info("CompuestoStrategy iniciando — cargando scores...")
        t0 = time.time()

        if feed is not None:
            candles = feed.get_candles(start or "2017-01-01",
                                       end   or "2030-01-01",
                                       symbol)
            self._compute_and_cache(candles)
        else:
            self._load_from_cache()

        log.info(
            "CompuestoStrategy lista",
            elapsed=f"{time.time()-t0:.1f}s",
            n=len(self._ts_to_idx),
        )

    def on_candle(self, candle: Candle, wallet: Wallet) -> Signal:
        """
        Lookup del score para esta vela y emisión de señal.
        O(1) — usa diccionario de timestamp → índice.
        """
        idx = self._ts_to_idx.get(candle.ts)
        if idx is None or self._score_bot is None:
            return HOLD

        sb = float(self._score_bot[idx])
        st = float(self._score_top[idx])

        # Cooldown en segundos (1 vela = 3600s)
        cooldown_s = self.cooldown * 3600

        if sb >= self.thr_bot and (candle.ts - self._last_bot_ts) >= cooldown_s:
            self._last_bot_ts = candle.ts
            return Signal(
                side   = SignalSide.BUY,
                price  = candle.close,
                reason = f"score_bot={sb:.1f}>={self.thr_bot}",
                score  = sb,
            )

        if st >= self.thr_top and (candle.ts - self._last_top_ts) >= cooldown_s:
            self._last_top_ts = candle.ts
            return Signal(
                side   = SignalSide.SELL,
                price  = candle.close,
                reason = f"score_top={st:.1f}>={self.thr_top}",
                score  = st,
            )

        return HOLD

    def on_stop(self, wallet: Wallet) -> None:
        log.info(
            "CompuestoStrategy detenida",
            velas_procesadas=self.candles_seen,
        )

    # ══════════════════════════════════════════════════════════════════════════
    # CÓMPUTO DEL PIPELINE
    # ══════════════════════════════════════════════════════════════════════════

    def _compute_and_cache(self, candles: List[Candle]) -> None:
        """Calcula el pipeline completo y cachea los resultados."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        close  = np.array([c.close   for c in candles], dtype=np.float64)
        high   = np.array([c.high    for c in candles], dtype=np.float64)
        low    = np.array([c.low     for c in candles], dtype=np.float64)
        open_  = np.array([c.open    for c in candles], dtype=np.float64)
        volume = np.array([c.volume  for c in candles], dtype=np.float64)
        taker  = np.array([c.taker_buy_base_vol or 0.0 for c in candles], dtype=np.float64)
        trades = np.array([c.trades_count or 0 for c in candles],         dtype=np.float64)
        ts_arr = np.array([c.ts for c in candles],                         dtype=np.int64)
        N      = len(candles)

        log.info(f"Pipeline compuesto: {N:,} velas")

        # ── DNA ────────────────────────────────────────────────────────────────
        dna = self._load_or_compute("dna", lambda: self._calc_dna(
            open_, high, low, close, volume, taker, trades, N))

        # ── Lyapunov + HFD ────────────────────────────────────────────────────
        lyap = self._load_or_compute("lyap", lambda: self._calc_lyapunov(close, N))
        _    = self._load_or_compute("hfd",  lambda: self._calc_hfd(close, N))

        # ── PE + TE ───────────────────────────────────────────────────────────
        pe_matrix = self._load_or_compute("pe_matrix", lambda: self._calc_pe(dna, close, N))
        te_arr    = self._load_or_compute("te_dc",     lambda: self._calc_te(dna, close, N))

        # ── RF ────────────────────────────────────────────────────────────────
        prob_bot, prob_top, labels = self._load_or_compute_triple(
            "prob_bot", "prob_top", "labels",
            lambda: self._calc_rf(close, dna, lyap,
                                  self._load_or_compute("hfd", lambda: None),
                                  pe_matrix, te_arr, N)
        )

        # ── Score compuesto ────────────────────────────────────────────────────
        sb, st = self._calc_score(
            dna, lyap, pe_matrix, te_arr,
            prob_bot, prob_top, labels, N
        )

        # Guardar scores finales
        np.save(self.cache_dir / "score_bot.npy", sb)
        np.save(self.cache_dir / "score_top.npy", st)

        self._score_bot = sb
        self._score_top = st
        self._ts_to_idx = {int(ts): i for i, ts in enumerate(ts_arr)}

    def _load_from_cache(self) -> None:
        """Carga scores desde cache. Lanza si no existen."""
        sb_path = self.cache_dir / "score_bot.npy"
        st_path = self.cache_dir / "score_top.npy"
        ts_path = self.cache_dir / "timestamps.npy"

        if not sb_path.exists() or not st_path.exists():
            raise FileNotFoundError(
                f"Cache de scores no encontrado en {self.cache_dir}. "
                "Pasar feed= a on_start() para computar desde cero."
            )

        self._score_bot = np.load(sb_path)
        self._score_top = np.load(st_path)

        if ts_path.exists():
            ts_arr = np.load(ts_path)
            self._ts_to_idx = {int(ts): i for i, ts in enumerate(ts_arr)}
        else:
            log.warning("cache sin timestamps.npy — señales pueden no alinearse")

        log.info("scores cargados desde cache",
                 n=len(self._score_bot), dir=str(self.cache_dir))

    def _load_or_compute(self, name: str, fn) -> np.ndarray:
        path = self.cache_dir / f"{name}.npy"
        if path.exists() and not self.force_recompute:
            arr = np.load(path)
            log.info(f"  cache hit: {name}", shape=arr.shape)
            return arr
        log.info(f"  calculando: {name}...")
        t0  = time.time()
        arr = fn()
        np.save(path, arr)
        log.info(f"  {name} listo", elapsed=f"{time.time()-t0:.1f}s")
        return arr

    def _load_or_compute_triple(self, n1, n2, n3, fn):
        p1 = self.cache_dir / f"{n1}.npy"
        p2 = self.cache_dir / f"{n2}.npy"
        p3 = self.cache_dir / f"{n3}.npy"
        if p1.exists() and p2.exists() and p3.exists() and not self.force_recompute:
            log.info(f"  cache hit: {n1}, {n2}, {n3}")
            return np.load(p1), np.load(p2), np.load(p3)
        log.info(f"  calculando: {n1}, {n2}, {n3}...")
        t0      = time.time()
        a1, a2, a3 = fn()
        for arr, n, p in [(a1,n1,p1),(a2,n2,p2),(a3,n3,p3)]:
            np.save(p, arr)
        log.info(f"  RF listo", elapsed=f"{time.time()-t0:.1f}s")
        return a1, a2, a3

    # ══════════════════════════════════════════════════════════════════════════
    # CÁLCULOS INTERNOS
    # ══════════════════════════════════════════════════════════════════════════

    def _calc_dna(self, open_, high, low, close, volume, taker, trades, N):
        from support.time_utils import to_epoch_s
        tr            = np.where(high - low == 0, 1e-9, high - low)
        body_ratio    = np.clip((close - open_) / tr, -1, 1)
        upper_wick    = np.clip((high - np.maximum(open_, close)) / tr, 0, 1)
        lower_wick    = np.clip((np.minimum(open_, close) - low)  / tr, 0, 1)
        delta_ratio   = np.clip(taker / (volume + 1e-9), 0, 1)
        roll_tr       = pd.Series(tr).rolling(self.DEFAULT_VENTANA_DNA, min_periods=1).mean().values
        range_rel     = np.clip(tr / (roll_tr + 1e-9), 0, 5)
        roll_tr2      = pd.Series(trades).rolling(self.DEFAULT_VENTANA_DNA, min_periods=1).mean().values
        trade_density = np.clip(trades / (roll_tr2 + 1e-9), 0, 5)
        dna = np.column_stack([body_ratio, upper_wick, lower_wick,
                                delta_ratio, range_rel, trade_density])
        return np.nan_to_num(dna.astype(np.float32))

    def _calc_lyapunov(self, close, N):
        from sklearn.neighbors import BallTree
        roll_m = pd.Series(close).rolling(200, min_periods=1).mean().values
        roll_s = pd.Series(close).rolling(200, min_periods=1).std().fillna(1).values
        cn     = (close - roll_m) / (roll_s + 1e-9)
        tau, dim = self.DEFAULT_TAU, self.DEFAULT_DIM
        lag = tau * (dim - 1)
        X   = np.column_stack([cn[i:N-lag+i] for i in range(0, lag+1, tau)])
        M   = X.shape[0]
        tree   = BallTree(X)
        lyap   = np.full(M, np.nan)
        W      = self.DEFAULT_W_LYAPUNOV
        BLOCK  = 2000
        for start in range(0, M - W, BLOCK):
            end_  = min(start + BLOCK, M - W)
            _, idxs = tree.query(X[start:end_], k=self.DEFAULT_K_VECINOS + 1)
            for bi, i in enumerate(range(start, end_)):
                nb    = idxs[bi, 1:]
                valid = nb[nb + W < M]
                if len(valid) == 0: continue
                d0 = np.linalg.norm(X[i]   - X[valid],   axis=1) + 1e-12
                dW = np.linalg.norm(X[i+W] - X[valid+W], axis=1) + 1e-12
                lyap[i] = np.mean(np.log(dW / d0)) / W
        lyap_full = np.full(N, np.nan)
        lyap_full[lag:lag+M] = lyap
        return pd.Series(lyap_full).ffill().bfill().values.astype(np.float32)

    def _calc_hfd(self, close, N):
        roll_m = pd.Series(close).rolling(200, min_periods=1).mean().values
        roll_s = pd.Series(close).rolling(200, min_periods=1).std().fillna(1).values
        cn     = (close - roll_m) / (roll_s + 1e-9)
        def higuchi(s):
            n, L = len(s), []
            for k in range(1, self.DEFAULT_KMAX_HFD + 1):
                Lk = []
                for m in range(1, k+1):
                    idx = np.arange(m-1, n, k)
                    if len(idx) < 2: continue
                    Lk.append(np.sum(np.abs(np.diff(s[idx]))) * (n-1) / ((len(idx)-1)*k*k))
                L.append(np.mean(Lk) if Lk else np.nan)
            L = np.array(L); ks = np.arange(1, self.DEFAULT_KMAX_HFD+1)
            v = ~np.isnan(L) & (L > 0)
            return np.polyfit(np.log(ks[v]), np.log(L[v]), 1)[0] if v.sum() >= 2 else np.nan
        hfd = np.full(N, np.nan)
        for i in range(self.DEFAULT_WIN_HFD, N):
            hfd[i] = higuchi(cn[i-self.DEFAULT_WIN_HFD:i])
        return pd.Series(hfd).ffill().bfill().values.astype(np.float32)

    def _calc_pe(self, dna, close, N):
        from itertools import permutations
        from math import log2
        order, delay, win = self.DEFAULT_PE_ORDER, self.DEFAULT_PE_DELAY, self.DEFAULT_PE_VENTANA
        perms = list(permutations(range(order)))
        p2i   = {p: i for i, p in enumerate(perms)}
        nfact = len(perms)
        step  = order * delay

        def pe_series(series):
            px = np.full(N, np.nan)
            for i in range(step, N):
                w = series[i-step:i:delay]
                px[i] = p2i[tuple(np.argsort(w).tolist())]
            pe = np.full(N, np.nan)
            for i in range(step + win, N):
                w     = px[i-win:i]
                valid = w[~np.isnan(w)].astype(int)
                if len(valid) < win // 2: continue
                cnt  = np.bincount(valid, minlength=nfact).astype(float)
                cnt /= cnt.sum()
                p    = cnt[cnt > 0]
                pe[i] = -np.sum(p * np.log2(p + 1e-12)) / log2(nfact)
            return pe

        channels = [close, dna[:,3], dna[:,2], dna[:,5]]
        result   = np.column_stack([pe_series(ch) for ch in channels])
        # Suavizar
        for j in range(result.shape[1]):
            result[:,j] = pd.Series(result[:,j]).ffill().bfill().rolling(12, min_periods=1).mean().values
        return result.astype(np.float32)

    def _calc_te(self, dna, close, N):
        from itertools import permutations
        from math import log2
        order, delay = self.DEFAULT_PE_ORDER, self.DEFAULT_PE_DELAY
        perms = list(permutations(range(order)))
        p2i   = {p: i for i, p in enumerate(perms)}
        nfact = len(perms)
        step  = order * delay
        WIN   = 48

        def idx_series(s):
            px = np.full(N, np.nan)
            for i in range(step, N):
                w = s[i-step:i:delay]
                px[i] = p2i[tuple(np.argsort(w).tolist())]
            return px

        px_d  = idx_series(dna[:,3])
        px_c  = idx_series(close)
        te    = np.full(N, np.nan)
        for i in range(step + WIN + 1, N):
            a = px_d[i-WIN:i]; b = px_c[i-WIN:i]; cf = px_c[i-WIN+1:i+1]
            m = min(len(a), len(b), len(cf))
            v = ~np.isnan(a[:m]) & ~np.isnan(b[:m]) & ~np.isnan(cf[:m])
            if v.sum() < WIN // 2: continue
            aa, bb, cc = a[:m][v].astype(int), b[:m][v].astype(int), cf[:m][v].astype(int)
            j3 = np.zeros((nfact, nfact, nfact)); j2 = np.zeros((nfact, nfact))
            for x, y, z in zip(aa, bb, cc):
                j3[x,y,z] += 1; j2[y,z] += 1
            j3 /= (j3.sum() + 1e-12); j2 /= (j2.sum() + 1e-12)
            tv = 0.0
            for x in range(nfact):
                for y in range(nfact):
                    for z in range(nfact):
                        p3 = j3[x,y,z]
                        if p3 < 1e-12: continue
                        p2  = j2[y,z]; pb = j2[y,:].sum()
                        if p2 < 1e-12 or pb < 1e-12: continue
                        tv += p3 * log2((p3 * pb) / (p2**2 + 1e-12) + 1e-12)
            te[i] = abs(tv)
        return pd.Series(te).ffill().bfill().values.astype(np.float32)

    def _calc_rf(self, close, dna, lyap, hfd, pe_matrix, te_arr, N):
        from scipy.signal import argrelextrema
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import TimeSeriesSplit

        # Labeling
        labels = np.zeros(N, dtype=int)
        for order in self.DEFAULT_LABEL_ORDERS:
            for bi in argrelextrema(close, np.less_equal, order=order)[0]:
                lo, hi = max(0,bi-order), min(N-1,bi+order)
                if (close[lo:hi].max() - close[bi]) / close[bi] > self.DEFAULT_LABEL_MIN_SWING:
                    labels[bi] = 1
            for ti in argrelextrema(close, np.greater_equal, order=order)[0]:
                lo, hi = max(0,ti-order), min(N-1,ti+order)
                if (close[ti] - close[lo:hi].min()) / close[ti] > self.DEFAULT_LABEL_MIN_SWING:
                    labels[ti] = -1
        y3 = np.where(labels==1,1,np.where(labels==-1,2,0))

        # Features
        dna_df = pd.DataFrame(dna, columns=[f'd{i}' for i in range(dna.shape[1])])
        feats  = [dna]
        for w in [12, 24, 48]:
            feats.append(dna_df.rolling(w, min_periods=1).mean().values)
            feats.append(dna_df.rolling(w, min_periods=1).std().fillna(0).values)
        for arr in [lyap, hfd if hfd is not None else lyap]:
            arr_c = pd.Series(arr).ffill().bfill().values
            feats.append(arr_c.reshape(-1,1))
            for w in [12, 24]:
                feats.append(pd.Series(arr_c).rolling(w,min_periods=1).mean().values.reshape(-1,1))
                feats.append(pd.Series(arr_c).rolling(w,min_periods=1).std().fillna(0).values.reshape(-1,1))
        pe_c = pd.DataFrame(pe_matrix).ffill().bfill().values
        te_c = pd.Series(te_arr).ffill().bfill().values
        feats.append(pe_c); feats.append(te_c.reshape(-1,1))
        X = np.column_stack([f if f.ndim==2 else f.reshape(-1,1) for f in feats])
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # Train
        START = 200
        scaler = StandardScaler().fit(X[START:])
        X_sc   = scaler.transform(X[START:])
        rng    = np.random.RandomState(42)
        bot_v  = np.where(y3[START:]==1)[0]
        top_v  = np.where(y3[START:]==2)[0]
        neu_v  = np.where(y3[START:]==0)[0]
        n_sig  = len(bot_v) + len(top_v)
        neu_s  = rng.choice(neu_v, min(n_sig*self.DEFAULT_NEUTROS_RATIO, len(neu_v)), replace=False)
        idx_b  = np.sort(np.concatenate([bot_v, top_v, neu_s]))
        X_bal  = X_sc[idx_b]; y_bal = y3[START:][idx_b]

        tscv   = TimeSeriesSplit(n_splits=self.DEFAULT_RF_CV_SPLITS)
        proba  = np.zeros((len(X[START:]), 3))
        for fold, (tr, te_) in enumerate(tscv.split(X_bal)):
            rf = RandomForestClassifier(
                n_estimators=self.DEFAULT_RF_ESTIMATORS,
                max_depth=self.DEFAULT_RF_DEPTH,
                min_samples_leaf=self.DEFAULT_RF_MIN_SAMPLES,
                class_weight='balanced', random_state=42, n_jobs=-1)
            rf.fit(X_bal[tr], y_bal[tr])
            proba += rf.predict_proba(X_sc)
            log.info(f"    RF fold {fold+1}/{self.DEFAULT_RF_CV_SPLITS} OK")
        proba /= self.DEFAULT_RF_CV_SPLITS

        pb = np.concatenate([np.full(START, 0.1), proba[:,1]])
        pt = np.concatenate([np.full(START, 0.1), proba[:,2]])
        lb = np.concatenate([np.zeros(START, dtype=int), y3[START:]])
        return pb.astype(np.float32), pt.astype(np.float32), lb.astype(np.int8)

    def _calc_score(self, dna, lyap, pe_matrix, te_arr,
                    prob_bot, prob_top, labels, N):
        from scipy.stats import percentileofscore
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import TimeSeriesSplit
        from scipy.ndimage import uniform_filter1d

        # Suavizar probabilities
        pb_sm = pd.Series(prob_bot).rolling(self.suavizado, min_periods=1).mean().values
        pt_sm = pd.Series(prob_top).rolling(self.suavizado, min_periods=1).mean().values

        # Lyapunov percentil
        lyap_c   = pd.Series(lyap).ffill().bfill().values
        lyap_pct = np.full(N, 0.5)
        for i in range(self.DEFAULT_WIN_LYAP_NORM, N):
            lyap_pct[i] = percentileofscore(lyap_c[i-self.DEFAULT_WIN_LYAP_NORM:i], lyap_c[i]) / 100.0
        lyap_pct = pd.Series(lyap_pct).ffill().bfill().values
        lyap_rev = np.abs(lyap_pct - 0.5) * 2
        lyap_bot = lyap_rev * (1 - lyap_pct)
        lyap_top = lyap_rev * lyap_pct

        # PE tensión
        w1, w2, w3, w4 = self.DEFAULT_PE_PESOS
        pe_comp = (w1*pe_matrix[:,0] + w2*pe_matrix[:,1] +
                   w3*pe_matrix[:,2] + w4*pe_matrix[:,3])
        pe_ten  = np.full(N, 0.5)
        for i in range(self.DEFAULT_WIN_PE_NORM, N):
            pe_ten[i] = 1.0 - percentileofscore(pe_comp[i-self.DEFAULT_WIN_PE_NORM:i], pe_comp[i]) / 100.0
        pe_ten = pd.Series(pe_ten).ffill().bfill().values

        # Delta divergencia
        delta  = pd.Series(dna[:,3]).rolling(12, min_periods=1).mean().values
        p_chg  = pd.Series(dna[:,0]).pct_change(12).fillna(0).values  # body como proxy
        d_chg  = pd.Series(delta).diff(12).fillna(0).values
        def rzsc(arr, w=200):
            s=pd.Series(arr); m=s.rolling(w,min_periods=1).mean()
            st_=s.rolling(w,min_periods=1).std().fillna(1).replace(0,1)
            return ((s-m)/st_).values
        pz=rzsc(p_chg); dz=rzsc(d_chg)
        def rp(arr, w=500):
            out=np.full(N, 0.5)
            for i in range(w, N):
                out[i]=percentileofscore(arr[i-w:i], arr[i])/100.0
            return out
        div_bot=rp(np.clip(-pz*dz,0,None)); div_top=rp(np.clip(pz*dz,0,None))

        # Morfología
        b6 =pd.Series(dna[:,0]).rolling(6, min_periods=1).mean().values
        b24=pd.Series(dna[:,0]).rolling(24,min_periods=1).mean().values
        morph_bot=rp(np.where(b24<-0.05,np.clip(b6-b24,0,None),0.0))
        morph_top=rp(np.where(b24> 0.05,np.clip(b24-b6,0,None),0.0))

        # Features compuestos
        y3 = np.where(labels==1,1,np.where(labels==-1,2,0))
        yb = (y3==1).astype(int); yt = (y3==2).astype(int)
        START = 600

        def build_X(pb, lb, pe, db, mb):
            ib=pb*pe; ilp=lb*pe; id_=pb*db; ipe=pe*db
            return np.column_stack([pb,lb,pe,db,mb,ib,ilp,id_,ipe])

        Xb = build_X(pb_sm, lyap_bot, pe_ten, div_bot, morph_bot)
        Xt = build_X(pt_sm, lyap_top, pe_ten, div_top, morph_top)

        sc_b = StandardScaler().fit(Xb[START:])
        sc_t = StandardScaler().fit(Xt[START:])

        def opt_w(Xsc, yv):
            ws=[]; tscv=TimeSeriesSplit(n_splits=self.DEFAULT_LR_C.__class__(5) if False else 5)
            for tr,te_ in tscv.split(Xsc):
                if len(np.unique(yv[tr]))<2 or (yv[tr]==1).sum()<3: continue
                try:
                    lr=LogisticRegression(C=self.DEFAULT_LR_C, class_weight='balanced',
                                          max_iter=1000, random_state=42)
                    lr.fit(Xsc[tr], yv[tr]); ws.append(lr.coef_[0])
                except: continue
            if not ws: return np.ones(Xsc.shape[1])/Xsc.shape[1]
            w=np.clip(np.mean(ws,axis=0),0,None); s=w.sum()
            return w/s if s>0 else np.ones(len(w))/len(w)

        wb = opt_w(sc_b.transform(Xb[START:]), yb[START:])
        wt = opt_w(sc_t.transform(Xt[START:]), yt[START:])

        raw_bot = uniform_filter1d(Xb @ wb, size=self.suavizado)
        raw_top = uniform_filter1d(Xt @ wt, size=self.suavizado)

        def adaptive_score(raw, w=None):
            w = w or self.ventana_score
            out = np.full(N, 50.0)
            for i in range(w, N):
                seg = raw[i-w:i]
                p10, p90 = np.percentile(seg, 10), np.percentile(seg, 90)
                out[i] = 50.0 if p90==p10 else float(np.clip((raw[i]-p10)/(p90-p10)*100,0,100))
            return out

        sb = adaptive_score(raw_bot)
        st = adaptive_score(raw_top)
        return sb.astype(np.float32), st.astype(np.float32)

    # ══════════════════════════════════════════════════════════════════════════
    # DESCRIBE
    # ══════════════════════════════════════════════════════════════════════════

    def describe(self) -> dict:
        return {
            "estrategia":     self.name,
            "thr_bot":        self.thr_bot,
            "thr_top":        self.thr_top,
            "cooldown_velas": self.cooldown,
            "suavizado":      self.suavizado,
            "ventana_score":  self.ventana_score,
            "N":              "adaptativo_score_compuesto",
            "rsi_length":     "N/A",
            "ath_caida_maxima": "N/A",
            "atl_subida_maxima": "N/A",
            "factor_caida":   "N/A",
            "factor_subida":  "N/A",
            "guardia_compra": True,
            "guardia_venta":  True,
        }
