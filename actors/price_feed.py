"""
price_feed.py — Actor 1: Fuente de precios
═══════════════════════════════════════════
Responsabilidad única: entregar velas OHLCV + datos de taker al sistema.

Interfaz abstracta PriceFeed
─────────────────────────────
  get_candles(start, end)  →  List[Candle]   # modo backtest / histórico
  subscribe(callback)      →  None            # modo live (stream de velas)

Implementaciones locales (simulación)
──────────────────────────────────────
  SQLiteFeed   →  lee desde btc_hourly.db  (configurado en config_local)
  CSVFeed      →  lee desde archivo .csv

Implementaciones producción (a codificar en etapa de producción)
─────────────────────────────────────────────────────────────────
  BinanceRESTFeed  →  histórico via REST API
  BinanceWSFeed    →  stream en tiempo real via WebSocket

Candle (dataclass)
───────────────────
Tipo interno canónico. Todos los actores y estrategias hablan Candle.
Los feeds convierten en sus propios bordes desde el formato de la fuente.
Timestamps siempre en UTC epoch seconds (int) via time_utils.
"""

from __future__ import annotations

import csv
import sqlite3
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Iterator, List, Optional

from support.logger    import get_logger
from support.time_utils import to_epoch_s, to_iso, TimeInput

log = get_logger("price_feed")


# ══════════════════════════════════════════════════════════════════════════════
# TIPO CANÓNICO: Candle
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class Candle:
    """
    Vela OHLCV canónica del sistema.

    Campos obligatorios: los 6 primeros.
    Campos opcionales: los datos de taker de Binance — presentes en SQLite
    pero ausentes en fuentes genéricas como CSV o exchanges alternativos.

    ts:   UTC epoch seconds (int) — timestamp de APERTURA de la vela
    """
    ts:                    int
    open:                  float
    high:                  float
    low:                   float
    close:                 float
    volume:                float

    # Datos de microestructura (Binance) — None si la fuente no los provee
    taker_buy_base_vol:    Optional[float] = field(default=None)
    taker_buy_quote_vol:   Optional[float] = field(default=None)
    quote_volume:          Optional[float] = field(default=None)
    trades_count:          Optional[int]   = field(default=None)

    # ── Propiedades calculadas ────────────────────────────────────────────────

    @property
    def delta_ratio(self) -> Optional[float]:
        """
        Presión compradora: taker_buy_base_vol / volume.
        None si los datos de taker no están disponibles.
        Clave para el método ④ (entropía de permutación multivariada).
        """
        if self.taker_buy_base_vol is None or self.volume == 0:
            return None
        return self.taker_buy_base_vol / self.volume

    @property
    def body(self) -> float:
        """Cuerpo de la vela: close - open."""
        return self.close - self.open

    @property
    def total_range(self) -> float:
        """Rango total: high - low."""
        return self.high - self.low

    @property
    def body_ratio(self) -> Optional[float]:
        """
        Ratio del cuerpo sobre el rango total [-1, 1].
        None si total_range es 0 (vela plana).
        """
        if self.total_range == 0:
            return None
        return self.body / self.total_range

    @property
    def upper_wick_ratio(self) -> Optional[float]:
        """Ratio de la mecha superior sobre el rango total [0, 1]."""
        if self.total_range == 0:
            return None
        top = max(self.open, self.close)
        return (self.high - top) / self.total_range

    @property
    def lower_wick_ratio(self) -> Optional[float]:
        """Ratio de la mecha inferior sobre el rango total [0, 1]."""
        if self.total_range == 0:
            return None
        bottom = min(self.open, self.close)
        return (bottom - self.low) / self.total_range

    def iso(self) -> str:
        """Timestamp como string ISO 8601 UTC."""
        return to_iso(self.ts)

    def __repr__(self) -> str:
        return (
            f"Candle({self.iso()}  "
            f"O={self.open:.2f} H={self.high:.2f} "
            f"L={self.low:.2f} C={self.close:.2f}  "
            f"vol={self.volume:.4f})"
        )


# ══════════════════════════════════════════════════════════════════════════════
# INTERFAZ ABSTRACTA
# ══════════════════════════════════════════════════════════════════════════════

class PriceFeed(ABC):
    """
    Contrato que deben cumplir todas las implementaciones de fuente de precios.
    La estrategia solo conoce esta interfaz — nunca las implementaciones.
    """

    @abstractmethod
    def get_candles(
        self,
        start: TimeInput,
        end:   TimeInput,
        symbol: str = "BTCUSDT",
    ) -> List[Candle]:
        """
        Retorna lista de velas en el rango [start, end] inclusive,
        ordenadas cronológicamente (más antigua primero).

        start / end: cualquier formato aceptado por time_utils.to_epoch_s().
        """

    @abstractmethod
    def subscribe(
        self,
        callback: Callable[[Candle], None],
        symbol:   str = "BTCUSDT",
    ) -> None:
        """
        Suscribe un callback que se llama cada vez que cierra una nueva vela.
        Modo live — bloquea o corre en background según la implementación.
        """

    def iter_candles(
        self,
        start:  TimeInput,
        end:    TimeInput,
        symbol: str = "BTCUSDT",
    ) -> Iterator[Candle]:
        """
        Iterador conveniente sobre get_candles().
        Útil en runners de backtest para procesar vela a vela sin cargar todo.
        """
        for candle in self.get_candles(start, end, symbol):
            yield candle


# ══════════════════════════════════════════════════════════════════════════════
# IMPLEMENTACIÓN LOCAL: SQLiteFeed
# ══════════════════════════════════════════════════════════════════════════════

class SQLiteFeed(PriceFeed):
    """
    Lee velas desde la base de datos SQLite generada por el downloader.
    Implementación de referencia para backtesting y optimización.

    Esquema esperado (tabla btc_hourly):
        timestamp               INTEGER  (epoch ms)
        datetime                TEXT
        open, high, low, close  REAL
        volume                  REAL
        quote_volume            REAL
        trades_count            INTEGER
        taker_buy_base_volume   REAL
        taker_buy_quote_volume  REAL
    """

    def __init__(self, db_path: str, table: str = "btc_hourly") -> None:
        self.db_path = db_path
        self.table   = table
        log.info("SQLiteFeed inicializado", db=db_path, table=table)

    # ── Interfaz pública ──────────────────────────────────────────────────────

    def get_candles(
        self,
        start:  TimeInput,
        end:    TimeInput,
        symbol: str = "BTCUSDT",
    ) -> List[Candle]:
        """
        Carga todas las velas del rango en memoria.
        Para rangos muy grandes (>50k velas) preferir iter_candles().
        """
        start_ms = to_epoch_s(start) * 1000
        end_ms   = to_epoch_s(end)   * 1000

        rows = self._query(start_ms, end_ms)
        candles = [self._row_to_candle(r) for r in rows]

        log.info(
            "velas cargadas",
            count=len(candles),
            start=to_iso(start),
            end=to_iso(end),
        )
        return candles

    def subscribe(
        self,
        callback: Callable[[Candle], None],
        symbol:   str = "BTCUSDT",
    ) -> None:
        """
        No implementado en SQLiteFeed — la DB es estática.
        En backtest el Clock itera las velas; subscribe() es para live.
        """
        raise NotImplementedError(
            "SQLiteFeed no soporta subscribe(). "
            "Usá LocalClock.tick() para iterar velas en backtest."
        )

    # ── Helpers privados ──────────────────────────────────────────────────────

    def _query(self, start_ms: int, end_ms: int) -> list:
        conn = sqlite3.connect(self.db_path)
        try:
            cur = conn.cursor()
            cur.execute(
                f"""
                SELECT
                    timestamp,
                    open, high, low, close, volume,
                    quote_volume, trades_count,
                    taker_buy_base_volume, taker_buy_quote_volume
                FROM   {self.table}
                WHERE  timestamp >= ? AND timestamp <= ?
                ORDER  BY timestamp ASC
                """,
                (start_ms, end_ms),
            )
            return cur.fetchall()
        finally:
            conn.close()

    @staticmethod
    def _row_to_candle(row: tuple) -> Candle:
        (ts_ms, open_, high, low, close, volume,
         quote_vol, trades, taker_base, taker_quote) = row
        return Candle(
            ts                  = ts_ms // 1000,
            open                = float(open_),
            high                = float(high),
            low                 = float(low),
            close               = float(close),
            volume              = float(volume),
            quote_volume        = float(quote_vol)   if quote_vol  is not None else None,
            trades_count        = int(trades)         if trades     is not None else None,
            taker_buy_base_vol  = float(taker_base)  if taker_base is not None else None,
            taker_buy_quote_vol = float(taker_quote) if taker_quote is not None else None,
        )


# ══════════════════════════════════════════════════════════════════════════════
# IMPLEMENTACIÓN LOCAL: CSVFeed
# ══════════════════════════════════════════════════════════════════════════════

class CSVFeed(PriceFeed):
    """
    Lee velas desde un archivo CSV.
    Columnas mínimas requeridas: timestamp (epoch ms o s), open, high, low, close, volume
    Columnas opcionales: trades_count, taker_buy_base_volume, taker_buy_quote_volume

    Útil para datos de exchanges sin SQLite o para tests con datos sintéticos.
    """

    # Nombres de columna aceptados (case-insensitive, el feed normaliza)
    _COL_ALIASES = {
        "ts":        ["timestamp", "ts", "time", "open_time"],
        "open":      ["open", "o"],
        "high":      ["high", "h"],
        "low":       ["low", "l"],
        "close":     ["close", "c"],
        "volume":    ["volume", "vol", "base_volume"],
        "trades":    ["trades_count", "trades", "number_of_trades"],
        "taker_base":  ["taker_buy_base_volume", "taker_buy_base_vol", "taker_base"],
        "taker_quote": ["taker_buy_quote_volume", "taker_buy_quote_vol", "taker_quote"],
    }

    def __init__(self, csv_path: str, delimiter: str = ",") -> None:
        self.csv_path  = csv_path
        self.delimiter = delimiter
        self._col_map: dict[str, str] = {}  # se construye al leer el header
        log.info("CSVFeed inicializado", path=csv_path)

    def get_candles(
        self,
        start:  TimeInput,
        end:    TimeInput,
        symbol: str = "BTCUSDT",
    ) -> List[Candle]:
        start_s = to_epoch_s(start)
        end_s   = to_epoch_s(end)

        candles: List[Candle] = []
        with open(self.csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter=self.delimiter)
            if self._col_map == {} and reader.fieldnames:
                self._build_col_map(reader.fieldnames)
            for row in reader:
                ts = self._parse_ts(row)
                if ts < start_s or ts > end_s:
                    continue
                candles.append(self._row_to_candle(row, ts))

        log.info("velas CSV cargadas", count=len(candles), path=self.csv_path)
        return candles

    def subscribe(
        self,
        callback: Callable[[Candle], None],
        symbol:   str = "BTCUSDT",
    ) -> None:
        raise NotImplementedError("CSVFeed no soporta subscribe().")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _build_col_map(self, fieldnames: list[str]) -> None:
        lower_fields = {f.lower(): f for f in fieldnames}
        for canonical, aliases in self._COL_ALIASES.items():
            for alias in aliases:
                if alias in lower_fields:
                    self._col_map[canonical] = lower_fields[alias]
                    break

    def _get(self, row: dict, key: str, default=None):
        col = self._col_map.get(key)
        return row.get(col, default) if col else default

    def _parse_ts(self, row: dict) -> int:
        raw = self._get(row, "ts", 0)
        return to_epoch_s(int(float(raw)))

    def _row_to_candle(self, row: dict, ts: int) -> Candle:
        def f(key): return float(self._get(row, key) or 0)
        def i(key): v = self._get(row, key); return int(v) if v else None
        def fo(key): v = self._get(row, key); return float(v) if v else None

        return Candle(
            ts                  = ts,
            open                = f("open"),
            high                = f("high"),
            low                 = f("low"),
            close               = f("close"),
            volume              = f("volume"),
            trades_count        = i("trades"),
            taker_buy_base_vol  = fo("taker_base"),
            taker_buy_quote_vol = fo("taker_quote"),
        )


# ══════════════════════════════════════════════════════════════════════════════
# STUBS de producción (se implementan en etapa live)
# ══════════════════════════════════════════════════════════════════════════════

class BinanceRESTFeed(PriceFeed):
    """
    Descarga histórico desde la API REST de Binance.
    Implementación completa en etapa de producción.
    """
    def get_candles(self, start, end, symbol="BTCUSDT"):
        raise NotImplementedError("BinanceRESTFeed pendiente de implementación.")

    def subscribe(self, callback, symbol="BTCUSDT"):
        raise NotImplementedError("BinanceRESTFeed no soporta subscribe(). Usar BinanceWSFeed.")


class BinanceWSFeed(PriceFeed):
    """
    Stream de velas en tiempo real via WebSocket de Binance.
    Implementación completa en etapa de producción.
    """
    def get_candles(self, start, end, symbol="BTCUSDT"):
        raise NotImplementedError("BinanceWSFeed no soporta get_candles(). Usar BinanceRESTFeed.")

    def subscribe(self, callback, symbol="BTCUSDT"):
        raise NotImplementedError("BinanceWSFeed pendiente de implementación.")


# ══════════════════════════════════════════════════════════════════════════════
# FACTORY — modo config decide qué implementación usar
# ══════════════════════════════════════════════════════════════════════════════

def build_price_feed() -> PriceFeed:
    """
    Construye la implementación correcta según mode_config y config_local.
    Llamar desde los runners — nunca instanciar feeds directamente.

    Modos disponibles (mode_config.py):
        USE_LIVE_FEED = False  →  SQLiteFeed (default backtest)
        USE_LIVE_FEED = True   →  BinanceWSFeed (producción)
    """
    try:
        import mode_config as MC
        use_live = getattr(MC, "USE_LIVE_FEED", False)
    except ImportError:
        use_live = False

    if use_live:
        log.info("PriceFeed modo LIVE → BinanceWSFeed")
        return BinanceWSFeed()

    # Modo local — leer ruta desde config_local
    try:
        import config_local as CL
        db_path = getattr(CL, "DB_PATH", "btc_hourly.db")
        table   = getattr(CL, "DB_TABLE", "btc_hourly")
    except ImportError:
        db_path = "btc_hourly.db"
        table   = "btc_hourly"

    log.info("PriceFeed modo LOCAL → SQLiteFeed", db=db_path)
    return SQLiteFeed(db_path=db_path, table=table)
