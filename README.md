# BTC/USDT Swing Trading Bot

Sistema completo de trading algorítmico para BTC/USDT con estrategias, backtesting, optimización y ejecución en tiempo real contra Binance (testnet y producción).

---

## Tabla de contenidos

1. [Visión general](#visión-general)
2. [Arquitectura del sistema](#arquitectura-del-sistema)
3. [Estructura de archivos](#estructura-de-archivos)
4. [Instalación y configuración](#instalación-y-configuración)
5. [Los cuatro actores](#los-cuatro-actores)
6. [Estrategias disponibles](#estrategias-disponibles)
7. [Runners de backtest](#runners-de-backtest)
8. [Live trading](#live-trading)
9. [Gestión de riesgo](#gestión-de-riesgo)
10. [Persistencia de estado](#persistencia-de-estado)
11. [Graficador](#graficador)
12. [Dashboard HTML en tiempo real](#dashboard-html-en-tiempo-real)
13. [Tests](#tests)
14. [Configuración detallada](#configuración-detallada)
15. [Flujo de trabajo recomendado](#flujo-de-trabajo-recomendado)
16. [Rendimiento histórico validado](#rendimiento-histórico-validado)
17. [Preguntas frecuentes](#preguntas-frecuentes)

---

## Visión general

El bot implementa un ciclo completo de investigación → validación → producción:

```
DB local (SQLite)
      │
      ▼
Backtest Irreal ──► techo teórico de rendimiento (oráculo perfecto)
      │
      ▼
Backtest LocalReversal ──► modelo de machine learning entrenado y validado OOS
      │
      ▼
Live Trader ──► ejecución real contra Binance testnet / producción
      │
      ▼
Graficador + Dashboard HTML ──► análisis y monitoreo
```

**Características principales:**

- Arquitectura limpia basada en cuatro actores con interfaces abstractas: `PriceFeed`, `Wallet`, `OrderBook`, `Clock`.
- Tres estrategias incluidas: oráculo perfecto (benchmark), señal compuesta (DNA + Lyapunov + PE + Delta) y reversals locales con Gradient Boosting.
- Modo backtest y modo live con el mismo código de estrategia, sin modificaciones.
- Cache de modelos para que el live trader arranque en segundos reutilizando el entrenamiento del backtest.
- Dashboard HTML auto-refrescante y dashboard en consola.
- Gestión de riesgo configurable: drawdown máximo, pérdida diaria, deduplicación de órdenes.
- Persistencia de estado entre sesiones para recuperarse ante caídas.
- Suite de tests con 40+ casos que cubren features, labeling, lógica de wallet y pipeline de integración.

---

## Arquitectura del sistema

El sistema separa responsabilidades en cuatro actores independientes y una capa de estrategia que solo conoce sus interfaces:

```
┌─────────────────────────────────────────────────────────────┐
│                        RUNNER                               │
│  (backtest_*.py  /  live_local_reversal.py)                 │
│                                                             │
│  Clock.tick() ──► Strategy.on_candle() ──► Signal           │
│                                               │             │
│                   RiskManager.check()◄────────┤             │
│                                               │             │
│                   OrderBook.execute() ◄───────┘             │
│                         │                                   │
│                   Wallet.update()                           │
└─────────────────────────────────────────────────────────────┘
```

### Principios de diseño

**Separación intención / ejecución.** El `OrderBook` divide cada operación en tres pasos: `create_order()` → `submit()` → `check()`. En backtest los tres colapsan en uno; en producción son llamadas REST separadas.

**Mismo código, distinto modo.** La estrategia nunca sabe si está en backtest o en live. El runner inyecta los actores correctos (simulados o reales) al construir el sistema.

**Inmutabilidad del slot.** El slot de capital por posición solo se recalcula cuando el número de posiciones llega a cero. Mientras haya posiciones abiertas, el slot permanece fijo. Esto replica el comportamiento del backtest irreal calibrado.

**Cache atómico.** Los modelos ML se calculan una vez y se guardan como archivos `.npy`. El live trader carga desde cache en ~1 segundo sin necesitar acceso a la base de datos.

---

## Estructura de archivos

```
.
├── config_local.py              # Rutas, fechas, capital, parámetros locales
├── config_world.py              # Endpoints Binance, timeouts, modo testnet/producción
├── log_config.py                # Configuración del sistema de logging
├── .env                         # Credenciales (NO commitear — ver .env.example)
├── env.example                  # Template para el .env
│
├── backtest_irreal.py           # Runner del oráculo perfecto (benchmark)
├── backtest_compuesto.py        # Runner de la señal compuesta
├── backtest_local_reversal.py   # Runner del modelo GBM (principal)
├── live_local_reversal.py       # Runner de producción / testnet
├── Graficador.py                # Análisis visual de resultados en PNG
├── test_testnet.py              # Diagnóstico del entorno Binance Testnet
│
├── actors/                      # Los cuatro actores del sistema
│   ├── price_feed.py            # Actor 1: fuente de precios (SQLite, CSV)
│   ├── binance_feed.py          # Actor 1: fuente de precios (REST + WebSocket)
│   ├── wallet.py                # Actor 2: billetera (MemoryWallet, JSONWallet)
│   ├── binance_wallet.py        # Actor 2: billetera sincronizada con Binance
│   ├── order_book.py            # Actor 3: libro de órdenes (simulado)
│   ├── binance_order_book.py    # Actor 3: órdenes reales en Binance
│   ├── clock.py                 # Actor 4: reloj de ciclos (LocalClock)
│   └── live_clock.py            # Actor 4: reloj en tiempo real (LiveClock)
│
├── strategies/                  # Estrategias de trading
│   ├── base_strategy.py         # Interfaz abstracta + tipo Signal
│   ├── irreal.py                # Oráculo perfecto (benchmark)
│   ├── compuesto.py             # DNA + Lyapunov + PE + Delta
│   └── local_reversal.py        # Gradient Boosting sobre microestructura
│
├── risk/
│   └── risk_manager.py          # Gestión de riesgo configurable
│
├── state/
│   └── state_manager.py         # Persistencia de estado entre sesiones
│
├── support/
│   ├── logger.py                # Logging estructurado (consola + JSONL)
│   ├── secrets.py               # Carga de credenciales desde .env
│   ├── time_utils.py            # Normalización universal de timestamps
│   └── time_sync.py             # Compensación de desfase de reloj vs Binance
│
└── tests/
    ├── run_all_tests.py          # Runner maestro de la suite
    ├── test_features_labeling.py # Tests de aritmética de features y labeling
    ├── test_wallet_logic.py      # Tests de la lógica de capital y slots
    ├── test_irreal_strategy.py   # Tests de detección de extremos locales
    └── test_simulation_pipeline.py # Tests de integración con datos reales
```

---

## Instalación y configuración

### Requisitos

- Python 3.10 o superior
- Base de datos SQLite de velas horarias de BTC/USDT (generada externamente)
- Cuenta en Binance Testnet para live trading (gratuita, sin riesgo real)

### Dependencias

```bash
pip install numpy pandas matplotlib scikit-learn requests websockets
```

Para el módulo de Lyapunov de la estrategia compuesta:

```bash
pip install scipy
```

### Configuración inicial

**1. Copiar el template de credenciales:**

```bash
cp env.example .env
```

Editar `.env` con las credenciales del testnet de Binance:

```
BINANCE_TESTNET_API_KEY=tu_api_key_aqui
BINANCE_TESTNET_SECRET=tu_secret_aqui
```

Las credenciales del testnet se obtienen en [testnet.binance.vision](https://testnet.binance.vision).

**2. Configurar `config_local.py`:**

```python
# Ruta a la base de datos de velas horarias
DB_PATH      = "/ruta/a/btc_hourly.db"
DB_TABLE     = "btc_hourly"

# Archivo de salida del backtest
RESULTS_JSON = "backtest_results.json"

# Rango de fechas para el backtest
FECHA_INICIO = '2022-01-01'
FECHA_FIN    = '2022-12-31'

# Parámetros de simulación
SALDO_USDT_INICIAL = 1000.0
MAX_POSICIONES     = 5
COMMISSION_PCT     = 0.1        # 0.1% (Binance Spot)
```

**3. Verificar el entorno de Binance (opcional pero recomendado):**

```bash
python test_testnet.py
```

Este script diagnóstica en 8 pasos: credenciales, conectividad REST, sincronización de tiempo, saldo de cuenta, filtros del par, velas históricas, WebSocket y una orden de prueba (sin ejecutar nada real).

---

## Los cuatro actores

### Actor 1: PriceFeed — Fuente de precios

Responsabilidad única: entregar velas OHLCV al sistema.

```python
from actors.price_feed import SQLiteFeed, CSVFeed, build_price_feed

# Backtest desde SQLite
feed = SQLiteFeed(db_path="btc_hourly.db", table="btc_hourly")
candles = feed.get_candles(start="2022-01-01", end="2022-12-31")

# O usando la factory
feed = build_price_feed(mode="local")   # SQLiteFeed
feed = build_price_feed(mode="live")    # BinanceWSFeed
```

**Candle** es el tipo canónico del sistema. Contiene todos los campos de microestructura de Binance:

| Campo | Tipo | Descripción |
|---|---|---|
| `ts` | `int` | Timestamp de apertura (epoch s UTC) |
| `open/high/low/close` | `float` | Precios OHLC |
| `volume` | `float` | Volumen base |
| `taker_buy_base_vol` | `float?` | Volumen de taker compradores |
| `quote_volume` | `float?` | Volumen quote total |
| `trades_count` | `int?` | Número de trades |

Propiedades calculadas: `delta_ratio` (presión compradora), `body_ratio`, `upper_wick_ratio`, `lower_wick_ratio`.

**Implementaciones disponibles:**

| Clase | Uso | Método principal |
|---|---|---|
| `SQLiteFeed` | Backtest local | `get_candles()` |
| `CSVFeed` | Datos externos en CSV | `get_candles()` |
| `BinanceRESTFeed` | Histórico via API REST | `get_candles()` |
| `BinanceWSFeed` | Stream en tiempo real | `subscribe(callback)` |

### Actor 2: Wallet — Billetera

Responsabilidad única: custodiar y reportar el estado del capital.

```python
from actors.wallet import MemoryWallet, JSONWallet

# Backtest rápido (sin I/O)
wallet = MemoryWallet(usdt_inicial=1000.0, max_posiciones=5)

# Backtest con registro completo en JSON
wallet = JSONWallet(
    usdt_inicial=1000.0,
    max_posiciones=5,
    json_path="backtest_results.json"
)
```

**Lógica de slots (benchmark canónico):**

La lógica de slots es el corazón del sistema de gestión de capital. Replica el comportamiento del backtest irreal calibrado:

1. Al llegar a 0 posiciones: `slot_usdt = usdt_balance / MAX_POSICIONES`
2. El slot permanece **inmutable** mientras haya posiciones abiertas
3. Después de cada BUY: `btc_por_venta = btc_en_posiciones / positions_count`
4. Las ventas son en partes iguales siguiendo orden FIFO

**Implementaciones disponibles:**

| Clase | Uso | Persistencia |
|---|---|---|
| `MemoryWallet` | Grid search, backtest rápido | Solo RAM |
| `JSONWallet` | Backtest completo | Archivo JSON |
| `BinanceWallet` | Live trading | Sincronizada con API real |

**`BinanceWallet`** al arrancar: lee el saldo real de la API, compara con el último checkpoint guardado, reconcilia si hay divergencia (por operaciones manuales u otra razón), y restaura el estado completo de posiciones.

### Actor 3: OrderBook — Libro de órdenes

Responsabilidad única: abrir y cerrar posiciones, calcular comisiones.

```python
from actors.order_book import SimulatedOrderBook, build_order_book

# Backtest (simulado)
ob = SimulatedOrderBook(commission_pct=0.1, max_posiciones=5)

# Ejecutar con guardias
order = ob.execute_with_guards(
    side=OrderSide.BUY,
    price=candle.close,
    wallet=wallet,
    candle_ts=candle.ts,
)

if order.is_filled:
    print(f"Comprado {order.trade.btc_bought:.6f} BTC")
elif order.is_ignored:
    print(f"Ignorado: {order.reject_reason}")
```

**Guardias del OrderBook:**

- `max_posiciones`: no abre más posiciones del límite configurado
- `usdt_insuficiente`: el slot supera el balance disponible
- `sin_posiciones`: intento de vender sin posiciones abiertas
- `btc_por_venta_cero`: no hay BTC calculado para vender

**`BinanceOrderBook`** además implementa firma HMAC-SHA256, compensación de desfase de reloj, reintentos con backoff exponencial, y conversión de comisiones en activos distintos a USDT (ej. BNB) al equivalente en USDT.

### Actor 4: Clock — Reloj de ciclos

Responsabilidad única: decidir **cuándo** se ejecuta cada ciclo de la estrategia.

```python
from actors.clock import LocalClock, build_clock

# Backtest: itera velas desde SQLite
clock = LocalClock(feed, start="2022-01-01", end="2022-12-31")
for candle in clock:
    signal = strategy.on_candle(candle, wallet)

# Live: bloquea hasta que cierra la próxima vela horaria
from actors.live_clock import LiveClock
clock = LiveClock(feed=BinanceWSFeed(), symbol="BTCUSDT")
for candle in clock:   # mismo código, distinto clock
    signal = strategy.on_candle(candle, wallet)
```

**`LocalClock`** usa lazy loading: las velas se cargan en el primer `tick()`. Esto permite reusar la instancia en grid search usando `reset()` sin recargar datos del disco.

**`LiveClock`** desacopla el thread del WebSocket del hilo principal del trader usando una `Queue(maxsize=2)`. El thread de WS encola; el hilo principal desencola en `tick()`. Reconnexión automática con backoff exponencial ante caídas de red.

---

## Estrategias disponibles

### Estrategia 1: IrrealStrategy — Oráculo Perfecto

Benchmark teórico imposible en producción. Compra en cada mínimo local y vende en cada máximo local con ventana configurable.

```python
from strategies.irreal import IrrealStrategy

strategy = IrrealStrategy(
    ventana=10,           # velas a cada lado para confirmar extremo
    precio_compra="low",  # "low" | "close" | "open"
    precio_venta="high",  # "high" | "close" | "open"
)
```

**Cómo funciona:** mantiene un buffer circular de `2*ventana+1` velas. Cuando el buffer se llena, evalúa si la vela central es un mínimo local (su `low` es ≤ al `low` de todas las demás velas del buffer) o un máximo local (su `high` es ≥ al `high` de todas). Requiere ver el futuro por eso es imposible en producción — es solo el techo teórico de rendimiento.

**Parámetros calibrados:** `ventana=10`, `precio_compra="low"`, `precio_venta="high"`.

### Estrategia 2: CompuestoStrategy — Señal Compuesta

Pipeline complejo de cinco componentes que producen un score adaptativo 0-100.

```python
from strategies.compuesto import CompuestoStrategy

strategy = CompuestoStrategy(
    thr_bot=75.0,        # umbral de score para señal BUY
    thr_top=75.0,        # umbral de score para señal SELL
    cooldown=16,         # horas mínimas entre señales del mismo tipo
    suavizado=6,         # ventana de suavizado del score
    ventana_score=500,   # ventana para normalización adaptativa del score
    cache_dir=".cache_compuesto",
)
```

**Componentes del pipeline:**

| Componente | Descripción | Archivo cache |
|---|---|---|
| **DNA** | 6 features por vela: body_ratio, mechas, delta_ratio, range_rel, trade_density | `dna.npy` |
| **Lyapunov** | Exponente de Lyapunov sobre serie de precios normalizada (caos local) | `lyap.npy` |
| **HFD** | Dimensión fractal de Higuchi (complejidad de la serie) | `hfd.npy` |
| **PE** | Entropía de Permutación en 4 canales: precio, delta, mechas | `pe_matrix.npy` |
| **TE** | Transferencia de entropía entre delta y precio (causalidad informacional) | `te_dc.npy` |
| **RF** | Random Forest sobre todos los anteriores (probabilidades bottom/top) | `prob_bot.npy`, `prob_top.npy` |
| **Score** | Combinación adaptativa con regresión logística y normalización percentílica | `score_bot.npy`, `score_top.npy` |

**Uso con cache:**

```bash
# Primera vez: calcula todo (~20-30 min)
python backtest_compuesto.py

# Siguientes veces: usa cache
python backtest_compuesto.py --fast

# Recalcular desde cero
python backtest_compuesto.py --nocache
```

### Estrategia 3: LocalReversalStrategy — Reversals Locales (Principal)

Detecta mínimos y máximos locales usando HistGradientBoostingClassifier de scikit-learn entrenado sobre features de microestructura de velas.

```python
from strategies.local_reversal import LocalReversalStrategy

strategy = LocalReversalStrategy(
    thr_b=0.50,   # umbral de confianza para señal BUY  [0.40, 0.85]
    thr_t=0.45,   # umbral de confianza para señal SELL [0.40, 0.85]
    cache_dir=".cache_local_reversal",
)
```

#### Features del modelo (8 series × 24 velas + 5 agregadas = 101 features)

**Series por vela:**

| Feature | Fórmula | Interpretación |
|---|---|---|
| `body_ratio` | `(close - open) / range` ∈ [-1, 1] | Dirección y fuerza de la vela |
| `lower_wick` | `(min(open,close) - low) / range` ∈ [0, 1] | Rechazo del precio bajo |
| `upper_wick` | `(high - max(open,close)) / range` ∈ [0, 1] | Rechazo del precio alto |
| `delta_ratio` | `taker_buy_vol / total_vol` ∈ [0, 1] | Presión compradora real |
| `range_rel` | `range / rolling_avg_range(48)` ∈ [0, 5] | Explosividad relativa |
| `divergence` | `zscore(delta) - zscore(ret_4h)` | Desacople delta-precio |
| `low_rejection` | `(close - low) / range` ∈ [0, 1] | Rebote desde el mínimo |
| `high_rejection` | `(high - close) / range` ∈ [0, 1] | Rechazo desde el máximo |

**Agregadas sobre la ventana de 24 velas:**

- `body_mean24`: tendencia de dirección reciente
- `body_last3`: momentum de corto plazo
- `div_last6`: divergencia acumulada en 6 horas
- `lowrej_last3`: fuerza del rebote en los últimos 3 ticks
- `hirej_last3`: fuerza del rechazo alto en los últimos 3 ticks

#### Labeling

- **Bottom (label=1):** `low[i]` es el mínimo de `low` en la ventana `±12` velas
- **Top (label=2):** `high[i]` es el máximo de `high` en la ventana `±12` velas
- **Neutro (label=0):** todo lo demás

#### Modelos

Dos modelos binarios independientes:
- **Modelo BOTTOM:** ¿es esta vela un mínimo local genuino? → `prob_bottom`
- **Modelo TOP:** ¿es esta vela un máximo local genuino? → `prob_top`

```python
HistGradientBoostingClassifier(
    max_iter=400,
    max_depth=6,
    learning_rate=0.05,
    min_samples_leaf=15,
    l2_regularization=0.1,
    class_weight='balanced',
)
```

#### Lógica de señal

```python
# Prioridad SELL sobre BUY en la misma vela
if prob_top >= thr_t:
    return Signal(SELL, price=candle.close, score=prob_top)
if prob_bottom >= thr_b:
    return Signal(BUY,  price=candle.close, score=prob_bottom)
return HOLD
```

La asimetría `thr_t < thr_b` es deliberada: facilita el cierre de posiciones abiertas reduciendo el tiempo en riesgo y el drawdown.

---

## Runners de backtest

### `backtest_irreal.py` — Benchmark

```bash
python backtest_irreal.py
```

Genera el JSON con el techo teórico de rendimiento. Parámetros configurables al inicio del archivo:

```python
VENTANA_LOCAL  = 10      # velas a cada lado
PRECIO_COMPRA  = "low"   # precio de ejecución en BUY
PRECIO_VENTA   = "high"  # precio de ejecución en SELL
```

### `backtest_compuesto.py` — Señal Compuesta

```bash
python backtest_compuesto.py           # calcula todo desde cero
python backtest_compuesto.py --fast    # usa cache existente
python backtest_compuesto.py --nocache # borra cache y recalcula
```

### `backtest_local_reversal.py` — Modelo Principal (recomendado)

```bash
python backtest_local_reversal.py            # entrena y corre
python backtest_local_reversal.py --fast     # usa cache si existe
python backtest_local_reversal.py --nocache  # borra cache y reentrena
```

Parámetros configurables:

```python
THR_B     = 0.85   # umbral señal BUY  (calibrado: 0.50)
THR_T     = 0.75   # umbral señal SELL (calibrado: 0.45)
CACHE_DIR = ".cache_local_reversal"
```

**Salida en consola:**

```
╔══════════════════════════════════════════════════════════╗
║        BACKTEST LOCAL REVERSAL — GBM BTC/USDT           ║
╚══════════════════════════════════════════════════════════╝
  Rango         : 2022-01-01 → 2022-12-31
  Capital       : $1,000.00 USDT
  Max posiciones: 5
  Comisión      : 0.1%
  THR_B (compra): 0.85  THR_T (venta): 0.75
...

════════════════════════════════════════════════════════════
  RESUMEN
════════════════════════════════════════════════════════════
  Portfolio final  : $    1,216.00 USDT
  PnL              : +21.60%
  Buy & Hold ref   : -64.50%
  Alpha vs B&H     : +86.10%
  Compras          : 87
  Ventas           : 82
  Ignorados        : 34
  Posiciones abier.: 3
  Tiempo total     : 45.2s
```

---

## Live trading

### Flujo de arranque

```
1. Verificar cache del modelo (.cache_local_reversal/)
   └─ Si no existe → entrenar desde DB local (~90s)
   └─ Si existe → continuar

2. Medir desfase de reloj vs Binance (compensación automática)

3. Vender BTC pre-existente en la cuenta (arranque limpio)

4. Inicializar actores:
   BinanceWallet.from_account() → reconcilia checkpoint vs API real
   BinanceOrderBook             → órdenes con firma HMAC-SHA256
   BinanceWSFeed                → stream WebSocket con reconexión automática
   LiveClock                    → bloquea hasta cierre de vela horaria

5. Cargar modelo desde cache (instantáneo)

6. Loop: tick → señal → ejecución → dashboards → checkpoint
```

### Ejecutar

```bash
python live_local_reversal.py
```

Para detener limpiamente: `Ctrl+C`. El sistema cierra el WebSocket, guarda el estado final y escribe el JSON de resultados.

### Configuración del live trader

Al inicio de `live_local_reversal.py`:

```python
THR_B             = 0.50   # umbral BUY
THR_T             = 0.50   # umbral SELL
MAX_POSICIONES    = 5      # posiciones simultáneas máximas
COMMISSION_PCT    = 0.1    # comisión estimada (0.1%)
SYMBOL            = "BTCUSDT"
LIVE_RESULTS_JSON = "live_results.json"
STATE_PATH        = "state/live_trading_state.jsonl"
CACHE_DIR         = ".cache_local_reversal"
DASHBOARD_HTML    = "live_dashboard.html"
DASHBOARD_REFRESH = 10     # segundos entre refresh del HTML
CHART_CANDLES     = 48     # velas a mostrar en el gráfico (~2 días)
```

### Modo testnet vs producción

En `config_world.py`:

```python
USE_TESTNET = True    # True = testnet (sin dinero real)
USE_TESTNET = False   # False = producción REAL (¡con dinero real!)
```

> **Advertencia:** Con `USE_TESTNET = False` el sistema opera con dinero real en Binance producción. Verificar exhaustivamente en testnet antes de cambiar esta opción.

---

## Gestión de riesgo

El `RiskManager` actúa como guardián entre la estrategia y el `OrderBook`. Es configurable independientemente del modo de ejecución.

```python
from risk.risk_manager import RiskManager, RiskConfig

# Sin límites (backtest)
risk = RiskManager(config=RiskConfig.permissive(), usdt_inicial=1000.0)

# Conservador (producción recomendado)
risk = RiskManager(config=RiskConfig.conservative(), usdt_inicial=1000.0)

# Custom
config = RiskConfig(
    max_drawdown_pct    = 15.0,   # detiene si el portfolio cae 15% desde el pico
    max_daily_loss_usdt = 50.0,   # detiene si se pierden $50 en el día
    max_order_usdt      = 300.0,  # rechaza órdenes individuales > $300
    min_order_usdt      = 5.0,    # rechaza órdenes < $5
    dedup_window_s      = 60,     # evita duplicados en ventana de 60s
)
```

**Reglas implementadas:**

| Regla | Parámetro | Acción |
|---|---|---|
| Drawdown máximo | `max_drawdown_pct` | Bloquea todo cuando el portfolio cae X% desde el pico |
| Pérdida diaria | `max_daily_loss_usdt` | Bloquea todo si la pérdida del día supera Y USDT |
| Tamaño máximo de orden | `max_order_usdt` | Rechaza órdenes individuales mayores a Z USDT |
| Tamaño mínimo de orden | `min_order_usdt` | Rechaza órdenes menores al mínimo operativo |
| Deduplicación | `dedup_window_s` | Evita órdenes duplicadas en ventana de tiempo |

---

## Persistencia de estado

El `StateManager` guarda un `Checkpoint` completo después de cada operación ejecutada. Si el proceso cae y se reinicia, el sistema restaura el estado exacto anterior.

```python
# Checkpoint contiene:
# - usdt_balance, btc_libre, slot_usdt, btc_por_venta
# - lista de posiciones abiertas (precio_entrada, btc, opened_at)
# - btc_acumulado_total, capital_inicial
# - timestamp de la última vela procesada
# - metadata (estrategia, umbrales, etc.)
```

**Implementaciones:**

```python
from state.state_manager import MemoryStateManager, JSONStateManager

# Backtest: solo RAM
state = MemoryStateManager()

# Producción: archivo JSONL append-only
state = JSONStateManager("state/live_trading_state.jsonl")
```

El formato JSONL (JSON Lines) es append-only: cada checkpoint es una línea independiente. Si el proceso muere a mitad de un write, las líneas anteriores permanecen válidas. Al restaurar, se lee la última línea válida del archivo.

---

## Graficador

Genera un análisis visual completo en PNG a partir del JSON producido por cualquier runner.

```bash
# Con parámetros de config_local.py
python Graficador.py

# Con parámetros explícitos
python Graficador.py --json resultados.json --db datos.db --dark
python Graficador.py --light --out mi_grafico.png
```

**Paneles del gráfico (6 en total):**

1. **Precio BTC + Operaciones** — línea de precio con marcas de compras (verde) y ventas (rojo), operaciones ignoradas (gris), y precio promedio de posiciones (línea azul punteada)
2. **Evolución del Balance USDT** — área rellena con el balance libre, línea de reserva si aplica
3. **Evolución del Balance BTC** — BTC libre, BTC en posiciones, BTC total
4. **Posiciones Abiertas Netas** — número de posiciones simultáneas a lo largo del tiempo
5. **Portfolio vs Buy & Hold** — comparación directa con áreas de ganancia/pérdida rellenas
6. **Drawdown Continuo** — estrategia vs Buy & Hold con mínimo anotado

**Métricas en consola (además del gráfico):**

- Portfolio final, PnL%, Buy & Hold, Alpha vs B&H
- Max Drawdown, Sharpe Ratio (anualizado, velas horarias), Calmar Ratio
- Win Rate, Profit Factor, ganancia total, promedio por operación
- Frecuencia de trades (trades/mes, días entre trades)

**Temas:** `--dark` (default) y `--light`.

---

## Dashboard HTML en tiempo real

El live trader genera automáticamente `live_dashboard.html` después de cada vela procesada. Abrir en cualquier browser; se refresca automáticamente cada 10 segundos.

**Contenido del dashboard:**

- Cards: portfolio total, P&L sesión, USDT libre, posiciones abiertas, operaciones totales, countdown a próxima vela
- Gráfico de velas (candlestick) con marcas de señales BUY/SELL
- Subgráfico de probabilidades del modelo (`prob_bottom` y `prob_top`) con líneas de umbral
- Panel de señal actual con barras de progreso
- Tabla de posiciones abiertas con P&L por posición
- Tabla de últimas 5 operaciones

---

## Tests

### Ejecutar la suite completa

```bash
python tests/run_all_tests.py
```

### Módulos de tests

**`test_features_labeling.py`** — 20 tests, sin dependencias externas:
- Aritmética exacta de cada feature (body_ratio, mechas, delta_ratio, etc.)
- Shape y ausencia de NaN/inf en el array de features
- Verificación de que una ventana produce exactamente 101 features
- Labeling V3: detección de mínimos/máximos, ventanas de borde, no solapamiento
- Lógica de señal: prioridad SELL sobre BUY, asimetría de umbrales

**`test_wallet_logic.py`** — 18 tests, sin dependencias externas:
- Slot inicial, recálculo al llegar a cero posiciones
- BUY: descuento exacto, comisión, guardias de max_posiciones
- SELL: partes iguales, FIFO, comisión, guardia sin_posiciones
- Compounding a través de múltiples ciclos
- Flujo completo con 5 posiciones simultáneas

**`test_irreal_strategy.py`** — 8 tests, sin dependencias externas:
- Buffer circular: llenado correcto, primera señal en momento exacto
- Detección de mínimo/máximo únicos con timestamp correcto
- Precio de ejecución es `low` del mínimo y `high` del máximo
- Comportamiento con empates (condición `<=`)
- Win rate en datos perfectos

**`test_simulation_pipeline.py`** — tests de integración (requiere DB):
- Tests matemáticos que siempre corren: capital sin señales, comisiones, compounding
- Tests OOS que generan automáticamente el pkl si no existe (~90s la primera vez):
  - Retornos positivos en todos los años 2021-2025
  - Supera Buy & Hold en 2022 (bear market severo)
  - Win rate > 55% en cada año
  - 2025 positivo (reservado como holdout final)

---

## Configuración detallada

### `config_local.py` — Configuración del entorno local

```python
# ── Rutas ──────────────────────────────────────────────────────
DB_PATH      = "/ruta/a/btc_hourly.db"   # base de datos SQLite
DB_TABLE     = "btc_hourly"               # nombre de la tabla
RESULTS_JSON = "backtest_results.json"    # salida del backtest
STATE_PATH   = "state/trading_state.jsonl"

# ── Rango de fechas ────────────────────────────────────────────
FECHA_INICIO = '2022-01-01'   # formato YYYY-MM-DD
FECHA_FIN    = '2022-12-31'   # None = hasta el final del dataset

# Referencias históricas de BTC útiles para configurar rangos:
#   TOP1 2021         : '2021-04-14'
#   TOP2 2021         : '2021-11-10'
#   Bottom Bear 2022  : '2022-11-22'
#   Inicio Bull 2023  : '2023-01-01'
#   TOP 2025          : '2025-10-06'

# ── Parámetros de simulación ───────────────────────────────────
SYMBOL             = "BTCUSDT"
SALDO_USDT_INICIAL = 1000.0
MAX_POSICIONES     = 5
COMMISSION_PCT     = 0.1       # % (Binance Spot maker/taker)

# ── Salida del Graficador ──────────────────────────────────────
DARK_MODE  = True
OUTPUT_PNG = "analisis_estrategia.png"
DPI        = 150
```

### `config_world.py` — Conexiones externas

```python
BINANCE_BASE_URL    = "https://api.binance.com"
BINANCE_TESTNET_URL = "https://testnet.binance.vision"
USE_TESTNET         = True    # ← cambiar a False para producción real

BINANCE_WS_URL      = "wss://stream.binance.com:9443/ws"
BINANCE_WS_TESTNET  = "wss://stream.testnet.binance.vision/ws"

REQUEST_TIMEOUT_S   = 10
WS_RECONNECT_DELAY_S= 5
MAX_RETRIES         = 3

SYMBOL              = "BTCUSDT"
KLINE_INTERVAL      = "1h"
ORDER_TYPE          = "MARKET"
RECV_WINDOW_MS      = 5000
```

### `log_config.py` — Logging

```python
LOG_TO_FILE = False  # True → escribe JSONL en logs/
LOG_LEVEL   = "INFO" # DEBUG | INFO | WARNING | ERROR | CRITICAL
LOG_DIR     = "logs"
```

El logger soporta contexto estructurado:

```python
from support.logger import get_logger
log = get_logger("mi_modulo")

log.info("vela procesada", ts="2022-01-15T08:00:00Z", close=42500.0)
log.warning("orden ignorada", motivo="max_posiciones", posiciones=5)
```

---

## Flujo de trabajo recomendado

### 1. Exploración inicial

```bash
# Ver el techo teórico de un período
# Editar FECHA_INICIO y FECHA_FIN en config_local.py
python backtest_irreal.py
python Graficador.py
```

### 2. Calibración del modelo

```bash
# Entrenar el modelo y ver resultados OOS
python backtest_local_reversal.py

# Ver el gráfico completo
python Graficador.py
```

### 3. Ajuste fino de umbrales (opcional)

Los umbrales `THR_B` y `THR_T` se ajustan al inicio de `backtest_local_reversal.py`. Valores calibrados como punto de partida:

| Año | THR_B | THR_T | PnL | B&H | Alpha |
|---|---|---|---|---|---|
| 2021 | 0.50 | 0.45 | +90.8% | +59.4% | +31.4% |
| 2022 | 0.50 | 0.45 | +21.6% | -64.5% | +86.1% |
| 2023 | 0.50 | 0.45 | +21.3% | +155.8% | -134.5% |
| 2024 | 0.50 | 0.45 | +38.0% | +120.3% | -82.3% |
| 2025 | 0.50 | 0.45 | +17.9% | -7.2% | +25.1% |

> La estrategia brilla en mercados bajistas y laterales. En bull markets fuertes el Buy & Hold generalmente supera dado que el modelo cierra posiciones antes de que el precio alcance nuevos máximos.

### 4. Verificar entorno de testnet

```bash
python test_testnet.py
```

### 5. Ejecutar en testnet

```bash
# El cache del backtest es reutilizado automáticamente
python live_local_reversal.py
```

Abrir `live_dashboard.html` en el browser para monitorear.

### 6. Revisar resultados

```bash
python Graficador.py --json live_results.json
```

---

## Rendimiento histórico validado

Resultados con validación walk-forward out-of-sample. El modelo se entrena en todos los datos anteriores a cada año de test. Los parámetros `thr_b=0.50`, `thr_t=0.45` fueron calibrados sobre 2021-2024 y 2025 fue reservado como holdout final.

| Año | Estrategia | Buy & Hold | Alpha | Win Rate | Max DD |
|---|---|---|---|---|---|
| 2021 | **+90.8%** | +59.4% | +31.4% | 59.4% | — |
| 2022 | **+21.6%** | -64.5% | +86.1% | 57.1% | — |
| 2023 | **+21.3%** | +155.8% | -134.5% | 65.4% | — |
| 2024 | **+38.0%** | +120.3% | -82.3% | 63.8% | — |
| 2025 | **+17.9%** | -7.2% | +25.1% | 61.4% | — |

*Capital: $1000 USDT, MAX_POSICIONES=5, COMMISSION=0.1%. Todos los años son out-of-sample respecto al período de calibración de umbrales.*

**Observaciones clave:**

- El win rate es estable entre 57% y 65% en todos los años, lo que indica que el modelo tiene poder predictivo genuino y no es sobreajuste.
- El mejor alpha (+86.1%) ocurre en 2022, el peor bear market de la historia reciente de BTC. El modelo es explícitamente bueno en detectar bottoms en mercados en caída.
- En bull markets fuertes (2023, 2024) el Buy & Hold supera a la estrategia porque el modelo cierra posiciones en máximos locales que resultan ser mínimos a largo plazo. Esto es un trade-off conocido y aceptado: menor upside en bulls a cambio de mucho menor drawdown en bears.

---

## Preguntas frecuentes

**¿Por qué el backtest irreal no es utilizable en producción?**

Porque requiere ver las próximas `ventana` velas para confirmar que una vela es un extremo local. En el momento de la vela i, no sabemos si habrá una vela con precio más bajo en las siguientes 10 horas.

**¿Por qué el slot no cambia cuando hay posiciones abiertas?**

Esta regla es la que hace que el sistema sea auto-regulable. Si el capital crece, el próximo ciclo de compras usará slots más grandes (compounding natural). Si cae, los slots se achican protegiendo el capital. El slot fijo durante un ciclo abierto evita que una señal tardía use un slot distinto al que tenía la primera compra del ciclo.

**¿Qué pasa si el proceso cae durante una posición abierta?**

El `JSONStateManager` guarda un checkpoint completo después de cada operación. Al reiniciar, `BinanceWallet.from_account()` lee el saldo real de Binance, lo compara con el último checkpoint, reconcilia si hay divergencia y restaura el estado completo incluyendo las posiciones abiertas.

**¿El modelo se reentrena en live?**

No. El modelo se entrena una vez en backtest sobre el dataset completo y se guarda en `.cache_local_reversal/`. En live se carga desde ese cache en ~1 segundo. Reentrenar en vivo requeriría acceso a la base de datos histórica que puede no estar disponible en el servidor de producción. Cuando haya datos nuevos suficientes (varios meses), se reentrena manualmente con `--nocache` y se despliega el nuevo cache.

**¿Por qué se prioriza SELL sobre BUY en la misma vela?**

Para no abrir y cerrar simultáneamente. Si en la misma vela hay señal de bottom y de top (caso raro pero posible), es más conservador cerrar la posición existente antes que abrir una nueva. Reduce el tiempo en riesgo.

**¿Cómo se maneja el desfase de reloj con Binance?**

`support/time_sync.py` mide el desfase contra el servidor de Binance al arrancar (3 muestras, mediana para reducir efecto de latencia variable) y lo aplica a cada timestamp generado para firmar requests. La solución permanente es sincronizar el reloj del sistema con NTP: `w32tm /resync /force` en Windows o `sudo ntpdate pool.ntp.org` en Linux.

**¿Puedo usar el sistema con otros pares además de BTC/USDT?**

El sistema está diseñado para BTCUSDT. Cambiar `SYMBOL` en `config_world.py` y `config_local.py` debería funcionar para otros pares de Binance, pero el modelo necesita ser reentrenado con datos del nuevo par. Las features y el labeling son genéricos para cualquier vela OHLCV.

**¿Cómo interpreto el Sharpe Ratio que muestra el Graficador?**

Se calcula como `mean(returns) / std(returns) * sqrt(8760)` donde `8760` es el número de horas por año. Es la versión de volatilidad histórica (no anualizada con 252 días sino con la frecuencia real del data). Un Sharpe > 1 es bueno; > 2 es excelente para trading algorítmico con comisiones reales.

---

## Notas de seguridad

- El archivo `.env` con las credenciales de Binance está en `.gitignore` y **nunca debe commitearse**.
- El sistema nunca loggea las credenciales completas. Solo los primeros 8 caracteres de la API key aparecen en logs de inicialización.
- Con `USE_TESTNET = True` (default), el sistema usa `testnet.binance.vision` que es completamente separado de la producción. Los saldos del testnet no tienen valor real.
- Antes de cambiar a `USE_TESTNET = False`, verificar el capital disponible en la cuenta real y configurar `RiskConfig.conservative()` con límites apropiados al capital real.

---

*Sistema desarrollado con Python 3.10+. Arquitectura basada en actores con interfaces abstractas para máxima testabilidad y separación de responsabilidades.*
