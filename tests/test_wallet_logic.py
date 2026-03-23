"""
tests/test_wallet_logic.py
══════════════════════════
Tests de la lógica de administración de capital (slots).

Reproduce la lógica exacta que existe en actors/wallet.py
verificada contra el JSON del backtest irreal
(backtest_results.json: $1000, 5 posiciones, comisión 0.1%).

Ejecutar:
    python tests/test_wallet_logic.py
    python -m pytest tests/test_wallet_logic.py -v
"""

import sys
from collections import deque
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ─────────────────────────────────────────────────────────────────────────────
# Simulador mínimo que replica MemoryWallet + SimulatedOrderBook
# (el mismo que se usó durante la calibración en la Etapa 4)
# ─────────────────────────────────────────────────────────────────────────────

COMMISSION = 0.001   # 0.1%
MAX_POS    = 5
CAPITAL    = 1000.0

class MiniWallet:
    """
    Replica exacta de la lógica de MemoryWallet relevante para los tests:
    - slot = usdt_balance / MAX_POSICIONES  (solo cuando positions=0)
    - slot INMUTABLE mientras haya posiciones
    - BUY: gasta slot, btc = (slot - comision) / precio
    - SELL: vende btc_en_posiciones / positions_count (partes iguales FIFO)
    - Ignorados: no modifican estado
    """

    def __init__(self, capital=CAPITAL, max_pos=MAX_POS, commission=COMMISSION):
        self.capital    = capital
        self.max_pos    = max_pos
        self.commission = commission
        self.usdt       = capital
        self.slot       = capital / max_pos
        self.positions  = deque()   # (precio_entrada, btc_comprado)
        self.trades_log = []

    @property
    def positions_count(self):
        return len(self.positions)

    def btc_en_posiciones(self):
        return sum(btc for _, btc in self.positions)

    def portfolio_value(self, precio_actual):
        return self.usdt + self.btc_en_posiciones() * precio_actual

    def buy(self, precio):
        """Intenta ejecutar BUY. Retorna True si ejecutado, False si ignorado."""
        if self.positions_count >= self.max_pos:
            self.trades_log.append({'tipo': 'BUY', 'ignorado': True,
                                    'motivo': f'max_posiciones({self.max_pos})'})
            return False
        if self.slot > self.usdt + 1e-9:
            self.trades_log.append({'tipo': 'BUY', 'ignorado': True,
                                    'motivo': 'usdt_insuficiente'})
            return False
        comision = self.slot * self.commission
        btc      = (self.slot - comision) / precio
        self.usdt -= self.slot
        self.positions.append((precio, btc))
        self.trades_log.append({'tipo': 'BUY', 'precio': precio,
                                'usdt_spent': self.slot, 'btc': btc,
                                'ignorado': False})
        return True

    def sell(self, precio):
        """Intenta ejecutar SELL. Retorna (usdt_recibido, ret) o (None, None)."""
        if self.positions_count == 0:
            self.trades_log.append({'tipo': 'SELL', 'ignorado': True,
                                    'motivo': 'sin_posiciones'})
            return None, None
        btc_total = self.btc_en_posiciones()
        btc_venta = btc_total / self.positions_count
        p_entrada = self.positions.popleft()[0]
        usdt_bruto = btc_venta * precio
        comision   = usdt_bruto * self.commission
        usdt_neto  = usdt_bruto - comision
        self.usdt += usdt_neto
        ret = (precio - p_entrada) / p_entrada
        if self.positions_count == 0:
            self.slot = self.usdt / self.max_pos
        self.trades_log.append({'tipo': 'SELL', 'precio': precio,
                                'btc': btc_venta, 'usdt_rec': usdt_neto,
                                'ret': ret, 'ignorado': False})
        return usdt_neto, ret


# ─────────────────────────────────────────────────────────────────────────────
# TESTS: slot inicial
# ─────────────────────────────────────────────────────────────────────────────

def test_slot_inicial_es_capital_dividido_max_pos():
    """slot = 1000 / 5 = 200."""
    w = MiniWallet(capital=1000.0, max_pos=5)
    assert abs(w.slot - 200.0) < 1e-9


def test_slot_inicial_otros_parametros():
    """Funciona con cualquier capital y max_pos."""
    w = MiniWallet(capital=2000.0, max_pos=4)
    assert abs(w.slot - 500.0) < 1e-9


# ─────────────────────────────────────────────────────────────────────────────
# TESTS: BUY
# ─────────────────────────────────────────────────────────────────────────────

def test_buy_descuenta_slot_exacto_de_usdt():
    """Cada BUY descuenta exactamente slot del balance USDT."""
    w    = MiniWallet()
    usdt_antes = w.usdt
    w.buy(50_000.0)
    assert abs(w.usdt - (usdt_antes - w.slot)) < 1e-9


def test_buy_btc_calculado_con_comision():
    """btc = (slot - slot*commission) / precio."""
    w    = MiniWallet()
    slot = w.slot          # 200.0
    w.buy(50_000.0)
    btc_esperado = (slot - slot * COMMISSION) / 50_000.0
    btc_real     = w.positions[0][1]
    assert abs(btc_real - btc_esperado) < 1e-10


def test_buy_incrementa_positions_count():
    w = MiniWallet()
    assert w.positions_count == 0
    w.buy(50_000.0)
    assert w.positions_count == 1
    w.buy(48_000.0)
    assert w.positions_count == 2


def test_buy_multiples_gastan_mismo_slot():
    """El slot es fijo entre compras del mismo ciclo."""
    w    = MiniWallet()
    w.buy(50_000.0)
    w.buy(48_000.0)
    t1 = [t for t in w.trades_log if not t['ignorado'] and t['tipo']=='BUY'][0]
    t2 = [t for t in w.trades_log if not t['ignorado'] and t['tipo']=='BUY'][1]
    assert abs(t1['usdt_spent'] - t2['usdt_spent']) < 1e-9


def test_buy_ignorado_si_max_posiciones():
    """BUY ignorado cuando ya se alcanzó el máximo de posiciones."""
    w = MiniWallet(max_pos=2)
    w.buy(50_000.0)
    w.buy(48_000.0)
    resultado = w.buy(46_000.0)   # debe ser ignorado
    assert resultado is False
    assert w.positions_count == 2
    ignorados = [t for t in w.trades_log if t['ignorado']]
    assert len(ignorados) == 1
    assert 'max_posiciones' in ignorados[0]['motivo']


def test_buy_ignorado_no_modifica_estado():
    """El balance y positions no cambian cuando el BUY es ignorado."""
    w = MiniWallet(max_pos=1)
    w.buy(50_000.0)
    usdt_antes = w.usdt
    pos_antes  = w.positions_count
    w.buy(48_000.0)   # ignorado
    assert w.usdt == usdt_antes
    assert w.positions_count == pos_antes


# ─────────────────────────────────────────────────────────────────────────────
# TESTS: slot inmutable mientras hay posiciones
# ─────────────────────────────────────────────────────────────────────────────

def test_slot_no_cambia_con_posiciones_abiertas():
    """El slot permanece fijo mientras haya al menos 1 posición abierta."""
    w          = MiniWallet()
    slot_init  = w.slot
    w.buy(50_000.0)                # positions_count=1
    assert abs(w.slot - slot_init) < 1e-9
    w.buy(48_000.0)                # positions_count=2
    assert abs(w.slot - slot_init) < 1e-9


# ─────────────────────────────────────────────────────────────────────────────
# TESTS: SELL y lógica FIFO + partes iguales
# ─────────────────────────────────────────────────────────────────────────────

def test_sell_btc_es_parte_igual_de_posiciones():
    """btc_vendido = btc_en_posiciones / positions_count."""
    w = MiniWallet()
    w.buy(50_000.0)
    w.buy(48_000.0)
    btc_total     = w.btc_en_posiciones()
    btc_esperado  = btc_total / 2          # 2 posiciones
    _, _          = w.sell(52_000.0)
    btc_vendido   = [t for t in w.trades_log if not t['ignorado']
                     and t['tipo']=='SELL'][0]['btc']
    assert abs(btc_vendido - btc_esperado) < 1e-10


def test_sell_fifo_cierra_posicion_mas_antigua():
    """SELL cierra la posición con precio de entrada más antiguo."""
    w = MiniWallet()
    w.buy(50_000.0)   # posición 1
    w.buy(48_000.0)   # posición 2
    w.sell(52_000.0)
    # Debe quedar la posición 2 (precio 48000)
    assert w.positions_count == 1
    assert abs(w.positions[0][0] - 48_000.0) < 1e-6


def test_sell_usdt_recibido_correcto():
    """usdt_recibido = btc_vendido * precio * (1 - commission)."""
    w = MiniWallet()
    w.buy(50_000.0)
    precio_venta  = 52_000.0
    btc_total     = w.btc_en_posiciones()
    btc_venta     = btc_total / 1           # 1 posición
    usdt_esperado = btc_venta * precio_venta * (1 - COMMISSION)
    usdt_rec, _   = w.sell(precio_venta)
    assert abs(usdt_rec - usdt_esperado) < 1e-8


def test_sell_ignorado_sin_posiciones():
    """SELL ignorado cuando no hay posiciones abiertas."""
    w = MiniWallet()
    usdt_antes = w.usdt
    resultado, _ = w.sell(52_000.0)
    assert resultado is None
    assert w.usdt == usdt_antes
    ignorados = [t for t in w.trades_log if t['ignorado']]
    assert len(ignorados) == 1
    assert ignorados[0]['motivo'] == 'sin_posiciones'


# ─────────────────────────────────────────────────────────────────────────────
# TESTS: recálculo del slot al llegar a cero posiciones
# ─────────────────────────────────────────────────────────────────────────────

def test_slot_se_recalcula_cuando_positions_llegan_a_cero():
    """Cuando positions_count llega a 0, slot = usdt_balance / max_pos."""
    w = MiniWallet()
    w.buy(50_000.0)
    # Vender con ganancia → capital > 1000
    w.sell(55_000.0)
    assert w.positions_count == 0
    slot_esperado = w.usdt / MAX_POS
    assert abs(w.slot - slot_esperado) < 1e-9
    assert w.slot > 200.0   # creció porque hubo ganancia


def test_slot_no_se_recalcula_si_quedan_posiciones():
    """Si quedan posiciones abiertas, el slot NO se toca."""
    w = MiniWallet()
    w.buy(50_000.0)
    w.buy(48_000.0)
    slot_antes = w.slot
    w.sell(52_000.0)   # queda 1 posición
    assert w.positions_count == 1
    assert abs(w.slot - slot_antes) < 1e-9


def test_slot_recalculo_compounding():
    """
    Ciclo completo: abrir 2 pos → cerrar 2 pos con ganancia.
    El nuevo slot debe ser > slot inicial (compounding natural).
    """
    w = MiniWallet()
    slot_inicial = w.slot    # 200.0
    w.buy(50_000.0)
    w.buy(49_000.0)
    w.sell(53_000.0)         # cierra posición 1 con ganancia
    w.sell(53_000.0)         # cierra posición 2, llega a 0 posiciones
    assert w.positions_count == 0
    assert w.slot > slot_inicial    # compounding ocurrió


# ─────────────────────────────────────────────────────────────────────────────
# TESTS: portfolio_value
# ─────────────────────────────────────────────────────────────────────────────

def test_portfolio_value_sin_posiciones():
    """Con 0 posiciones, portfolio = usdt_balance."""
    w = MiniWallet()
    assert abs(w.portfolio_value(50_000.0) - 1000.0) < 1e-9


def test_portfolio_value_con_posiciones():
    """portfolio = usdt_libre + btc_en_posiciones * precio_actual."""
    w = MiniWallet()
    w.buy(50_000.0)
    precio_actual = 52_000.0
    esperado      = w.usdt + w.btc_en_posiciones() * precio_actual
    assert abs(w.portfolio_value(precio_actual) - esperado) < 1e-6


# ─────────────────────────────────────────────────────────────────────────────
# TESTS: simulación completa del flujo del irreal
# ─────────────────────────────────────────────────────────────────────────────

def test_flujo_completo_5_posiciones():
    """
    Simula el flujo completo: abrir 5 posiciones (1 slot cada una) y
    cerrarlas todas. Verifica que se consumen exactamente 5 slots = 1000 USDT.
    """
    w = MiniWallet()
    precio = 50_000.0

    for _ in range(5):
        resultado = w.buy(precio)
        assert resultado is True

    assert w.positions_count == 5
    assert abs(w.usdt) < 1e-6    # usdt agotado (5 × 200 = 1000)

    # La sexta compra debe ser ignorada
    resultado = w.buy(precio)
    assert resultado is False

    # Cerrar todas
    for _ in range(5):
        rec, ret = w.sell(precio * 1.05)   # +5% en cada una
        assert rec is not None

    assert w.positions_count == 0
    assert w.portfolio_value(precio) > 1000.0   # ganó capital


def test_compounding_tres_ciclos():
    """
    3 ciclos de compra/venta con ganancia. El capital debe crecer
    cada ciclo y el slot también (compounding natural).
    """
    w           = MiniWallet()
    slots_ciclo = []

    for ciclo in range(3):
        slots_ciclo.append(w.slot)
        w.buy(50_000.0 + ciclo * 1000)
        w.sell(55_000.0 + ciclo * 1000)   # siempre con ganancia
        assert w.positions_count == 0

    # Cada ciclo el slot creció
    assert slots_ciclo[1] > slots_ciclo[0]
    assert slots_ciclo[2] > slots_ciclo[1]


# ─────────────────────────────────────────────────────────────────────────────
# Runner sin pytest
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [(k, v) for k, v in globals().items() if k.startswith("test_")]
    passed = failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  ✓ {name}")
            passed += 1
        except AssertionError as e:
            print(f"  ✗ {name}: {e}")
            failed += 1
        except Exception as e:
            print(f"  ✗ {name}: {type(e).__name__}: {e}")
            failed += 1
    print(f"\n{'='*50}")
    print(f"  {passed} passed, {failed} failed")
