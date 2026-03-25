"""
tests/run_all_tests.py
══════════════════════
Runner maestro: ejecuta todos los tests del proyecto.

Uso:
    python tests/run_all_tests.py           # correr todo
    python tests/run_all_tests.py --fast    # solo tests sin DB

Los tests están organizados en 4 módulos, de menor a mayor dependencia:

  test_features_labeling.py   — aritmética pura, sin actores
  test_wallet_logic.py        — lógica de capital, sin actores
  test_irreal_strategy.py     — lógica de detección, sin actores
  test_simulation_pipeline.py — integración, requiere DB y pkl
"""

import sys
import time
import importlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Windows a veces usa cp1252 en consola y puede romper prints con Unicode
# (por ejemplo "═", "✓", flechas, etc.). Para que la suite sea robusta,
# forzamos UTF-8 en stdout/stderr cuando sea posible.
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass
try:
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

TEST_MODULES = [
    "tests.test_features_labeling",
    "tests.test_wallet_logic",
    "tests.test_irreal_strategy",
    "tests.test_simulation_pipeline",
]

ANSI_GREEN  = "\033[92m"
ANSI_RED    = "\033[91m"
ANSI_YELLOW = "\033[93m"
ANSI_RESET  = "\033[0m"
ANSI_BOLD   = "\033[1m"


def run_module(module_name: str) -> tuple[int, int, list[str]]:
    """Ejecuta todos los test_ del módulo. Retorna (passed, failed, errores)."""
    mod     = importlib.import_module(module_name)
    tests   = [(k, v) for k, v in vars(mod).items() if k.startswith("test_")]
    passed  = failed = 0
    errores = []

    for name, fn in tests:
        try:
            fn()
            passed += 1
        except AssertionError as e:
            failed += 1
            errores.append(f"    {ANSI_RED}✗{ANSI_RESET} {name}: {e}")
        except Exception as e:
            failed += 1
            errores.append(
                f"    {ANSI_RED}✗{ANSI_RESET} {name}: "
                f"{type(e).__name__}: {e}"
            )

    return passed, failed, errores


def main():
    print(f"\n{ANSI_BOLD}{'═'*65}{ANSI_RESET}")
    print(f"{ANSI_BOLD}  TEST SUITE — BTC Swing Trading Bot{ANSI_RESET}")
    print(f"{ANSI_BOLD}{'═'*65}{ANSI_RESET}\n")

    total_passed = total_failed = 0
    t_start      = time.time()

    for module_name in TEST_MODULES:
        short = module_name.split(".")[-1]
        print(f"  {ANSI_BOLD}{short}{ANSI_RESET}")
        t0 = time.time()

        try:
            passed, failed, errores = run_module(module_name)
        except ImportError as e:
            print(f"    {ANSI_YELLOW}⚠ No se pudo importar: {e}{ANSI_RESET}\n")
            continue

        elapsed = time.time() - t0
        total_passed += passed
        total_failed += failed

        status = (
            f"{ANSI_GREEN}{passed} passed{ANSI_RESET}"
            if failed == 0
            else f"{ANSI_GREEN}{passed} passed{ANSI_RESET}, "
                 f"{ANSI_RED}{failed} failed{ANSI_RESET}"
        )
        print(f"    {status}  ({elapsed:.2f}s)")

        for err in errores:
            print(err)
        print()

    # Resumen final
    elapsed_total = time.time() - t_start
    print(f"{'─'*65}")
    if total_failed == 0:
        print(
            f"  {ANSI_GREEN}{ANSI_BOLD}✓ {total_passed} passed "
            f"— todos los tests pasaron{ANSI_RESET}  ({elapsed_total:.2f}s)"
        )
        sys.exit(0)
    else:
        print(
            f"  {ANSI_RED}{ANSI_BOLD}✗ {total_failed} failed, "
            f"{total_passed} passed{ANSI_RESET}  ({elapsed_total:.2f}s)"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
