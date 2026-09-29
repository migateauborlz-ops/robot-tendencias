"""
Pruebas del estimador de rezago entre mercados (Modulo 3).

No consultan Google Trends: trabajan sobre series sinteticas y simulaciones, asi
que corren sin red y sin costo.

Run:  ./venv/Scripts/python.exe test_trend_lag.py
"""
import sys

import numpy as np
import pandas as pd

from src.cross_validation import TrendValidator, MIN_LAG_CORRELATION

PASSED = 0
FAILED = 0


def check(name, condition, detail=""):
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"  [OK]   {name}")
    else:
        FAILED += 1
        print(f"  [FALLA] {name}" + (f" -> {detail}" if detail else ""))


# Curva tipica de una tendencia: sube, hace pico y decae.
CURVA = np.array([10, 12, 15, 20, 28, 40, 55, 70, 80, 85, 88, 90, 87, 84, 80,
                  75, 70, 66, 62, 60, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49],
                 dtype=float)


def desplazar(serie, dias):
    if dias == 0:
        return serie.copy()
    return np.concatenate([np.full(dias, serie[0]), serie[:-dias]])


def test_recupera_el_rezago():
    print("\n1. Recuperacion de un rezago conocido")
    for k in (0, 1, 3, 5, 7, 10):
        lag, corr = TrendValidator.estimate_lag_days(
            pd.Series(CURVA), pd.Series(desplazar(CURVA, k)))
        check(f"Desplazamiento real de {k} dias", lag == k,
              f"estimado {lag} (r={corr})")


def test_series_insuficientes():
    print("\n2. Series demasiado cortas o vacias")
    lag, corr = TrendValidator.estimate_lag_days(pd.Series(CURVA[:10]),
                                                 pd.Series(CURVA[:10]))
    check("Menos de 16 puntos devuelve None", lag is None, f"{lag}")
    check("Serie None no lanza excepcion",
          TrendValidator.estimate_lag_days(None, pd.Series(CURVA)) == (None, 0.0))
    plana = pd.Series(np.full(30, 42.0))
    lag, corr = TrendValidator.estimate_lag_days(plana, plana)
    check("Serie constante no reporta rezago", lag is None, f"{lag} (r={corr})")


def test_control_de_falsos_positivos():
    print("\n3. Control de correlacion espuria")
    rng = np.random.default_rng(7)
    falsos = 0
    total = 400
    for _ in range(total):
        a = pd.Series(np.cumsum(rng.normal(0, 1, 30)))
        b = pd.Series(np.cumsum(rng.normal(0, 1, 30)))
        if TrendValidator.estimate_lag_days(a, b)[0] is not None:
            falsos += 1
    tasa = 100.0 * falsos / total
    print(f"       tasa observada de falsos positivos: {tasa:.1f}%")
    check("Falsos positivos por debajo del 10% sobre paseos aleatorios",
          tasa < 10.0, f"{tasa:.1f}%")
    check(f"El umbral documentado es {MIN_LAG_CORRELATION}",
          abs(MIN_LAG_CORRELATION - 0.60) < 1e-9)


def test_ruido_sobre_senal():
    print("\n4. Señal real con ruido moderado")
    rng = np.random.default_rng(3)
    detectados = 0
    for _ in range(50):
        ruido_a = rng.normal(0, 3, len(CURVA))
        ruido_b = rng.normal(0, 3, len(CURVA))
        a = pd.Series(CURVA + ruido_a)
        b = pd.Series(desplazar(CURVA, 5) + ruido_b)
        lag, _ = TrendValidator.estimate_lag_days(a, b)
        if lag is not None and abs(lag - 5) <= 1:
            detectados += 1
    print(f"       aciertos con ruido: {detectados}/50")
    check("Detecta el rezago de 5 dias en la mayoria de los casos ruidosos",
          detectados >= 35, f"{detectados}/50")


def test_configuracion_de_geografia():
    print("\n5. Configuracion de geografia")
    v = TrendValidator(geo="", geo_local="CO", compare_geos=True)
    check("Deteccion mundial por defecto", v.geo == "")
    check("Mercado local configurable", v.geo_local == "CO")
    check("Comparacion activada", v.compare_geos is True)

    v2 = TrendValidator(geo="CO", geo_local="CO", compare_geos=True)
    check("Comparar un mercado consigo mismo se desactiva sola",
          v2.compare_geos is False)

    v3 = TrendValidator(geo="US", geo_local="CO", compare_geos=False)
    check("La comparacion se puede apagar", v3.compare_geos is False)
    check("La geografia de deteccion se respeta", v3.geo == "US")

    # TrendReq pide una cookie a Google dentro de su constructor. Si el
    # validador la creara al instanciarse, estas mismas comprobaciones -- que
    # son puramente de configuracion -- dependerian de que Google responda, y
    # en un runner de CI eso falla de forma intermitente.
    check("Construir el validador no abre sesion con Google",
          v.pytrends is None and v3.pytrends is None,
          f"pytrends={v.pytrends!r}")
    check("La geografia de la sesion arranca sin fijar",
          v._last_geo is None, f"_last_geo={v._last_geo!r}")


if __name__ == "__main__":
    print("=" * 68)
    print("Pruebas del estimador de rezago entre mercados (sin red, sin costo)")
    print("=" * 68)
    test_recupera_el_rezago()
    test_series_insuficientes()
    test_control_de_falsos_positivos()
    test_ruido_sobre_senal()
    test_configuracion_de_geografia()
    print("\n" + "=" * 68)
    print(f"RESULTADO: {PASSED} correctas, {FAILED} fallidas")
    print("=" * 68)
    sys.exit(1 if FAILED else 0)
