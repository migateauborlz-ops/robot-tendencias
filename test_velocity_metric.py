"""
Pruebas de la metrica de velocidad de interaccion (Modulo 4).

Contrastan la metrica nueva contra la anterior usando la muestra real de 30
publicaciones descargada el 2026-08-20, que esta guardada en data/raw_dumps/.
No llaman a ninguna API: no cuestan nada.

Run:  ./venv/Scripts/python.exe test_velocity_metric.py
"""
import glob
import json
import sys
from datetime import timedelta

import pandas as pd

from src.config import Config
from src.database import StorageAndScoring, MIN_AGE_DAYS

PASSED = 0
FAILED = 0
SKIPPED = 0


def check(name, condition, detail=""):
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"  [OK]   {name}")
    else:
        FAILED += 1
        print(f"  [FALLA] {name}" + (f" -> {detail}" if detail else ""))


def skip(motivo, comprobaciones):
    """
    Registra las comprobaciones que no se pudieron ejecutar.

    La muestra real vive en data/raw_dumps/, que no se versiona porque contiene
    material raspado. En CI esas comprobaciones no corren, y sin contarlas la
    suite reportaba "todo correcto" con tres verificaciones menos: el total
    bajaba de 130 a 127 sin que nadie lo notara.
    """
    global SKIPPED
    SKIPPED += comprobaciones
    print(f"  [OMITIDA] {motivo} ({comprobaciones} comprobaciones)")


NOW = pd.Timestamp.now(tz="UTC")


def fila(likes, shares, dias):
    return {"likes_count": likes, "shares_count": shares,
            "timestamp": NOW - timedelta(days=dias)}


def test_velocidad_basica():
    print("\n1. Calculo de la velocidad")
    v = StorageAndScoring._engagement_velocity(fila(1000, 0, 10), NOW)
    check("1.000 interacciones en 10 dias = 100/dia", abs(v - 100) < 0.5, f"{v:.2f}")

    v = StorageAndScoring._engagement_velocity(fila(900, 100, 10), NOW)
    check("Los shares suman a la interaccion", abs(v - 100) < 0.5, f"{v:.2f}")

    viejo = StorageAndScoring._engagement_velocity(fila(8_400_000, 0, 1518), NOW)
    nuevo = StorageAndScoring._engagement_velocity(fila(1_700_000, 0, 56), NOW)
    check("Un post de 56 dias con 1,7M supera a uno de 1.518 dias con 8,4M",
          nuevo > viejo, f"nuevo={nuevo:,.0f}/dia viejo={viejo:,.0f}/dia")

    v = StorageAndScoring._engagement_velocity(fila(10, 0, 0), NOW)
    check("La antiguedad tiene piso: no divide por cero",
          abs(v - (10 / MIN_AGE_DAYS)) < 0.01, f"{v:.2f}")

    v = StorageAndScoring._engagement_velocity(
        {"likes_count": 5000, "shares_count": 0, "timestamp": None}, NOW)
    esperado = 5000 / float(Config.INGESTION_WINDOW_DAYS)
    check("Sin fecha asume la antiguedad maxima de la ventana",
          abs(v - esperado) < 0.01, f"{v:.2f} vs {esperado:.2f}")

    v = StorageAndScoring._engagement_velocity(
        {"likes_count": None, "shares_count": None, "timestamp": NOW}, NOW)
    check("Contadores nulos no lanzan excepcion", v == 0.0, f"{v}")


def test_normalizacion():
    print("\n2. Normalizacion a escala 0-1")
    s = StorageAndScoring._normalize_velocity(pd.Series([100.0, 50.0, 10.0, 0.0]))
    check("El maximo obtiene 1,0", abs(s.iloc[0] - 1.0) < 1e-9)
    check("El cero obtiene 0,0", abs(s.iloc[3]) < 1e-9)
    check("Es monotona decreciente", list(s) == sorted(s, reverse=True))
    check("Todo queda dentro de [0,1]", bool(((s >= 0) & (s <= 1)).all()))

    # El motivo del log: con una cola larga, la normalizacion lineal aplasta todo.
    cola = pd.Series([100000.0, 500.0, 300.0, 100.0])
    lineal = cola / cola.max()
    logaritmica = StorageAndScoring._normalize_velocity(cola)
    check("El log evita que la cola se aplaste a cero",
          logaritmica.iloc[1] > lineal.iloc[1] * 10,
          f"log={logaritmica.iloc[1]:.3f} lineal={lineal.iloc[1]:.5f}")

    s0 = StorageAndScoring._normalize_velocity(pd.Series([0.0, 0.0]))
    check("Serie toda en cero no produce division invalida", bool((s0 == 0).all()))


def test_contra_muestra_real():
    print("\n3. Contraste sobre la muestra real de 30 publicaciones")
    archivos = sorted(glob.glob("data/raw_dumps/tiktok_videos_*.json"))
    if not archivos:
        skip("no hay muestra en data/raw_dumps", 3)
        return

    items = json.load(open(archivos[-1], encoding="utf-8"))
    filas = []
    for it in items:
        ts = it.get("createTimeISO")
        if not ts:
            continue
        filas.append({"likes_count": it.get("diggCount") or 0,
                      "shares_count": it.get("shareCount") or 0,
                      "timestamp": pd.to_datetime(ts, utc=True)})
    df = pd.DataFrame(filas)
    if df.empty:
        skip("la muestra no trae fechas", 3)
        return

    df["edad"] = (NOW - df["timestamp"]).dt.total_seconds() / 86400.0
    df["velocidad"] = df.apply(lambda r: StorageAndScoring._engagement_velocity(r, NOW), axis=1)

    top_likes = df.nlargest(1, "likes_count").iloc[0]
    top_vel = df.nlargest(1, "velocidad").iloc[0]
    print(f"  Lider por likes acumulados : {int(top_likes.likes_count):,} likes, "
          f"{top_likes.edad:.0f} dias")
    print(f"  Lider por velocidad        : {int(top_vel.likes_count):,} likes, "
          f"{top_vel.edad:.0f} dias")

    check("La metrica nueva elige una publicacion mas reciente que la anterior",
          top_vel.edad < top_likes.edad,
          f"{top_vel.edad:.0f} vs {top_likes.edad:.0f} dias")

    edad_media_top5_likes = df.nlargest(5, "likes_count")["edad"].mean()
    edad_media_top5_vel = df.nlargest(5, "velocidad")["edad"].mean()
    print(f"  Edad media del top 5 por likes     : {edad_media_top5_likes:>7.0f} dias")
    print(f"  Edad media del top 5 por velocidad : {edad_media_top5_vel:>7.0f} dias")
    check("El top 5 por velocidad es en promedio mas reciente",
          edad_media_top5_vel < edad_media_top5_likes)

    dentro = int((df["edad"] <= Config.INGESTION_WINDOW_DAYS).sum())
    print(f"  Publicaciones dentro de la ventana de {Config.INGESTION_WINDOW_DAYS} dias: "
          f"{dentro} de {len(df)}")
    check("La ventana de 30 dias admite mas publicaciones que la de 7",
          dentro > int((df["edad"] <= 7).sum()),
          f"{dentro} vs {int((df['edad'] <= 7).sum())}")


def test_formula_completa():
    print("\n4. Formula del Opportunity Score")
    store = StorageAndScoring.__new__(StorageAndScoring)  # sin tocar la base de datos
    df_nlp = pd.DataFrame([
        {**fila(20000, 0, 10), "extracted_products": ["lampara led"],
         "purchase_intent_score": 0.8},
        {**fila(100, 0, 200), "extracted_products": ["reloj viejo"],
         "purchase_intent_score": 0.1},
    ])
    df_val = pd.DataFrame([
        {"product_name": "lampara led", "trend_growth": 0.5},
        {"product_name": "reloj viejo", "trend_growth": 0.0},
    ])
    out = store.calculate_opportunity_score(df_nlp, df_val)
    check("Devuelve una fila por producto validado", len(out) == 2, f"{len(out)}")
    check("Incluye la columna de velocidad", "engagement_velocity" in out.columns)
    check("El producto emergente queda primero",
          out.iloc[0]["product_name"] == "lampara led",
          str(out["product_name"].tolist()))
    check("El puntaje se mantiene dentro de [0,1]",
          bool(((out["opportunity_score"] >= 0) & (out["opportunity_score"] <= 1)).all()))

    fila_top = out.iloc[0]
    esperado = (0.4 * fila_top["viral_metric_score"]
                + 0.4 * fila_top["purchase_intent_score"]
                + 0.2 * fila_top["norm_trend_growth"])
    check("Los pesos siguen siendo 0,4 / 0,4 / 0,2",
          abs(fila_top["opportunity_score"] - esperado) < 1e-9)


if __name__ == "__main__":
    print("=" * 68)
    print("Pruebas de la metrica de velocidad (sin costo de API)")
    print("=" * 68)
    test_velocidad_basica()
    test_normalizacion()
    test_contra_muestra_real()
    test_formula_completa()
    print("\n" + "=" * 68)
    print(f"RESULTADO: {PASSED} correctas, {FAILED} fallidas, {SKIPPED} omitidas")
    print("=" * 68)
    sys.exit(1 if FAILED else 0)
