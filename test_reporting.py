"""
Pruebas de la capa de reportes (Modulo 5).

Cubren lo que el tablero le afirma a una PYME: que la descomposicion del score
suma exactamente el score, que una ventana de adopcion no medible se comunica
como tal y no como cero dias, y que la lectura nunca modifica la base. Usan
SQLite en archivos temporales; no llaman a ninguna API ni requieren Streamlit.

Run:  ./venv/Scripts/python.exe test_reporting.py
"""
import os
import shutil
import sys
import tempfile

import pandas as pd
from sqlalchemy import create_engine, inspect, text

from src.database import (StorageAndScoring, SCORE_WEIGHTS, Base,
                          FinalTrends, RawSocialData)
from src.reporting import (TrendReport, export_columns, TREND_COLUMNS,
                           LAG_MEASURED, LAG_NOT_MEASURABLE, _adoption_label)

PASSED = 0
FAILED = 0
_TMPDIRS = []


def check(name, condition, detail=""):
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"  [OK]   {name}")
    else:
        FAILED += 1
        print(f"  [FALLA] {name}")
        if detail:
            print(f"          {detail}")


def _tmp_db():
    """Devuelve una URL SQLAlchemy a una base temporal vacia."""
    d = tempfile.mkdtemp(prefix="reporting_test_")
    _TMPDIRS.append(d)
    return "sqlite:///" + os.path.join(d, "trends.db").replace("\\", "/")


def _poblar(db_url, filas, raw=None):
    """Crea el esquema real y siembra final_trends (y opcionalmente raw)."""
    engine = create_engine(db_url, future=True)
    Base.metadata.create_all(engine)
    from sqlalchemy.orm import sessionmaker
    Session = sessionmaker(bind=engine)
    with Session() as s:
        for f in filas:
            s.add(FinalTrends(**f))
        for r in (raw or []):
            s.add(RawSocialData(**r))
        s.commit()
    engine.dispose()


# ---------------------------------------------------------------------------

def test_base_vacia():
    print("\n[1] Base sin datos")

    db = _tmp_db()
    rep = TrendReport(db)
    df = rep.load_trends()

    check("no lanza excepcion con la base vacia", isinstance(df, pd.DataFrame))
    check("devuelve un frame vacio", df.empty, f"filas={len(df)}")
    check("conserva las columnas declaradas",
          list(df.columns) == TREND_COLUMNS,
          f"columnas={list(df.columns)}")

    cob = rep.coverage()
    check("cobertura en cero sin datos",
          cob["posts"] == 0 and cob["productos_validados"] == 0,
          str(cob))
    check("cobertura de comentarios es 0.0 y no NaN",
          cob["cobertura_comentarios"] == 0.0)


def test_no_escribe():
    print("\n[2] La lectura no altera el esquema")

    db = _tmp_db()
    rep = TrendReport(db)
    rep.load_trends()
    rep.coverage()

    engine = create_engine(db, future=True)
    tablas = inspect(engine).get_table_names()
    engine.dispose()

    check("no se crearon tablas al leer", tablas == [], f"tablas={tablas}")


def test_descomposicion_suma():
    print("\n[3] La descomposicion del score suma el score")

    viral, intent, growth = 0.80, 0.55, 0.30
    score = (SCORE_WEIGHTS["viral"] * viral
             + SCORE_WEIGHTS["intent"] * intent
             + SCORE_WEIGHTS["trend"] * growth)

    db = _tmp_db()
    _poblar(db, [dict(product_name="lampara led", viral_metric_score=viral,
                      purchase_intent_score=intent,
                      google_trend_growth_pct=growth,
                      engagement_velocity=1200.0, trend_geo="WORLD",
                      opportunity_score=score, lag_days=6,
                      lag_correlation=0.71, adoption_gap=0.22)])

    fila = TrendReport(db).load_trends().iloc[0]
    suma = (fila["aporte_viralidad"] + fila["aporte_intencion"]
            + fila["aporte_tendencia"])

    check("los tres aportes suman el opportunity_score",
          abs(suma - fila["opportunity_score"]) < 1e-9,
          f"suma={suma:.6f} score={fila['opportunity_score']:.6f}")
    check("el aporte de viralidad usa el peso correcto",
          abs(fila["aporte_viralidad"] - SCORE_WEIGHTS["viral"] * viral) < 1e-9)


def test_crecimiento_extremo_se_recorta():
    print("\n[4] Un crecimiento de busquedas desbordado no rompe la suma")

    # El pipeline recorta el crecimiento a 1.0 antes de puntuar; si el reporte
    # no aplicara el mismo recorte, un producto con 400% de crecimiento
    # mostraria aportes que suman mas que su propio score.
    viral, intent, growth = 0.10, 0.20, 4.00
    score = (SCORE_WEIGHTS["viral"] * viral
             + SCORE_WEIGHTS["intent"] * intent
             + SCORE_WEIGHTS["trend"] * 1.0)

    db = _tmp_db()
    _poblar(db, [dict(product_name="botella termica", viral_metric_score=viral,
                      purchase_intent_score=intent,
                      google_trend_growth_pct=growth,
                      engagement_velocity=50.0, trend_geo="WORLD",
                      opportunity_score=score)])

    fila = TrendReport(db).load_trends().iloc[0]
    suma = (fila["aporte_viralidad"] + fila["aporte_intencion"]
            + fila["aporte_tendencia"])

    check("el aporte de busquedas se recorta en 0.2",
          abs(fila["aporte_tendencia"] - SCORE_WEIGHTS["trend"]) < 1e-9,
          f"aporte={fila['aporte_tendencia']:.6f}")
    check("la suma sigue coincidiendo con el score",
          abs(suma - fila["opportunity_score"]) < 1e-9,
          f"suma={suma:.6f} score={fila['opportunity_score']:.6f}")


def test_ventana_no_medible():
    print("\n[5] Una ventana no medible se comunica, no se muestra como cero")

    db = _tmp_db()
    _poblar(db, [
        dict(product_name="con ventana", viral_metric_score=0.5,
             purchase_intent_score=0.5, google_trend_growth_pct=0.2,
             engagement_velocity=10.0, opportunity_score=0.44,
             lag_days=9, lag_correlation=0.68),
        dict(product_name="sin ventana", viral_metric_score=0.5,
             purchase_intent_score=0.5, google_trend_growth_pct=0.2,
             engagement_velocity=10.0, opportunity_score=0.43,
             lag_days=None, lag_correlation=0.31),
    ])

    df = TrendReport(db).load_trends().set_index("product_name")

    check("el producto con rezago se marca como medido",
          df.loc["con ventana", "confianza_ventana"] == LAG_MEASURED)
    check("conserva el numero de dias",
          df.loc["con ventana", "ventana_dias"] == 9)
    check("los dias quedan como entero, no como flotante",
          str(df["ventana_dias"].dtype) == "Int64",
          f"dtype={df['ventana_dias'].dtype}")
    check("el producto sin rezago se marca como no medible",
          df.loc["sin ventana", "confianza_ventana"] == LAG_NOT_MEASURABLE)
    # El riesgo concreto es que una ventana no medida llegue al tablero como 0
    # y se lea como "no hay margen", que es lo contrario de "no se sabe".
    faltante = df.loc["sin ventana", "ventana_dias"]
    check("no inventa cero dias cuando no hay medicion",
          pd.isna(faltante) and not (faltante is not pd.NA and faltante == 0),
          f"valor={faltante!r}")


def test_etiquetas_de_adopcion():
    print("\n[6] Etiquetas de la brecha de adopcion")

    check("brecha positiva = aun no llega",
          _adoption_label(0.40) == "aun no llega al mercado local")
    check("brecha negativa = el local va adelante",
          _adoption_label(-0.40) == "el mercado local ya va adelante")
    check("brecha nula = mismo ritmo",
          _adoption_label(0.0) == "ambos mercados al mismo ritmo")
    check("brecha ausente se declara sin comparacion",
          _adoption_label(None) == "sin comparacion local")
    check("NaN se trata como ausencia",
          _adoption_label(float("nan")) == "sin comparacion local")


def test_cobertura_de_comentarios():
    print("\n[7] Cobertura de comentarios")

    raw = [
        dict(platform="tiktok", video_id="1", comment_text="me encanta"),
        dict(platform="tiktok", video_id="2", comment_text="donde lo compro"),
        dict(platform="tiktok", video_id="3", comment_text=""),
        dict(platform="tiktok", video_id="4", comment_text=None),
        dict(platform="tiktok", video_id="5", comment_text="   "),
    ]

    db = _tmp_db()
    _poblar(db, [], raw=raw)

    cob = TrendReport(db).coverage()

    check("cuenta todas las publicaciones", cob["posts"] == 5,
          f"posts={cob['posts']}")
    check("solo cuenta las que traen texto real",
          cob["posts_con_comentarios"] == 2,
          f"con={cob['posts_con_comentarios']}")
    check("la cadena de espacios no cuenta como comentario",
          abs(cob["cobertura_comentarios"] - 0.4) < 1e-9,
          f"cobertura={cob['cobertura_comentarios']}")


def test_exportacion():
    print("\n[8] Exportacion a CSV")

    db = _tmp_db()
    _poblar(db, [dict(product_name="mini proyector", viral_metric_score=0.6,
                      purchase_intent_score=0.7, google_trend_growth_pct=0.5,
                      engagement_velocity=900.0, trend_geo="WORLD",
                      opportunity_score=0.62, lag_days=4,
                      lag_correlation=0.66, adoption_gap=0.18)])

    exportado = export_columns(TrendReport(db).load_trends())

    check("exporta una fila", len(exportado) == 1)
    check("no filtra columnas internas de normalizacion",
          "viral_metric_score" not in exportado.columns
          and "aporte_viralidad" not in exportado.columns,
          f"columnas={list(exportado.columns)}")
    check("incluye la confianza junto a la ventana",
          "ventana_dias" in exportado.columns
          and "confianza_ventana" in exportado.columns)
    check("el frame vacio conserva el encabezado",
          not export_columns(pd.DataFrame()).columns.empty)


if __name__ == "__main__":
    print("=" * 68)
    print("Pruebas de la capa de reportes (sin costo de API)")
    print("=" * 68)
    try:
        test_base_vacia()
        test_no_escribe()
        test_descomposicion_suma()
        test_crecimiento_extremo_se_recorta()
        test_ventana_no_medible()
        test_etiquetas_de_adopcion()
        test_cobertura_de_comentarios()
        test_exportacion()
    finally:
        for d in _TMPDIRS:
            shutil.rmtree(d, ignore_errors=True)
    print("\n" + "=" * 68)
    print(f"RESULTADO: {PASSED} correctas, {FAILED} fallidas")
    print("=" * 68)
    sys.exit(1 if FAILED else 0)
