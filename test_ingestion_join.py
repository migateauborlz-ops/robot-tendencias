"""
Regression tests for the comment-to-video join in Module 1.

These tests run entirely offline: they do not call Apify and therefore cost
nothing. They reproduce the failure that left comment_text empty in 38 of 38
records and verify that the corrected resolution logic recovers the link.

Run with:  python test_ingestion_join.py
"""
import sys
import pandas as pd

from src.data_ingestion import DataIngestion

PASSED = 0
FAILED = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"  [OK]   {name}")
    else:
        FAILED += 1
        print(f"  [FALLA] {name}" + (f" -> {detail}" if detail else ""))


def videos_frame() -> pd.DataFrame:
    """Two videos as returned by clockworks/tiktok-scraper (ids as strings)."""
    return pd.DataFrame([
        {"platform": "tiktok", "video_id": "7607749319017483551",
         "url": "https://www.tiktok.com/@dailyfulfillsource/video/7607749319017483551",
         "description": "3 winning dropshipping products", "likes_count": 4100,
         "shares_count": 12, "timestamp": pd.Timestamp("2026-02-19", tz="UTC")},
        {"platform": "tiktok", "video_id": "7608445610051538197",
         "url": "https://www.tiktok.com/@smartlivinghub48/video/7608445610051538197",
         "description": "TikTok made me try this", "likes_count": 41800,
         "shares_count": 255, "timestamp": pd.Timestamp("2026-02-19", tz="UTC")},
    ])


def resolve(item: dict) -> str:
    """Mirrors the resolution order used inside fetch_tiktok_comments."""
    video_id = str(DataIngestion._first_present(
        item, ("aweme_id", "awemeId", "video_id", "videoId", "postId", "post_id"), ""))
    if not video_id:
        url_value = DataIngestion._first_present(
            item, ("videoWebUrl", "submittedVideoUrl", "postUrl", "videoUrl",
                   "webVideoUrl", "url"), "")
        video_id = DataIngestion._video_id_from_url(url_value)
    return video_id


def test_url_parsing():
    print("\n1. Extraccion del id desde la URL")
    check("URL canonica de TikTok",
          DataIngestion._video_id_from_url(
              "https://www.tiktok.com/@user/video/7607749319017483551")
          == "7607749319017483551")
    check("URL con parametros de consulta",
          DataIngestion._video_id_from_url(
              "https://www.tiktok.com/@user/video/7608445610051538197?is_from_webapp=1")
          == "7608445610051538197")
    check("URL sin id de video devuelve cadena vacia",
          DataIngestion._video_id_from_url("https://www.tiktok.com/@user") == "")
    check("Valor nulo no lanza excepcion",
          DataIngestion._video_id_from_url(None) == "")


def test_first_present():
    print("\n2. Seleccion del primer campo disponible")
    check("Toma la primera clave con valor",
          DataIngestion._first_present({"b": "x"}, ("a", "b", "c")) == "x")
    check("Ignora claves con cadena vacia",
          DataIngestion._first_present({"a": "", "b": "y"}, ("a", "b")) == "y")
    check("Ignora claves con None",
          DataIngestion._first_present({"a": None, "b": "z"}, ("a", "b")) == "z")
    check("Devuelve el valor por defecto si no hay ninguna",
          DataIngestion._first_present({}, ("a", "b"), default=0) == 0)


def test_resolution_across_schemas():
    print("\n3. Resolucion del video padre segun el esquema del actor")
    check("Actor que expone aweme_id",
          resolve({"aweme_id": "7607749319017483551", "text": "donde lo compro?"})
          == "7607749319017483551")
    check("Actor que expone videoId en camelCase",
          resolve({"videoId": "7608445610051538197", "text": "necesito el link"})
          == "7608445610051538197")
    check("Actor que solo expone la URL del video (fallo original)",
          resolve({"videoWebUrl": "https://www.tiktok.com/@u/video/7607749319017483551",
                   "text": "cuanto vale?"}) == "7607749319017483551")
    check("Actor que devuelve el id como entero",
          resolve({"aweme_id": 7607749319017483551}) == "7607749319017483551")
    check("Item sin ningun identificador queda sin resolver",
          resolve({"text": "huerfano"}) == "")


def test_join_recovers_comments():
    print("\n4. Union de comentarios con el DataFrame de videos")
    videos = videos_frame()

    # Escenario A: el actor devuelve el id como entero (el de videos es cadena).
    comments_int = pd.DataFrame([
        {"video_id": 7607749319017483551, "comment_text": "donde lo compro?"},
        {"video_id": 7607749319017483551, "comment_text": "pasame el link"},
        {"video_id": 7608445610051538197, "comment_text": "cuanto vale?"},
    ])
    merged = DataIngestion.attach_comments(videos, comments_int)
    filled = int((merged["comment_text"].str.strip() != "").sum())
    check("Ids enteros vs cadena: se unen los 2 videos", filled == 2,
          f"cubiertos {filled} de 2")
    check("Los comentarios del mismo video se concatenan",
          "donde lo compro? | pasame el link" in merged.iloc[0]["comment_text"])

    # Escenario B: sin comentarios, la columna existe y queda vacia.
    merged_empty = DataIngestion.attach_comments(videos, pd.DataFrame())
    check("Sin comentarios la columna existe y esta vacia",
          "comment_text" in merged_empty.columns
          and (merged_empty["comment_text"] == "").all())

    # Escenario C: cobertura parcial, no se pierden filas.
    comments_partial = pd.DataFrame([
        {"video_id": "7607749319017483551", "comment_text": "lo necesito"},
    ])
    merged_partial = DataIngestion.attach_comments(videos, comments_partial)
    check("Cobertura parcial conserva todas las filas de video",
          len(merged_partial) == len(videos))
    check("Cobertura parcial cubre exactamente 1 video",
          int((merged_partial["comment_text"].str.strip() != "").sum()) == 1)

    # Escenario D: reproduccion del fallo historico.
    comments_broken = pd.DataFrame([
        {"video_id": "", "comment_text": "donde lo compro?"},
        {"video_id": "", "comment_text": "pasame el link"},
    ])
    merged_broken = DataIngestion.attach_comments(videos, comments_broken)
    check("Ids vacios producen cobertura cero (fallo historico reproducido)",
          int((merged_broken["comment_text"].str.strip() != "").sum()) == 0)


def test_coverage_alert(capture: list):
    print("\n5. Alerta de cobertura")
    import logging

    class Capture(logging.Handler):
        def emit(self, record):
            capture.append((record.levelname, record.getMessage()))

    logger = logging.getLogger("src.data_ingestion")
    handler = Capture()
    logger.addHandler(handler)
    try:
        videos = videos_frame()
        empty = DataIngestion.attach_comments(videos, pd.DataFrame())
        DataIngestion._report_comment_coverage(empty, requested=2)
    finally:
        logger.removeHandler(handler)

    errors = [m for lvl, m in capture if lvl == "ERROR"]
    check("Se emite un ERROR cuando la cobertura es cero", len(errors) == 1,
          f"errores emitidos: {len(errors)}")
    check("El mensaje explica el impacto en la intencion de compra",
          any("purchase-intent" in m for m in errors))


if __name__ == "__main__":
    print("=" * 68)
    print("Pruebas de regresion - union comentarios/videos (sin costo de API)")
    print("=" * 68)
    test_url_parsing()
    test_first_present()
    test_resolution_across_schemas()
    test_join_recovers_comments()
    test_coverage_alert([])
    print("\n" + "=" * 68)
    print(f"RESULTADO: {PASSED} correctas, {FAILED} fallidas")
    print("=" * 68)
    sys.exit(1 if FAILED else 0)
