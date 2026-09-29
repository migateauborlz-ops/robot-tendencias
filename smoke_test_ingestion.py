"""
Prueba de humo de la capa de ingesta (Modulo 1).

Ejecuta una extraccion deliberadamente pequena contra Apify para verificar que
los comentarios vuelven a enlazarse con sus videos, ANTES de gastar creditos en
una corrida grande. No escribe nada en la base de datos.

Uso:
    ./venv/Scripts/python.exe smoke_test_ingestion.py
    ./venv/Scripts/python.exe smoke_test_ingestion.py --query "#AmazonFinds" --videos 3

Al terminar deja el payload crudo de cada actor en data/raw_dumps/ para poder
confirmar como se llaman realmente los campos que devuelve cada scraper.
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from src.data_ingestion import DataIngestion

# La consola de Windows usa cp1252 y los comentarios traen emojis.
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass


def main() -> int:
    parser = argparse.ArgumentParser(description="Prueba de humo de la ingesta")
    parser.add_argument("--query", default="#TikTokMadeMeBuyIt",
                        help="Hashtag a consultar (por defecto #TikTokMadeMeBuyIt)")
    parser.add_argument("--videos", type=int, default=3,
                        help="Cuantos videos descargar (mantener bajo: cuesta creditos)")
    parser.add_argument("--comments", type=int, default=10,
                        help="Comentarios por video")
    args = parser.parse_args()

    print("=" * 72)
    print("PRUEBA DE HUMO DE INGESTA")
    print(f"  hashtag           : {args.query}")
    print(f"  videos            : {args.videos}")
    print(f"  comentarios/video : {args.comments}")
    print("  destino           : ninguno (no se escribe en la base de datos)")
    print("=" * 72)

    ingestion = DataIngestion(debug_dump=True)

    videos = ingestion.fetch_tiktok_videos([args.query], max_items=args.videos,
                                           comments_per_post=args.comments)
    if videos.empty:
        print()
        if ingestion.errors:
            # Distinguir "no hubo datos" de "no se pudo hablar con la API": el
            # mensaje anterior culpaba al hashtag incluso cuando fallaba el TLS.
            print("RESULTADO: FALLA DE CONEXION O DE API. El actor no devolvio")
            print("respuesta util. Errores registrados:")
            for err in ingestion.errors:
                print(f"  - {err}")
            if any("CERTIFICATE_VERIFY_FAILED" in e for e in ingestion.errors):
                print()
                print("  Diagnostico: un antivirus o proxy esta inspeccionando el")
                print("  trafico TLS y Python no confia en su CA raiz. Ejecute:")
                print("    ./venv/Scripts/python.exe fix_ssl_certs.py")
        else:
            print("RESULTADO: el actor respondio pero no devolvio filas dentro de la")
            print("ventana de 7 dias. Revise el hashtag o amplie la ventana.")
        print(f"\nCosto de esta prueba: ${ingestion.total_cost_usd:.4f} USD")
        return 1

    print(f"\nVideos recuperados: {len(videos)}")
    for _, row in videos.iterrows():
        print(f"  - {row['video_id']}  likes={row['likes_count']}  {row['url']}")

    comments = ingestion.fetch_comments_from_datasets()
    if comments.empty:
        print("\nLa corrida de videos no dejo comentarios; probando el actor "
              "independiente como respaldo...")
        urls = videos["url"].dropna().tolist()
        comments = ingestion.fetch_tiktok_comments(
            urls, max_comments_per_video=args.comments)

    print(f"\nComentarios recuperados: {len(comments)}")
    if not comments.empty:
        distintos = comments["video_id"].astype(str).nunique()
        print(f"Videos distintos referenciados por los comentarios: {distintos}")
        print("\nMuestra:")
        for _, row in comments.head(5).iterrows():
            texto = str(row["comment_text"])[:70]
            print(f"  [{row['video_id']}] {texto}")

    merged = DataIngestion.attach_comments(videos, comments)
    con_comentarios = int((merged["comment_text"].astype(str).str.strip() != "").sum())
    total = len(merged)
    cobertura = 100.0 * con_comentarios / total if total else 0.0

    print("\n" + "=" * 72)
    print("VEREDICTO")
    print(f"  Cobertura de comentarios : {con_comentarios}/{total} videos ({cobertura:.0f}%)")
    print(f"  Costo de esta prueba     : ${ingestion.total_cost_usd:.4f} USD")

    dumps = sorted(Path("data/raw_dumps").glob("*.json")) if Path("data/raw_dumps").exists() else []
    if dumps:
        print("\n  Payloads crudos guardados:")
        for d in dumps[-4:]:
            print(f"    {d}")
        # Mostrar las claves reales del primer comentario: es lo que permite
        # confirmar el nombre del campo que identifica al video padre.
        for d in dumps:
            if "comments" in d.name:
                try:
                    items = json.loads(d.read_text(encoding="utf-8"))
                    if items:
                        print(f"\n  Claves reales del actor de comentarios ({d.name}):")
                        print("    " + ", ".join(sorted(items[0].keys())))
                except Exception as err:
                    print(f"    (no se pudo leer {d.name}: {err})")
                break

    if con_comentarios == 0:
        print("\n  ESTADO: FALLA. Los comentarios no se enlazaron con ningun video.")
        print("  Revise las claves listadas arriba y agreguelas a COMMENT_ID_KEYS o")
        print("  COMMENT_URL_KEYS en src/data_ingestion.py. No lance la corrida grande.")
        print("=" * 72)
        return 1

    print("\n  ESTADO: CORRECTO. La capa de ingesta ya entrega comentarios.")
    print("  Puede proceder con la corrida de volumen.")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
