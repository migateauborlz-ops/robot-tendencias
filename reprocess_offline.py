"""
Reprocesa el corpus ya almacenado sin volver a llamar a Apify.

Lee raw_social_data, vuelve a pasar el NLP, valida contra Google Trends y
recalcula el Opportunity Score. La ingesta es el unico paso que cuesta dinero,
asi que este script permite iterar sobre el algoritmo cuantas veces haga falta
con costo cero.

Uso:
    ./venv/Scripts/python.exe reprocess_offline.py
    ./venv/Scripts/python.exe reprocess_offline.py --skip-trends   # solo NLP
"""
import argparse
import logging
import sys

import pandas as pd

from src.database import StorageAndScoring
from src.nlp_layer import NLPLayer
from src.cross_validation import TrendValidator

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Reprocesa el corpus almacenado")
    parser.add_argument("--skip-trends", action="store_true",
                        help="Omite la validacion con Google Trends")
    parser.add_argument("--candidates", type=int, default=5,
                        help="Cuantos candidatos enviar a Google Trends")
    parser.add_argument("--top_n", type=int, default=15)
    parser.add_argument("--geo", default=None,
                        help="Mercado de deteccion. Cadena vacia = mundial.")
    parser.add_argument("--geo-local", dest="geo_local", default=None)
    parser.add_argument("--no-compare-geos", dest="compare_geos",
                        action="store_false", default=None)
    args = parser.parse_args()

    store = StorageAndScoring()
    df_raw = pd.read_sql(
        "SELECT video_id, platform, url, description, comment_text, "
        "likes_count, shares_count, timestamp FROM raw_social_data",
        store.engine)

    if df_raw.empty:
        print("La tabla raw_social_data esta vacia. Ejecute primero main.py.")
        return 1

    print(f"Publicaciones en la base: {len(df_raw)}")

    nlp = NLPLayer()
    df_processed = nlp.process_dataframe(df_raw)
    store.save_processed_entities(df_processed)

    entidades = set()
    for lista in df_processed["extracted_products"]:
        entidades.update(lista or [])
    print(f"Entidades unicas extraidas: {len(entidades)}")
    print(f"Idiomas detectados: "
          f"{dict(sorted(nlp.language_counts.items(), key=lambda kv: -kv[1]))}")

    if args.skip_trends:
        print("\nValidacion con Google Trends omitida (--skip-trends).")
        return 0

    # La preseleccion vive en StorageAndScoring para que este script y main.py
    # no puedan divergir en como ordenan los candidatos.
    candidatos = StorageAndScoring.select_candidates(df_processed,
                                                     limit=args.candidates)
    print(f"\nCandidatos enviados a Google Trends ({len(candidatos)}):")
    for c in candidatos:
        print(f"  - {c}")

    validator = TrendValidator(geo=args.geo, geo_local=args.geo_local,
                               compare_geos=args.compare_geos)
    df_validation = validator.validate_products(candidatos)
    if df_validation.empty:
        print("\nNingun candidato supero la validacion de tendencia.")
        return 0

    df_final = store.calculate_opportunity_score(df_processed, df_validation)
    if df_final.empty:
        print("\nNo se pudo calcular el Opportunity Score.")
        return 0

    store.save_final_trends(df_final, top_n=args.top_n)
    print(f"\nTOP {min(args.top_n, len(df_final))} OPORTUNIDADES")
    print("-" * 78)
    cols = ["product_name", "engagement_velocity", "viral_metric_score",
            "purchase_intent_score", "norm_trend_growth", "opportunity_score"]
    cols = [c for c in cols if c in df_final.columns]
    print(df_final[cols].head(args.top_n).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
