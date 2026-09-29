"""
Exporta los resultados validados a CSV.

Existe para que la orquestacion automatica pueda publicar lo que el pipeline
encontro sin publicar la base de datos entera. `raw_social_data` guarda
comentarios raspados y los identificadores de las publicaciones de las que
salieron; subir el archivo .db como artefacto de CI trasladaria esos datos
personales al almacenamiento de GitHub, que es justamente lo que la politica
del proyecto evita. `final_trends`, en cambio, solo contiene nombres de
producto y metricas agregadas.

Uso:
    python exportar_resultados.py --salida resultados.csv
"""
import argparse
import sys

from src.config import Config
from src.reporting import TrendReport, export_columns

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def exportar(salida: str, db_url: str = None, limite: int = None) -> int:
    reporte = TrendReport(db_url or Config.DB_PATH)
    df = reporte.load_trends(limit=limite)
    tabla = export_columns(df)

    # utf-8-sig para que Excel respete los acentos al abrir el archivo.
    tabla.to_csv(salida, index=False, encoding="utf-8-sig")

    cobertura = reporte.coverage()
    print(f"Publicaciones analizadas : {cobertura['posts']}")
    print(f"Cobertura de comentarios : {cobertura['cobertura_comentarios']:.0%}")
    print(f"Productos validados      : {len(tabla)}")
    print(f"Archivo generado         : {salida}")

    if tabla.empty:
        # No es un fallo: significa que ningun producto supero la validacion de
        # tendencias. La corrida fue correcta y el resultado es informativo.
        print("\nNingun producto supero la validacion. Revise pipeline.log "
              "para ver en que etapa se descartaron los candidatos.")
    return len(tabla)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--salida", default="resultados.csv",
                        help="Ruta del CSV de salida.")
    parser.add_argument("--db", default=None,
                        help="Cadena de conexion. Por defecto Config.DB_PATH.")
    parser.add_argument("--limite", type=int, default=None,
                        help="Maximo de productos a exportar.")
    args = parser.parse_args()

    exportar(args.salida, db_url=args.db, limite=args.limite)
