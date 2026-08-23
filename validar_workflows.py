"""
Valida los archivos de GitHub Actions antes de subirlos.

Comprobar que el YAML parsea no basta, y este proyecto lo aprendio de la peor
forma. La linea

    CONSULTAS: ${{ github.event.inputs.consultas || '#TikTokMadeMeBuyIt #Otro' }}

es YAML perfectamente valido: como escalar plano, un ' #' inicia un comentario,
asi que el segundo hashtag y todo lo que sigue desaparecen y el valor queda en
"${{ github.event.inputs.consultas || '#TikTokMadeMeBuyIt", con la expresion sin
cerrar. Un validador de YAML no protesta -- el documento esta bien formado --
pero GitHub rechaza el workflow completo, y como el archivo no se dispara con
push, el unico sintoma es una corrida fallida de cero segundos sin mensaje.

De ahi las dos comprobaciones: que el documento parsee, y que ninguna expresion
haya quedado partida al hacerlo. La segunda es la que atrapa esta clase de
error.

Uso:
    python validar_workflows.py
"""
import glob
import sys

try:
    import yaml
except ImportError:
    print("Se requiere PyYAML para validar los workflows.")
    sys.exit(1)


def recorrer(nodo, ruta="raiz"):
    """Genera (ruta, texto) por cada cadena del documento."""
    if isinstance(nodo, dict):
        for k, v in nodo.items():
            yield from recorrer(v, f"{ruta}.{k}")
    elif isinstance(nodo, list):
        for i, v in enumerate(nodo):
            yield from recorrer(v, f"{ruta}[{i}]")
    elif isinstance(nodo, str):
        yield ruta, nodo


def revisar(ruta_archivo: str) -> list:
    """Devuelve la lista de problemas encontrados en un workflow."""
    problemas = []
    try:
        doc = yaml.safe_load(open(ruta_archivo, encoding="utf-8"))
    except yaml.YAMLError as err:
        return [f"YAML invalido: {err}"]

    if not isinstance(doc, dict):
        return ["El archivo no define un mapa en su raiz."]

    for ruta, texto in recorrer(doc):
        abiertas = texto.count("${{")
        cerradas = texto.count("}}")
        if abiertas != cerradas:
            problemas.append(
                f"{ruta}: expresion sin cerrar ({abiertas} '${{{{' frente a "
                f"{cerradas} '}}}}'). Casi siempre es un '#' precedido de "
                f"espacio en un valor sin comillas.\n"
                f"      valor: {texto!r}")

    return problemas


def main() -> int:
    archivos = sorted(glob.glob(".github/workflows/*.yml")
                      + glob.glob(".github/workflows/*.yaml"))
    if not archivos:
        print("No se encontro ningun workflow en .github/workflows/.")
        return 0

    print("=" * 72)
    print("Validacion de workflows de GitHub Actions")
    print("=" * 72)

    fallidos = 0
    for archivo in archivos:
        problemas = revisar(archivo)
        if problemas:
            fallidos += 1
            print(f"  FALLA  {archivo}")
            for p in problemas:
                print(f"      {p}")
        else:
            print(f"  OK     {archivo}")

    print("=" * 72)
    print(f"{len(archivos) - fallidos} de {len(archivos)} workflows validos")
    print("=" * 72)
    return 1 if fallidos else 0


if __name__ == "__main__":
    sys.exit(main())
