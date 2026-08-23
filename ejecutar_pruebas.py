"""
Ejecuta todas las suites de pruebas del proyecto y agrega el resultado.

Cada suite es un script independiente que imprime una linea
`RESULTADO: N correctas, M fallidas` y termina con codigo distinto de cero si
algo fallo. Este corredor las descubre por patron, las ejecuta aisladas en
subprocesos y suma los totales.

Una suite que termina sin imprimir esa linea se cuenta como ERROR, no como
exito. Ese caso importa: un archivo llamado test_*.py que no comprueba nada y
siempre sale con codigo 0 pasaria inadvertido como suite aprobada y daria una
falsa sensacion de cobertura.

Uso:
    python ejecutar_pruebas.py
    python ejecutar_pruebas.py --detalle     # muestra la salida completa
"""
import argparse
import glob
import re
import subprocess
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

RESULTADO = re.compile(r"RESULTADO:\s*(\d+)\s+correctas,\s*(\d+)\s+fallidas")


def ejecutar(ruta: str, detalle: bool = False):
    """Corre una suite y devuelve (correctas, fallidas, ok, salida)."""
    proc = subprocess.run([sys.executable, ruta], capture_output=True,
                          text=True, encoding="utf-8", errors="replace")
    salida = (proc.stdout or "") + (proc.stderr or "")

    match = None
    for m in RESULTADO.finditer(salida):
        match = m  # la ultima linea es el total de la suite

    if match is None:
        return 0, 0, False, salida

    correctas, fallidas = int(match.group(1)), int(match.group(2))
    ok = proc.returncode == 0 and fallidas == 0
    return correctas, fallidas, ok, salida


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detalle", action="store_true",
                        help="Imprime la salida completa de cada suite.")
    args = parser.parse_args()

    suites = sorted(glob.glob("test_*.py"))
    if not suites:
        print("No se encontro ninguna suite (test_*.py).")
        return 1

    print("=" * 72)
    print("Suite de pruebas del robot de tendencias")
    print("=" * 72)

    total_ok = total_fail = 0
    rotas = []

    for ruta in suites:
        correctas, fallidas, ok, salida = ejecutar(ruta, args.detalle)
        total_ok += correctas
        total_fail += fallidas

        if args.detalle:
            print(f"\n----- {ruta} -----")
            print(salida.rstrip())
        elif not ok:
            # Solo se imprime el detalle de lo que fallo, para que el log de CI
            # se pueda leer sin desplegar cientos de lineas correctas.
            fallos = [l for l in salida.splitlines() if "[FALLA]" in l]
            if fallos:
                print(f"\n----- {ruta} -----")
                for l in fallos:
                    print(l)

        if correctas == 0 and fallidas == 0:
            estado = "ERROR (no reporto resultados)"
            rotas.append(ruta)
            if not args.detalle:
                print(f"\n----- {ruta} -----")
                print(salida.rstrip()[-1500:])
        else:
            estado = f"{correctas:3d} correctas, {fallidas} fallidas"

        print(f"  {'OK ' if ok else 'FALLA'}  {ruta:<32} {estado}")

    print("\n" + "=" * 72)
    print(f"TOTAL: {total_ok} correctas, {total_fail} fallidas, "
          f"{len(suites)} suites")
    if rotas:
        print(f"SUITES SIN RESULTADO: {', '.join(rotas)}")
    print("=" * 72)

    return 1 if (total_fail or rotas) else 0


if __name__ == "__main__":
    sys.exit(main())
