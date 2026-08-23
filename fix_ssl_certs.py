"""
Agrega al bundle de certifi las CA raiz de inspeccion TLS instaladas en Windows.

Por que hace falta:
    Algunos antivirus y proxies corporativos (Avast, ESET, Zscaler, Netskope)
    inspeccionan el trafico HTTPS reemplazando el certificado del servidor por
    uno propio, firmado con una raiz que ellos instalan en el almacen de Windows.
    Windows confia en ella, pero Python no la ve: httpx y requests validan contra
    el bundle de certifi, que solo trae las CA publicas. El resultado es
    CERTIFICATE_VERIFY_FAILED aunque la red funcione perfectamente.

Que hace este script:
    Busca en el almacen de Windows unicamente las raices de inspeccion TLS
    conocidas y las agrega al bundle de certifi si faltan. NO desactiva la
    verificacion y NO importa el almacen completo: amplia la confianza solo con
    la CA que esta interceptando la conexion.

Uso:
    ./venv/Scripts/python.exe fix_ssl_certs.py --check     # diagnostica
    ./venv/Scripts/python.exe fix_ssl_certs.py             # aplica
    ./venv/Scripts/python.exe fix_ssl_certs.py --match Zscaler   # otra marca
    ./venv/Scripts/python.exe fix_ssl_certs.py --restore   # deshace el cambio

Nota: el bundle vive dentro de venv/. Si se recrea el entorno virtual hay que
volver a ejecutar este script.
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

# Marcas conocidas de software que hace inspeccion TLS.
INTERCEPTORES = [
    "Avast", "AVG", "ESET", "Kaspersky", "Bitdefender", "Malwarebytes",
    "Zscaler", "Netskope", "Blue Coat", "Broadcom ProxySG", "Fortinet",
    "FortiGate", "Sophos", "McAfee Web Gateway", "Forcepoint", "Palo Alto",
    "Cisco Umbrella", "Charles Proxy", "Fiddler", "BurpSuite",
]

PS_ENUMERAR = r"""
$resultado = @()
foreach ($ruta in @('Cert:\LocalMachine\Root','Cert:\CurrentUser\Root',
                    'Cert:\LocalMachine\CA','Cert:\CurrentUser\CA')) {
  try {
    Get-ChildItem $ruta -ErrorAction SilentlyContinue | ForEach-Object {
      $resultado += [PSCustomObject]@{
        Subject = $_.Subject
        Thumb   = $_.Thumbprint
        B64     = [System.Convert]::ToBase64String($_.RawData)
      }
    }
  } catch {}
}
$resultado | ConvertTo-Json -Compress -Depth 3
"""


def certificados_de_windows() -> list:
    """Enumera las CA del almacen de Windows con su subject y su DER en base64."""
    try:
        proc = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", PS_ENUMERAR],
            capture_output=True, text=True, timeout=120,
        )
    except Exception as err:
        print(f"ERROR: no se pudo consultar el almacen de Windows: {err}")
        return []

    salida = (proc.stdout or "").strip()
    if not salida:
        print("ERROR: el almacen de Windows no devolvio certificados.")
        return []
    try:
        datos = json.loads(salida)
    except json.JSONDecodeError as err:
        print(f"ERROR: respuesta ilegible del almacen de Windows: {err}")
        return []
    return datos if isinstance(datos, list) else [datos]


def a_pem(b64: str) -> str:
    lineas = [b64[i:i + 64] for i in range(0, len(b64), 64)]
    return ("-----BEGIN CERTIFICATE-----\n" + "\n".join(lineas)
            + "\n-----END CERTIFICATE-----")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Agrega raices de inspeccion TLS al bundle de certifi")
    parser.add_argument("--check", action="store_true",
                        help="Solo diagnostica, no modifica nada")
    parser.add_argument("--match", action="append", default=None,
                        help="Texto adicional a buscar en el Subject (repetible)")
    parser.add_argument("--restore", action="store_true",
                        help="Restaura el bundle original desde el respaldo")
    args = parser.parse_args()

    try:
        import certifi
    except ImportError:
        print("ERROR: certifi no esta instalado en este entorno.")
        return 1

    bundle = Path(certifi.where())
    respaldo = bundle.with_suffix(bundle.suffix + ".original")
    print(f"Bundle de certifi : {bundle}")

    if args.restore:
        if not respaldo.exists():
            print("No hay respaldo que restaurar.")
            return 1
        shutil.copy2(respaldo, bundle)
        print(f"Bundle restaurado desde {respaldo}")
        return 0

    if not bundle.exists():
        print("ERROR: el bundle no existe.")
        return 1

    contenido = bundle.read_text(encoding="utf-8", errors="replace")
    contenido_plano = contenido.replace("\n", "")
    print(f"Certificados actuales en el bundle : "
          f"{contenido.count('-----BEGIN CERTIFICATE-----')}")

    patrones = list(INTERCEPTORES) + list(args.match or [])
    certs = certificados_de_windows()
    if not certs:
        return 1
    print(f"Certificados revisados en el almacen de Windows : {len(certs)}")

    candidatos = {}
    for c in certs:
        subject = str(c.get("Subject", ""))
        if any(pat.lower() in subject.lower() for pat in patrones):
            candidatos[c.get("Thumb")] = (subject, c.get("B64", ""))

    if not candidatos:
        print("\nNo se encontro ninguna CA de inspeccion TLS conocida.")
        print("Si su red usa otra marca, ejecute con --match \"<nombre>\".")
        return 1

    print(f"\nCA de inspeccion TLS encontradas : {len(candidatos)}")
    faltantes = []
    for thumb, (subject, b64) in candidatos.items():
        presente = b64.replace("\n", "") in contenido_plano
        estado = "ya presente en certifi" if presente else "FALTA"
        print(f"  - {subject}")
        print(f"      huella {thumb} -> {estado}")
        if not presente:
            faltantes.append((subject, b64))

    if not faltantes:
        print("\nNada que hacer: certifi ya confia en esas raices.")
        return 0

    if args.check:
        print(f"\nModo --check: {len(faltantes)} certificado(s) por agregar. "
              "Ejecute sin --check para aplicar.")
        return 0

    if not respaldo.exists():
        shutil.copy2(bundle, respaldo)
        print(f"\nRespaldo creado en : {respaldo}")

    with bundle.open("a", encoding="utf-8") as handle:
        handle.write("\n\n# === CA de inspeccion TLS importadas del almacen de Windows ===\n")
        for subject, b64 in faltantes:
            handle.write(f"# {subject}\n")
            handle.write(a_pem(b64) + "\n")

    total = bundle.read_text(encoding="utf-8", errors="replace").count(
        "-----BEGIN CERTIFICATE-----")
    print(f"Agregado(s) {len(faltantes)} certificado(s). Total ahora : {total}")
    print("\nPara deshacer: ./venv/Scripts/python.exe fix_ssl_certs.py --restore")
    print("\nNota sobre pip: pip no usa el bundle de certifi sino una copia propia")
    print("embebida, asi que seguira fallando. Para instalar paquetes, pasele el")
    print("bundle ya corregido:")
    print(f'  ./venv/Scripts/python.exe -m pip install --cert "{bundle}" <paquete>')
    return 0


if __name__ == "__main__":
    sys.exit(main())
