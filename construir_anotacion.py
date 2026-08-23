"""
Genera los libros de anotacion manual para los tres anotadores.

Produce, en la carpeta anotacion/:
  - Anotacion_<Nombre>.xlsx  (uno por anotador, contenido identico)
  - clave_maestra.csv        (lo que predijo el sistema; NO se entrega a nadie
                              hasta que la anotacion este cerrada)

Los libros no muestran la prediccion del sistema: si el anotador la viera,
tenderia a confirmarla y las metricas quedarian infladas.
"""
import json
import random
import sqlite3
import sys
from pathlib import Path

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation

SEMILLA = 20260822          # muestreo reproducible
ANOTADORES = ["Daniela_Trujillo", "Miguel_Reina", "Gabriel_Zambrano"]
N_NO_MARCADOS = 250         # comentarios del estrato "sin intencion segun el sistema"
N_POSTS_RECALL = 20

FUENTE = "Arial"
AZUL = "1F3864"
GRIS = "F2F2F2"
AMARILLO = "FFF2CC"
BORDE = Border(*[Side(style="thin", color="BFBFBF")] * 4)


def cargar_datos():
    """Lee el corpus y arma los tres conjuntos a anotar."""
    con = sqlite3.connect("data/trends.db")

    entidades = set()
    for (j,) in con.execute("SELECT extracted_products FROM processed_entities"):
        entidades.update(json.loads(j or "[]"))
    entidades = sorted(entidades)

    posts = pd.read_sql(
        "SELECT video_id, url, description, comment_text, likes_count "
        "FROM raw_social_data WHERE TRIM(COALESCE(comment_text,'')) <> ''", con)

    # Comentarios unicos conservando su publicacion de origen
    vistos, comentarios = set(), []
    for _, fila in posts.iterrows():
        for texto in str(fila["comment_text"]).split("|"):
            texto = texto.strip()
            if texto and texto.lower() not in vistos:
                vistos.add(texto.lower())
                comentarios.append({"video_id": fila["video_id"], "comentario": texto})

    con.close()
    return entidades, comentarios, posts


def marcar_intencion(comentarios):
    """Aplica el clasificador actual para estratificar la muestra."""
    from src.nlp_layer import NLPLayer
    nlp = NLPLayer()
    for c in comentarios:
        puntaje, altos = nlp.analyze_purchase_intent([c["comentario"]])
        c["sistema"] = "SI" if altos else "NO"
    return comentarios


def muestrear(comentarios):
    """
    Muestreo estratificado.

    Se toman TODOS los comentarios que el sistema marco como intencion (para
    medir precision sobre el total del estrato) y una muestra aleatoria de los
    no marcados (para estimar los falsos negativos y, con ellos, el recall).
    """
    marcados = [c for c in comentarios if c["sistema"] == "SI"]
    no_marcados = [c for c in comentarios if c["sistema"] == "NO"]
    rng = random.Random(SEMILLA)
    muestra_no = rng.sample(no_marcados, min(N_NO_MARCADOS, len(no_marcados)))

    seleccion = marcados + muestra_no
    rng.shuffle(seleccion)      # mezclados, para que no se note el estrato
    return seleccion, len(marcados), len(no_marcados), len(muestra_no)


def encabezado(hoja, titulo, subtitulo, anchos):
    hoja.sheet_view.showGridLines = False
    hoja["A1"] = titulo
    hoja["A1"].font = Font(name=FUENTE, size=14, bold=True, color=AZUL)
    hoja["A2"] = subtitulo
    hoja["A2"].font = Font(name=FUENTE, size=10, italic=True, color="595959")
    hoja.freeze_panes = "A6"
    for i, ancho in enumerate(anchos, start=1):
        hoja.column_dimensions[get_column_letter(i)].width = ancho


def fila_encabezado(hoja, fila, columnas):
    for i, nombre in enumerate(columnas, start=1):
        celda = hoja.cell(row=fila, column=i, value=nombre)
        celda.font = Font(name=FUENTE, size=10, bold=True, color="FFFFFF")
        celda.fill = PatternFill("solid", fgColor=AZUL)
        celda.alignment = Alignment(horizontal="center", vertical="center",
                                    wrap_text=True)
        celda.border = BORDE


def validacion(hoja, opciones, rango):
    dv = DataValidation(type="list", formula1=f'"{opciones}"', allow_blank=True,
                        showDropDown=False)
    dv.error = "Elija un valor de la lista."
    dv.errorTitle = "Valor no permitido"
    hoja.add_data_validation(dv)
    dv.add(rango)


def construir_libro(anotador, entidades, comentarios, posts_recall):
    wb = Workbook()

    # ---------------- Instrucciones ----------------
    ins = wb.active
    ins.title = "Instrucciones"
    ins.sheet_view.showGridLines = False
    ins.column_dimensions["A"].width = 3
    ins.column_dimensions["B"].width = 110

    lineas = [
        (f"Anotacion manual del corpus — {anotador.replace('_', ' ')}", "titulo"),
        ("", None),
        ("Este libro tiene tres tareas. Complete solo las celdas amarillas.", "normal"),
        ("Trabaje solo: no consulte con los demas anotadores hasta que los tres "
         "hayan terminado. La medida de concordancia pierde sentido si se ponen "
         "de acuerdo antes.", "normal"),
        ("", None),
        ("Tarea A — Entidades  (hoja A_Entidades)", "sub"),
        ("Para cada texto extraido por el sistema, responda: ¿nombra un producto "
         "fisico que una PYME podria comprar y revender?", "normal"),
        ("   SI       es un producto o una categoria concreta de producto", "normal"),
        ("   NO       no lo es (frases, lugares, saludos, conceptos abstractos)", "normal"),
        ("   DUDOSO   no lo puede decidir con la informacion disponible", "normal"),
        ("", None),
        ("Tarea B — Intencion de compra  (hoja B_Intencion)", "sub"),
        ("Para cada comentario, responda: ¿expresa que la persona quiere adquirir "
         "el producto?", "normal"),
        ("   SI       pregunta precio, pide el link, dice que lo quiere o ya lo pidio", "normal"),
        ("   NO       elogia, comenta, bromea o critica, sin querer comprarlo", "normal"),
        ("   DUDOSO   ambiguo o ininteligible", "normal"),
        ("", None),
        ("Tarea C — Productos por publicacion  (hoja C_Recall)  [opcional]", "sub"),
        ("Lea el texto de la publicacion y escriba los productos que usted "
         "reconoce, separados por punto y coma. Sirve para medir lo que el "
         "sistema deja pasar.", "normal"),
        ("", None),
        ("Consulte el Manual de anotacion para los casos de frontera. Ante la "
         "duda, marque DUDOSO en vez de adivinar: esas filas se revisan aparte.", "normal"),
    ]
    fila = 2
    for texto, estilo in lineas:
        if texto:
            celda = ins.cell(row=fila, column=2, value=texto)
            if estilo == "titulo":
                celda.font = Font(name=FUENTE, size=14, bold=True, color=AZUL)
            elif estilo == "sub":
                celda.font = Font(name=FUENTE, size=11, bold=True, color=AZUL)
            else:
                celda.font = Font(name=FUENTE, size=10)
                celda.alignment = Alignment(wrap_text=True, vertical="top")
        fila += 1

    # ---------------- Tarea A ----------------
    ha = wb.create_sheet("A_Entidades")
    encabezado(ha, "Tarea A — ¿Es un producto?",
               "Marque la columna C. Ejemplo en la fila 6.", [8, 46, 16, 40])
    ha["C4"] = "Anotadas:"
    ha["C4"].font = Font(name=FUENTE, size=9, bold=True)
    ha["D4"] = f'=COUNTIF(C7:C{6 + len(entidades)},"?*")&" de {len(entidades)}"'
    ha["D4"].font = Font(name=FUENTE, size=9, bold=True, color=AZUL)
    fila_encabezado(ha, 5, ["ID", "Texto extraido por el sistema",
                            "¿Es producto?", "Observacion (opcional)"])

    ejemplo = ["EJ", "ninja coffee maker", "SI", "marca + tipo de producto"]
    for i, v in enumerate(ejemplo, start=1):
        c = ha.cell(row=6, column=i, value=v)
        c.font = Font(name=FUENTE, size=10, italic=True, color="808080")
        c.border = BORDE

    for n, ent in enumerate(entidades):
        f = 7 + n
        ha.cell(row=f, column=1, value=f"A{n+1:03d}").font = Font(name=FUENTE, size=10)
        ha.cell(row=f, column=2, value=ent).font = Font(name=FUENTE, size=10)
        celda = ha.cell(row=f, column=3)
        celda.fill = PatternFill("solid", fgColor=AMARILLO)
        for col in range(1, 5):
            ha.cell(row=f, column=col).border = BORDE
    validacion(ha, "SI,NO,DUDOSO", f"C7:C{6 + len(entidades)}")

    # ---------------- Tarea B ----------------
    hb = wb.create_sheet("B_Intencion")
    encabezado(hb, "Tarea B — ¿Expresa intencion de compra?",
               "Marque la columna C. Ejemplo en la fila 6.", [8, 76, 16, 34])
    hb["C4"] = "Anotados:"
    hb["C4"].font = Font(name=FUENTE, size=9, bold=True)
    hb["D4"] = f'=COUNTIF(C7:C{6 + len(comentarios)},"?*")&" de {len(comentarios)}"'
    hb["D4"].font = Font(name=FUENTE, size=9, bold=True, color=AZUL)
    fila_encabezado(hb, 5, ["ID", "Comentario", "¿Intencion de compra?",
                            "Observacion (opcional)"])

    ejemplo = ["EJ", "cuanto vale? me interesa", "SI", "pregunta el precio"]
    for i, v in enumerate(ejemplo, start=1):
        c = hb.cell(row=6, column=i, value=v)
        c.font = Font(name=FUENTE, size=10, italic=True, color="808080")
        c.border = BORDE

    for n, com in enumerate(comentarios):
        f = 7 + n
        hb.cell(row=f, column=1, value=f"B{n+1:03d}").font = Font(name=FUENTE, size=10)
        celda_txt = hb.cell(row=f, column=2, value=com["comentario"][:300])
        celda_txt.font = Font(name=FUENTE, size=10)
        celda_txt.alignment = Alignment(wrap_text=True, vertical="top")
        hb.cell(row=f, column=3).fill = PatternFill("solid", fgColor=AMARILLO)
        for col in range(1, 5):
            hb.cell(row=f, column=col).border = BORDE
    validacion(hb, "SI,NO,DUDOSO", f"C7:C{6 + len(comentarios)}")

    # ---------------- Tarea C ----------------
    hc = wb.create_sheet("C_Recall")
    encabezado(hc, "Tarea C — Productos presentes en la publicacion (opcional)",
               "Escriba en la columna C los productos separados por punto y coma.",
               [8, 84, 44])
    fila_encabezado(hc, 5, ["ID", "Texto de la publicacion",
                            "Productos que usted reconoce"])
    c = hc.cell(row=6, column=2, value="3 winning dropshipping products for 2026 ...")
    c.font = Font(name=FUENTE, size=10, italic=True, color="808080")
    c = hc.cell(row=6, column=3, value="organizador de closet; lampara led")
    c.font = Font(name=FUENTE, size=10, italic=True, color="808080")
    hc.cell(row=6, column=1, value="EJ").font = Font(name=FUENTE, size=10,
                                                     italic=True, color="808080")

    for n, (_, post) in enumerate(posts_recall.iterrows()):
        f = 7 + n
        hc.cell(row=f, column=1, value=f"C{n+1:03d}").font = Font(name=FUENTE, size=10)
        texto = f'{post["description"]} || {str(post["comment_text"])[:400]}'
        celda = hc.cell(row=f, column=2, value=texto[:700])
        celda.font = Font(name=FUENTE, size=9)
        celda.alignment = Alignment(wrap_text=True, vertical="top")
        hc.cell(row=f, column=3).fill = PatternFill("solid", fgColor=AMARILLO)
        hc.row_dimensions[f].height = 70
        for col in range(1, 4):
            hc.cell(row=f, column=col).border = BORDE

    salida = Path("anotacion") / f"Anotacion_{anotador}.xlsx"
    wb.save(salida)
    return salida


def main() -> int:
    Path("anotacion").mkdir(exist_ok=True)

    entidades, comentarios, posts = cargar_datos()
    print(f"Entidades a anotar : {len(entidades)}")
    print(f"Comentarios unicos : {len(comentarios)}")

    comentarios = marcar_intencion(comentarios)
    muestra, n_si, n_no, n_muestra_no = muestrear(comentarios)
    print(f"\nMuestreo estratificado de comentarios")
    print(f"  marcados por el sistema      : {n_si} (se anotan todos)")
    print(f"  no marcados                  : {n_no} (se muestrean {n_muestra_no})")
    print(f"  total a anotar               : {len(muestra)}")

    posts_recall = posts.sample(n=min(N_POSTS_RECALL, len(posts)),
                                random_state=SEMILLA)

    # Clave maestra: se guarda aparte para no contaminar la anotacion.
    clave = pd.DataFrame([
        {"id": f"B{i+1:03d}", "video_id": c["video_id"],
         "comentario": c["comentario"], "prediccion_sistema": c["sistema"]}
        for i, c in enumerate(muestra)])
    clave.to_csv("anotacion/clave_maestra_comentarios.csv", index=False,
                 encoding="utf-8-sig")
    pd.DataFrame({"id": [f"A{i+1:03d}" for i in range(len(entidades))],
                  "entidad": entidades}).to_csv(
        "anotacion/clave_maestra_entidades.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"estrato": ["marcados", "no_marcados"],
                  "poblacion": [n_si, n_no],
                  "muestreados": [n_si, n_muestra_no]}).to_csv(
        "anotacion/estratos.csv", index=False, encoding="utf-8-sig")

    print()
    for anotador in ANOTADORES:
        ruta = construir_libro(anotador, entidades, muestra, posts_recall)
        print(f"  generado: {ruta}")

    print("\nClaves maestras en anotacion/clave_maestra_*.csv "
          "(no compartir hasta cerrar la anotacion).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
