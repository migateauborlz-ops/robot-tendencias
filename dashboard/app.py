"""
Tablero de oportunidades comerciales (OE4).

Capa de entrega del robot de tendencias: traduce lo que el pipeline dejo en la
base de datos a una decision que una PYME pueda tomar. Toda la logica de datos
vive en src/reporting.py; aqui solo hay presentacion.

Ejecucion:
    streamlit run dashboard/app.py
"""

import sys
from pathlib import Path

# Streamlit agrega al sys.path la carpeta del script, no la raiz del proyecto,
# asi que `import src...` falla si no se corrige antes de importar.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from src.config import Config
from src.reporting import TrendReport, export_columns, LAG_MEASURED

st.set_page_config(page_title="Robot de tendencias",
                   page_icon="📈", layout="wide")


# El TTL evita que el tablero quede congelado tras una corrida del pipeline sin
# obligar al usuario a reiniciar el proceso.
@st.cache_data(ttl=60, show_spinner=False)
def _cargar(db_url: str):
    reporte = TrendReport(db_url)
    return reporte.load_trends(), reporte.coverage()


# --------------------------------------------------------------------------
# Barra lateral
# --------------------------------------------------------------------------
st.sidebar.title("Configuracion")
db_url = st.sidebar.text_input(
    "Base de datos", value=Config.DB_PATH,
    help="Cadena de conexion SQLAlchemy. Por defecto la que usa el pipeline.")

if st.sidebar.button("Recargar datos"):
    st.cache_data.clear()

try:
    df, cobertura = _cargar(db_url)
except Exception as err:
    # La ruta de la base es editable, asi que una cadena de conexion invalida
    # es una entrada de usuario esperable, no un fallo del programa.
    st.error(f"No se pudo abrir la base de datos indicada.\n\n`{err}`")
    st.stop()

st.sidebar.caption(
    "El tablero solo lee. Para generar datos nuevos ejecute el pipeline:\n\n"
    "`python main.py --queries \"#TikTokMadeMeBuyIt\" --items 30`")

# --------------------------------------------------------------------------
# Encabezado
# --------------------------------------------------------------------------
st.title("Oportunidades comerciales detectadas")
st.caption("Deteccion temprana de productos en tendencia para comercio "
           "electronico — Maestria en Analitica de Datos, Universidad Central")

if df.empty:
    st.info(
        "**Todavia no hay resultados.** La base de datos existe pero no "
        "contiene productos validados.\n\n"
        "Ejecute el pipeline y vuelva a cargar el tablero. Si acaba de "
        "ejecutarlo y no ve nada, revise `pipeline.log`: lo mas frecuente es "
        "que ningun producto haya superado la validacion de Google Trends.")
    st.stop()

# --------------------------------------------------------------------------
# Panel de procedencia
# --------------------------------------------------------------------------
st.subheader("De donde salen estos resultados")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Publicaciones analizadas", f"{cobertura['posts']:,}")
c2.metric("Cobertura de comentarios",
          f"{cobertura['cobertura_comentarios']:.0%}",
          help="Porcentaje de publicaciones que llegaron con comentarios. La "
               "intencion de compra se calcula sobre comentarios: si esta "
               "cifra es baja, el ranking depende casi solo de la viralidad.")
c3.metric("Productos validados", f"{cobertura['productos_validados']:,}")
c4.metric("Ventana de ingesta", f"{cobertura['ventana_dias']} dias")

if cobertura["cobertura_comentarios"] < 0.5 and cobertura["posts"]:
    st.warning(
        f"Solo {cobertura['cobertura_comentarios']:.0%} de las publicaciones "
        "trajeron comentarios. Los puntajes de intencion de compra descansan "
        "sobre una base reducida; interprete el ranking con cautela.")

if cobertura["ultima_validacion"] is not None and pd.notna(cobertura["ultima_validacion"]):
    antiguedad = (pd.Timestamp.utcnow().tz_localize(None)
                  - pd.Timestamp(cobertura["ultima_validacion"]).tz_localize(None)).days
    st.caption(f"Ultima validacion: "
               f"{pd.Timestamp(cobertura['ultima_validacion']):%Y-%m-%d %H:%M} "
               f"({antiguedad} dias atras)")

st.divider()

# --------------------------------------------------------------------------
# Ranking
# --------------------------------------------------------------------------
st.subheader("Ranking de oportunidades")

izq, der = st.columns([3, 2])

with izq:
    st.dataframe(
        df[["product_name", "opportunity_score", "purchase_intent_score",
            "google_trend_growth_pct", "estado_adopcion", "ventana_dias",
            "confianza_ventana"]],
        hide_index=True,
        use_container_width=True,
        column_config={
            "product_name": st.column_config.TextColumn("Producto"),
            "opportunity_score": st.column_config.ProgressColumn(
                "Score", min_value=0.0, max_value=1.0, format="%.3f"),
            "purchase_intent_score": st.column_config.NumberColumn(
                "Intencion", format="%.3f"),
            "google_trend_growth_pct": st.column_config.NumberColumn(
                "Crecim. busquedas", format="%.1f%%"),
            "estado_adopcion": st.column_config.TextColumn("Adopcion local"),
            "ventana_dias": st.column_config.NumberColumn(
                "Ventana (dias)", format="%d"),
            "confianza_ventana": st.column_config.TextColumn("Confianza"),
        })

with der:
    st.markdown("**Composicion del puntaje**")
    st.caption("Cada barra suma exactamente el score del producto. Sirve para "
               "distinguir un producto que la gente quiere comprar de uno que "
               "solo aparecio en un video popular.")
    composicion = df.set_index("product_name")[
        ["aporte_viralidad", "aporte_intencion", "aporte_tendencia"]]
    composicion.columns = ["Viralidad (40%)", "Intencion (40%)", "Busquedas (20%)"]
    st.bar_chart(composicion, horizontal=True, height=max(220, 34 * len(df)))

st.divider()

# --------------------------------------------------------------------------
# Detalle
# --------------------------------------------------------------------------
st.subheader("Detalle del producto")

elegido = st.selectbox("Producto", df["product_name"].tolist())
fila = df[df["product_name"] == elegido].iloc[0]

d1, d2, d3 = st.columns(3)
d1.metric("Opportunity Score", f"{fila['opportunity_score']:.3f}")
d2.metric("Intencion de compra", f"{fila['purchase_intent_score']:.3f}")
d3.metric("Velocidad de interaccion",
          f"{fila['engagement_velocity']:,.0f}/dia",
          help="Interacciones acumuladas por dia desde la publicacion. "
               "Sustituye a los likes totales, que premiaban publicaciones "
               "antiguas ya saturadas.")

mercado = fila["trend_geo"] or "mundial"
if fila["confianza_ventana"] == LAG_MEASURED:
    st.success(
        f"**Ventana de adopcion: {int(fila['ventana_dias'])} dias.** "
        f"El interes de busqueda en Colombia reproduce la curva de "
        f"{mercado} con ese rezago "
        f"(correlacion {fila['lag_correlation']:.2f} sobre primeras "
        f"diferencias). Ese es el margen estimado para conseguir inventario "
        f"antes de que la demanda local aparezca.")
else:
    st.info(
        f"**Ventana de adopcion no medible.** La correlacion entre "
        f"{mercado} y Colombia no supera el umbral de significancia, asi que "
        f"cualquier rezago que se reportara seria indistinguible del azar. "
        f"El producto sigue siendo valido como oportunidad, pero sin una "
        f"estimacion confiable de cuanto tiempo hay.")

if pd.notna(fila["adoption_gap"]):
    st.caption(
        f"Brecha de adopcion: {fila['adoption_gap'] * 100:+.1f} puntos "
        f"porcentuales entre {mercado} y Colombia — {fila['estado_adopcion']}.")

st.divider()

# --------------------------------------------------------------------------
# Exportacion
# --------------------------------------------------------------------------
exportable = export_columns(df)
st.download_button(
    "Descargar resultados en CSV",
    data=exportable.to_csv(index=False).encode("utf-8-sig"),
    file_name="oportunidades.csv",
    mime="text/csv",
    help="Codificado en UTF-8 con BOM para que Excel respete los acentos.")

st.caption(
    "El tablero no muestra comentarios ni identificadores de las "
    "publicaciones originales. Solo se presentan agregados, en aplicacion del "
    "principio de minimizacion de la Ley 1581 de 2012.")
