"""
Module 5: Reporting Layer.

Reads what the pipeline already stored and derives the fields the dashboard
displays. It lives apart from the Streamlit app on purpose: the UI cannot be
unit tested without a browser and a running server, but every rule that decides
what a PYME is told about a product can be, and those rules are the part that
would mislead someone if it were wrong.

Two constraints shape this module:

1. It never writes. The dashboard is a consumer of the database, and opening it
   in read/write mode would let a stray Streamlit rerun create tables or lock
   the file while the pipeline is running.
2. It never surfaces raw social text. `raw_social_data` holds scraped comments
   and the identifiers of the posts they came from; under the minimisation
   principle of Ley 1581 de 2012 the operational product has no reason to
   display them, so coverage is reported as counts only.
"""

import logging
from typing import Optional

import pandas as pd
from sqlalchemy import create_engine, inspect, text

from .config import Config
from .database import SCORE_WEIGHTS

logger = logging.getLogger(__name__)

# Columns the dashboard expects from `final_trends`. Declared explicitly so an
# empty database yields a frame with the right shape instead of raising, which
# is the state the app is in before the first pipeline run.
TREND_COLUMNS = [
    "product_name",
    "opportunity_score",
    "viral_metric_score",
    "purchase_intent_score",
    "google_trend_growth_pct",
    "engagement_velocity",
    "trend_geo",
    "trend_growth_local",
    "adoption_gap",
    "lag_days",
    "lag_correlation",
    "validated_at",
]

# Labels shown to the user for the reliability of the adoption window.
LAG_MEASURED = "ventana medida"
LAG_NOT_MEASURABLE = "no medible"


class TrendReport:
    """Read-only view over the pipeline's results."""

    def __init__(self, db_url: Optional[str] = None):
        self.db_url = db_url or Config.DB_PATH
        # `future=True` keeps SQLAlchemy 2.x semantics explicit. No
        # `create_all` here: the reporting layer must not alter the schema.
        self.engine = create_engine(self.db_url, future=True)

    def _has_table(self, name: str) -> bool:
        try:
            return inspect(self.engine).has_table(name)
        except Exception as err:  # a missing or unreadable file is not fatal
            logger.warning("No se pudo inspeccionar la base de datos: %s", err)
            return False

    def load_trends(self, limit: Optional[int] = None) -> pd.DataFrame:
        """
        Returns the scored products, best opportunity first.

        An empty frame with the declared columns is returned when the pipeline
        has not run yet, so the dashboard can render its empty state without
        special-casing an exception.
        """
        if not self._has_table("final_trends"):
            return pd.DataFrame(columns=TREND_COLUMNS)

        query = "SELECT {} FROM final_trends ORDER BY opportunity_score DESC".format(
            ", ".join(TREND_COLUMNS))
        if limit:
            query += f" LIMIT {int(limit)}"

        try:
            df = pd.read_sql(text(query), self.engine)
        except Exception as err:
            logger.error("Fallo la lectura de final_trends: %s", err)
            return pd.DataFrame(columns=TREND_COLUMNS)

        if df.empty:
            return pd.DataFrame(columns=TREND_COLUMNS)

        df["validated_at"] = pd.to_datetime(df["validated_at"], errors="coerce")
        return self.annotate(df)

    @staticmethod
    def annotate(df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds the derived fields the dashboard explains to the user.

        The score decomposition matters more than the score itself: a PYME
        deciding whether to stock a product needs to know whether it ranked high
        because people are asking to buy it or merely because a video was
        popular, and those two cases warrant different decisions.
        """
        if df is None or df.empty:
            return df

        out = df.copy()

        # Contribution of each term to the final score, in score points. These
        # sum to opportunity_score by construction.
        out["aporte_viralidad"] = (
            SCORE_WEIGHTS["viral"] * pd.to_numeric(
                out["viral_metric_score"], errors="coerce").fillna(0.0))
        out["aporte_intencion"] = (
            SCORE_WEIGHTS["intent"] * pd.to_numeric(
                out["purchase_intent_score"], errors="coerce").fillna(0.0))
        # The stored growth is the raw percentage; the score used it clipped to
        # the 0-1 range, so the contribution has to be clipped the same way or
        # the parts would not add up to the whole.
        out["aporte_tendencia"] = (
            SCORE_WEIGHTS["trend"] * pd.to_numeric(
                out["google_trend_growth_pct"], errors="coerce"
            ).fillna(0.0).clip(lower=0.0, upper=1.0))

        # `lag_days` is stored as NULL precisely when the correlation between
        # both markets did not clear the significance threshold, so its absence
        # already encodes "this window is not distinguishable from noise". We
        # translate that into words rather than showing an empty cell the user
        # would read as zero days.
        out["confianza_ventana"] = out["lag_days"].map(
            lambda v: LAG_NOT_MEASURABLE if pd.isna(v) else LAG_MEASURED)

        # Nullable integer rather than float: `lag_days` arrives as float64
        # because SQL NULLs force that dtype, and a plain `.map` returning None
        # would be coerced straight back to NaN. Int64 keeps the days an integer
        # while representing "not measured" as pd.NA, which renders as an empty
        # cell instead of a misleading 0.
        out["ventana_dias"] = pd.to_numeric(
            out["lag_days"], errors="coerce").astype("Int64")

        # A positive gap means the detection market is accelerating faster than
        # the local one: the trend has not landed here yet, which is the window
        # the project exists to exploit. A negative gap means it already did.
        out["estado_adopcion"] = out["adoption_gap"].map(_adoption_label)

        return out

    def coverage(self) -> dict:
        """
        Data-provenance panel: how much material the conclusions rest on.

        Comment coverage is the headline number because the purchase-intent
        signal is computed from comments; a run where most posts arrived without
        them produces scores driven almost entirely by engagement, and the user
        deserves to see that before acting on the ranking.
        """
        summary = {
            "posts": 0,
            "posts_con_comentarios": 0,
            "cobertura_comentarios": 0.0,
            "entidades": 0,
            "productos_validados": 0,
            "ventana_dias": Config.INGESTION_WINDOW_DAYS,
            "ultima_validacion": None,
        }

        if self._has_table("raw_social_data"):
            try:
                with self.engine.connect() as conn:
                    row = conn.execute(text(
                        "SELECT COUNT(*) AS total, "
                        "SUM(CASE WHEN comment_text IS NOT NULL "
                        "     AND TRIM(comment_text) != '' THEN 1 ELSE 0 END) AS con "
                        "FROM raw_social_data")).mappings().first()
                if row:
                    summary["posts"] = int(row["total"] or 0)
                    summary["posts_con_comentarios"] = int(row["con"] or 0)
                    if summary["posts"]:
                        summary["cobertura_comentarios"] = (
                            summary["posts_con_comentarios"] / summary["posts"])
            except Exception as err:
                logger.warning("No se pudo calcular la cobertura: %s", err)

        if self._has_table("processed_entities"):
            try:
                with self.engine.connect() as conn:
                    summary["entidades"] = int(conn.execute(text(
                        "SELECT COUNT(*) FROM processed_entities")).scalar() or 0)
            except Exception as err:
                logger.warning("No se pudo contar entidades: %s", err)

        if self._has_table("final_trends"):
            try:
                with self.engine.connect() as conn:
                    summary["productos_validados"] = int(conn.execute(text(
                        "SELECT COUNT(*) FROM final_trends")).scalar() or 0)
                    last = conn.execute(text(
                        "SELECT MAX(validated_at) FROM final_trends")).scalar()
                summary["ultima_validacion"] = (
                    pd.to_datetime(last, errors="coerce") if last else None)
            except Exception as err:
                logger.warning("No se pudo leer final_trends: %s", err)

        return summary


def _adoption_label(gap) -> str:
    """Turns the raw adoption gap into the phrase the dashboard shows."""
    if gap is None or pd.isna(gap):
        return "sin comparacion local"
    if gap > 0.05:
        return "aun no llega al mercado local"
    if gap < -0.05:
        return "el mercado local ya va adelante"
    return "ambos mercados al mismo ritmo"


def export_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Selects the columns handed to a PYME as CSV.

    Internal normalisations (viral_metric_score, the per-term contributions) are
    left out: they are useful to explain a ranking on screen but meaningless in
    a spreadsheet without the accompanying text.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=[
            "producto", "score", "intencion_compra", "crecimiento_busquedas",
            "mercado_deteccion", "estado_adopcion", "ventana_dias",
            "confianza_ventana", "validado_en"])

    return pd.DataFrame({
        "producto": df["product_name"],
        "score": df["opportunity_score"].round(4),
        "intencion_compra": df["purchase_intent_score"].round(4),
        "crecimiento_busquedas": df["google_trend_growth_pct"].round(4),
        "mercado_deteccion": df["trend_geo"],
        "estado_adopcion": df["estado_adopcion"],
        "ventana_dias": df["ventana_dias"],
        "confianza_ventana": df["confianza_ventana"],
        "validado_en": df["validated_at"],
    })
