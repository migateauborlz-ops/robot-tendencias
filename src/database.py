import os
import logging
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, JSON
from sqlalchemy.orm import declarative_base, sessionmaker

from .config import Config

logger = logging.getLogger(__name__)

Base = declarative_base()

# Floor on post age when computing velocity: a post published minutes ago
# would otherwise divide by ~0 and dominate the ranking on a handful of likes.
MIN_AGE_DAYS = 0.5


def _opt_float(value):
    """Casts to float preserving None, so an unmeasured metric is not stored as 0."""
    try:
        return None if value is None or pd.isna(value) else float(value)
    except Exception:
        return None


def _opt_int(value):
    try:
        return None if value is None or pd.isna(value) else int(value)
    except Exception:
        return None

# --- Database Schema ---

class RawSocialData(Base):
    """Stores the raw ingested data from social platforms."""
    __tablename__ = 'raw_social_data'
    
    id = Column(Integer, primary_key=True)
    platform = Column(String(50))
    video_id = Column(String(100), unique=True, index=True)
    url = Column(String(500))
    description = Column(String)
    comment_text = Column(String)
    likes_count = Column(Integer)
    shares_count = Column(Integer)
    timestamp = Column(DateTime)
    ingested_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

class ProcessedEntities(Base):
    """Stores the NLP processed insights for each video/post."""
    __tablename__ = 'processed_entities'
    
    id = Column(Integer, primary_key=True)
    video_id = Column(String(100), index=True) # Foreign key relation implied
    extracted_products = Column(JSON)          # Storing list of products as JSON
    purchase_intent_score = Column(Float)
    high_intent_comments_count = Column(Integer)
    processed_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

class FinalTrends(Base):
    """Stores the final validated and scored products."""
    __tablename__ = 'final_trends'
    
    id = Column(Integer, primary_key=True)
    product_name = Column(String(100), index=True)
    engagement_velocity = Column(Float)   # interacciones por dia
    viral_metric_score = Column(Float)
    purchase_intent_score = Column(Float)
    google_trend_growth_pct = Column(Float)      # mercado de deteccion
    trend_geo = Column(String(10))               # geografia de deteccion
    trend_growth_local = Column(Float)           # mismo termino en el mercado local
    adoption_gap = Column(Float)                 # deteccion - local, en puntos
    lag_days = Column(Integer)                   # rezago estimado del mercado local
    lag_correlation = Column(Float)              # correlacion en ese rezago
    opportunity_score = Column(Float, index=True)
    validated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

# --- Database & Scoring Manager ---

class StorageAndScoring:
    """
    Module 4: Scoring and Database Storage
    Handles SQLite via SQLAlchemy and computes the final Opportunity Score.
    """
    def __init__(self, db_url: str = Config.DB_PATH):
        # By default, Config.DB_PATH points to sqlite:///../data/trends.db
        # Ensure the directory exists if it's a file path
        if db_url.startswith('sqlite:///'):
            db_path = db_url.replace('sqlite:///', '')
            os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
            
        logger.info(f"Connecting to database: {db_url}")
        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        
    def save_raw_data(self, df: pd.DataFrame):
        """Saves the output from the Ingestion Pipeline."""
        if df.empty:
            return
            
        with self.Session() as session:
            for _, row in df.iterrows():
                # Upsert logic based on video_id
                existing = session.query(RawSocialData).filter_by(video_id=row['video_id']).first()
                if not existing:
                    new_record = RawSocialData(
                        platform=row.get('platform'),
                        video_id=row.get('video_id'),
                        url=row.get('url'),
                        description=row.get('description'),
                        comment_text=row.get('comment_text'),
                        likes_count=int(row.get('likes_count') or 0),
                        shares_count=int(row.get('shares_count') or 0),
                        timestamp=row.get('timestamp')
                    )
                    session.add(new_record)
            session.commit()
            logger.info("Raw social data saved to database.")

    def save_processed_entities(self, df: pd.DataFrame):
        """Saves the output from the NLP Layer."""
        if df.empty:
            return
            
        with self.Session() as session:
            for _, row in df.iterrows():
                existing = session.query(ProcessedEntities).filter_by(video_id=row.get('video_id')).first()
                if existing:
                    existing.extracted_products = row.get('extracted_products', [])
                    existing.purchase_intent_score = float(row.get('purchase_intent_score') or 0.0)
                    existing.high_intent_comments_count = int(row.get('high_intent_comments_count') or 0)
                    existing.processed_at = datetime.now(timezone.utc)
                else:
                    new_record = ProcessedEntities(
                        video_id=row.get('video_id'),
                        extracted_products=row.get('extracted_products', []),
                        purchase_intent_score=float(row.get('purchase_intent_score') or 0.0),
                        high_intent_comments_count=int(row.get('high_intent_comments_count') or 0)
                    )
                    session.add(new_record)
            session.commit()
            logger.info("Processed entities saved to database.")

    @staticmethod
    def _engagement_velocity(row, now: pd.Timestamp) -> float:
        """
        Engagement accumulated per day since the post was published.

        A post with 16 likes in two days is growing faster than one with 300,000
        likes over four years; accumulated counts cannot tell them apart.
        """
        published = None
        try:
            raw_ts = row.get("timestamp")
            if raw_ts is not None and not pd.isna(raw_ts):
                published = pd.to_datetime(raw_ts, utc=True)
        except Exception:
            published = None

        if published is None:
            # Without a date we cannot compute a rate. Assume the oldest post the
            # window admits, which penalises rather than rewards the unknown.
            age_days = float(Config.INGESTION_WINDOW_DAYS)
        else:
            age_days = (now - published).total_seconds() / 86400.0
            age_days = max(age_days, MIN_AGE_DAYS)

        engagement = float(row.get("likes_count") or 0) + float(row.get("shares_count") or 0)
        return engagement / age_days

    @staticmethod
    def _normalize_velocity(velocities: pd.Series) -> pd.Series:
        """
        Maps engagement velocity to the 0-1 range used by the Opportunity Score.

        Engagement on social platforms is heavy-tailed: a plain min-max division
        would collapse every product except the single fastest to nearly zero. A
        log transform stabilises the variance first, which is the standard
        treatment for this kind of distribution.
        """
        v = pd.to_numeric(velocities, errors="coerce").fillna(0.0).clip(lower=0.0)
        log_v = np.log1p(v)
        max_log = log_v.max()
        if not max_log or max_log <= 0:
            return pd.Series([0.0] * len(v), index=v.index)
        return (log_v / max_log).clip(upper=1.0)

    @classmethod
    def select_candidates(cls, df_nlp: pd.DataFrame, limit: int = 5) -> list:
        """
        Picks the entities worth spending a Google Trends query on.

        Trends is rate limited, so only a handful of candidates can be validated
        per run and the choice matters. Raw velocity cannot be used directly: it
        reaches six figures while intent tops out at 1.0, so every entity coming
        from the same viral post tied at the same score regardless of quality.
        Velocity is therefore normalised before being combined, and the number of
        distinct posts mentioning an entity is added as evidence of a real trend.
        """
        if df_nlp is None or df_nlp.empty:
            return []

        now = pd.Timestamp.now(tz="UTC")
        rows = []
        for _, row in df_nlp.iterrows():
            velocity = cls._engagement_velocity(row, now)
            intent = float(row.get("purchase_intent_score") or 0.0)
            for product in (row.get("extracted_products") or []):
                if isinstance(product, str) and len(product) > 2:
                    rows.append({"product_name": product.lower(),
                                 "velocity": velocity, "intent": intent})

        if not rows:
            return []

        df = pd.DataFrame(rows)
        agg = df.groupby("product_name").agg(
            velocity=("velocity", "sum"),
            intent=("intent", "mean"),
            mentions=("product_name", "size"),
        ).reset_index()

        agg["norm_velocity"] = cls._normalize_velocity(agg["velocity"])
        # Mentions are capped: appearing in three posts is already strong evidence,
        # and beyond that the signal should not outweigh intent.
        agg["norm_mentions"] = (agg["mentions"] / 3.0).clip(upper=1.0)
        agg["proxy"] = (0.4 * agg["norm_velocity"]
                        + 0.4 * agg["intent"]
                        + 0.2 * agg["norm_mentions"])

        agg = agg.sort_values("proxy", ascending=False)
        return agg["product_name"].head(limit).tolist()

    def calculate_opportunity_score(self, df_nlp: pd.DataFrame, df_validation: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates the final Opportunity Score for validated products.
        Formula: Score = (0.4 * Viral_Metric) + (0.4 * Purchase_Intent_Score) + (0.2 * Google_Trend_Growth)
        where Viral_Metric is now derived from engagement VELOCITY (interactions
        per day since publication), not from accumulated likes.
        """
        if df_validation.empty or df_nlp.empty:
            return pd.DataFrame()
            
        now = pd.Timestamp.now(tz="UTC")

        # 1. Flatten NLP DataFrame to Product level (many-to-many relationship)
        # We need to calculate aggregate metrics per unique product extracted
        product_metrics = []
        for _, row in df_nlp.iterrows():
            products = row.get("extracted_products", [])
            velocity = self._engagement_velocity(row, now)
            for p in products:
                product_metrics.append({
                    "product_name": p,
                    "likes_count": row.get("likes_count", 0),
                    "shares_count": row.get("shares_count", 0),
                    "engagement_velocity": velocity,
                    "purchase_intent_score": row.get("purchase_intent_score", 0.0)
                })

        df_product_level = pd.DataFrame(product_metrics)
        if df_product_level.empty:
            return pd.DataFrame()

        # Group by product and average/sum metrics
        df_agg = df_product_level.groupby("product_name").agg({
            "likes_count": "sum",
            "shares_count": "sum",
            "engagement_velocity": "sum",
            "purchase_intent_score": "mean"
        }).reset_index()

        # 2. Merge with Validation Metrics (Google Trends)
        final_df = pd.merge(df_validation, df_agg, on="product_name", how="inner")

        if final_df.empty:
            return pd.DataFrame()

        # 3. Calculate the Viral Metric from engagement VELOCITY, not accumulated
        # likes. Measured on 2026-08-20 over a 30-post hashtag sample, accumulated
        # likes rank four-year-old posts above everything else (8.4M likes at 1,518
        # days), which is the opposite of early detection. Dividing engagement by
        # the age of the post surfaces what is growing now.
        final_df["viral_metric_score"] = self._normalize_velocity(
            final_df["engagement_velocity"])

        # 4. Normalize Google Trend Growth (Cap exorbitant growths like 500% to 1.0)
        # Anything above 100% growth (1.0) gets max score.
        final_df["norm_trend_growth"] = final_df["trend_growth"].clip(lower=0.0, upper=1.0)
        
        # 5. The Formula
        final_df["opportunity_score"] = (
            (0.4 * final_df["viral_metric_score"]) + 
            (0.4 * final_df["purchase_intent_score"]) + 
            (0.2 * final_df["norm_trend_growth"])
        )
        
        # Sort top descending
        final_df = final_df.sort_values(by="opportunity_score", ascending=False)
        return final_df

    def save_final_trends(self, df: pd.DataFrame, top_n: int = 10):
        """Saves the top N final scored products."""
        if df.empty:
            return
            
        top_df = df.head(top_n)
        
        with self.Session() as session:
            for _, row in top_df.iterrows():
                # Avoid duplicates across runs, update or insert (Upsert)
                existing = session.query(FinalTrends).filter_by(product_name=row['product_name']).first()
                if existing:
                    existing.trend_geo = row.get('trend_geo')
                    existing.trend_growth_local = _opt_float(row.get('trend_growth_local'))
                    existing.adoption_gap = _opt_float(row.get('adoption_gap'))
                    existing.lag_days = _opt_int(row.get('lag_days'))
                    existing.lag_correlation = _opt_float(row.get('lag_correlation'))
                    existing.engagement_velocity = float(row.get('engagement_velocity', 0))
                    existing.viral_metric_score = float(row.get('viral_metric_score', 0))
                    existing.purchase_intent_score = float(row.get('purchase_intent_score', 0))
                    existing.google_trend_growth_pct = float(row.get('trend_growth', 0))
                    existing.opportunity_score = float(row.get('opportunity_score', 0))
                    existing.validated_at = datetime.now(timezone.utc)
                else:
                    new_record = FinalTrends(
                        product_name=row['product_name'],
                        trend_geo=row.get('trend_geo'),
                        trend_growth_local=_opt_float(row.get('trend_growth_local')),
                        adoption_gap=_opt_float(row.get('adoption_gap')),
                        lag_days=_opt_int(row.get('lag_days')),
                        lag_correlation=_opt_float(row.get('lag_correlation')),
                        engagement_velocity=float(row.get('engagement_velocity', 0)),
                        viral_metric_score=float(row.get('viral_metric_score', 0)),
                        purchase_intent_score=float(row.get('purchase_intent_score', 0)),
                        google_trend_growth_pct=float(row.get('trend_growth', 0)),
                        opportunity_score=float(row.get('opportunity_score', 0))
                    )
                    session.add(new_record)
            session.commit()
            logger.info(f"Top {len(top_df)} final trends saved to database.")

if __name__ == "__main__":
    # Test Execution
    try:
        # DB_PATH will be sqlite:///../data/trends.db, handling the path traversal safely relative to script execution.
        store = StorageAndScoring("sqlite:///data/trends.db")
        
        # Mock Data
        nlp_data = {
            "video_id": ["7607749319017483551"],
            "extracted_products": [["led lamp", "mini projector"]],
            "purchase_intent_score": [0.85],
            "likes_count": [150000],
            "shares_count": [5000],
            "high_intent_comments_count": [20]
        }
        df_nlp = pd.DataFrame(nlp_data)
        
        val_data = {
            "product_name": ["led lamp", "mini projector"],
            "trend_growth": [0.35, 1.20],
            "trend_volume": [65.4, 88.2]
        }
        df_val = pd.DataFrame(val_data)
        
        print("Scoring Products...")
        scored_df = store.calculate_opportunity_score(df_nlp, df_val)
        
        print("\nFinal Opportunity Scores:")
        print(scored_df[["product_name", "opportunity_score", "viral_metric_score", "purchase_intent_score", "norm_trend_growth"]])
        
        print("\nSaving to Database...")
        store.save_final_trends(scored_df)
        print("Success! Schema created and data saved.")
        
    except Exception as err:
        print(f"Test failed: {err}")
