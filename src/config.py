import os
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging: console for live feedback, file for the audit trail.
# The file handler matters because a silent ingestion failure only shows up as a
# WARNING/ERROR line, and without persistence there is no evidence afterwards.
_LOG_FILE = os.getenv("PIPELINE_LOG", "pipeline.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(_LOG_FILE, mode="a", encoding="utf-8"),
    ],
)

class Config:
    """Central configuration class for the application."""
    
    # API Keys
    APIFY_API_TOKEN = os.getenv("APIFY_API_TOKEN", "")
    
    # Database Configuration
    DB_PATH = os.getenv("DB_PATH", "sqlite:///data/trends.db")
    
    # Apify Settings
    APIFY_ACTOR_TIKTOK_VIDEOS = os.getenv("APIFY_ACTOR_TIKTOK_VIDEOS", "clockworks/tiktok-scraper")
    APIFY_ACTOR_TIKTOK_COMMENTS = os.getenv("APIFY_ACTOR_TIKTOK_COMMENTS", "clockworks/tiktok-comments-scraper")
    APIFY_ACTOR_INSTAGRAM = os.getenv("APIFY_ACTOR_INSTAGRAM", "apify/instagram-scraper")
    
    # Max items to scrape per run to control costs
    MAX_SCRAPE_ITEMS = int(os.getenv("MAX_SCRAPE_ITEMS", "100"))

    # NLP models.
    # SENTIMENT_MODEL must be a checkpoint fine-tuned for classification. The
    # previous default, distilbert-base-multilingual-cased, is a base language
    # model: it loads fine but scores every text at roughly 0.55 with a generic
    # LABEL_0, so purchase intent was pure noise. The nlptown checkpoint is
    # multilingual (Spanish included), trained on product reviews, and emits
    # star ratings that the scoring branch already understands.
    SPACY_MODEL = os.getenv("SPACY_MODEL", "es_core_news_sm")
    SENTIMENT_MODEL = os.getenv(
        "SENTIMENT_MODEL", "nlptown/bert-base-multilingual-uncased-sentiment")

    # Google Trends geography.
    # TRENDS_GEO is the market where a trend is DETECTED. It defaults to worldwide
    # because the social corpus comes from global hashtags: measured on
    # 2026-08-20, products viral there had little or no search volume in Colombia
    # ("nicetown curtains" returned no data at all with geo='CO').
    # TRENDS_GEO_LOCAL is the target market whose adoption lag is the commercial
    # opportunity. Set TRENDS_COMPARE_GEOS=0 to validate only against TRENDS_GEO.
    TRENDS_GEO = os.getenv("TRENDS_GEO", "")            # "" = mundial
    TRENDS_GEO_LOCAL = os.getenv("TRENDS_GEO_LOCAL", "CO")
    TRENDS_COMPARE_GEOS = os.getenv("TRENDS_COMPARE_GEOS", "1") not in ("0", "false", "False")
    TRENDS_TIMEFRAME = os.getenv("TRENDS_TIMEFRAME", "today 1-m")

    # Recency window applied to ingested posts.
    # Measured on 2026-08-20 over a 30-post hashtag sample: posts under 7 days old
    # had a median of 16 likes and no comments at all, because engagement has not
    # accumulated yet. A 30-day window keeps posts that are still emerging while
    # giving them time to gather the conversation the NLP layer needs.
    INGESTION_WINDOW_DAYS = int(os.getenv("INGESTION_WINDOW_DAYS", "30"))
    
    @classmethod
    def validate(cls):
        """Validates that all required configuration variables are set."""
        if not cls.APIFY_API_TOKEN:
            logging.warning("APIFY_API_TOKEN is missing. Please set it in your .env file.")

# Validate config on import to fail fast
Config.validate()
