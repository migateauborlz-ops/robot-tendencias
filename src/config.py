import os
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

class Config:
    """Central configuration class for the application."""
    
    # API Keys
    APIFY_API_TOKEN = os.getenv("APIFY_API_TOKEN", "")
    
    # Database Configuration
    DB_PATH = os.getenv("DB_PATH", "sqlite:///../data/trends.db")
    
    # Apify Settings
    APIFY_ACTOR_TIKTOK_VIDEOS = os.getenv("APIFY_ACTOR_TIKTOK_VIDEOS", "clockworks/tiktok-scraper")
    APIFY_ACTOR_TIKTOK_COMMENTS = os.getenv("APIFY_ACTOR_TIKTOK_COMMENTS", "clockworks/tiktok-comments-scraper")
    APIFY_ACTOR_INSTAGRAM = os.getenv("APIFY_ACTOR_INSTAGRAM", "apify/instagram-scraper")
    
    # Max items to scrape per run to control costs
    MAX_SCRAPE_ITEMS = int(os.getenv("MAX_SCRAPE_ITEMS", "100"))
    
    @classmethod
    def validate(cls):
        """Validates that all required configuration variables are set."""
        if not cls.APIFY_API_TOKEN:
            logging.warning("APIFY_API_TOKEN is missing. Please set it in your .env file.")

# Validate config on import to fail fast
Config.validate()
