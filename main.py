import logging
import argparse
import sys
from datetime import datetime
import pandas as pd

# The Windows console defaults to cp1252 and the final report prints emoji, which
# crashed the run AFTER the results had already been written to the database.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from src.config import Config
from src.data_ingestion import DataIngestion
from src.nlp_layer import NLPLayer
from src.cross_validation import TrendValidator
from src.database import StorageAndScoring

# Configure logging for the main pipeline execution
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("pipeline.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def run_pipeline(queries: list[str], max_items: int = 10, top_n_save: int = 10,
                 geo: str = None, geo_local: str = None,
                 compare_geos: bool = None):
    """
    Executes the full Social Commerce Trend Detection Pipeline.
    
    1. Data Ingestion (Apify Scrapers)
    2. NLP & Intelligence (Text cleaning, NER, Intent Scoring)
    3. Cross-Validation (Google Trends)
    4. Scoring & Database Storage (SQLite)
    """
    logger.info(f"=========== Pipeline Started at {datetime.now()} ===========")
    logger.info(f"Target Queries: {queries}")
    
    # ---------------------------------------------------------
    # Module 1: Data Ingestion
    # ---------------------------------------------------------
    ingestion = DataIngestion()
    df_raw = ingestion.run_ingestion_pipeline(queries, max_items=max_items)
    
    if df_raw.empty:
        logger.warning("Pipeline halted: Data Ingestion returned an empty dataset.")
        return
        
    logger.info(f"-> Extracted {len(df_raw)} social media posts.")
    
    # Initialize Storage early to save Raw Data
    store = StorageAndScoring()
    store.save_raw_data(df_raw)

    # ---------------------------------------------------------
    # Module 2: The NLP & Intelligence Layer
    # ---------------------------------------------------------
    logger.info("-> Starting NLP Analysis...")
    nlp = NLPLayer()
    df_processed = nlp.process_dataframe(df_raw)
    
    if df_processed.empty:
        logger.warning("Pipeline halted: NLP Layer resulted in empty processed data.")
        return
        
    store.save_processed_entities(df_processed)
    
    # Candidate preselection lives in StorageAndScoring so that main.py and
    # reprocess_offline.py cannot drift apart in how they rank entities.
    all_products = StorageAndScoring.select_candidates(df_processed, limit=5)

    logger.info(f"-> Sliced to {len(all_products)} top candidate products for validation.")
    
    if not all_products:
        logger.warning("Pipeline halted: No product entities were extracted.")
        return

    # ---------------------------------------------------------
    # Module 3: Cross-Validation Layer (Google Trends)
    # ---------------------------------------------------------
    logger.info("-> Starting Google Trends Validation...")
    validator = TrendValidator(geo=geo, geo_local=geo_local,
                               compare_geos=compare_geos)
    df_validation = validator.validate_products(list(all_products))
    
    if df_validation.empty:
        logger.warning(f"Pipeline halted: 0 products passed Google Trends cross-validation (0 products out of {len(all_products)} had positive 30-day search growth > 0%).")
        return
        
    logger.info(f"-> {len(df_validation)} products passed validation.")

    # ---------------------------------------------------------
    # Module 4: Scoring and Database Storage
    # ---------------------------------------------------------
    logger.info("-> Calculating Opportunity Scores...")
    # Calculate the formula based on NLP metrics and Validation metrics
    df_final = store.calculate_opportunity_score(df_processed, df_validation)
    
    if df_final.empty:
        logger.warning("Pipeline halted: Scoring resulted in an empty dataset.")
        return
        
    # Save the Top 10 to the Database
    store.save_final_trends(df_final, top_n=top_n_save)
    
    logger.info(f"=========== Pipeline Completed Successfully! ===========")
    
    # Print the terminal summary
    print("\n" + "="*60)
    print(f"🏆 TOP {min(top_n_save, len(df_final))} SOCIAL COMMERCE OPPORTUNITIES 🏆")
    print("="*60)
    
    display_df = df_final.head(top_n_save)[["product_name", "opportunity_score", "viral_metric_score", "purchase_intent_score", "norm_trend_growth"]]
    print(display_df.to_string(index=False))
    print("\nRaw data and metrics securely saved to SQLite Database (data/trends.db)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Social Commerce Trend Detection Bot")
    parser.add_argument("--queries", nargs="+", default=["#TikTokMadeMeBuyItColombia", "#ProductosVirales", "#ComprasColombia"], 
                        help="List of hashtags or search queries to run the scraper on.")
    parser.add_argument("--items", type=int, default=5, 
                        help="Maximum number of posts to scrape per query (default 5 for testing).")
    parser.add_argument("--top_n", type=int, default=10, 
                        help="Number of top trends to save and display.")
    parser.add_argument("--geo", default=None,
                        help="Mercado de deteccion en Google Trends. Cadena vacia = "
                             "mundial. Por defecto Config.TRENDS_GEO.")
    parser.add_argument("--geo-local", dest="geo_local", default=None,
                        help="Mercado objetivo contra el que se mide el rezago "
                             "(por defecto CO).")
    parser.add_argument("--no-compare-geos", dest="compare_geos",
                        action="store_false", default=None,
                        help="Valida solo en el mercado de deteccion, sin medir rezago.")
                        
    args = parser.parse_args()
    
    try:
        run_pipeline(args.queries, max_items=args.items, top_n_save=args.top_n,
                     geo=args.geo, geo_local=args.geo_local,
                     compare_geos=args.compare_geos)
    except KeyboardInterrupt:
        print("\nPipeline execution halted by user.")
