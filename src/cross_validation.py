import time
import logging
from typing import Dict, Any, Optional, Tuple
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)
import numpy as np
from pytrends.request import TrendReq
from pytrends.exceptions import TooManyRequestsError

logger = logging.getLogger(__name__)

class TrendValidator:
    """
    Module 3: Cross-Validation Layer
    Uses Google Trends (pytrends) to cross-reference extracted products against search demand.
    Filters products with flat or decreasing trajectories, validating only those with upward momentum.
    """
    def __init__(self, geo: str = 'US', timeframe: str = 'today 1-m', max_retries: int = 3, backoff_factor: int = 5):
        """
        Initializes the Trend Validator.
        
        Args:
            geo: Geographic location for trends (default 'US', use '' for global).
            timeframe: Window of data (default 'today 1-m' for the last 30 days).
            max_retries: How many times to retry on a 429 Too Many Requests error.
            backoff_factor: Multiplier for exponential backoff during retries.
        """
        self.geo = geo
        self.timeframe = timeframe
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        
        # Initialize pytrends. We use timeout to avoid hanging requests.
        self.pytrends = TrendReq(hl='en-US', tz=360, timeout=(10, 25))

    def fetch_interest_over_time(self, keyword: str) -> Optional[pd.DataFrame]:
        """
        Fetches the interest over time data for a specific keyword with exponential backoff.
        Combats Pytrends aggressive rate limiting.
        """
        for attempt in range(self.max_retries):
            try:
                # We build payload exactly for the specific keyword
                self.pytrends.build_payload([keyword], cat=0, timeframe=self.timeframe, geo=self.geo, gprop='')
                df = self.pytrends.interest_over_time()
                
                # Sleep momentarily to respect API limits even on success
                time.sleep(2)
                
                if df.empty:
                    logger.debug(f"Pytrends returned empty DataFrame for '{keyword}'.")
                    return None
                    
                # Drop the isPartial column if it exists
                if 'isPartial' in df.columns:
                    df = df.drop(columns=['isPartial'])
                    
                return df
                
            except TooManyRequestsError:
                wait_time = self.backoff_factor * (2 ** attempt)
                logger.warning(f"Pytrends rate limit hit. Retrying '{keyword}' in {wait_time}s (Attempt {attempt+1}/{self.max_retries})")
                time.sleep(wait_time)
            except Exception as e:
                logger.error(f"Unexpected error fetching trends for '{keyword}': {e}")
                break
                
        logger.error(f"Failed to fetch trends for '{keyword}' after {self.max_retries} attempts.")
        return None

    def calculate_growth_metrics(self, df: pd.DataFrame, keyword: str) -> Tuple[bool, float, float]:
        """
        Analyzes the trend 30-day time series to determine if the product has an upward trajectory.
        
        Returns:
            Tuple(Passed_Validation: bool, Growth_Percentage: float, Avg_Volume: float)
        """
        if df is None or df.empty or keyword not in df.columns:
            return False, 0.0, 0.0
            
        series = df[keyword]
        
        # We need at least 14 days of data to compare last week vs previous weeks
        if len(series) < 14:
            return False, 0.0, series.mean()
            
        avg_volume = series.mean()
        
        # Filter very low absolute volume keywords to prevent noisy spikes
        if avg_volume < 10:
            return False, 0.0, avg_volume
            
        # Slope calculation using straightforward linear regression over the 30 days
        x = np.arange(len(series))
        y = series.values
        slope, _ = np.polyfit(x, y, 1)
        
        # Growth calculation: Last 7 days vs Previous 7 days
        last_7_days_avg = series[-7:].mean()
        previous_7_days_avg = series[-14:-7].mean()
        
        if previous_7_days_avg == 0:
            growth_pct = 1.0 if last_7_days_avg > 0 else 0.0 # 100% growth if it went from 0 to something
        else:
            growth_pct = (last_7_days_avg - previous_7_days_avg) / previous_7_days_avg
            
        # Validation rules:
        # 1. Trajectory must be upward (positive slope over 30 days)
        # 2. Must exhibit recent growth (e.g., > 0% growth in the last week for calibration)
        is_valid = slope > 0 and growth_pct > 0.0
        
        # Return percentage multiplied by 100 for readability in score algorithms
        return is_valid, growth_pct, avg_volume

    def validate_products(self, products: list[str]) -> pd.DataFrame:
        """
        Iterates over a list of extracted products, fetches their trends, and filters out saturated ones.
        Returns a DataFrame summarizing the validation results of passing products.
        """
        logger.info(f"Validating {len(products)} unique products against Google Trends...")
        results = []
        
        for idx, product in enumerate(products):
            # Clean product name for search
            search_term = product.strip().lower()
            if not search_term or len(search_term) < 3:
                continue
                
            logger.info(f"[{idx+1}/{len(products)}] Checking trends for: '{search_term}'")
            trend_df = self.fetch_interest_over_time(search_term)
            
            is_valid, growth, avg_vol = self.calculate_growth_metrics(trend_df, search_term)
            
            if is_valid:
                logger.info(f"✅ '{search_term}' passed validation! (Growth: {growth:.1%}, Vol: {avg_vol:.1f})")
                results.append({
                    "product_name": search_term,
                    "trend_growth": growth,
                    "trend_volume": avg_vol
                })
            else:
                logger.debug(f"❌ '{search_term}' failed validation. Flat or decreasing.")
                
        logger.info(f"Cross-Validation complete. {len(results)} out of {len(products)} products passed.")
        return pd.DataFrame(results)

if __name__ == "__main__":
    # Test execution
    try:
        validator = TrendValidator()
        # Test with a mix of an established trend and an unlikely trend
        test_products = ["stanley cup", "fidget spinner", "vintage typewriter"]
        validation_df = validator.validate_products(test_products)
        
        print("\nValidated Validation DataFrame:")
        if not validation_df.empty:
            print(validation_df.to_string())
        else:
            print("No products passed validation within the current 30-day timeframe.")
            
    except Exception as err:
        print(f"Test failed: {err}")
