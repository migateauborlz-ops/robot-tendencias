import time
import logging
from typing import Dict, Any, Optional, Tuple
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)
import numpy as np
from pytrends.request import TrendReq
from pytrends.exceptions import TooManyRequestsError

from .config import Config

logger = logging.getLogger(__name__)

# Minimum correlation required to report an adoption lag. It is the empirical
# 95th percentile of the null distribution: 2,000 pairs of independent random
# walks of 30 points, correlated on first differences across 15 candidate lags,
# reached 0.602 at that percentile. Anything below is indistinguishable from
# chance. See estimate_lag_days.
MIN_LAG_CORRELATION = 0.60

class TrendValidator:
    """
    Module 3: Cross-Validation Layer
    Uses Google Trends (pytrends) to cross-reference extracted products against search demand.
    Filters products with flat or decreasing trajectories, validating only those with upward momentum.
    """
    def __init__(self, geo: str = None, timeframe: str = None, max_retries: int = 3,
                 backoff_factor: int = 5, geo_local: str = None,
                 compare_geos: bool = None):
        """
        Initializes the Trend Validator.

        Args:
            geo: Market where the trend is detected. Defaults to Config.TRENDS_GEO
                ("" means worldwide).
            timeframe: Window of data (default 'today 1-m' for the last 30 days).
            max_retries: How many times to retry on a 429 Too Many Requests error.
            backoff_factor: Multiplier for exponential backoff during retries.
            geo_local: Target market to measure adoption against. The gap between
                this market and `geo` is the window the product is meant to exploit.
            compare_geos: When True the term is also queried in `geo_local`, so the
                lag between both markets can be reported.
        """
        self.geo = Config.TRENDS_GEO if geo is None else geo
        self.geo_local = Config.TRENDS_GEO_LOCAL if geo_local is None else geo_local
        self.compare_geos = (Config.TRENDS_COMPARE_GEOS
                             if compare_geos is None else compare_geos)
        self.timeframe = timeframe or Config.TRENDS_TIMEFRAME
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor

        # Comparing a market against itself would only double the cost in time.
        if self.geo_local == self.geo:
            self.compare_geos = False

        logger.info("Trend validation geography: detection='%s', local='%s', "
                    "comparison=%s",
                    self.geo or "mundial", self.geo_local or "mundial",
                    "on" if self.compare_geos else "off")

        # Initialize pytrends. We use timeout to avoid hanging requests.
        self.pytrends = TrendReq(hl='en-US', tz=360, timeout=(10, 25))

    def fetch_interest_over_time(self, keyword: str,
                                 geo: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        Fetches the interest over time data for a specific keyword with exponential backoff.
        Combats Pytrends aggressive rate limiting.
        """
        for attempt in range(self.max_retries):
            try:
                # We build payload exactly for the specific keyword
                target_geo = self.geo if geo is None else geo
                self.pytrends.build_payload([keyword], cat=0,
                                            timeframe=self.timeframe,
                                            geo=target_geo, gprop='')
                
                # Strict 15s delay to prevent 429 Too Many Requests errors
                logger.debug(f"Sleeping 15s before PyTrends request for '{keyword}'...")
                time.sleep(15)
                
                df = self.pytrends.interest_over_time()
                
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

    @staticmethod
    def estimate_lag_days(leading: pd.Series, following: pd.Series,
                          max_lag: int = 14) -> Tuple[Optional[int], float]:
        """
        Estimates how many days the `following` market trails the `leading` one.

        This is the empirical measure of the adoption window the project exploits:
        if interest in Colombia reproduces the global curve k days later, k is the
        margin a local retailer has to stock the product before demand arrives.

        Two precautions make the estimate trustworthy:

        1. The correlation is computed on FIRST DIFFERENCES, not on the levels.
           Search-interest series behave like random walks, and correlating the
           levels of two independent random walks produces the classic spurious
           regression: over 2,000 simulated pairs the 95th percentile of the best
           correlation across 15 lags reached 0.888. On first differences the same
           percentile drops to 0.602.
        2. A lag is only reported when the correlation clears
           MIN_LAG_CORRELATION, the empirical 95th percentile of that null
           distribution. Below it the propagation is indistinguishable from noise
           and the method returns None rather than a spurious number.

        Returns (lag in days or None, correlation at the best lag).
        """
        if leading is None or following is None:
            return None, 0.0

        a = pd.to_numeric(leading, errors="coerce").dropna().to_numpy(dtype=float)
        b = pd.to_numeric(following, errors="coerce").dropna().to_numpy(dtype=float)
        n = min(len(a), len(b))
        if n < 16:   # 15 differences leave enough overlap after shifting
            return None, 0.0

        # First differences remove the shared trend that inflates the correlation.
        a = np.diff(a[-n:])
        b = np.diff(b[-n:])
        n = len(a)

        best_lag, best_corr = None, -1.0
        for lag in range(0, min(max_lag, n - 10) + 1):
            x = a[:n - lag] if lag else a
            y = b[lag:] if lag else b
            if len(x) < 10 or x.std() == 0 or y.std() == 0:
                continue
            corr = float(np.corrcoef(x, y)[0, 1])
            if corr > best_corr:
                best_lag, best_corr = lag, corr

        best_corr = round(max(best_corr, 0.0), 3)
        if best_lag is None or best_corr < MIN_LAG_CORRELATION:
            return None, best_corr
        return best_lag, best_corr

    def validate_products(self, products: list[str]) -> pd.DataFrame:
        """
        Iterates over a list of extracted products, fetches their trends, and filters out saturated ones.
        Returns a DataFrame summarizing the validation results of passing products.
        """
        logger.info("Validating %d unique products against Google Trends (geo='%s')...",
                    len(products), self.geo or "mundial")
        results = []

        for idx, product in enumerate(products):
            # Clean product name for search
            search_term = product.strip().lower()
            if not search_term or len(search_term) < 3:
                continue

            logger.info(f"[{idx+1}/{len(products)}] Checking trends for: '{search_term}'")
            trend_df = self.fetch_interest_over_time(search_term, geo=self.geo)

            is_valid, growth, avg_vol = self.calculate_growth_metrics(trend_df, search_term)

            if not is_valid:
                logger.debug(f"❌ '{search_term}' failed validation. Flat or decreasing.")
                continue

            record = {
                "product_name": search_term,
                "trend_growth": growth,
                "trend_volume": avg_vol,
                "trend_geo": self.geo or "WORLD",
                "trend_growth_local": None,
                "trend_volume_local": None,
                "adoption_gap": None,
                "lag_days": None,
                "lag_correlation": None,
            }

            # Only products that already passed are compared against the local
            # market: querying the rest would spend rate-limit budget on terms
            # that will be discarded anyway.
            if self.compare_geos:
                local_df = self.fetch_interest_over_time(search_term, geo=self.geo_local)
                _, local_growth, local_vol = self.calculate_growth_metrics(
                    local_df, search_term)
                record["trend_growth_local"] = local_growth
                record["trend_volume_local"] = local_vol
                # Positive gap: the detection market is moving faster than the
                # local one, so the trend has not landed here yet.
                record["adoption_gap"] = growth - local_growth
                if trend_df is not None and local_df is not None:
                    lag, corr = self.estimate_lag_days(
                        trend_df[search_term], local_df[search_term])
                    record["lag_days"] = lag
                    record["lag_correlation"] = corr
                logger.info("   %s vs %s: growth %.1f%% vs %.1f%% | gap %.1f pp | "
                            "lag %s d (r=%s)",
                            self.geo or "WORLD", self.geo_local,
                            growth * 100, local_growth * 100,
                            record["adoption_gap"] * 100,
                            record["lag_days"], record["lag_correlation"])

            logger.info(f"✅ '{search_term}' passed validation! (Growth: {growth:.1%}, Vol: {avg_vol:.1f})")
            results.append(record)

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
