import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta, timezone
import pandas as pd
from apify_client import ApifyClient
from apify_client._errors import ApifyApiError

from .config import Config

logger = logging.getLogger(__name__)

class DataIngestion:
    """
    Module 1: Data Ingestion
    Connects to the Apify API to fetch social media data based on specific queries.
    Supports TikTok Videos, TikTok Comments, and Instagram Scrapers.
    """
    
    def __init__(self, api_token: Optional[str] = None):
        self.token = api_token or Config.APIFY_API_TOKEN
        if not self.token:
            logger.warning("API token is missing. Some methods may fail.")
             
        self.client = ApifyClient(self.token)
        
        # Load Actor IDs from Config
        self.actor_tiktok_videos = Config.APIFY_ACTOR_TIKTOK_VIDEOS
        self.actor_tiktok_comments = Config.APIFY_ACTOR_TIKTOK_COMMENTS
        self.actor_instagram = Config.APIFY_ACTOR_INSTAGRAM
        
        # Define the timeframe constraint
        self.seven_days_ago = datetime.now(timezone.utc) - timedelta(days=7)

    def fetch_tiktok_videos(self, queries: List[str], max_items: int = Config.MAX_SCRAPE_ITEMS) -> pd.DataFrame:
        """Fetches TikTok videos matching the queries."""
        logger.info(f"Fetching TikTok videos for queries: {queries}")
        all_results = []
        
        for query in queries:
            run_input = {
                "hashtags": [query.replace("#", "")],
                "resultsPerPage": max_items,
                "shouldDownloadVideos": False,
            }
            try:
                run = self.client.actor(self.actor_tiktok_videos).call(run_input=run_input)
                dataset_items = self.client.dataset(run["defaultDatasetId"]).iterate_items()
                
                for item in dataset_items:
                    timestamp_str = item.get("createTimeISO")
                    try:
                        post_time = pd.to_datetime(timestamp_str, utc=True) if timestamp_str else None
                    except Exception:
                        post_time = None
                    
                    if post_time and post_time >= self.seven_days_ago:
                        processed_item = {
                            "platform": "tiktok",
                            "video_id": item.get("id", ""),
                            "url": item.get("webVideoUrl", ""),
                            "description": item.get("text", ""),
                            "likes_count": item.get("diggCount", 0),
                            "shares_count": item.get("shareCount", 0),
                            "timestamp": post_time,
                            "commentsCount": item.get("commentCount", 0)
                        }
                        all_results.append(processed_item)
            except ApifyApiError as e:
                logger.error(f"Apify API error (TikTok Videos) for query '{query}': {e}")
                
        return pd.DataFrame(all_results)

    def fetch_tiktok_comments(self, video_urls: List[str], max_comments_per_video: int = 20) -> pd.DataFrame:
        """Fetches comments for specific TikTok video URLs."""
        if not video_urls:
            return pd.DataFrame()
            
        logger.info(f"Fetching TikTok comments for {len(video_urls)} videos...")
        all_results = []
        
        run_input = {
            "postURLs": video_urls,
            "commentsPerPost": max_comments_per_video
        }
        
        try:
            run = self.client.actor(self.actor_tiktok_comments).call(run_input=run_input)
            dataset_items = self.client.dataset(run["defaultDatasetId"]).iterate_items()
            
            for item in dataset_items:
                if "error" in item:
                    continue  # Skip private/deleted videos
                    
                processed_item = {
                    "video_id": item.get("aweme_id", ""),  # TikTok internal ID for video
                    "comment_text": item.get("text", ""),
                    "comment_likes": item.get("digg_count", 0),
                    "comment_replies": item.get("reply_comment_total", 0)
                }
                all_results.append(processed_item)
        except ApifyApiError as e:
            logger.error(f"Apify API error (TikTok Comments): {e}")

        return pd.DataFrame(all_results)

    def fetch_instagram_posts(self, queries: List[str], max_items: int = Config.MAX_SCRAPE_ITEMS) -> pd.DataFrame:
        """Fetches Instagram posts matching the queries."""
        logger.info(f"Fetching Instagram posts for queries: {queries}")
        all_results = []
        
        for query in queries:
            run_input = {
                "search": query.replace("#", ""),
                "searchType": "hashtag",
                "resultsType": "posts",
                "resultsLimit": max_items
            }
            try:
                run = self.client.actor(self.actor_instagram).call(run_input=run_input)
                dataset_items = self.client.dataset(run["defaultDatasetId"]).iterate_items()
                
                for item in dataset_items:
                    timestamp_str = item.get("timestamp")
                    try:
                        post_time = pd.to_datetime(timestamp_str, utc=True) if timestamp_str else None
                    except Exception:
                        post_time = None
                    
                    if post_time and post_time >= self.seven_days_ago:
                        processed_item = {
                            "platform": "instagram",
                            "video_id": item.get("id", ""),
                            "url": item.get("url", ""),
                            "description": item.get("caption", ""),
                            "likes_count": item.get("likesCount", 0),
                            "shares_count": 0,  # IG API generally doesn't expose shares natively
                            "timestamp": post_time,
                            "commentsCount": item.get("commentsCount", 0)
                        }
                        
                        # Some IG actors nest caption text
                        if isinstance(processed_item["description"], dict):
                            processed_item["description"] = processed_item["description"].get("text", "")
                            
                        all_results.append(processed_item)
            except ApifyApiError as e:
                logger.error(f"Apify API error (Instagram) for query '{query}': {e}")
                
        return pd.DataFrame(all_results)
        
    def run_ingestion_pipeline(self, queries: List[str], max_items: int = 5) -> pd.DataFrame:
        """
        Orchestrates the extraction from all configured platforms and merges the data.
        Returns the dataset mapped to the expected schema.
        """
        logger.info(f"Starting Multi-Platform Ingestion Pipeline for queries: {queries}")
        
        # 1. Fetch TikTok Videos
        tiktok_df = self.fetch_tiktok_videos(queries, max_items=max_items)
        
        # 2. Fetch TikTok Comments (if we found videos)
        tiktok_comments_df = pd.DataFrame()
        if not tiktok_df.empty:
            urls = tiktok_df["url"].dropna().tolist()
            # Fetch at most 50 comments per video for efficiency during testing
            tiktok_comments_df = self.fetch_tiktok_comments(urls[:5], max_comments_per_video=20)
            
            # Merge comments into a single string per video (basic aggregation for NLP)
            # Or keep them separate if NLP module expects one row per comment
            # The assignment asks for: 'video_id', 'description', 'comment_text', ...
            # Let's aggregate comments for simplicity, separated by ' | '
            if not tiktok_comments_df.empty:
                agg_comments = tiktok_comments_df.groupby("video_id")["comment_text"].apply(lambda x: " | ".join(x.astype(str))).reset_index()
                tiktok_df = pd.merge(tiktok_df, agg_comments, on="video_id", how="left")
                tiktok_df["comment_text"] = tiktok_df["comment_text"].fillna("")
            else:
                tiktok_df["comment_text"] = ""
        else:
            tiktok_df["comment_text"] = ""
            
        # 3. Fetch Instagram Posts
        ig_df = self.fetch_instagram_posts(queries, max_items=max_items)
        ig_df["comment_text"] = "" # Basic IG scrapers might not nest all comments, we leave it empty or extract top comments if available
        
        # 4. Combine and standardize
        combined_df = pd.concat([tiktok_df, ig_df], ignore_index=True)
        
        expected_columns = [
            "platform", "video_id", "url", "description", "comment_text", 
            "likes_count", "shares_count", "timestamp"
        ]
        
        if combined_df.empty:
            return pd.DataFrame(columns=expected_columns)
            
        # Ensure only common fields are exported
        for col in expected_columns:
            if col not in combined_df.columns:
                combined_df[col] = None
                
        cleaned_df = combined_df[expected_columns]
        logger.info(f"Ingestion Pipeline Complete. Total Rows: {len(cleaned_df)}")
        return cleaned_df

if __name__ == "__main__":
    import json
    try:
        ingestion = DataIngestion()
        test_df = ingestion.run_ingestion_pipeline(["#ViralProduct"], max_items=2)
        print(f"\nFinal DataFrame Shape: {test_df.shape}")
        if not test_df.empty:
            print(test_df.head(2).to_string())
    except Exception as err:
        print(f"Test failed: {err}")
