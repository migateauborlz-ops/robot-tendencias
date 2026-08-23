import json
import logging
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterable
from datetime import datetime, timedelta, timezone
import pandas as pd
from apify_client import ApifyClient
from apify_client._errors import ApifyApiError

from .config import Config

logger = logging.getLogger(__name__)

# Different TikTok comment actors expose the parent video under different keys.
# We probe them in order instead of assuming a single schema.
COMMENT_ID_KEYS = ("aweme_id", "awemeId", "video_id", "videoId", "postId", "post_id")
COMMENT_URL_KEYS = ("videoWebUrl", "submittedVideoUrl", "postUrl", "videoUrl",
                    "webVideoUrl", "url")
COMMENT_TEXT_KEYS = ("text", "comment", "commentText", "content")
COMMENT_LIKES_KEYS = ("digg_count", "diggCount", "likesCount", "likes")
COMMENT_REPLIES_KEYS = ("reply_comment_total", "replyCommentTotal", "repliesCount")

# TikTok canonical URLs embed the numeric video id: /@user/video/<id>
VIDEO_ID_IN_URL = re.compile(r"/video/(\d+)")
# Apify dataset URLs embed the dataset id: .../datasets/<id>/items
DATASET_ID_IN_URL = re.compile(r"/datasets?/([A-Za-z0-9]{6,})")


class DataIngestion:
    """
    Module 1: Data Ingestion
    Connects to the Apify API to fetch social media data based on specific queries.
    Supports TikTok Videos, TikTok Comments, and Instagram Scrapers.
    """

    def __init__(self, api_token: Optional[str] = None, debug_dump: bool = False):
        self.token = api_token or Config.APIFY_API_TOKEN
        if not self.token:
            logger.warning("API token is missing. Some methods may fail.")

        self.client = ApifyClient(self.token)

        # Load Actor IDs from Config
        self.actor_tiktok_videos = Config.APIFY_ACTOR_TIKTOK_VIDEOS
        self.actor_tiktok_comments = Config.APIFY_ACTOR_TIKTOK_COMMENTS
        self.actor_instagram = Config.APIFY_ACTOR_INSTAGRAM

        # Define the timeframe constraint. See Config.INGESTION_WINDOW_DAYS for why
        # this is 30 days and not 7.
        self.window_days = Config.INGESTION_WINDOW_DAYS
        self.window_start = datetime.now(timezone.utc) - timedelta(days=self.window_days)

        # Accumulated Apify spend for this instance, so a run can report its cost.
        self.total_cost_usd = 0.0

        # Actor failures recorded during this run. An empty result set means
        # something very different depending on whether this list is empty.
        self.errors: List[str] = []

        # Companion comment datasets reported by the video scraper.
        self.comment_dataset_ids: set = set()

        # When enabled, the raw payload of each actor is written to disk so the
        # real field names can be inspected without guessing.
        self.debug_dump = debug_dump
        self.dump_dir = Path("data/raw_dumps")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _first_present(item: Dict[str, Any], keys: Iterable[str], default: Any = "") -> Any:
        """Returns the first key present in `item` with a non-empty value."""
        for key in keys:
            value = item.get(key)
            if value not in (None, ""):
                return value
        return default

    @staticmethod
    def _video_id_from_url(url: Any) -> str:
        """Extracts the numeric TikTok video id embedded in a canonical URL."""
        match = VIDEO_ID_IN_URL.search(str(url or ""))
        return match.group(1) if match else ""

    @staticmethod
    def _dataset_id_from_url(url: Any) -> str:
        """Extracts the Apify dataset id from a dataset URL."""
        match = DATASET_ID_IN_URL.search(str(url or ""))
        return match.group(1) if match else ""

    def fetch_comments_from_datasets(self) -> pd.DataFrame:
        """
        Reads the companion comment datasets produced by the video scraper.

        Reading an existing dataset does not start an actor, so this step adds no
        extra cost beyond the comments already paid for during the video run.
        """
        if not self.comment_dataset_ids:
            logger.info("No companion comment datasets were reported by the scraper.")
            return pd.DataFrame()

        all_results: List[Dict[str, Any]] = []
        unresolved = 0

        for dataset_id in sorted(self.comment_dataset_ids):
            try:
                raw_items = list(self.client.dataset(dataset_id).iterate_items())
            except Exception as e:
                self.errors.append(f"Comments dataset {dataset_id}: {type(e).__name__}: {e}")
                logger.error("Could not read comments dataset %s: %s", dataset_id, e)
                continue

            self._dump(f"tiktok_comments_dataset_{dataset_id}", raw_items)
            logger.info("Comments dataset %s returned %d items.", dataset_id, len(raw_items))

            for item in raw_items:
                if "error" in item:
                    continue
                video_id = str(self._first_present(item, COMMENT_ID_KEYS, ""))
                if not video_id:
                    video_id = self._video_id_from_url(
                        self._first_present(item, COMMENT_URL_KEYS, ""))
                if not video_id:
                    unresolved += 1
                    continue

                all_results.append({
                    "video_id": video_id,
                    "comment_text": self._first_present(item, COMMENT_TEXT_KEYS, ""),
                    "comment_likes": self._first_present(item, COMMENT_LIKES_KEYS, 0),
                    "comment_replies": self._first_present(item, COMMENT_REPLIES_KEYS, 0),
                })

        if unresolved:
            logger.warning("%d comments could not be linked to a video. Inspect "
                           "data/raw_dumps to find the parent-video field.", unresolved)

        logger.info("Collected %d comments from %d companion dataset(s), covering "
                    "%d distinct videos.",
                    len(all_results), len(self.comment_dataset_ids),
                    len({r["video_id"] for r in all_results}) if all_results else 0)
        return pd.DataFrame(all_results)

    def _track_cost(self, run: Optional[Dict[str, Any]], actor: str) -> None:
        """Logs and accumulates the cost of a single actor run."""
        usage = float((run or {}).get("usageTotalUsd", 0.0) or 0.0)
        self.total_cost_usd += usage
        logger.info(
            "Apify Actor '%s' cost: $%.4f USD (accumulated this run: $%.4f USD)",
            actor, usage, self.total_cost_usd,
        )

    def _dump(self, name: str, items: List[Dict[str, Any]]) -> None:
        """Writes raw actor output to disk for schema inspection."""
        if not self.debug_dump:
            return
        try:
            self.dump_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = self.dump_dir / f"{name}_{stamp}.json"
            with path.open("w", encoding="utf-8") as handle:
                json.dump(items[:50], handle, ensure_ascii=False, indent=2, default=str)
            logger.info("Raw payload for '%s' written to %s (%d items)",
                        name, path, min(len(items), 50))
        except Exception as err:  # dumping must never break the pipeline
            logger.warning("Could not write raw dump for '%s': %s", name, err)

    # ------------------------------------------------------------------
    # Extractors
    # ------------------------------------------------------------------
    def fetch_tiktok_videos(self, queries: List[str],
                            max_items: int = Config.MAX_SCRAPE_ITEMS,
                            comments_per_post: int = 0) -> pd.DataFrame:
        """
        Fetches TikTok videos matching the queries.

        When `comments_per_post` is greater than zero the scraper also collects
        comments and publishes them to a companion dataset, whose URL it returns
        in the `commentsDatasetURL` field of every item. Requesting comments here
        is cheaper and more reliable than a second actor, because the comments are
        already associated with the post that produced them.
        """
        logger.info(f"Fetching TikTok videos for queries: {queries}")
        all_results = []

        for query in queries:
            run_input = {
                "hashtags": [query.replace("#", "")],
                "resultsPerPage": max_items,
                "shouldDownloadVideos": False,
            }
            if comments_per_post > 0:
                run_input["commentsPerPost"] = comments_per_post
            try:
                run = self.client.actor(self.actor_tiktok_videos).call(run_input=run_input)
                if not run or "defaultDatasetId" not in run:
                    logger.error("TikTok video actor returned no dataset for query '%s'. "
                                 "Response: %r", query, run)
                    continue
                self._track_cost(run, self.actor_tiktok_videos)

                raw_items = list(self.client.dataset(run["defaultDatasetId"]).iterate_items())
                self._dump(f"tiktok_videos_{query.replace('#', '')}", raw_items)

                kept = 0
                for item in raw_items:
                    # The scraper reports the companion comments dataset per item.
                    ds_url = self._first_present(
                        item, ("commentsDatasetURL", "commentsDatasetUrl"), "")
                    ds_id = self._dataset_id_from_url(ds_url)
                    if ds_id:
                        self.comment_dataset_ids.add(ds_id)

                    timestamp_str = item.get("createTimeISO")
                    try:
                        post_time = pd.to_datetime(timestamp_str, utc=True) if timestamp_str else None
                    except Exception:
                        post_time = None

                    if post_time and post_time >= self.window_start:
                        url = item.get("webVideoUrl", "")
                        # Prefer the actor's own id, fall back to the id in the URL.
                        video_id = str(item.get("id", "") or self._video_id_from_url(url))
                        processed_item = {
                            "platform": "tiktok",
                            "video_id": video_id,
                            "url": url,
                            "description": item.get("text", ""),
                            "likes_count": item.get("diggCount", 0),
                            "shares_count": item.get("shareCount", 0),
                            "timestamp": post_time,
                            "commentsCount": item.get("commentCount", 0),
                        }
                        all_results.append(processed_item)
                        kept += 1

                logger.info("Query '%s': %d items returned, %d within the %d-day window.",
                            query, len(raw_items), kept, self.window_days)
            except ApifyApiError as e:
                self.errors.append(f"TikTok videos ({query}): {e}")
                logger.error(f"Apify API error (TikTok Videos) for query '{query}': {e}")
            except Exception as e:
                self.errors.append(f"TikTok videos ({query}): {type(e).__name__}: {e}")
                logger.exception("Unexpected error fetching TikTok videos for '%s': %s",
                                 query, e)

        return pd.DataFrame(all_results)

    def fetch_tiktok_comments(self, video_urls: List[str],
                              max_comments_per_video: int = 20,
                              batch_size: int = 20) -> pd.DataFrame:
        """
        Fetches comments for specific TikTok video URLs.

        The parent video is resolved from whichever identifier the actor exposes,
        falling back to the numeric id embedded in the video URL. This avoids the
        silent join failure that occurs when the comment actor and the video actor
        use different field names for the same video.
        """
        if not video_urls:
            return pd.DataFrame()

        logger.info("Fetching TikTok comments for %d videos (batches of %d)...",
                    len(video_urls), batch_size)
        all_results: List[Dict[str, Any]] = []
        unresolved = 0

        for start in range(0, len(video_urls), batch_size):
            batch = video_urls[start:start + batch_size]
            run_input = {
                "postURLs": batch,
                "commentsPerPost": max_comments_per_video,
            }

            try:
                run = self.client.actor(self.actor_tiktok_comments).call(run_input=run_input)
                if not run or "defaultDatasetId" not in run:
                    logger.error("TikTok comment actor returned no dataset for batch "
                                 "starting at %d. Response: %r", start, run)
                    continue
                self._track_cost(run, self.actor_tiktok_comments)

                raw_items = list(self.client.dataset(run["defaultDatasetId"]).iterate_items())
                self._dump(f"tiktok_comments_batch{start}", raw_items)
                logger.info("Batch starting at %d returned %d raw comment items.",
                            start, len(raw_items))

                for item in raw_items:
                    if "error" in item:
                        continue  # Skip private/deleted videos

                    # 1) explicit id field, 2) id parsed from any URL field
                    video_id = str(self._first_present(item, COMMENT_ID_KEYS, ""))
                    if not video_id:
                        url_value = self._first_present(item, COMMENT_URL_KEYS, "")
                        video_id = self._video_id_from_url(url_value)
                    if not video_id:
                        unresolved += 1
                        continue

                    all_results.append({
                        "video_id": video_id,
                        "comment_text": self._first_present(item, COMMENT_TEXT_KEYS, ""),
                        "comment_likes": self._first_present(item, COMMENT_LIKES_KEYS, 0),
                        "comment_replies": self._first_present(item, COMMENT_REPLIES_KEYS, 0),
                    })
            except ApifyApiError as e:
                self.errors.append(f"TikTok comments: {e}")
                logger.error(f"Apify API error (TikTok Comments): {e}")
            except Exception as e:
                self.errors.append(f"TikTok comments: {type(e).__name__}: {e}")
                logger.exception("Unexpected error fetching TikTok comments: %s", e)

        if unresolved:
            logger.warning("%d comment items could not be linked to a video. "
                           "Run with debug_dump=True and inspect data/raw_dumps to see "
                           "which field carries the parent video id.", unresolved)

        logger.info("Collected %d comments linked to %d distinct videos.",
                    len(all_results),
                    len({r["video_id"] for r in all_results}) if all_results else 0)
        return pd.DataFrame(all_results)

    def fetch_instagram_posts(self, queries: List[str],
                              max_items: int = Config.MAX_SCRAPE_ITEMS) -> pd.DataFrame:
        """Fetches Instagram posts matching the queries."""
        logger.info(f"Fetching Instagram posts for queries: {queries}")
        all_results = []

        for query in queries:
            run_input = {
                "search": query.replace("#", ""),
                "searchType": "hashtag",
                "resultsType": "posts",
                "resultsLimit": max_items,
            }
            try:
                run = self.client.actor(self.actor_instagram).call(run_input=run_input)
                if not run or "defaultDatasetId" not in run:
                    logger.error("Instagram actor returned no dataset for query '%s'. "
                                 "Response: %r", query, run)
                    continue
                self._track_cost(run, self.actor_instagram)

                raw_items = list(self.client.dataset(run["defaultDatasetId"]).iterate_items())
                self._dump(f"instagram_{query.replace('#', '')}", raw_items)

                for item in raw_items:
                    timestamp_str = item.get("timestamp")
                    try:
                        post_time = pd.to_datetime(timestamp_str, utc=True) if timestamp_str else None
                    except Exception:
                        post_time = None

                    if post_time and post_time >= self.window_start:
                        processed_item = {
                            "platform": "instagram",
                            "video_id": str(item.get("id", "")),
                            "url": item.get("url", ""),
                            "description": item.get("caption", ""),
                            "likes_count": item.get("likesCount", 0),
                            "shares_count": 0,  # IG API generally doesn't expose shares natively
                            "timestamp": post_time,
                            "commentsCount": item.get("commentsCount", 0),
                        }

                        # Some IG actors nest caption text
                        if isinstance(processed_item["description"], dict):
                            processed_item["description"] = processed_item["description"].get("text", "")

                        all_results.append(processed_item)
            except ApifyApiError as e:
                self.errors.append(f"Instagram ({query}): {e}")
                logger.error(f"Apify API error (Instagram) for query '{query}': {e}")
            except Exception as e:
                self.errors.append(f"Instagram ({query}): {type(e).__name__}: {e}")
                logger.exception("Unexpected error fetching Instagram posts for '%s': %s",
                                 query, e)

        return pd.DataFrame(all_results)

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------
    def run_ingestion_pipeline(self, queries: List[str], max_items: int = 5,
                               max_comment_videos: int = 25,
                               comments_per_video: int = 20) -> pd.DataFrame:
        """
        Orchestrates the extraction from all configured platforms and merges the data.
        Returns the dataset mapped to the expected schema.
        """
        logger.info(f"Starting Multi-Platform Ingestion Pipeline for queries: {queries}")

        # 1. Fetch TikTok Videos, asking the same run for comments.
        tiktok_df = self.fetch_tiktok_videos(queries, max_items=max_items,
                                             comments_per_post=comments_per_video)

        # 2. Fetch TikTok Comments (if we found videos)
        if not tiktok_df.empty:
            tiktok_df["video_id"] = tiktok_df["video_id"].astype(str)
            urls = tiktok_df["url"].dropna().tolist()[:max_comment_videos]

            # Preferred path: comments produced by the video run itself.
            comments_df = self.fetch_comments_from_datasets()

            # Fallback: the standalone comments actor, only if the integrated
            # path produced nothing.
            if comments_df.empty and urls:
                logger.info("Falling back to the standalone comments actor.")
                comments_df = self.fetch_tiktok_comments(
                    urls, max_comments_per_video=comments_per_video)

            tiktok_df = self.attach_comments(tiktok_df, comments_df)
            self._report_comment_coverage(tiktok_df, requested=len(urls))
        else:
            tiktok_df = pd.DataFrame(columns=["platform", "video_id", "url", "description",
                                              "likes_count", "shares_count", "timestamp",
                                              "comment_text"])

        # 3. Fetch Instagram Posts
        ig_df = self.fetch_instagram_posts(queries, max_items=max_items)
        # Basic IG scrapers do not nest comments; the field is kept for schema parity.
        ig_df["comment_text"] = ""

        # 4. Combine and standardize
        combined_df = pd.concat([tiktok_df, ig_df], ignore_index=True)

        expected_columns = [
            "platform", "video_id", "url", "description", "comment_text",
            "likes_count", "shares_count", "timestamp",
        ]

        if combined_df.empty:
            logger.warning("Ingestion produced no rows for queries: %s", queries)
            return pd.DataFrame(columns=expected_columns)

        # Ensure only common fields are exported
        for col in expected_columns:
            if col not in combined_df.columns:
                combined_df[col] = None

        cleaned_df = combined_df[expected_columns]
        logger.info("Ingestion Pipeline Complete. Total Rows: %d | Apify cost: $%.4f USD",
                    len(cleaned_df), self.total_cost_usd)
        return cleaned_df

    @staticmethod
    def attach_comments(tiktok_df: pd.DataFrame, comments_df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregates comments per video and joins them onto the video DataFrame.

        Both sides are cast to string before the join: the video actor and the
        comment actor may return the same id as int and as str respectively, and
        pandas would silently match nothing.
        """
        if comments_df is None or comments_df.empty:
            tiktok_df = tiktok_df.copy()
            tiktok_df["comment_text"] = ""
            return tiktok_df

        tiktok_df = tiktok_df.copy()
        comments_df = comments_df.copy()
        tiktok_df["video_id"] = tiktok_df["video_id"].astype(str)
        comments_df["video_id"] = comments_df["video_id"].astype(str)

        agg_comments = (comments_df
                        .groupby("video_id")["comment_text"]
                        .apply(lambda x: " | ".join(s for s in x.astype(str) if s.strip()))
                        .reset_index())

        merged = pd.merge(tiktok_df, agg_comments, on="video_id", how="left")
        merged["comment_text"] = merged["comment_text"].fillna("")
        return merged

    @staticmethod
    def _report_comment_coverage(tiktok_df: pd.DataFrame, requested: int) -> None:
        """
        Reports how many videos actually carry comments.

        A silent zero here is the failure mode that invalidated the previous run:
        the purchase-intent component receives no input and scores 0.0 for every
        product, without any exception being raised.
        """
        total = len(tiktok_df)
        if total == 0:
            return
        filled = int((tiktok_df["comment_text"].astype(str).str.strip() != "").sum())
        pct = 100.0 * filled / total
        logger.info("Comment coverage: %d/%d TikTok videos carry comments (%.0f%%).",
                    filled, total, pct)

        if requested > 0 and filled == 0:
            logger.error(
                "COMMENT INGESTION FAILED: comments were requested for %d videos but "
                "none could be joined back. The purchase-intent component will score "
                "0.0 for every product. Re-run with debug_dump=True and inspect "
                "data/raw_dumps/ to identify the parent-video field name.", requested)
        elif requested > 0 and filled < requested / 2:
            logger.warning(
                "Comment coverage is below half of what was requested (%d of %d). "
                "Results for purchase intent will be weakly supported.", filled, requested)


if __name__ == "__main__":
    try:
        ingestion = DataIngestion(debug_dump=True)
        test_df = ingestion.run_ingestion_pipeline(["#ViralProduct"], max_items=2)
        print(f"\nFinal DataFrame Shape: {test_df.shape}")
        if not test_df.empty:
            print(test_df.head(2).to_string())
    except Exception as err:
        print(f"Test failed: {err}")
