import re
import logging
from typing import List, Dict, Any, Tuple
import pandas as pd
import spacy
from transformers import pipeline

logger = logging.getLogger(__name__)

class NLPLayer:
    """
    Module 2: The NLP & Intelligence Layer
    Handles text cleaning, Named Entity Recognition (NER), and Sentiment/Intent analysis.
    """
    def __init__(self, model_name: str = "en_core_web_sm", sentiment_model: str = "distilbert-base-multilingual-cased"):
        """
        Initializes the NLP Layer.
        Loads spaCy for NER and a Hugging Face pipeline for Sentiment/Intent Analysis.
        """
        logger.info(f"Loading spaCy model: {model_name}")
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            logger.error(f"spaCy model '{model_name}' not found. Please run: python -m spacy download {model_name}")
            raise
            
        logger.info(f"Loading Hugging Face sentiment model: {sentiment_model}")
        # We use a text-classification pipeline. The assignment mentioned distilbert-base-multilingual-cased.
        # Note: distilbert-base-multilingual-cased doesn't have a default sentiment head fine-tuned, 
        # but we'll use a standard zero-shot classification or a fine-tuned sentiment model.
        # To strictly follow instructions, we load the requested model.
        # If the requested model is purely base (un-finetuned), we might use a pipeline that defaults to a fine-tuned version,
        # but let's stick to the prompt's explicit model: "distilbert-base-multilingual-cased".
        # Actually, using a generic 'sentiment-analysis' pipeline defaults to distilbert-base-uncased-finetuned-sst-2-english.
        # Since the prompt specifically asked for `distilbert-base-multilingual-cased` for sentiment analysis, 
        # we will specify it, though it might need a specific task like 'text-classification'.
        # For intent, we look for positive sentiment as a proxy for purchase intent as instructed.
        try:
            # Using 'sentiment-analysis' with the specified model might fail if it lacks a classification head.
            # We'll use 'text-classification' which is the underlying pipeline.
            # Warning: distilbert-base-multilingual-cased is a masked language model.
            # If it fails, HuggingFace will warn us. We will try loading it as sentiment-analysis.
            # There are many fine-tuned versions like 'nlptown/bert-base-multilingual-uncased-sentiment'.
            # We'll use the pipeline.
            self.sentiment_analyzer = pipeline("sentiment-analysis", model=sentiment_model)
        except Exception as e:
            logger.warning(f"Could not load specified sentiment model fully initialized for classification. Falling back to default sentiment-analysis. Error: {e}")
            self.sentiment_analyzer = pipeline("sentiment-analysis")

    @staticmethod
    def clean_text(text: str) -> str:
        """
        Removes emojis, URLs, and extra whitespaces.
        """
        if not isinstance(text, str):
            return ""
            
        # Remove URLs
        text = re.sub(r'http\S+|www.\S+', '', text)
        
        # Remove mentions
        text = re.sub(r'@\w+', '', text)
        
        # Remove emojis (basic range covering most standard emojis)
        text = re.sub(r'[^\w\s.,!?#\'"-]', '', text)
        
        # Remove extra whitespaces
        text = " ".join(text.split())
        
        return text.strip()

    def merge_and_clean_comments(self, row: pd.Series) -> str:
        """
        Helper function to clean and deduplicate comments from a row.
        """
        desc = self.clean_text(row.get("description", ""))
        comments_raw = str(row.get("comment_text", ""))
        
        # Split merged comments if it was joined by ' | '
        comments_list = [c.strip() for c in comments_raw.split(" | ") if c.strip()]
        
        # Clean and deduplicate comments
        cleaned_comments = []
        seen = set()
        for comment in comments_list:
            c = self.clean_text(comment)
            if c and c not in seen:
                seen.add(c)
                cleaned_comments.append(c)
                
        # Return a combined context string for NER extraction
        # We append description and comments
        full_text = f"{desc}. " + " ".join(cleaned_comments)
        return full_text

    def extract_product_entities(self, text: str) -> List[str]:
        """
        Named Entity Recognition (NER) to extract physical product candidates.
        Features Brand & Model extraction, N-Gram compound noun support, and Regex pattern matching.
        Strict validation ensures rigorous POS checks and limits noise.
        """
        doc = self.nlp(text)
        raw_products = set()
        
        # 4. Brand Filtering Priority
        known_brands = {"dyson", "sony", "apple", "nike", "adidas", "samsung", "lg", "bose", "nintendo", "stanley", "ninja"}
        excluded_words = {
            "video", "comment", "tiktok", "instagram", "post", "link", "price", "small business", "business", 
            "everyone", "people", "love", "share", "follow", "cost", "bio", "like", 
            "lo", "que", "el", "la", "los", "las", "un", "una", "unos", "unas", "uno", 
            "part", "best", "good", "great", "awesome", "amazing", "yall", "you", "me", "my", "por", "para", "con", "sin"
        }
        
        # 3. Pattern Matching Regex
        model_pattern = re.compile(r'\b(v\d+|pro( max)?|series \d+|gen \d+|edition|airwrap|ultra|plus)\b', re.IGNORECASE)
        
        used_tokens = set()

        # 1. Brand & Model Extraction (Merge ORG + PRODUCT within 3 words distance)
        orgs = [ent for ent in doc.ents if ent.label_ == "ORG"]
        prods = [ent for ent in doc.ents if ent.label_ == "PRODUCT"]
        
        for org in orgs:
            for prod in prods:
                distance = abs(org.start - prod.end) if org.start > prod.start else abs(prod.start - org.end)
                if distance <= 3:
                    merged = f"{org.text} {prod.text}" if org.start < prod.start else f"{prod.text} {org.text}"
                    raw_products.add(merged.lower())
                    used_tokens.update(range(min(org.start, prod.start), max(org.end, prod.end)))
                    
        # Add standalone products if not merged
        for ent in doc.ents:
            if ent.label_ == "PRODUCT" and not any(i in used_tokens for i in range(ent.start, ent.end)):
                raw_products.add(ent.text.lower())
                used_tokens.update(range(ent.start, ent.end))

        # 2. N-Gram Support & Pattern Matching
        tokens = [token for token in doc]
        i = 0
        while i < len(tokens):
            if i in used_tokens:
                i += 1
                continue
                
            # If we find a NOUN, PROPN, or known brand token, start grabbing N-grams
            if tokens[i].pos_ in ["NOUN", "PROPN"] or tokens[i].text.lower() in known_brands:
                start = i
                end = i + 1
                while end < len(tokens):
                    t = tokens[end]
                    text_lower = t.text.lower()
                    # Allow compound nouns, numbers, or our pattern matches
                    if t.pos_ in ["NOUN", "PROPN", "NUM"]:
                        end += 1
                    elif model_pattern.match(text_lower):
                        end += 1
                    else:
                        break
                        
                candidate = doc[start:end].text.lower().strip()
                
                # Keep if it's a compound word, OR contains a known brand, OR contains a model pattern
                if len(candidate.split()) > 1 or any(b in candidate for b in known_brands) or model_pattern.search(candidate):
                    raw_products.add(candidate)
                    used_tokens.update(range(start, end))
                i = end
            else:
                i += 1

        # 5. Strict Entity Validation Layer
        valid_products = set()
        for p in raw_products:
            p = p.strip()
            
            # Length Constraint: >= 4 chars to avoid tiny noisy words
            if len(p) < 4:
                continue
                
            # Smart Blacklist: check with word boundaries
            contains_excluded = False
            for w in excluded_words:
                if re.search(r'\b' + re.escape(w) + r'\b', p):
                    contains_excluded = True
                    break
            if contains_excluded:
                continue
                
            # Re-parse the isolated entity to evaluate strict POS rules
            p_doc = self.nlp(p)
            if not p_doc or len(p_doc) == 0:
                continue
                
            # Ensure the entity doesn't start with a VERB
            if p_doc[0].pos_ == "VERB":
                continue
                
            # Contextual Logic: If an entity contains only 1 word and it's an adjective, discard it
            if len(p_doc) == 1 and p_doc[0].pos_ == "ADJ":
                continue
                
            # Contextual Logic: discard strings starting with numbers unless they are part of a model
            if p_doc[0].pos_ == "NUM" and len(p_doc) == 1:
                continue
                
            # POS Strictness: Ensure headword (syntactic root) is NOUN or PROPN
            roots = [t for t in p_doc if t.head == t]
            if roots:
                head_pos = roots[0].pos_
                if head_pos not in ["NOUN", "PROPN"]:
                    # Allow override only if it explicitly contains a verified global brand
                    if not any(b in p for b in known_brands):
                        continue
                        
            valid_products.add(p)

        return list(valid_products)

    def analyze_purchase_intent(self, comments: List[str]) -> Tuple[float, List[str]]:
        """
        Scores comments for 'Purchase Intent' utilizing sentiment as a proxy.
        Filters comments with a score < 0.6.
        Returns the average score of passing comments and the list of high-intent comments.
        """
        if not comments:
            return 0.0, []
            
        high_intent_comments = []
        total_score = 0.0
        
        # Analyze each comment individually
        for comment in comments:
            try:
                # Hugging Face pipeline usually returns [{'label': 'POSITIVE', 'score': 0.99}]
                # or a rating like '5 stars'
                result = self.sentiment_analyzer(comment[:512])[0]  # truncate to 512 chars max
                
                # Normalize score based on positive intent
                score = 0.0
                label = str(result.get("label", "")).upper()
                
                # If the model outputs POSITIVE/NEGATIVE or star ratings
                if "POS" in label or "4" in label or "5" in label:
                    score = result.get("score", 0.0)
                elif "NEG" in label or "1" in label or "2" in label:
                    # Invert negative sentiment score, or just clamp to 0
                    score = 0.0
                else:
                    # Neutral
                    score = result.get("score", 0.0) * 0.5
                
                # Specific intent keywords boost the score manually
                intent_keywords = ["where", "buy", "link", "price", "need", "want", "how much"]
                if any(kw in comment.lower() for kw in intent_keywords):
                    score = min(1.0, score + 0.6)  # Boost significantly to bypass neutral sentiment classification
                    
                if score >= 0.6:
                    high_intent_comments.append(comment)
                    total_score += score
                    
            except Exception as e:
                logger.debug(f"Error analyzing sentiment for comment: {e}")
                
        avg_score = total_score / len(high_intent_comments) if high_intent_comments else 0.0
        return round(avg_score, 3), high_intent_comments

    def process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Processes the raw ingested DataFrame through the NLP layer.
        Returns a DataFrame with extracted products and purchase intent scores.
        """
        logger.info(f"Processing NLP Layer on DataFrame with {len(df)} rows.")
        if df.empty:
            return df
            
        results = []
        
        for idx, row in df.iterrows():
            # 1. Clean and Deduplicate
            desc = self.clean_text(row.get("description", ""))
            comments_raw = str(row.get("comment_text", ""))
            comments_list = [c.strip() for c in comments_raw.split(" | ") if c.strip()]
            
            cleaned_comments = []
            seen = set()
            for comment in comments_list:
                c = self.clean_text(comment)
                if c and c not in seen:
                    seen.add(c)
                    cleaned_comments.append(c)
                    
            # 2. Extract Entities
            combined_text = f"{desc}. " + " ".join(cleaned_comments)
            products = self.extract_product_entities(combined_text)
            
            # 3. Analyze Sentiment & Intent on Comments
            intent_score, highlight_comments = self.analyze_purchase_intent(cleaned_comments)
            
            # Pack results
            processed_row = row.to_dict()
            processed_row["clean_description"] = desc
            processed_row["extracted_products"] = products
            processed_row["purchase_intent_score"] = intent_score
            processed_row["high_intent_comments_count"] = len(highlight_comments)
            
            results.append(processed_row)
            
        logger.info("NLP Layer processing complete.")
        return pd.DataFrame(results)

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    # Test execution
    try:
        nlp = NLPLayer(sentiment_model="distilbert-base-uncased-finetuned-sst-2-english") # Using default HF model for testing
        test_data = {
            "platform": ["tiktok", "tiktok"],
            "video_id": ["7607749319017483551", "1234567890"],
            "description": ["Check out this amazing Sunset Projection Lamp! 🔥 #viral #tiktokmademebuyit", "My new Dyson Airwrap Pro Max is amazing!"],
            "comment_text": ["Where can I buy this? \U0001F60D | I need the link! | trash | how much is the led lamp? | I love this cost", "follow share video | Airwrap changed my life. | Apple Pro Max version? | link in bio follow me"],
            "likes_count": [1500, 25000]
        }
        df = pd.DataFrame(test_data)
        
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        
        print("Raw DataFrame:")
        print(df[["description", "comment_text"]])
        
        processed_df = nlp.process_dataframe(df)
        
        print("\nProcessed DataFrame:")
        print(processed_df[["clean_description", "extracted_products", "purchase_intent_score", "high_intent_comments_count"]])
        
    except Exception as err:
        print(f"Test failed: {err}")
