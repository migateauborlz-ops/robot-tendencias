import re
import logging
from typing import List, Dict, Any, Tuple
import pandas as pd
import spacy
from transformers import pipeline

from .config import Config

logger = logging.getLogger(__name__)

# Purchase-intent cues. The Colombian corpus is bilingual, so the English-only
# list silently scored "cuanto vale? lo necesito ya" as zero intent.
INTENT_KEYWORDS = [
    # English
    "where", "buy", "link", "price", "need", "want", "how much", "cost",
    "order", "shipping", "in stock", "sold out", "add to cart", "checkout",
    # Espanol
    "donde", "dónde", "compro", "comprar", "compre", "compré", "vendes",
    "venden", "precio", "vale", "cuesta", "cuanto", "cuánto", "necesito",
    "quiero", "lo pido", "pedido", "envio", "envío", "envian", "envían",
    "contra entrega", "disponible", "hay", "pasame", "pásame", "el link",
    "informacion", "información", "info", "me interesa", "lo llevo",
]

# Explicit transactional questions: the strongest signal available.
EXPLICIT_PURCHASE_QUESTION = re.compile(
    r"\b(?:"
    r"cu[aá]nto\s+(?:vale|cuesta|sale|es)|qu[eé]\s+precio|a\s+c[oó]mo|"
    r"d[oó]nde\s+(?:lo|la|los|las)?\s*(?:compro|consigo|venden|puedo\s+comprar)|"
    r"c[oó]mo\s+(?:lo|la)\s+(?:compro|pido|consigo)|"
    r"pas(?:a|á)me\s+el\s+link|link\s+por\s+fa|"
    r"how\s+much|where\s+(?:can|do)\s+i\s+(?:buy|get)|"
    r"drop\s+the\s+link|send\s+the\s+link|link\s+please"
    r")\b", re.IGNORECASE)

# Weaker cues: stating a need, a wish, or asking about availability/shipping.
INTENT_CUE_PATTERN = re.compile(
    r"\b(?:"
    r"necesito|lo\s+quiero|la\s+quiero|me\s+interesa|lo\s+llevo|lo\s+pido|"
    r"comprar|compro|pedido|precio|env[ií]o|env[ií]os|contra\s+entrega|"
    r"disponible|vendes|venden|"
    r"need\s+(?:it|this)|want\s+(?:it|this)|buy(?:ing)?|link|price|"
    r"order(?:ed|ing)?|shipping|in\s+stock|sold\s+out|add\s+to\s+cart|checkout"
    r")\b", re.IGNORECASE)

# A comment is filed as high intent above this score.
INTENT_THRESHOLD = 0.6

# Sentiment models that emit generic LABEL_0/LABEL_1 have no trained
# classification head: their scores are noise. See _validate_sentiment_model.
GENERIC_LABEL_PREFIX = "LABEL_"

# spaCy pipeline per language. Only the small models are listed: they are the
# ones the project installs, and accuracy matters less here than applying the
# correct grammar.
SPACY_MODEL_BY_LANGUAGE = {
    "es": "es_core_news_sm",
    "en": "en_core_web_sm",
}

# Nouns that are grammatically valid heads but never name a physical product.
# Without this filter the extractor returns discourse fragments such as
# "way lol", "ur idea" or "shipping costs", which are noun phrases but not goods.
ABSTRACT_HEAD_NOUNS = {
    # English
    "way", "ways", "thing", "things", "time", "times", "day", "days", "year",
    "years", "idea", "ideas", "tip", "tips", "cost", "costs", "price", "prices",
    "deal", "deals", "discount", "discounts", "quality", "life", "love",
    "people", "man", "woman", "guy", "guys", "lol", "lmao", "video", "videos",
    "comment", "comments", "reason", "reasons", "question", "questions",
    "answer", "answers", "problem", "problems", "fact", "facts", "part", "parts",
    "kind", "kinds", "type", "types", "lot", "lots", "bit", "one", "ones",
    "thanks", "thank", "everyone", "everybody", "someone", "somebody", "nobody",
    "today", "tomorrow", "yesterday", "week", "weeks", "month", "months",
    "sale", "sales", "shipping", "delivery", "order", "orders", "link", "links",
    "business", "work", "money", "help", "name", "names", "place", "places",
    "world", "stuff", "content", "account", "page", "story", "stories", "post",
    "posts", "fyp", "tutorial", "review", "reviews", "hack", "hacks", "test",
    # Espanol
    "cosa", "cosas", "forma", "formas", "manera", "maneras", "tiempo", "dia",
    "dias", "día", "días", "gente", "persona", "personas", "hombre", "mujer",
    "precio", "precios", "costo", "costos", "oferta", "ofertas", "descuento",
    "descuentos", "calidad", "vida", "amor", "razon", "razón", "pregunta",
    "respuesta", "problema", "hecho", "tipo", "tipos", "semana", "mes", "meses",
    "envio", "envios", "envío", "envíos", "entrega", "pedido", "pedidos",
    "plata", "dinero", "negocio", "trabajo", "nombre", "lugar", "mundo",
    "comentario", "comentarios", "publicacion", "publicación", "cuenta",
    "pagina", "página", "medida", "medidas", "consejo", "consejos", "resena",
    "reseña", "prueba", "gracias",
}

# Parts of speech that disqualify a candidate: a product name does not contain
# verbs, pronouns, adverbs or interjections.
FORBIDDEN_POS = {"VERB", "AUX", "PRON", "ADV", "INTJ", "SCONJ", "CCONJ", "PART"}

# Characters outside the Latin scripts. The corpus mixes languages and the
# extractor was returning strings like "реально работает" and "surfaces о",
# where the second word carries a Cyrillic character.
NON_LATIN = re.compile(r"[^\x00-\x7FÀ-ɏ\s]")

# Long digit runs are phone numbers, account ids or document numbers. The
# extractor produced "catalogo whatsap 318XXXXXXX" from a spam comment, which is
# personal data under Ley 1581 de 2012 and must never reach the database: the
# project commits to storing products, not identifiers.
SECUENCIA_IDENTIFICADORA = re.compile(r"\d{7,}")

class NLPLayer:
    """
    Module 2: The NLP & Intelligence Layer
    Handles text cleaning, Named Entity Recognition (NER), and Sentiment/Intent analysis.
    """
    def __init__(self, model_name: str = None, sentiment_model: str = None):
        """
        Initializes the NLP Layer.
        Loads spaCy for NER and a Hugging Face pipeline for Sentiment/Intent Analysis.
        """
        model_name = model_name or Config.SPACY_MODEL
        sentiment_model = sentiment_model or Config.SENTIMENT_MODEL

        logger.info(f"Loading spaCy model: {model_name}")
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            logger.error(f"spaCy model '{model_name}' not found. Please run: python -m spacy download {model_name}")
            raise

        # Grammar rules are language specific. Applying the Spanish model to
        # English text (or the reverse) makes the part-of-speech tags meaningless
        # and the extractor returns sentence fragments instead of products. The
        # corpus is mixed -- of 33 posts measured on 2026-08-20, 24 were English,
        # 2 Spanish and 7 undetermined -- so each text is routed to its own model.
        self._models = {Config.SPACY_MODEL.split("_")[0]: self.nlp}
        self._missing_models = set()
        self.language_counts = {}

        # The sentiment model must be one that was fine-tuned for classification.
        # A base language model loads without error but emits generic LABEL_0
        # labels with near-constant scores, which silently turns the purchase
        # intent component into noise. _validate_sentiment_model checks for that.
        logger.info(f"Loading Hugging Face sentiment model: {sentiment_model}")
        try:
            self.sentiment_analyzer = pipeline("sentiment-analysis", model=sentiment_model)
        except Exception as e:
            logger.warning("Could not load '%s' (%s). Falling back to the default "
                           "sentiment pipeline.", sentiment_model, e)
            self.sentiment_analyzer = pipeline("sentiment-analysis")

        self.sentiment_model_name = sentiment_model
        self._validate_sentiment_model()

    def _validate_sentiment_model(self) -> None:
        """
        Verifies that the loaded model actually discriminates sentiment.

        Probes the classifier with an unambiguously positive and an unambiguously
        negative sentence. A model without a trained classification head returns
        the same generic label and almost the same score for both, which is what
        happened with distilbert-base-multilingual-cased: "donde lo compro" scored
        0.5505 and "que porqueria" scored 0.5507, a difference of 0.0002.
        """
        try:
            positivo = self.sentiment_analyzer("Me encanta, es excelente, lo recomiendo")[0]
            negativo = self.sentiment_analyzer("Es terrible, una porqueria, no sirve")[0]
        except Exception as e:
            logger.warning("Could not validate the sentiment model: %s", e)
            return

        etiquetas = {str(positivo.get("label", "")), str(negativo.get("label", ""))}
        separacion = abs(float(positivo.get("score", 0)) - float(negativo.get("score", 0)))

        if any(l.upper().startswith(GENERIC_LABEL_PREFIX) for l in etiquetas):
            logger.error(
                "SENTIMENT MODEL NOT USABLE: '%s' returns generic labels %s. It has "
                "no trained classification head, so purchase-intent scores would be "
                "noise. Set SENTIMENT_MODEL in .env to a fine-tuned checkpoint such "
                "as nlptown/bert-base-multilingual-uncased-sentiment.",
                self.sentiment_model_name, sorted(etiquetas))
        elif len(etiquetas) == 1 and separacion < 0.05:
            logger.error(
                "SENTIMENT MODEL NOT USABLE: '%s' gives the same label and nearly "
                "the same score (difference %.4f) to opposite sentences.",
                self.sentiment_model_name, separacion)
        else:
            logger.info("Sentiment model check passed: labels %s, separation %.3f.",
                        sorted(etiquetas), separacion)

    # ------------------------------------------------------------------
    # Language routing
    # ------------------------------------------------------------------
    @staticmethod
    def detect_language(text: str) -> str:
        """
        Returns the ISO code of the language of `text`, or "" when undecidable.

        Very short comments carry too little signal, so anything under 20
        characters is left undetermined rather than guessed.
        """
        clean = (text or "").strip()
        if len(clean) < 20:
            return ""
        try:
            from langdetect import detect, DetectorFactory
            DetectorFactory.seed = 0   # deterministic output across runs
            return detect(clean)
        except Exception:
            return ""

    def is_plausible_product(self, candidate: str, language: str = "") -> bool:
        """
        Decides whether a candidate string can name a physical product.

        The n-gram extractor happily returns any run of nouns, so grammatically
        valid but commercially meaningless phrases get through. This layer keeps
        only candidates whose head is a concrete noun and that contain no verbs,
        pronouns or foreign script.
        """
        text = (candidate or "").strip()
        if not text:
            return False

        # Mixed-script strings come from cross-language noise, not product names.
        if NON_LATIN.search(text):
            return False

        # Data minimisation: never let a phone number or document id through.
        if SECUENCIA_IDENTIFICADORA.search(text):
            return False

        # Require real words, not punctuation or digit soup.
        letters = sum(ch.isalpha() for ch in text)
        if letters < 4:
            return False

        doc = self._model_for(language)(text)
        if len(doc) == 0:
            return False

        if any(t.pos_ in FORBIDDEN_POS for t in doc):
            return False

        # The head is the last nominal token in English compounds and usually the
        # first one in Spanish, so scan for any nominal and prefer the last.
        head = None
        for token in doc:
            if token.pos_ in ("NOUN", "PROPN"):
                head = token
        if head is None:
            return False

        if head.like_num:
            return False

        for form in (head.lemma_, head.text):
            if (form or "").lower() in ABSTRACT_HEAD_NOUNS:
                return False

        return True

    def _model_for(self, language: str):
        """
        Returns the spaCy pipeline for `language`, falling back to the default.

        Models are loaded on first use and cached; a language with no installed
        model is reported once and then handled by the default pipeline.
        """
        lang = (language or "").lower()[:2]
        if not lang or lang not in SPACY_MODEL_BY_LANGUAGE:
            return self.nlp
        if lang in self._models:
            return self._models[lang]
        if lang in self._missing_models:
            return self.nlp

        model_name = SPACY_MODEL_BY_LANGUAGE[lang]
        try:
            self._models[lang] = spacy.load(model_name)
            logger.info("Loaded spaCy model '%s' for language '%s'.", model_name, lang)
            return self._models[lang]
        except OSError:
            self._missing_models.add(lang)
            logger.warning("spaCy model '%s' is not installed; text in '%s' will be "
                           "processed with the default pipeline and its entities will "
                           "be unreliable. Install it with: python -m spacy download %s",
                           model_name, lang, model_name)
            return self.nlp

    @staticmethod
    def clean_text(text: str) -> str:
        """
        Removes emojis, URLs, and extra whitespaces.
        """
        if not isinstance(text, str):
            return ""
            
        # Remove URLs
        text = re.sub(r'http\S+|www.\S+', '', text)
        
        # Remove hashtags and mentions completely
        text = re.sub(r'[#@]\S+', '', text)
        text = re.sub(r'#.*?(?=\s|$)', '', text)
        
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

    def extract_product_entities(self, text: str, language: str = "") -> List[str]:
        """
        Named Entity Recognition (NER) to extract physical product candidates.
        Features Brand & Model extraction, N-Gram compound noun support, and Regex pattern matching.
        Strict validation ensures rigorous POS checks and limits noise.
        """
        doc = self._model_for(language)(text)
        raw_products = set()
        
        # 4. Brand Filtering Priority
        known_brands = {"dyson", "sony", "apple", "nike", "adidas", "samsung", "lg", "bose", "nintendo", "stanley", "ninja"}
        excluded_words = {
            "video", "comment", "tiktok", "instagram", "post", "link", "price", "small business", "business", 
            "everyone", "people", "love", "share", "follow", "cost", "bio", "like", 
            "lo", "que", "el", "la", "los", "las", "un", "una", "unos", "unas", "uno", 
            "part", "best", "good", "great", "awesome", "amazing", "yall", "you", "me", "my", "por", "para", "con", "sin",
            "jajaja", "plata", "literal", "oigan", "entrega", "envíos", "envios"
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
                
            # Max Word Constraint: A valid product entity should be between 1 and 4 words
            if len(p.split()) > 4:
                continue
                
            # Smart Blacklist: check with word boundaries
            contains_excluded = False
            for w in excluded_words:
                if re.search(r'\b' + re.escape(w) + r'\b', p):
                    contains_excluded = True
                    break
            if contains_excluded:
                continue
                
            # Re-parse the isolated entity to evaluate strict POS rules,
            # using the pipeline of the detected language.
            p_doc = self._model_for(language)(p)
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
                        
            # Semantic gate: the candidate must be able to name a physical
                        
            # product, not merely be a grammatical noun phrase.
                        
            if not self.is_plausible_product(p, language):
                        
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

        for comment in comments:
            try:
                # Purchase intent must be EXPRESSED, not inferred from mood. Using
                # sentiment as a proxy scored "this is so funny hahaha" at 0.736,
                # a comment with no commercial value whatsoever. The lexical cue is
                # therefore the gate and sentiment only modulates its strength.
                cue = self._intent_cue_strength(comment)
                if cue <= 0.0:
                    continue

                score = cue * (0.6 + 0.4 * self._sentiment_factor(comment))

                if score >= INTENT_THRESHOLD:
                    high_intent_comments.append(comment)

                # The average runs over EVERY comment, not only the ones that pass.
                # Averaging just the winners made a post with 1 of 50 interested
                # comments score the same as one with 50 of 50, which is precisely
                # the distinction the Opportunity Score needs to make.
                total_score += score

            except Exception as e:
                logger.debug(f"Error analyzing intent for comment: {e}")

        avg_score = total_score / len(comments) if comments else 0.0
        return round(avg_score, 3), high_intent_comments

    @staticmethod
    def _intent_cue_strength(comment: str) -> float:
        """
        Strength of the purchase-intent cue expressed in the comment.

        Returns 1.0 for an explicit transactional question ("cuanto vale",
        "where can I buy it"), 0.7 for a weaker cue such as stating a need, and
        0.0 when nothing commercial is expressed. Matching uses word boundaries:
        plain substring search made "need" fire inside "needle" and "hay" fire
        inside "hayan".
        """
        text = (comment or "").lower()
        if EXPLICIT_PURCHASE_QUESTION.search(text):
            return 1.0
        if INTENT_CUE_PATTERN.search(text):
            return 0.7
        return 0.0

    def _sentiment_factor(self, comment: str) -> float:
        """Maps the sentiment classifier onto 0.0 negative, 0.5 neutral, 1.0 positive."""
        try:
            result = self.sentiment_analyzer(comment[:512])[0]
        except Exception as e:
            logger.debug(f"Error analyzing sentiment: {e}")
            return 0.5

        label = str(result.get("label", "")).upper()
        if "POS" in label or "4" in label or "5" in label:
            return 1.0
        if "NEG" in label or "1" in label or "2" in label:
            return 0.0
        return 0.5

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
                    
            # 2. Extract Entities using the pipeline of the detected language
            combined_text = f"{desc}. " + " ".join(cleaned_comments)
            language = self.detect_language(combined_text)
            self.language_counts[language or "sin_dato"] = (
                self.language_counts.get(language or "sin_dato", 0) + 1)
            products = self.extract_product_entities(combined_text, language)
            
            # 3. Analyze Sentiment & Intent on Comments
            intent_score, highlight_comments = self.analyze_purchase_intent(cleaned_comments)
            
            # Pack results
            processed_row = row.to_dict()
            processed_row["clean_description"] = desc
            processed_row["extracted_products"] = products
            processed_row["purchase_intent_score"] = intent_score
            processed_row["high_intent_comments_count"] = len(highlight_comments)
            
            results.append(processed_row)
            
        logger.info("Language distribution of the corpus: %s",
                    dict(sorted(self.language_counts.items(),
                                key=lambda kv: -kv[1])))
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
