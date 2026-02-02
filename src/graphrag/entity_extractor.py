"""Entity extraction for GraphRAG (no LLM)."""

import re
import unicodedata
from typing import Any

# Arabic diacritics (harakat/tashkeel) Unicode ranges
ARABIC_DIACRITICS = [
    "\u064B", "\u064C", "\u064D", "\u064E", "\u064F",
    "\u0650", "\u0651", "\u0652", "\u0653", "\u0654",
    "\u0655", "\u0656", "\u0657", "\u0658", "\u0670"
]

# Load Arabic stopwords from library (fallback to basic set)
try:
    from arabicstopwords import stopwords_list
    ARABIC_STOPWORDS = set(stopwords_list())
except ImportError:
    # Fallback: Common Arabic stopwords (حروف وأدوات)
    ARABIC_STOPWORDS = {
        "في", "من", "إلى", "على", "عن", "مع", "بـ", "ل", "ك",
        "و", "أو", "ف", "ثم", "لكن", "أن", "إن",
        "ال", "هذا", "هذه", "ذلك", "تلك", "هؤلاء", "أولئك",
        "هو", "هي", "هم", "هن", "أنت", "أنتم", "أنا", "نحن",
        "كان", "يكون", "ليس", "قد", "لم", "لن", "لا",
        "ما", "التي", "الذي", "اللذان", "اللتان", "كل", "بعض",
        "كيف", "لماذا", "متى", "أين", "هل", "نعم", "غير", "سوى",
        "إلا", "حتى", "منذ", "بين", "عند", "قبل", "بعد"
    }


def strip_arabic_diacritics(text: str) -> str:
    """
    Remove Arabic diacritics (harakat) for entity normalization.
    This is ONLY for GraphRAG entity deduplication - original text keeps diacritics.
    """
    for diacritic in ARABIC_DIACRITICS:
        text = text.replace(diacritic, "")
    text = "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")
    return text


def extract_entities(text: str, lang: str = "en") -> list[str]:
    """
    Extract entities from text. Uses spaCy for English; fallback for Arabic.
    
    For Arabic: diacritics are stripped for entity normalization to avoid duplicates
    (e.g., كَتَبَ and كُتُب both become كتب for GraphRAG purposes).
    The original text with harakat is preserved in chunks.

    Args:
        text: Input text.
        lang: Language code ('en', 'ar', or 'unknown').

    Returns:
        List of entity strings (deduplicated, filtered, normalized).
    """
    if not text or not text.strip():
        return []

    if lang == "en":
        return _extract_entities_spacy(text)
    return _extract_entities_fallback(text)


def _extract_entities_spacy(text: str) -> list[str]:
    """Use spaCy for English NER and noun chunks."""
    try:
        import spacy
    except ImportError:
        return _extract_entities_fallback(text)

    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        return _extract_entities_fallback(text)

    doc = nlp(text[:100000])
    entities: set[str] = set()

    for ent in doc.ents:
        if ent.label_ in ("PERSON", "ORG", "GPE", "LOC", "PRODUCT", "EVENT", "WORK_OF_ART"):
            e = ent.text.strip()
            if len(e) >= 2 and len(e) <= 80:
                entities.add(e)

    for chunk in doc.noun_chunks:
        c = chunk.text.strip()
        if 3 <= len(c) <= 60 and c.lower() not in ("the", "a", "an", "this", "that"):
            entities.add(c)

    return list(entities)[:100]


def _extract_entities_fallback(text: str) -> list[str]:
    """Fallback: capitalized phrases, numbers, Arabic-like patterns."""
    from collections import Counter
    
    entities: set[str] = set()

    capitalized = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", text)
    for c in capitalized:
        if 2 <= len(c) <= 80:
            entities.add(c)

    arabic_words = re.findall(r"[\u0600-\u06FF]{2,}", text)
    if arabic_words:
        normalized = [strip_arabic_diacritics(a) for a in arabic_words]
        filtered = []
        for w in normalized:
            if 3 <= len(w) <= 60:
                if w not in ARABIC_STOPWORDS:
                    filtered.append(w)
        counter = Counter(filtered)
        top_entities = [w for w, count in counter.most_common(20) if count >= 2]
        entities.update(top_entities)

    return list(entities)[:30]


def detect_language(text: str) -> str:
    """Detect language for entity extraction."""
    try:
        import langdetect
        return langdetect.detect(text)
    except Exception:
        if re.search(r"[\u0600-\u06FF]", text):
            return "ar"
        return "en"
