import spacy
from spacy.tokens import Span
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from typing import Dict, Any
from tqdm import tqdm

# Register confidence extension if not already present
if not Span.has_extension("confidence"):
    Span.set_extension("confidence", default=None)

# Load transformer model
nlp = spacy.load("en_core_web_trf")
analyzer = SentimentIntensityAnalyzer()

# Performs NLP analysis on the given text
def analyze_text(text: str) -> Dict[str, Any]:
    doc = nlp(text)

    # Sentiment
    sentiment = analyzer.polarity_scores(text)
    sentiment_confidence = round(abs(sentiment["compound"]), 2)

    # Add progress bar while processing entities
    print("\n[+] Processing Named Entities...")

    # Named Entities
    entities = []
    for ent in tqdm(doc.ents, desc="NER Analysis"):
        confidence = ent._.confidence if ent._.confidence is not None else 0.0
        entities.append({
            "text": ent.text,
            "label": ent.label_,
            "confidence": round(confidence, 2)
        })

    return {
        "sentiment": sentiment,
        "sentiment_confidence": sentiment_confidence,
        "entities": entities
    }