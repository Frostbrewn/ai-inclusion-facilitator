import spacy
from spacy.tokens import DocBin
import json

def json_to_spacy(json_path, spacy_path):
    nlp = spacy.blank("en")
    if "ner" not in nlp.pipe_names:
        nlp.add_pipe("ner")  # ← IMPORTANT!
    
    db = DocBin()
    with open(json_path, "r", encoding="utf-8") as f:
        training_data = json.load(f)

    for record in training_data:
        text = record["text"]
        entities = record["entities"]
        doc = nlp.make_doc(text)
        ents = []
        for ent in entities:
            start = ent["start"]
            end = ent["end"]
            label = ent["label"]
            span = doc.char_span(start, end, label=label)
            if span is None:
                print(f"⚠️ Skipping invalid span: {text[start:end]} [{label}]")
            else:
                ents.append(span)
        doc.ents = ents
        db.add(doc)

    db.to_disk(spacy_path)
    print(f"✅ Saved {len(db)} docs to {spacy_path}")

# Call the function
json_to_spacy("ner_data/train.json", "ner_data/train.spacy")
json_to_spacy("ner_data/dev.json", "ner_data/dev.spacy")