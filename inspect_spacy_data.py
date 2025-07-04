import spacy
from spacy.tokens import DocBin

def inspect_spacy_data(path):
    print(f"Loading data from: {path}")
    doc_bin = DocBin().from_disk(path)
    docs = list(doc_bin.get_docs(spacy.blank("en").vocab))
    print(f"Number of docs in {path}: {len(docs)}")
    for i, doc in enumerate(docs):
        print(f"\nDoc {i} text:\n{doc.text}")
        print("Entities:")
        for ent in doc.ents:
            print(f" - Text: {ent.text}, Label: {ent.label_}")

if __name__ == "__main__":
    inspect_spacy_data("ner_data/train.spacy")
    inspect_spacy_data("ner_data/dev.spacy")