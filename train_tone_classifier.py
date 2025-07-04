import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

# Example training data
texts = [
    "I am very happy today!",
    "This is so sad and depressing.",
    "I feel angry about this.",
    "What a wonderful experience!",
    "I'm frustrated with the service.",
    "It was a delightful day.",
]
labels = ["positive", "negative", "negative", "positive", "negative", "positive"]

# Build a pipeline: vectorizer + classifier
model = make_pipeline(TfidfVectorizer(), LogisticRegression())

# Train the model
model.fit(texts, labels)

# Save the model to tone_model.pkl
with open("model/tone_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved to model/tone_model.pkl")