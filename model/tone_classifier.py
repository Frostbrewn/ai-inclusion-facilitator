import pickle

class ToneClassifier:
    def __init__(self, model_path="model/tone_model.pkl"):
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)

    def predict_with_confidence(self, text):
        # Get predicted label
        label = self.model.predict([text])[0]
        # Get prediction probabilities (confidence)
        probs = self.model.predict_proba([text])[0]
        # Find confidence of the predicted label
        label_index = list(self.model.classes_).index(label)
        confidence = probs[label_index]
        return label, confidence