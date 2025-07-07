import pickle
import os

class ToneClassifier:
    # Defensive programming for file check
    def __init__(self, model_path="outputs/tone_model.pkl"):

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"\n\033[91m[ERROR]\033[0m Tone model not found at: {model_path}. Please ensure the model is trained and saved.")
        
        try:
            with open(model_path, "rb") as f:
                self.model = pickle.load(f)
        except Exception as e:
            raise RuntimeError(f"\n\033[91m[ERROR]\033[0m Failed to load tone model from {model_path}: {e}")

    def predict_with_confidence(self, text):

        if not hasattr(self.model, "predict_proba"):
            raise AttributeError("\n\033[91m[ERROR]\033[0m Loaded model does not support probability estimation.")
        
        # Get predicted label
        label = self.model.predict([text])[0]
        # Get prediction probabilities (confidence)
        probs = self.model.predict_proba([text])[0]
        # Find confidence of the predicted label
        label_index = list(self.model.classes_).index(label)
        confidence = probs[label_index]
        return label, confidence