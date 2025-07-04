import subprocess
import sys
from pathlib import Path

def train_ner_model():
    # Ensure paths are correct
    config_path = Path("ner_training/config.cfg")
    train_path = Path("ner_data/train.spacy")
    dev_path = Path("ner_data/dev.spacy")
    output_dir = Path("ner_output")

    if not config_path.exists():
        raise FileNotFoundError("❌ config.cfg not found.")
    if not train_path.exists() or not dev_path.exists():
        raise FileNotFoundError("❌ .spacy training/dev files not found.")
    
    print("Starting training...")
    subprocess.run([
        sys.executable,  # THIS ensures the correct Python interpreter runs spacy train
        "-m", "spacy", "train",
        str(config_path),
        "--output", str(output_dir),
        "--paths.train", str(train_path),
        "--paths.dev", str(dev_path),
    ])

if __name__ == "__main__":
    train_ner_model()