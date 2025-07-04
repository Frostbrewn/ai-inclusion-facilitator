# AI Inclusion Facilitator

**Project Overview**  
An AI-powered tool designed to transcribe audio, perform Named Entity Recognition (NER), and classify tone from text to support inclusive communication and feedback analysis.

---

## Features

- **Audio transcription** using OpenAI Whisper  
- **Named Entity Recognition (NER)** with spaCy custom-trained model  
- **Tone Classification** with a custom scikit-learn model  
- **Confidence score output** for model predictions  
- Simple web interface or CLI to input audio/text and receive analyzed output  

---

## Installation

```bash
git clone https://github.com/Frostbrewn/ai-inclusion-facilitator.git
cd ai-inclusion-facilitator
python -m venv venv
source venv/bin/activate    # macOS/Linux
venv\Scripts\activate       # Windows
pip install -r requirements.txt
```

**Train models (optional)**  


> ⚠️ **Custom train the NER and tone classifier models if needed:**

```bash
python -m spacy train ner_training/config.cfg --output ner_output --paths.train ner_data/train.spacy --paths.dev ner_data/dev.spacy
python train_tone_classifier.py
```

**Run the app**
```bash
python run.py
```
