from app.config import AUDIO_FILE_PATH
from app.transcription import transcribe_audio
from app.nlp_analysis import analyze_text
from app.participation import generate_nudges
from app.utils import assign_fake_speakers
from model.tone_classifier import ToneClassifier
import warnings

warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")


def main():

    print(f"\n[+] Transcribing: {AUDIO_FILE_PATH}")
    text, segments = transcribe_audio(AUDIO_FILE_PATH)

    print("\n[+] Transcript Preview:")
    preview = text[:500] if isinstance(text, str) else str(text)[:500]
    print(preview + "...")

    print("\n[+] NLP Analysis:\n")
    analysis = analyze_text(text)

    print(f"  ▸ Sentiment: {analysis['sentiment']}\n")

    entities = analysis.get("entities", [])
    if entities:
        print("  ▸ Named Entities:\n")
        for ent in analysis["entities"]:
            print(f"     - {ent['text']} ({ent['label']}) [confidence: {ent['confidence']:.2f}]")
            print("-" * 45)
    else:
        print("  ▸ No named entities found.")
        print("-" * 45)


    classifier = ToneClassifier()

    print("\n[+] Tone Classification per segment:")
    for i, segment in enumerate(segments):
        seg_text = segment.get("text", "")
        seg_label, seg_confidence = classifier.predict_with_confidence(seg_text)
        print(f"  \nSegment {i+1}: Predicted tone: {seg_label} (Confidence: {seg_confidence:.2f})")
        print(f"    Text snippet: {seg_text[:60]}...")
        print("-" * 45)


    print("\n[+] Participation Report:")
    segments = assign_fake_speakers(segments, num_speakers=5)
    generate_nudges(segments)
    print("-" * 45)
    print("\n")

if __name__ == "__main__":
    main()