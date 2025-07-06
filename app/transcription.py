# import whisper
# from app.config import WHISPER_MODEL_SIZE

# def transcribe_audio(file_path):
#     model = whisper.load_model(WHISPER_MODEL_SIZE)
#     result = model.transcribe(file_path)
#     return result['text'], result.get('segments', [])

import whisper
from typing import Tuple, List, Dict, Any, cast
from app.config import WHISPER_MODEL_SIZE
from tqdm import tqdm
import json
import os


def transcribe_audio(file_path: str) -> Tuple[str, List[Dict[str, Any]]]:

    # Transcribes a given audio file to text using OpenAI Whisper
    try:
        print(f"[Whisper] Loading model: {WHISPER_MODEL_SIZE}")
        model = whisper.load_model(WHISPER_MODEL_SIZE)

        print(f"[Whisper] Transcribing: {file_path}")
        result = model.transcribe(file_path, verbose=False)

        # Making sure of model output being text and segments only
        raw_text = result.get("text", "")
        text = raw_text if isinstance(raw_text, str) else ""

        raw_segments = result.get("segments", [])
        segments = raw_segments if isinstance(raw_segments, list) else []

        # Show progress bar through segments
        for seg in tqdm(segments, desc="\nTranscribing", unit="segment"):
            pass

        os.makedirs("outputs", exist_ok=True)
        base_filename = os.path.splitext(os.path.basename(file_path))[0]
        output_json_path = os.path.join("outputs", f"{base_filename}_transcription.json")
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=4, ensure_ascii=False)
        print(f"[Whisper] Transcription JSON saved to: {output_json_path}")


        return text, segments

    except Exception as e:
        print(f"[ERROR] Failed to transcribe: {e}")
        return "", []