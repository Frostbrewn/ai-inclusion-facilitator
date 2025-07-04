import random
from typing import List, Dict, Any

def assign_fake_speakers(segments: List[Dict[str, Any]], num_speakers: int = 5) -> List[Dict[str, Any]]:

    # Assigns random fake speakers to Whisper segments for testing participation logic
    speakers = [f"Speaker {i+1}" for i in range(num_speakers)]

    for segment in segments:
        segment["speaker"] = random.choice(speakers)

    return segments