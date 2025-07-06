from app.config import SPEAKING_TIME_THRESHOLD

def generate_nudges(speaker_segments):
    speakers = {}
    for segment in speaker_segments:
        speaker = segment.get("speaker") or "Unknown"
        duration = segment["end"] - segment["start"]
        speakers[speaker] = speakers.get(speaker, 0) + duration

    total = sum(speakers.values())
    for person, time in speakers.items():
        percent = (time / total) * 100
        print(f"{person}: {percent:.1f}% of speaking time")
        if percent < SPEAKING_TIME_THRESHOLD:
            print(f"\n\033[93m[nudge]\033[0m Consider inviting {person} to speak more.\n")
        else:
            print(f"\n✅ The speaker participated above the threshold!\n")