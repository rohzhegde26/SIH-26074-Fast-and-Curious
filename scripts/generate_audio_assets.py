"""
scripts/generate_audio_assets.py
Generates 7 canonical, lightweight Mandya Kannada audio files for offline PWA precaching.
Text scripts use natural, spoken Mandya agricultural phrasing without bureaucratic artifacts.
Kept punchy and direct to ensure clear village speaker projection and stay under 500 KB total.
"""
import os
from gtts import gTTS

AUDIO_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend", "audio")
os.makedirs(AUDIO_DIR, exist_ok=True)

AUDIO_SCRIPTS = {
    "ragi_veg_rain_kn.mp3": (
        "ರಾಗಿ ಬೆಳೆಗೆ ಈಗ ಯೂರಿಯಾ ಗೊಬ್ಬರ ಹಾಕಬೇಡಿ. ಸಂಜೆ ಜೋರು ಮಳೆಗೆ ಗೊಬ್ಬರ ಕೊಚ್ಚಿ ಹೋಗಿ ನಷ್ಟವಾಗುತ್ತದೆ."
    ),
    "ragi_harvest_rot_kn.mp3": (
        "ಇಂದೇ ರಾಗಿ ಕೊಯ್ಲು ಮುಗಿಸಿ ಒಣ ಜಾಗದಲ್ಲಿ ಭದ್ರಪಡಿಸಿ. ಸಂಜೆ ಮಳೆಗೆ ಕಾಳು ಕೊಳೆಯುವ ಅಪಾಯವಿದೆ."
    ),
    "ragi_sow_dry_kn.mp3": (
        "ರಾಗಿ ಬಿತ್ತನೆಗೆ ಒಣ ಹವೆ ಮುಂದುವರಿಯಲಿದೆ. ಮಣ್ಣಿನ ಹದ ನೋಡಿ ಅಗತ್ಯವಿದ್ದರೆ ಮಾತ್ರ ನೀರು ಹಾಯಿಸಿ."
    ),
    "paddy_veg_rain_kn.mp3": (
        "ಭತ್ತದ ಗದ್ದೆಯಲ್ಲಿ ಬಸಿಗಾಲುವೆ ಸರಿಪಡಿಸಿ. ಮಳೆಯ ನಂತರ ಬೆಂಕಿ ರೋಗ ಬಾರದಂತೆ ನಿಗಾವಹಿಸಿ."
    ),
    "paddy_harvest_rot_kn.mp3": (
        "ಭತ್ತದ ಕಟಾವು ಮುಗಿದಿದ್ದರೆ ತಕ್ಷಣ ಕಾಳು ಸುರಕ್ಷಿತ ಜಾಗಕ್ಕೆ ಸಾಗಿಸಿ. ಮಳೆಯಿಂದ ಕಾಳು ಕಪ್ಪಾಗುವ ಅಪಾಯವಿದೆ."
    ),
    "dry_window_safe_kn.mp3": (
        "ನಾಳೆ ಒಣ ಹವೆ ಇರುತ್ತದೆ. ಔಷಧಿ ಸಿಂಪಡಣೆ ಹಾಗೂ ಕಳೆ ಕೀಳಲು ಧೈರ್ಯವಾಗಿ ಕೂಲಿಗಳನ್ನು ಕರೆಯಬಹುದು."
    ),
    "heavy_cloudburst_kn.mp3": (
        "ನಾಳೆ ಸಂಜೆ ಜೋರು ಮಳೆ ಬರುವ ಸಾಧ್ಯತೆ ಇದೆ. ಸಿಂಪಡಣೆ ಅಥವಾ ಕಳೆ ಕೆಲಸಕ್ಕೆ ಕೂಲಿ ಕರೆಯಬೇಡಿ."
    ),
}

def generate_all():
    total_bytes = 0
    print("Generating audio assets into:", AUDIO_DIR)
    for filename, script in AUDIO_SCRIPTS.items():
        filepath = os.path.join(AUDIO_DIR, filename)
        print(f"Synthesizing {filename}...")
        tts = gTTS(text=script, lang="kn")
        tts.save(filepath)
        size = os.path.getsize(filepath)
        total_bytes += size
        print(f"  -> {filename}: {size / 1024:.1f} KB")

    print(f"\nTotal audio payload: {total_bytes / 1024:.1f} KB ({total_bytes} bytes)")
    assert total_bytes < 500 * 1024, f"Audio exceeds 500 KB ceiling! ({total_bytes} bytes)"
    print(f"[SUCCESS] All audio assets successfully generated ({total_bytes / 1024:.1f} KB < 500 KB).")

if __name__ == "__main__":
    generate_all()
