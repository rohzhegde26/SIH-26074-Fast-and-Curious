"""
tests/test_audio_assets.py
Automated test verifying that:
1. Every (crop, stage, risk) combination in the matrix resolves to an audio file.
2. The resolved audio file exists on disk in frontend/audio/.
3. Every audio file is non-empty and has a valid MP3 file size.
4. Total precached audio size across all files is strictly under 500 KB.
5. All audio files are registered in service-worker.js precache manifest.
"""
import os
import pytest
from src.advisory.audio_matrix import (
    get_all_matrix_combinations,
    resolve_audio_file,
    CROPS,
    STAGES,
    RISKS,
)

AUDIO_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend", "audio")
SW_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend", "service-worker.js")

def test_all_combinations_resolve_to_existing_file():
    combos = get_all_matrix_combinations()
    assert len(combos) == len(CROPS) * len(STAGES) * len(RISKS)
    assert len(combos) == 24

    missing = []
    resolved_files = set()

    for crop, stage, risk in combos:
        filename = resolve_audio_file(crop, stage, risk)
        resolved_files.add(filename)
        path = os.path.join(AUDIO_DIR, filename)
        if not os.path.exists(path):
            missing.append((crop, stage, risk, filename))
        else:
            size = os.path.getsize(path)
            assert size > 1000, f"Audio file {filename} is suspiciously small ({size} bytes)"

    assert not missing, f"Audio files missing for matrix combos: {missing}"
    assert len(resolved_files) >= 4, "Expected at least 4 distinct audio scenarios resolved"

def test_total_audio_payload_under_500kb():
    total_bytes = 0
    files = [f for f in os.listdir(AUDIO_DIR) if f.endswith(".mp3")]
    assert len(files) >= 7, f"Expected at least 7 audio files, found {len(files)}"

    for f in files:
        total_bytes += os.path.getsize(os.path.join(AUDIO_DIR, f))

    # Assert under 500 KB (512,000 bytes)
    assert total_bytes < 500 * 1024, f"Total audio payload ({total_bytes} bytes) exceeds 500 KB ceiling!"

def test_audio_files_in_service_worker_precache():
    assert os.path.exists(SW_FILE), "Service worker file missing"
    with open(SW_FILE, "r", encoding="utf-8") as f:
        content = f.read()

    files = [f for f in os.listdir(AUDIO_DIR) if f.endswith(".mp3")]
    for audio_name in files:
        expected_entry = f"/audio/{audio_name}"
        assert expected_entry in content, f"Service worker precache missing {expected_entry}"
