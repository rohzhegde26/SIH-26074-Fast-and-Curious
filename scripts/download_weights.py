"""Download the final v3 model weights (GitHub release v3.0) into models/final/ and verify their SHA-256.

  python scripts/download_weights.py
"""
import hashlib
import sys
import urllib.request
from pathlib import Path

RELEASE = "https://github.com/rohzhegde26/SIH-26074-Fast-and-Curious/releases/download/v3.0"
FILES = {   # name: (sha256, bytes)
    "det.pt": ("738ec2fcefe00184a2935fcd62c508ebe84f8d9297d8cc3564b83ca230f3e120", 115174925),
    "det_s1.pt": ("00fa722384dbdb107ca7b5f1cb3b2199ec1be66cb3046757dfae6ca7cf62c260", 115174925),
    "det_s2.pt": ("df807bacaab7f1080f16e10efddddee046a3356e93acc6bad6dc0954838bc92f", 115174925),
    "diff.pt": ("d8b79cc1e19a0f6057b903c82300237d00b823ba6c58fcd38e4cc71e97221f4a", 71170432),
}
OUT = Path(__file__).resolve().parents[1] / "models" / "final"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ok = True
    for name, (digest, size) in FILES.items():
        dst = OUT / name
        if dst.exists() and dst.stat().st_size == size and sha256(dst) == digest:
            print(f"{name}: already present")
            continue
        print(f"{name}: downloading {size / 1e6:.0f} MB ...", flush=True)
        urllib.request.urlretrieve(f"{RELEASE}/{name}", dst)
        good = sha256(dst) == digest
        ok &= good
        print(f"{name}: {'ok' if good else 'CHECKSUM MISMATCH'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
