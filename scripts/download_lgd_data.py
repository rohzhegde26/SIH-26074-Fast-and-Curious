#!/usr/bin/env python3
"""
Automated download script for LGD Panchayats geospatial dataset.

Downloads `LGD_panchayats.parquet` into `data/raw/geodata/` with progress
reporting and robust error handling.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional

DEFAULT_URL = "https://github.com/placeholder/LGD/releases/download/v1/LGD_panchayats.parquet"
DEFAULT_OUTPUT = Path("data/raw/geodata/LGD_panchayats.parquet")
CHUNK_SIZE = 1024 * 128  # 128 KB chunks


def format_bytes(num_bytes: int | float) -> str:
    """Format bytes into a human-readable string."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:3.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} PB"


def _download_with_requests(url: str, dest_path: Path) -> bool:
    """Download using `requests` library with tqdm or byte counter."""
    import requests

    headers = {"User-Agent": "SIH-26074-Downloader/1.0"}
    response = requests.get(url, stream=True, timeout=30, headers=headers)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    part_path = dest_path.with_suffix(".parquet.part")

    try:
        from tqdm import tqdm

        with open(part_path, "wb") as f, tqdm(
            desc=dest_path.name,
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            miniters=1,
        ) as bar:
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))
    except ImportError:
        # Fallback to byte counter if tqdm is not installed
        downloaded = 0
        last_print = 0.0
        start_time = time.time()
        with open(part_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    now = time.time()
                    if now - last_print > 0.5 or (total_size and downloaded == total_size):
                        last_print = now
                        elapsed = max(now - start_time, 0.001)
                        speed = downloaded / elapsed
                        if total_size:
                            pct = (downloaded / total_size) * 100
                            sys.stdout.write(
                                f"\rDownloading {dest_path.name}: {format_bytes(downloaded)} / "
                                f"{format_bytes(total_size)} ({pct:.1f}%) @ {format_bytes(speed)}/s"
                            )
                        else:
                            sys.stdout.write(
                                f"\rDownloading {dest_path.name}: {format_bytes(downloaded)} @ {format_bytes(speed)}/s"
                            )
                        sys.stdout.flush()
        print()

    if part_path.exists():
        if dest_path.exists():
            dest_path.unlink()
        part_path.rename(dest_path)
    return True


def _download_with_urllib(url: str, dest_path: Path) -> bool:
    """Download using standard library `urllib.request` with tqdm or byte counter."""
    import urllib.error
    import urllib.request

    req = urllib.request.Request(
        url,
        headers={"User-Agent": "SIH-26074-Downloader/1.0"},
    )
    with urllib.request.urlopen(req, timeout=30) as response:
        content_len_header = response.headers.get("Content-Length")
        total_size = int(content_len_header) if content_len_header else 0
        part_path = dest_path.with_suffix(".parquet.part")

        try:
            from tqdm import tqdm

            with open(part_path, "wb") as f, tqdm(
                desc=dest_path.name,
                total=total_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                miniters=1,
            ) as bar:
                while True:
                    chunk = response.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    f.write(chunk)
                    bar.update(len(chunk))
        except ImportError:
            downloaded = 0
            last_print = 0.0
            start_time = time.time()
            with open(part_path, "wb") as f:
                while True:
                    chunk = response.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    now = time.time()
                    if now - last_print > 0.5 or (total_size and downloaded == total_size):
                        last_print = now
                        elapsed = max(now - start_time, 0.001)
                        speed = downloaded / elapsed
                        if total_size:
                            pct = (downloaded / total_size) * 100
                            sys.stdout.write(
                                f"\rDownloading {dest_path.name}: {format_bytes(downloaded)} / "
                                f"{format_bytes(total_size)} ({pct:.1f}%) @ {format_bytes(speed)}/s"
                            )
                        else:
                            sys.stdout.write(
                                f"\rDownloading {dest_path.name}: {format_bytes(downloaded)} @ {format_bytes(speed)}/s"
                            )
                        sys.stdout.flush()
            print()

        if part_path.exists():
            if dest_path.exists():
                dest_path.unlink()
            part_path.rename(dest_path)
    return True


def download_lgd_data(
    url: str = DEFAULT_URL,
    dest_path: Path = DEFAULT_OUTPUT,
    force: bool = False,
) -> bool:
    """
    Download the LGD Panchayats parquet dataset.

    Parameters
    ----------
    url : str
        Source URL to download from.
    dest_path : Path
        Destination filepath.
    force : bool
        If True, re-download even if destination file exists.

    Returns
    -------
    bool
        True if download succeeded or file already exists and not forced.
    """
    dest_path = Path(dest_path)
    dest_dir = dest_path.parent

    # Ensure parent directory exists
    dest_dir.mkdir(parents=True, exist_ok=True)

    if dest_path.exists() and not force:
        size_str = format_bytes(dest_path.stat().st_size)
        print(f"[OK] Destination file already exists: {dest_path} ({size_str})")
        print("     Use --force to redownload.")
        return True

    print(f"[*] Downloading LGD Panchayats dataset...")
    print(f"    Source URL  : {url}")
    print(f"    Destination : {dest_path}")

    part_path = dest_path.with_suffix(".parquet.part")

    try:
        # Prefer requests if installed, otherwise fallback to urllib
        try:
            import requests  # noqa: F401
            _download_with_requests(url, dest_path)
        except ImportError:
            _download_with_urllib(url, dest_path)

        final_size = dest_path.stat().st_size
        print(f"[SUCCESS] Download completed: {dest_path} ({format_bytes(final_size)})")
        return True

    except Exception as exc:
        # Clean up partial download file if present
        if part_path.exists():
            try:
                part_path.unlink()
            except OSError:
                pass

        print(f"[ERROR] Failed to download {dest_path.name} from {url}", file=sys.stderr)
        print(f"        Details: {exc}", file=sys.stderr)
        return False


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download LGD Panchayats parquet dataset for SIH 26074 pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--url",
        default=os.environ.get("LGD_DATA_URL", DEFAULT_URL),
        help="URL to download LGD_panchayats.parquet from",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Target output file path",
    )
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Force download even if file already exists",
    )

    args = parser.parse_args()
    success = download_lgd_data(url=args.url, dest_path=args.output, force=args.force)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
