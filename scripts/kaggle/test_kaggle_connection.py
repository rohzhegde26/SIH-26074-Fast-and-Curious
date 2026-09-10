"""
scripts/kaggle/test_kaggle_connection.py

Verifies Kaggle API connectivity and credentials.
Automatically detects if kaggle.json is in Downloads and moves it to ~/.kaggle/ if needed.
"""

import os
import shutil
import sys
from pathlib import Path


def setup_and_verify_kaggle() -> bool:
    home = Path.home()
    kaggle_dir = home / ".kaggle"
    kaggle_json_target = kaggle_dir / "kaggle.json"
    downloads_json = home / "Downloads" / "kaggle.json"

    print("=" * 60)
    print("Kaggle API Setup & Connectivity Diagnostic")
    print("=" * 60)

    access_token_target = kaggle_dir / "access_token"

    has_token = (
        kaggle_json_target.exists()
        or access_token_target.exists()
        or "KAGGLE_API_TOKEN" in os.environ
    )

    if not has_token:
        if downloads_json.exists():
            print(f"[*] Found kaggle.json in Downloads: {downloads_json}")
            kaggle_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(downloads_json, kaggle_json_target)
            print(f"[+] Successfully copied to: {kaggle_json_target}")
        else:
            print("[-] Kaggle token NOT FOUND!")
            print(f"    Expected at: {access_token_target} or {kaggle_json_target}")
            print(f"    Or via env var KAGGLE_API_TOKEN")
            return False
    else:
        if access_token_target.exists():
            print(f"[+] Found Kaggle Personal Access Token at: {access_token_target}")
        elif kaggle_json_target.exists():
            print(f"[+] Found legacy credentials at: {kaggle_json_target}")
        elif "KAGGLE_API_TOKEN" in os.environ:
            print("[+] Found KAGGLE_API_TOKEN in environment variables.")

    # Verify kaggle package
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        print("[-] python 'kaggle' package is missing. Run: pip install kaggle")
        return False

    # Authenticate
    try:
        api = KaggleApi()
        api.authenticate()
        print("[+] Authentication SUCCESSFUL!")
        
        # Test an API call
        print("[*] Testing API with a light request...")
        kernels = api.kernels_list(mine=True, page=1, page_size=5)
        print(f"[+] Connected to Kaggle! Found {len(kernels)} existing kernel(s).")
        print("\nYour Kaggle GPU environment is ready to be used by the agent!")
        return True
    except Exception as e:
        print(f"[-] Authentication failed: {e}")
        return False


if __name__ == "__main__":
    success = setup_and_verify_kaggle()
    sys.exit(0 if success else 1)
