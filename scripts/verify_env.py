"""Verify every FLAIR dependency imports. Run this first in any session."""

import importlib
import sys

REQUIRED = [
    "torch",
    "diffusers",
    "transformers",
    "spacy",
    "skfuzzy",
    "lpips",
    "numpy",
    "scipy",
    "skimage",
]

failed = []
for name in REQUIRED:
    try:
        module = importlib.import_module(name)
        version = getattr(module, "__version__", "(no __version__)")
        print(f"  OK    {name:<14} {version}")
    except ImportError as exc:
        failed.append(name)
        print(f"  FAIL  {name:<14} {exc}")

try:
    import spacy

    spacy.load("en_core_web_sm")
    print("  OK    en_core_web_sm")
except Exception as exc:  # noqa: BLE001
    failed.append("en_core_web_sm")
    print(f"  FAIL  en_core_web_sm  {exc}")

if failed:
    print(f"\nMissing: {', '.join(failed)}")
    print("Install, then RESTART the session before running anything else.")
    sys.exit(1)

print("\nEnvironment OK.")
