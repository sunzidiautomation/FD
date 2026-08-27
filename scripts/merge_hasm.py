"""Merge multiple independently calibrated HASMs into a single complete HASM.

Use this to combine outputs when running different attributes in parallel
across multiple Kaggle/Colab notebooks or GPUs.

Examples:
    python scripts/merge_hasm.py \\
        --inputs out_color/hasm.npz out_identity/hasm.npz out_size/hasm.npz \\
        --out merged_calibration/

    python scripts/merge_hasm.py \\
        --dir ./calibration_parts/ \\
        --out ./merged_calibration/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flair_t2i.demo.report import write_report
from flair_t2i.demo.sweep import DemoPaths
from flair_t2i.hasm import HASM


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge multiple HASM calibration files.")
    parser.add_argument(
        "--inputs",
        "-i",
        type=Path,
        nargs="+",
        help="Paths to individual hasm.npz files to merge.",
    )
    parser.add_argument(
        "--dir",
        "-d",
        type=Path,
        help="Directory containing individual hasm*.npz files to search and merge.",
    )
    parser.add_argument(
        "--out",
        "-o",
        type=Path,
        default=Path("merged_calibration"),
        help="Output directory to write merged hasm.npz, basm.npz, and index.html.",
    )
    args = parser.parse_args()

    paths_to_load: list[Path] = []
    if args.inputs:
        paths_to_load.extend(args.inputs)
    if args.dir:
        if not args.dir.exists():
            print(f"Error: directory {args.dir} does not exist.")
            sys.exit(1)
        paths_to_load.extend(sorted(args.dir.rglob("hasm*.npz")))

    if not paths_to_load:
        print("Error: No input files provided. Use --inputs or --dir.")
        sys.exit(1)

    print(f"Loading {len(paths_to_load)} HASM file(s):")
    hasms: list[HASM] = []
    for p in paths_to_load:
        print(f"  - {p}")
        hasms.append(HASM.load(p))

    merged = HASM.merge(hasms)
    args.out.mkdir(parents=True, exist_ok=True)

    hasm_out = args.out / "hasm.npz"
    basm_out = args.out / "basm.npz"
    merged.save(hasm_out)
    merged.to_basm().save(basm_out)

    print(f"\nSuccessfully merged {len(merged.attributes)} attribute(s):")
    for attr in merged.attributes:
        unit, score = merged.top_k(attr, 1)[0]
        print(f"  {attr.value:<10} peak at B{unit.block:<2} H{unit.head:<2} (score: {score:.3f})")

    # Render unified HTML report
    demo_paths = DemoPaths(args.out)
    report_path = write_report(merged, demo_paths, title="FLAIR — Merged Multi-Attribute Calibration")
    print(f"\nWrote merged matrix to: {hasm_out}")
    print(f"Wrote merged BASM to:   {basm_out}")
    print(f"Wrote HTML report to:   {report_path}")


if __name__ == "__main__":
    main()
