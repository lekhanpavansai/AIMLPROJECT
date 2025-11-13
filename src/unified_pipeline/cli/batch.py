from __future__ import annotations

import argparse
from pathlib import Path

from unified_pipeline.pipeline import run_pipeline_sync


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline target speaker diarization and ASR.")
    parser.add_argument("--mixture", type=Path, required=True, help="Path to mixture audio wav file.")
    parser.add_argument("--target", type=Path, required=True, help="Path to target speaker reference wav.")
    parser.add_argument("--output", type=Path, required=True, help="Directory to store outputs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    transcripts = run_pipeline_sync(args.mixture, args.target, args.output)
    print(f"Processed {len(transcripts)} transcript segments. Output stored in {args.output}")


if __name__ == "__main__":
    main()


