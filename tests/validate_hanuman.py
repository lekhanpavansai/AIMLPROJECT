"""Specific validation for Hanuman audio file."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import sys
from pathlib import Path

# Add tests to path
sys.path.insert(0, str(Path(__file__).parent))

from unified_pipeline.pipeline import run_pipeline_sync
from validate_pipeline import (
    validate_audio_file,
    validate_diarization_json,
    validate_timeline_files,
    validate_target_audio,
)


def validate_hanuman() -> None:
    """Validate pipeline with Hanuman audio files."""
    print("=" * 60)
    print("Hanuman Audio Validation")
    print("=" * 60)
    
    # Find Hanuman files
    outputs_dir = Path("outputs")
    hanuman_files = list(outputs_dir.glob("*hanuman*.wav"))
    sample_files = list(outputs_dir.glob("*sample*.wav"))
    
    if not hanuman_files:
        print("\n[ERROR] No Hanuman audio files found in outputs/")
        print("Looking for files matching: *hanuman*.wav")
        sys.exit(1)
    
    if not sample_files:
        print("\n[ERROR] No sample/target audio files found in outputs/")
        print("Looking for files matching: *sample*.wav")
        sys.exit(1)
    
    # Use first Hanuman file as mixture
    mixture_path = hanuman_files[0]
    print(f"\n[INFO] Found mixture: {mixture_path.name}")
    
    # Use first sample file as target
    target_path = sample_files[0]
    print(f"[INFO] Found target: {target_path.name}")
    
    output_dir = Path("outputs/hanuman_validation")
    
    print("\n" + "=" * 60)
    print("Step 1: Validating Input Files")
    print("=" * 60)
    
    mix_ok, mix_msg = validate_audio_file(mixture_path)
    target_ok, target_msg = validate_audio_file(target_path)
    
    if not mix_ok:
        print(f"[FAIL] Mixture: {mix_msg}")
        sys.exit(1)
    print(f"[OK] Mixture: {mix_msg}")
    print(f"  File size: {mixture_path.stat().st_size / (1024*1024):.2f} MB")
    
    if not target_ok:
        print(f"[FAIL] Target: {target_msg}")
        sys.exit(1)
    print(f"[OK] Target: {target_msg}")
    print(f"  File size: {target_path.stat().st_size / (1024*1024):.2f} MB")
    
    print("\n" + "=" * 60)
    print("Step 2: Running Pipeline")
    print("=" * 60)
    
    try:
        transcripts = run_pipeline_sync(mixture_path, target_path, output_dir)
        print(f"[OK] Pipeline completed successfully")
        print(f"[INFO] Generated {len(transcripts)} transcript segments")
    except Exception as e:
        print(f"[FAIL] Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("Step 3: Validating Outputs")
    print("=" * 60)
    
    # Validate diarization.json
    diar_path = output_dir / "diarization.json"
    diar_ok, diar_msg, diar_data = validate_diarization_json(diar_path)
    speakers = []
    total_duration = 0
    avg_conf = 0
    
    if diar_ok:
        print(f"[OK] diarization.json: {diar_msg}")
        print(f"  Segments: {len(diar_data)}")
        if diar_data:
            speakers = list(set(seg.get("speaker", "unknown") for seg in diar_data))
            total_duration = max((seg.get("end", 0) for seg in diar_data), default=0)
            avg_conf = sum(seg.get("confidence", 0) for seg in diar_data) / len(diar_data)
            print(f"  Speakers: {len(speakers)} ({', '.join(speakers)})")
            print(f"  Duration: {total_duration:.2f}s")
            print(f"  Avg Confidence: {avg_conf:.3f}")
    else:
        print(f"[FAIL] diarization.json: {diar_msg}")
    
    # Validate timeline files
    timeline_ok, timeline_msg = validate_timeline_files(output_dir)
    if timeline_ok:
        print(f"[OK] Timeline files: {timeline_msg}")
    else:
        print(f"[WARN] Timeline files: {timeline_msg}")
    
    # Validate target audio
    target_audio_ok, target_audio_msg = validate_target_audio(output_dir)
    if target_audio_ok:
        target_wav = output_dir / "target_speaker.wav"
        size_mb = target_wav.stat().st_size / (1024*1024)
        print(f"[OK] target_speaker.wav: {target_audio_msg}")
        print(f"  File size: {size_mb:.2f} MB")
    else:
        print(f"[FAIL] target_speaker.wav: {target_audio_msg}")
    
    # Check separated speakers
    print("\n" + "=" * 60)
    print("Step 4: Separated Speakers")
    print("=" * 60)
    speakers_dir = output_dir / "speakers"
    speakers_metadata_path = output_dir / "speakers_metadata.json"
    
    if speakers_dir.exists():
        speaker_files = list(speakers_dir.glob("speaker_*.wav"))
        print(f"[OK] Found {len(speaker_files)} separated speaker audio files")
        for speaker_file in sorted(speaker_files):
            size_kb = speaker_file.stat().st_size / 1024
            print(f"  - {speaker_file.name}: {size_kb:.2f} KB")
    else:
        print(f"[WARN] Speakers directory missing: {speakers_dir}")
    
    if speakers_metadata_path.exists():
        try:
            with open(speakers_metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            print(f"[OK] Speakers metadata: {metadata.get('total_speakers', 0)} speakers")
            target_speakers = [s for s in metadata.get("speakers", []) if s.get("is_target", False)]
            if target_speakers:
                print(f"  [INFO] Target speaker: speaker_{target_speakers[0]['index']} (similarity: {target_speakers[0].get('similarity', 0):.3f})")
        except Exception as e:
            print(f"[WARN] Failed to read speakers metadata: {e}")
    else:
        print(f"[WARN] Speakers metadata missing: {speakers_metadata_path}")
    
    print("\n" + "=" * 60)
    print("Step 5: File Summary")
    print("=" * 60)
    
    output_files = [
        ("diarization.json", output_dir / "diarization.json"),
        ("timeline.json", output_dir / "timeline.json"),
        ("timeline.html", output_dir / "timeline.html"),
        ("target_speaker.wav", output_dir / "target_speaker.wav"),
        ("speakers_metadata.json", speakers_metadata_path),
    ]
    
    for name, path in output_files:
        if path.exists():
            size_kb = path.stat().st_size / 1024
            if size_kb > 1024:
                print(f"[FILE] {name}: {size_kb/1024:.2f} MB")
            else:
                print(f"[FILE] {name}: {size_kb:.2f} KB")
        else:
            print(f"[MISSING] {name}")
    
    print("\n" + "=" * 60)
    print("Validation Complete")
    print("=" * 60)
    print(f"\n[INFO] Output directory: {output_dir}")
    print(f"[INFO] Open timeline.html in browser to view visualization")
    
    # Save detailed results
    results = {
        "mixture_file": str(mixture_path),
        "target_file": str(target_path),
        "output_dir": str(output_dir),
        "segments": len(transcripts),
        "diarization_segments": len(diar_data) if diar_ok else 0,
        "speakers": list(speakers) if diar_ok and diar_data else [],
        "total_duration": total_duration if diar_ok and diar_data else 0,
        "avg_confidence": avg_conf if diar_ok and diar_data else 0,
    }
    
    results_file = output_dir / "validation_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    
    print(f"[INFO] Detailed results saved to: {results_file}")


if __name__ == "__main__":
    validate_hanuman()

