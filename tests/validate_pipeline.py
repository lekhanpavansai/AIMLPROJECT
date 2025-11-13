"""Pipeline validation script to test and verify current implementation."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from unified_pipeline.pipeline import run_pipeline_sync


def validate_audio_file(path: Path) -> tuple[bool, str]:
    """Check if audio file exists and is readable."""
    if not path.exists():
        return False, f"File does not exist: {path}"
    
    if not path.suffix.lower() == ".wav":
        return False, f"Expected .wav file, got: {path.suffix}"
    
    try:
        # Try to read file size
        size = path.stat().st_size
        if size == 0:
            return False, "File is empty"
        if size < 1000:  # Less than 1KB is suspicious
            return False, f"File seems too small: {size} bytes"
    except Exception as e:
        return False, f"Cannot read file: {e}"
    
    return True, "OK"


def validate_diarization_json(json_path: Path) -> tuple[bool, str, dict[str, Any]]:
    """Validate diarization.json structure and content."""
    if not json_path.exists():
        return False, "diarization.json does not exist", {}
    
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        return False, f"Invalid JSON: {e}", {}
    except Exception as e:
        return False, f"Cannot read file: {e}", {}
    
    if not isinstance(data, list):
        return False, "Expected JSON array", {}
    
    required_fields = ["speaker", "start", "end", "text", "confidence"]
    issues = []
    
    for idx, segment in enumerate(data):
        if not isinstance(segment, dict):
            issues.append(f"Segment {idx} is not a dict")
            continue
        
        for field in required_fields:
            if field not in segment:
                issues.append(f"Segment {idx} missing field: {field}")
        
        # Validate types
        if "start" in segment and not isinstance(segment["start"], (int, float)):
            issues.append(f"Segment {idx} 'start' must be numeric")
        if "end" in segment and not isinstance(segment["end"], (int, float)):
            issues.append(f"Segment {idx} 'end' must be numeric")
        if "confidence" in segment and not isinstance(segment["confidence"], (int, float)):
            issues.append(f"Segment {idx} 'confidence' must be numeric")
        if "start" in segment and "end" in segment:
            if segment["end"] <= segment["start"]:
                issues.append(f"Segment {idx} end <= start")
    
    if issues:
        return False, "; ".join(issues), data
    
    return True, f"Valid JSON with {len(data)} segments", data


def validate_timeline_files(output_dir: Path) -> tuple[bool, str]:
    """Check if timeline files exist and are readable."""
    json_path = output_dir / "timeline.json"
    html_path = output_dir / "timeline.html"
    
    issues = []
    
    if not json_path.exists():
        issues.append("timeline.json missing")
    else:
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                json.load(f)
        except Exception as e:
            issues.append(f"timeline.json invalid: {e}")
    
    if not html_path.exists():
        issues.append("timeline.html missing")
    else:
        try:
            content = html_path.read_text(encoding="utf-8")
            if len(content) < 100:
                issues.append("timeline.html seems too small")
        except Exception as e:
            issues.append(f"timeline.html unreadable: {e}")
    
    if issues:
        return False, "; ".join(issues)
    
    return True, "Both timeline files valid"


def validate_target_audio(output_dir: Path) -> tuple[bool, str]:
    """Check if target_speaker.wav exists and is valid."""
    target_path = output_dir / "target_speaker.wav"
    return validate_audio_file(target_path)


def run_validation_test(
    mixture_path: Path,
    target_path: Path,
    output_dir: Path,
    test_name: str,
) -> dict[str, Any]:
    """Run a single validation test."""
    print(f"\n{'='*60}")
    print(f"Test: {test_name}")
    print(f"{'='*60}")
    
    results: dict[str, Any] = {
        "test_name": test_name,
        "mixture": str(mixture_path),
        "target": str(target_path),
        "output_dir": str(output_dir),
        "passed": True,
        "errors": [],
        "warnings": [],
    }
    
    # Step 1: Validate input files
    print("\n[1/5] Validating input files...")
    mix_ok, mix_msg = validate_audio_file(mixture_path)
    target_ok, target_msg = validate_audio_file(target_path)
    
    if not mix_ok:
        results["errors"].append(f"Mixture: {mix_msg}")
        results["passed"] = False
        print(f"  [FAIL] Mixture: {mix_msg}")
    else:
        print(f"  [OK] Mixture: {mix_msg}")
    
    if not target_ok:
        results["errors"].append(f"Target: {target_msg}")
        results["passed"] = False
        print(f"  [FAIL] Target: {target_msg}")
    else:
        print(f"  [OK] Target: {target_msg}")
    
    if not results["passed"]:
        return results
    
    # Step 2: Run pipeline
    print("\n[2/5] Running pipeline...")
    try:
        transcripts = run_pipeline_sync(mixture_path, target_path, output_dir)
        print(f"  [OK] Pipeline completed: {len(transcripts)} segments")
        results["segment_count"] = len(transcripts)
    except Exception as e:
        results["errors"].append(f"Pipeline failed: {e}")
        results["passed"] = False
        print(f"  [FAIL] Pipeline failed: {e}")
        return results
    
    # Step 3: Validate outputs
    print("\n[3/5] Validating outputs...")
    
    # Check diarization.json
    diar_ok, diar_msg, diar_data = validate_diarization_json(output_dir / "diarization.json")
    if diar_ok:
        print(f"  [OK] diarization.json: {diar_msg}")
        results["diarization_segments"] = len(diar_data)
    else:
        print(f"  [FAIL] diarization.json: {diar_msg}")
        results["errors"].append(f"diarization.json: {diar_msg}")
        results["passed"] = False
    
    # Check timeline files
    timeline_ok, timeline_msg = validate_timeline_files(output_dir)
    if timeline_ok:
        print(f"  [OK] Timeline files: {timeline_msg}")
    else:
        print(f"  [WARN] Timeline files: {timeline_msg}")
        results["warnings"].append(f"Timeline: {timeline_msg}")
    
    # Check target audio
    target_audio_ok, target_audio_msg = validate_target_audio(output_dir)
    if target_audio_ok:
        print(f"  [OK] target_speaker.wav: {target_audio_msg}")
    else:
        print(f"  [FAIL] target_speaker.wav: {target_audio_msg}")
        results["errors"].append(f"target_speaker.wav: {target_audio_msg}")
        results["passed"] = False
    
    # Step 4: Analyze content
    print("\n[4/5] Analyzing content...")
    if diar_data:
        speakers = set(seg.get("speaker", "unknown") for seg in diar_data)
        total_duration = max((seg.get("end", 0) for seg in diar_data), default=0)
        avg_confidence = sum(seg.get("confidence", 0) for seg in diar_data) / len(diar_data)
        
        print(f"  [INFO] Speakers found: {len(speakers)} ({', '.join(speakers)})")
        print(f"  [INFO] Total duration: {total_duration:.2f}s")
        print(f"  [INFO] Avg confidence: {avg_confidence:.3f}")
        
        results["speakers"] = list(speakers)
        results["total_duration"] = total_duration
        results["avg_confidence"] = avg_confidence
    
    # Step 5: Check separated speakers
    print("\n[5/6] Checking separated speakers...")
    speakers_dir = output_dir / "speakers"
    speakers_metadata_path = output_dir / "speakers_metadata.json"
    
    if speakers_dir.exists():
        speaker_files = list(speakers_dir.glob("speaker_*.wav"))
        print(f"  [OK] Found {len(speaker_files)} separated speaker audio files")
        results["separated_speakers_count"] = len(speaker_files)
        for speaker_file in sorted(speaker_files):
            size_kb = speaker_file.stat().st_size / 1024
            print(f"    - {speaker_file.name}: {size_kb:.2f} KB")
    else:
        print(f"  [WARN] Speakers directory missing: {speakers_dir}")
        results["warnings"].append("speakers directory missing")
    
    if speakers_metadata_path.exists():
        try:
            with open(speakers_metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            print(f"  [OK] Speakers metadata: {metadata.get('total_speakers', 0)} speakers")
            target_speakers = [s for s in metadata.get("speakers", []) if s.get("is_target", False)]
            if target_speakers:
                print(f"  [INFO] Target speaker: speaker_{target_speakers[0]['index']}")
            results["speakers_metadata"] = metadata
        except Exception as e:
            print(f"  [WARN] Failed to read speakers metadata: {e}")
            results["warnings"].append(f"speakers_metadata.json read error: {e}")
    else:
        print(f"  [WARN] Speakers metadata missing: {speakers_metadata_path}")
        results["warnings"].append("speakers_metadata.json missing")
    
    # Step 6: Check file sizes
    print("\n[6/6] Checking file sizes...")
    files_to_check = [
        ("diarization.json", output_dir / "diarization.json"),
        ("timeline.json", output_dir / "timeline.json"),
        ("timeline.html", output_dir / "timeline.html"),
        ("target_speaker.wav", output_dir / "target_speaker.wav"),
        ("speakers_metadata.json", speakers_metadata_path),
    ]
    
    for name, path in files_to_check:
        if path.exists():
            size = path.stat().st_size
            size_kb = size / 1024
            print(f"  [FILE] {name}: {size_kb:.2f} KB")
            results[f"{name}_size_kb"] = size_kb
        else:
            print(f"  [WARN] {name}: missing")
            results["warnings"].append(f"{name} missing")
    
    return results


def main() -> None:
    """Run validation tests."""
    print("=" * 60)
    print("Pipeline Validation Suite")
    print("=" * 60)
    
    # Test cases
    test_cases = [
        {
            "name": "Synthetic Audio Test",
            "mixture": Path("data/mixture_audio.wav"),
            "target": Path("data/target_sample.wav"),
            "output": Path("outputs/validation_test1"),
        },
        {
            "name": "Real Audio Test (Vocals)",
            "mixture": Path("outputs/1_hanuman-song-maker-song-abhi-404570_(Vocals).wav"),
            "target": Path("outputs/1_sample-6s_(Vocals).wav"),
            "output": Path("outputs/validation_test2"),
        },
    ]
    
    all_results = []
    
    for test_case in test_cases:
        if not test_case["mixture"].exists() or not test_case["target"].exists():
            print(f"\n[SKIP] Skipping {test_case['name']}: input files not found")
            continue
        
        result = run_validation_test(
            test_case["mixture"],
            test_case["target"],
            test_case["output"],
            test_case["name"],
        )
        all_results.append(result)
    
    # Summary
    print("\n" + "=" * 60)
    print("Validation Summary")
    print("=" * 60)
    
    passed = sum(1 for r in all_results if r["passed"])
    total = len(all_results)
    
    print(f"\nTests passed: {passed}/{total}")
    
    for result in all_results:
        status = "[PASS]" if result["passed"] else "[FAIL]"
        print(f"\n{status} - {result['test_name']}")
        
        if result["errors"]:
            print("  Errors:")
            for error in result["errors"]:
                print(f"    - {error}")
        
        if result["warnings"]:
            print("  Warnings:")
            for warning in result["warnings"]:
                print(f"    - {warning}")
        
        if "segment_count" in result:
            print(f"  Segments: {result['segment_count']}")
        if "speakers" in result:
            print(f"  Speakers: {len(result['speakers'])}")
        if "avg_confidence" in result:
            print(f"  Avg Confidence: {result['avg_confidence']:.3f}")
    
    # Save results
    results_file = Path("outputs/validation_results.json")
    results_file.parent.mkdir(parents=True, exist_ok=True)
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n[INFO] Detailed results saved to: {results_file}")
    
    # Exit code
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()

