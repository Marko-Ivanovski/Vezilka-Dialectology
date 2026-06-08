"""
fix_chunk_alignment.py - Fix text-audio misalignment in Vezilka corpus chunks.

Problem: YouTube SRT captions have timing ahead of actual speech, causing each
chunk's text to include words actually spoken in the next chunk's audio.

Solution:
1. Build a de-duplicated full transcript per video (merge overlapping text)
2. Run faster-whisper on each chunk to determine actual word boundaries
3. Re-split the full transcript based on what's actually audible per chunk
4. Update metadata.json and training_data.jsonl with corrected text + pause info

Usage:
  python fix_chunk_alignment.py --analyze              # Run Whisper (slow, GPU)
  python fix_chunk_alignment.py --apply --dry-run      # Preview changes
  python fix_chunk_alignment.py --apply                # Write corrected files
  python fix_chunk_alignment.py --analyze --apply      # Both in sequence

  python fix_chunk_alignment.py --analyze --video "sobranie/1. Собранието..."  # Single video

Resume-safe: already-analyzed videos are skipped on re-run.
"""

import argparse
import json
import hashlib
import sys
import os
import time
from pathlib import Path
from difflib import SequenceMatcher

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


CORPUS_ROOT = Path(__file__).parent.parent / "govori"
CACHE_DIR = Path(__file__).parent.parent / "reports" / "alignment_cache"
REPORTS_DIR = Path(__file__).parent.parent / "reports"


# ─── Utilities ───────────────────────────────────────────────────────────────

def find_all_video_dirs(root: Path) -> list[Path]:
    dirs = []
    for dirpath, _, filenames in os.walk(root):
        if "metadata.json" in filenames and "training_data.jsonl" in filenames:
            audio_dir = Path(dirpath) / "audio"
            if audio_dir.exists():
                dirs.append(Path(dirpath))
    return sorted(dirs)


def get_cache_path(video_dir: Path) -> Path:
    rel = str(video_dir.relative_to(CORPUS_ROOT))
    safe_name = hashlib.md5(rel.encode("utf-8")).hexdigest()
    return CACHE_DIR / f"{safe_name}.json"


def read_metadata(path: Path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"  ERROR reading {path}: {e}")
        return None


def read_jsonl(path: Path):
    lines = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for raw in f:
                raw = raw.strip()
                if raw:
                    lines.append(json.loads(raw))
    except Exception as e:
        print(f"  ERROR reading {path}: {e}")
        return None
    return lines


# ─── Phase 1: Analyze (Whisper) ─────────────────────────────────────────────

def analyze_video(video_dir: Path, model) -> dict | None:
    """Run Whisper on all chunks in a video directory."""
    metadata = read_metadata(video_dir / "metadata.json")

    if not metadata or len(metadata) < 2:
        return None

    audio_dir = video_dir / "audio"
    chunks = metadata[1:]
    results = []

    for chunk in chunks:
        audio_file = chunk.get("audio_file", "")
        wav_path = audio_dir / audio_file

        if not wav_path.exists():
            results.append({
                "audio_file": audio_file,
                "whisper_words": [],
                "word_timestamps": [],
                "speech_start_s": None,
                "speech_end_s": None,
            })
            continue

        segments, info = model.transcribe(
            str(wav_path),
            language="mk",
            word_timestamps=True,
            vad_filter=False,
        )

        words = []
        word_timestamps = []

        for segment in segments:
            if segment.words:
                for w in segment.words:
                    cleaned = w.word.strip()
                    if cleaned:
                        words.append(cleaned)
                        word_timestamps.append({
                            "word": cleaned,
                            "start": round(w.start, 3),
                            "end": round(w.end, 3),
                        })

        speech_start = word_timestamps[0]["start"] if word_timestamps else None
        speech_end = word_timestamps[-1]["end"] if word_timestamps else None

        results.append({
            "audio_file": audio_file,
            "whisper_words": words,
            "word_timestamps": word_timestamps,
            "speech_start_s": speech_start,
            "speech_end_s": speech_end,
        })

    return {
        "video_path": str(video_dir.relative_to(CORPUS_ROOT)),
        "chunks": results,
    }


def run_analyze(args):
    """Phase 1: Run Whisper on all videos, cache results."""
    if WhisperModel is None:
        print("ERROR: faster-whisper not installed. Run: pip install faster-whisper")
        sys.exit(1)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    video_dirs = find_all_video_dirs(CORPUS_ROOT)
    print(f"Found {len(video_dirs)} video directories")

    if args.video:
        video_dirs = [v for v in video_dirs if args.video in str(v.relative_to(CORPUS_ROOT))]
        print(f"Filtered to {len(video_dirs)} matching '--video {args.video}'")

    to_process = []
    for vd in video_dirs:
        if not get_cache_path(vd).exists():
            to_process.append(vd)

    print(f"Already cached: {len(video_dirs) - len(to_process)}")
    print(f"To process: {len(to_process)}")

    if not to_process:
        print("\nAll videos already analyzed. Use --apply to fix alignment.")
        return

    total_chunks = 0
    for vd in to_process:
        meta = read_metadata(vd / "metadata.json")
        if meta:
            total_chunks += max(0, len(meta) - 1)

    print(f"Total chunks to transcribe: {total_chunks}")
    print(f"Estimated time (~10x realtime on GTX 1080 with 'small'): "
          f"~{total_chunks * 30 / 10 / 3600:.1f} hours\n")

    print(f"Loading Whisper model '{args.model}' on {args.device}...")
    model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type)
    print("Model loaded.\n")

    start_time = time.time()
    processed_chunks = 0

    if tqdm:
        iterator = tqdm(to_process, desc="Videos", unit="video")
    else:
        iterator = to_process

    for video_dir in iterator:
        rel = str(video_dir.relative_to(CORPUS_ROOT))

        if tqdm:
            iterator.set_postfix_str(rel[-50:])
        else:
            print(f"  [{to_process.index(video_dir)+1}/{len(to_process)}] {rel}")

        result = analyze_video(video_dir, model)

        if result:
            cache_path = get_cache_path(video_dir)
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            processed_chunks += len(result["chunks"])

    elapsed = time.time() - start_time
    print(f"\nAnalysis complete in {elapsed/3600:.1f}h ({processed_chunks} chunks)")
    print("Run with --apply to fix alignment.")


# ─── Phase 2: Apply ─────────────────────────────────────────────────────────

def find_overlap_length(words_a: list[str], words_b: list[str], max_check: int = 100) -> int:
    """Find longest suffix of words_a that equals prefix of words_b."""
    limit = min(len(words_a), len(words_b), max_check)
    best = 0

    for length in range(1, limit + 1):
        if words_a[-length:] == words_b[:length]:
            best = length

    return best


def build_full_transcript(chunks_text: list[str]) -> list[str]:
    """
    De-duplicate overlapping text between consecutive chunks.
    Returns the full word list representing the continuous transcript.
    """
    if not chunks_text:
        return []

    all_words = [text.split() for text in chunks_text]

    if not all_words[0]:
        full_words = []
    else:
        full_words = list(all_words[0])

    for i in range(1, len(all_words)):
        if not all_words[i]:
            continue
        overlap_len = find_overlap_length(full_words, all_words[i])
        full_words.extend(all_words[i][overlap_len:])

    return full_words


def normalize_word(w: str) -> str:
    """Normalize for fuzzy comparison."""
    punct = '.,!?;:\"\'-' + chr(8211) + chr(8212) + '[]{}' + chr(171) + chr(187) + chr(8222) + chr(8220) + chr(8221)
    return w.lower().strip(punct)


def fuzzy_word_match(a: str, b: str) -> float:
    """Score similarity between two words (0.0 - 1.0)."""
    na, nb = normalize_word(a), normalize_word(b)
    if na == nb:
        return 1.0
    if not na or not nb:
        return 0.0
    return SequenceMatcher(None, na, nb).ratio()


def score_boundary(full_words: list[str], candidate_end: int, cursor: int,
                   whisper_words: list[str]) -> float:
    """
    Score a candidate split point by comparing boundary words.
    Checks both the tail of the current chunk and head of the next chunk.
    """
    if candidate_end <= cursor:
        return -1.0

    n_check = min(5, len(whisper_words))
    if n_check == 0:
        return 0.0

    whisper_tail = whisper_words[-n_check:]
    candidate_tail_start = max(cursor, candidate_end - n_check)
    candidate_tail = full_words[candidate_tail_start:candidate_end]

    if not candidate_tail:
        return -1.0

    score = 0.0
    pairs = min(len(candidate_tail), len(whisper_tail))
    for j in range(pairs):
        ct_idx = len(candidate_tail) - 1 - j
        wt_idx = len(whisper_tail) - 1 - j
        score += fuzzy_word_match(candidate_tail[ct_idx], whisper_tail[wt_idx])

    score /= max(pairs, 1)

    distance_from_estimate = abs(candidate_end - (cursor + len(whisper_words)))
    score -= distance_from_estimate * 0.005

    return score


def find_best_split(full_words: list[str], cursor: int,
                    whisper_words: list[str], search_radius: int = 20) -> int:
    """
    Find the best split point in full_words for this chunk.
    Returns end index (exclusive).
    """
    if not whisper_words:
        return cursor

    estimated_end = cursor + len(whisper_words)

    if estimated_end >= len(full_words):
        return len(full_words)

    window_start = max(cursor + 1, estimated_end - search_radius)
    window_end = min(len(full_words), estimated_end + search_radius)

    best_end = min(estimated_end, len(full_words))
    best_score = -999.0

    for candidate_end in range(window_start, window_end + 1):
        s = score_boundary(full_words, candidate_end, cursor, whisper_words)
        if s > best_score:
            best_score = s
            best_end = candidate_end

    return best_end


def align_chunks(full_words: list[str], whisper_per_chunk: list[list[str]]) -> list[tuple[int, int]]:
    """
    Assign spans of full_words to each chunk based on Whisper word counts + fuzzy matching.
    Returns list of (start_idx, end_idx) into full_words.
    """
    n_chunks = len(whisper_per_chunk)
    cursor = 0
    assignments = []

    for chunk_idx in range(n_chunks):
        whisper_words = whisper_per_chunk[chunk_idx]

        if chunk_idx == n_chunks - 1:
            assignments.append((cursor, len(full_words)))
            break

        if not whisper_words:
            assignments.append((cursor, cursor))
            continue

        end = find_best_split(full_words, cursor, whisper_words)
        assignments.append((cursor, end))
        cursor = end

    return assignments


def apply_video(video_dir: Path, cache_data: dict, dry_run: bool) -> dict:
    """Apply alignment fix to a single video."""
    metadata_path = video_dir / "metadata.json"
    jsonl_path = video_dir / "training_data.jsonl"

    metadata = read_metadata(metadata_path)
    jsonl = read_jsonl(jsonl_path)

    if not metadata or not jsonl or len(metadata) < 2:
        return {"skipped": True, "reason": "invalid metadata"}

    header = metadata[0]
    meta_chunks = metadata[1:]

    if len(meta_chunks) != len(jsonl):
        return {"skipped": True, "reason": f"count mismatch: meta={len(meta_chunks)}, jsonl={len(jsonl)}"}

    if len(meta_chunks) != len(cache_data["chunks"]):
        return {"skipped": True, "reason": "cache chunk count != metadata chunk count"}

    # Step 1: Build full transcript
    chunks_text = [mc.get("text", "") for mc in meta_chunks]
    full_words = build_full_transcript(chunks_text)

    # Step 2: Get Whisper words per chunk
    whisper_per_chunk = [c["whisper_words"] for c in cache_data["chunks"]]

    # Sanity check: if total whisper words is way off from full_words, flag it
    total_whisper = sum(len(w) for w in whisper_per_chunk)
    ratio = total_whisper / max(len(full_words), 1)
    if ratio < 0.5 or ratio > 2.0:
        return {
            "skipped": True,
            "reason": f"word count ratio suspicious: whisper={total_whisper}, transcript={len(full_words)}, ratio={ratio:.2f}"
        }

    # Step 3: Align
    assignments = align_chunks(full_words, whisper_per_chunk)

    # Step 4: Build corrected entries
    changes = []
    new_meta_chunks = []
    new_jsonl_lines = []

    for i, (start_idx, end_idx) in enumerate(assignments):
        new_text = " ".join(full_words[start_idx:end_idx])
        old_text = meta_chunks[i].get("text", "")

        chunk_cache = cache_data["chunks"][i]
        speech_start = chunk_cache.get("speech_start_s")
        speech_end = chunk_cache.get("speech_end_s")

        new_mc = {**meta_chunks[i], "text": new_text}
        if speech_start is not None:
            new_mc["speech_start_ms"] = int(speech_start * 1000)
        if speech_end is not None:
            new_mc["speech_end_ms"] = int(speech_end * 1000)

        new_meta_chunks.append(new_mc)
        new_jsonl_lines.append({**jsonl[i], "text": new_text})

        if old_text.split() != new_text.split():
            old_words = old_text.split()
            new_words = new_text.split()
            changes.append({
                "chunk": i,
                "audio_file": meta_chunks[i].get("audio_file", ""),
                "old_word_count": len(old_words),
                "new_word_count": len(new_words),
                "words_removed_end": len(old_words) - len(new_words) if len(old_words) > len(new_words) else 0,
                "words_added_start": len(new_words) - len(old_words) if len(new_words) > len(old_words) else 0,
                "old_tail": " ".join(old_words[-5:]),
                "new_tail": " ".join(new_words[-5:]),
            })

    # Step 5: Write
    if not dry_run and changes:
        new_metadata = [header] + new_meta_chunks
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(new_metadata, f, ensure_ascii=False, indent=2)

        with open(jsonl_path, "w", encoding="utf-8") as f:
            for entry in new_jsonl_lines:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    return {
        "skipped": False,
        "chunks_total": len(meta_chunks),
        "chunks_changed": len(changes),
        "changes": changes[:20],
    }


def run_apply(args):
    """Phase 2: Use cached Whisper data to fix alignment."""
    video_dirs = find_all_video_dirs(CORPUS_ROOT)
    print(f"Found {len(video_dirs)} video directories")

    if args.video:
        video_dirs = [v for v in video_dirs if args.video in str(v.relative_to(CORPUS_ROOT))]
        print(f"Filtered to {len(video_dirs)} matching '--video {args.video}'")

    mode = "DRY-RUN" if args.dry_run else "APPLY"
    print(f"Mode: {mode}\n")

    total_changed = 0
    total_chunks_changed = 0
    skipped = 0
    no_cache = 0
    reports = []

    for video_dir in video_dirs:
        rel = str(video_dir.relative_to(CORPUS_ROOT))
        cache_path = get_cache_path(video_dir)

        if not cache_path.exists():
            no_cache += 1
            continue

        with open(cache_path, "r", encoding="utf-8") as f:
            cache_data = json.load(f)

        result = apply_video(video_dir, cache_data, dry_run=args.dry_run)

        if result.get("skipped"):
            skipped += 1
            print(f"  SKIP  {rel}: {result.get('reason')}")
        elif result["chunks_changed"] > 0:
            total_changed += 1
            total_chunks_changed += result["chunks_changed"]
            verb = "WOULD FIX" if args.dry_run else "FIXED"
            print(f"  {verb}  {rel}: {result['chunks_changed']}/{result['chunks_total']} chunks adjusted")

        reports.append({"path": rel, **result})

    print(f"\n{'=' * 60}")
    print(f"SUMMARY ({mode})")
    print(f"{'=' * 60}")
    print(f"  Videos processed:         {len(video_dirs) - no_cache - skipped}")
    print(f"  Videos with changes:      {total_changed}")
    print(f"  Total chunks adjusted:    {total_chunks_changed}")
    print(f"  Skipped (errors):         {skipped}")
    print(f"  Missing cache (--analyze): {no_cache}")

    REPORTS_DIR.mkdir(exist_ok=True)
    suffix = "dryrun" if args.dry_run else "applied"
    report_path = REPORTS_DIR / f"alignment_report_{suffix}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(reports, f, ensure_ascii=False, indent=2)
    print(f"\n  Report: {report_path}")

    if args.dry_run and total_chunks_changed:
        print("\n  Run without --dry-run to write changes.")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Fix chunk text-audio alignment in Vezilka corpus.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--analyze", action="store_true",
                        help="Run Whisper analysis on chunks (slow, uses GPU)")
    parser.add_argument("--apply", action="store_true",
                        help="Apply fixes using cached Whisper analysis (fast)")
    parser.add_argument("--dry-run", action="store_true",
                        help="With --apply: preview changes without writing files")
    parser.add_argument("--model", default="small",
                        help="Whisper model size (default: small)")
    parser.add_argument("--device", default="cuda",
                        help="Device for Whisper (default: cuda)")
    parser.add_argument("--compute-type", default="float16",
                        help="Compute type (default: float16)")
    parser.add_argument("--video", default="",
                        help="Process only video dirs matching this substring")
    args = parser.parse_args()

    if not args.analyze and not args.apply:
        parser.print_help()
        print("\nSpecify --analyze, --apply, or both.")
        sys.exit(1)

    if args.analyze:
        run_analyze(args)

    if args.apply:
        run_apply(args)


if __name__ == "__main__":
    main()
