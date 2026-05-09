import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


CORPUS_ROOT = Path(__file__).parent.parent / "govori"
REPORTS_DIR = Path(__file__).parent.parent / "reports"

SPEAKER_MARKER = ">>"
MUSIC_INDICATORS = ["[музика]", "[music]", "♪", "♫", "[аплауз]", "[applause]"]


def strip_markers(text: str) -> str:
    cleaned = text.replace(SPEAKER_MARKER, "")
    return re.sub(r" {2,}", " ", cleaned).strip()


def is_music(text: str) -> bool:
    lower = text.lower()
    return any(indicator in lower for indicator in MUSIC_INDICATORS)


def classify_chunk(text: str) -> tuple[str, str]:
    """Return (decision, cleaned_text). Decision is 'DELETE' or 'KEEP'."""
    if is_music(text):
        return "DELETE", text
    cleaned = strip_markers(text)
    if not cleaned:
        return "DELETE", cleaned
    return "KEEP", cleaned


@dataclass
class DirResult:
    path: str
    skipped: bool = False
    skip_reason: str = ""
    chunks_before: int = 0
    chunks_deleted: int = 0
    chunks_stripped: int = 0
    deleted_indices: list = field(default_factory=list)


def read_metadata(path: Path) -> list | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"  ERROR reading {path}: {e}")
        return None


def read_jsonl(path: Path) -> list | None:
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


def process_video_dir(video_dir: Path, apply: bool) -> DirResult:
    rel = str(video_dir.relative_to(CORPUS_ROOT))
    result = DirResult(path=rel)

    metadata_path = video_dir / "metadata.json"
    jsonl_path = video_dir / "training_data.jsonl"

    if not metadata_path.exists() or not jsonl_path.exists():
        result.skipped = True
        result.skip_reason = "missing metadata.json or training_data.jsonl"
        return result

    metadata = read_metadata(metadata_path)
    jsonl = read_jsonl(jsonl_path)

    if metadata is None or jsonl is None:
        result.skipped = True
        result.skip_reason = "file read error"
        return result

    if len(metadata) < 2:
        result.skipped = True
        result.skip_reason = "metadata.json has no chunks"
        return result

    header = metadata[0]
    meta_chunks = metadata[1:]

    if len(meta_chunks) != len(jsonl):
        result.skipped = True
        result.skip_reason = (
            f"count mismatch: metadata={len(meta_chunks)}, jsonl={len(jsonl)}"
        )
        return result

    result.chunks_before = len(meta_chunks)

    new_meta_chunks = []
    new_jsonl_lines = []

    for i, (mc, jl) in enumerate(zip(meta_chunks, jsonl)):
        text = mc.get("text", "")
        decision, cleaned = classify_chunk(text)

        if decision == "DELETE":
            result.chunks_deleted += 1
            result.deleted_indices.append(i)
            continue

        had_marker = SPEAKER_MARKER in text
        if had_marker:
            result.chunks_stripped += 1

        new_mc = {**mc, "text": cleaned}
        new_jl = {**jl, "text": cleaned}
        new_meta_chunks.append(new_mc)
        new_jsonl_lines.append(new_jl)

    if apply and (result.chunks_deleted > 0 or result.chunks_stripped > 0):
        new_metadata = [header] + new_meta_chunks
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(new_metadata, f, ensure_ascii=False, indent=2)

        with open(jsonl_path, "w", encoding="utf-8") as f:
            for entry in new_jsonl_lines:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    return result


def find_all_video_dirs(root: Path) -> list[Path]:
    dirs = []
    for dirpath, _, filenames in os.walk(root):
        if "metadata.json" in filenames or "training_data.jsonl" in filenames:
            dirs.append(Path(dirpath))
    return sorted(dirs)


def main():
    parser = argparse.ArgumentParser(
        description="Clean corpus: delete music/empty chunks and strip '>>' markers."
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write changes to disk. Without this flag, runs as dry-run.",
    )
    args = parser.parse_args()

    if not CORPUS_ROOT.exists():
        print(f"ERROR: Corpus root not found: {CORPUS_ROOT}")
        sys.exit(1)

    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"Mode: {mode}")
    print(f"Scanning: {CORPUS_ROOT}\n")

    video_dirs = find_all_video_dirs(CORPUS_ROOT)
    results: list[DirResult] = []

    for video_dir in video_dirs:
        result = process_video_dir(video_dir, apply=args.apply)
        results.append(result)

        if result.skipped:
            print(f"  SKIP  {result.path}")
            print(f"        Reason: {result.skip_reason}")
            continue

        if result.chunks_deleted == 0 and result.chunks_stripped == 0:
            continue

        status = "WROTE" if args.apply else "WOULD"
        print(f"  {result.path}")
        if result.chunks_deleted:
            indices_preview = result.deleted_indices[:10]
            more = len(result.deleted_indices) - 10
            preview_str = ", ".join(str(i) for i in indices_preview)
            if more > 0:
                preview_str += f" ... +{more} more"
            print(f"    {status} delete {result.chunks_deleted} chunk(s) [indices: {preview_str}]")
        if result.chunks_stripped:
            print(f"    {status} strip '>>' from {result.chunks_stripped} chunk(s)")

    skipped = [r for r in results if r.skipped]
    changed = [r for r in results if not r.skipped and (r.chunks_deleted or r.chunks_stripped)]
    clean = [r for r in results if not r.skipped and not r.chunks_deleted and not r.chunks_stripped]

    total_deleted = sum(r.chunks_deleted for r in results)
    total_stripped = sum(r.chunks_stripped for r in results)
    total_before = sum(r.chunks_before for r in results)

    print()
    print("=" * 60)
    print(f"SUMMARY ({mode})")
    print("=" * 60)
    print(f"  Directories scanned:  {len(results)}")
    print(f"  Directories skipped:  {len(skipped)}")
    print(f"  Directories changed:  {len(changed)}")
    print(f"  Directories clean:    {len(clean)}")
    print()
    print(f"  Total chunks before:  {total_before}")
    print(f"  Chunks deleted:       {total_deleted}  (music/empty)")
    print(f"  Chunks stripped '>>': {total_stripped}")
    print(f"  Chunks after:         {total_before - total_deleted}")

    if not args.apply and (total_deleted or total_stripped):
        print()
        print("  Run with --apply to write changes.")

    REPORTS_DIR.mkdir(exist_ok=True)
    report_name = "clean_report_apply.json" if args.apply else "clean_report_dryrun.json"
    report_path = REPORTS_DIR / report_name
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(
            [
                {
                    "path": r.path,
                    "skipped": r.skipped,
                    "skip_reason": r.skip_reason,
                    "chunks_before": r.chunks_before,
                    "chunks_deleted": r.chunks_deleted,
                    "chunks_stripped": r.chunks_stripped,
                    "deleted_indices": r.deleted_indices,
                }
                for r in results
            ],
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\n  Report saved: {report_path}")


if __name__ == "__main__":
    main()
