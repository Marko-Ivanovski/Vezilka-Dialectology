import json
import os
import sys
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


CORPUS_ROOT = Path(__file__).parent.parent / "govori"
REPORTS_DIR = Path(__file__).parent.parent / "reports"

VALID_GOVORI = {
    "skopski", "bitolski", "ohridski", "strushki", "prilepski",
    "veleshki", "kichevsko-porechki", "gevgeliski", "kumanovski",
    "ovchepolski", "shtipski", "kochanski", "tetovski", "kratovski",
    "krivopalanechki", "maleshevski", "strumichki", "kavadrechki",
    "negotinski", "valandovski", "debarski", "galichki", "gostivarski",
    "prespanski", "literaturen",
}

MIN_TEXT_LENGTH = 15
MIN_WAV_SIZE_BYTES = 5000
SPEAKER_MARKER = ">>"


@dataclass
class ChunkIssue:
    chunk_index: int
    audio_file: str
    issue: str
    detail: str = ""


@dataclass
class VideoReport:
    path: str
    govor: str = ""
    total_chunks: int = 0
    passed: bool = True
    issues: list = field(default_factory=list)

    def add_issue(self, issue: ChunkIssue):
        self.issues.append(issue)
        self.passed = False


def validate_metadata_structure(metadata: list, report: VideoReport, video_dir: Path):
    if not isinstance(metadata, list) or len(metadata) < 2:
        report.add_issue(ChunkIssue(
            chunk_index=-1, audio_file="",
            issue="STRUCTURE", detail="metadata.json must be a list with header + chunks"
        ))
        return []

    header = metadata[0]
    if "govor" not in header:
        report.add_issue(ChunkIssue(
            chunk_index=-1, audio_file="",
            issue="MISSING_GOVOR", detail="First element missing 'govor' key"
        ))
    else:
        report.govor = header["govor"]
        if header["govor"] not in VALID_GOVORI:
            report.add_issue(ChunkIssue(
                chunk_index=-1, audio_file="",
                issue="UNKNOWN_GOVOR", detail=f"'{header['govor']}' not in known dialects"
            ))

    chunks = metadata[1:]
    report.total_chunks = len(chunks)
    required_fields = {"audio_file", "text", "start_ms", "end_ms", "duration_s"}

    for i, chunk in enumerate(chunks):
        missing = required_fields - set(chunk.keys())
        if missing:
            report.add_issue(ChunkIssue(
                chunk_index=i, audio_file=chunk.get("audio_file", "?"),
                issue="MISSING_FIELDS", detail=f"Missing: {missing}"
            ))
            continue

        expected_duration = (chunk["end_ms"] - chunk["start_ms"]) / 1000
        if abs(chunk["duration_s"] - expected_duration) > 0.1:
            report.add_issue(ChunkIssue(
                chunk_index=i, audio_file=chunk["audio_file"],
                issue="DURATION_MISMATCH",
                detail=f"duration_s={chunk['duration_s']} but end-start={expected_duration}"
            ))

    for i in range(1, len(chunks)):
        if chunks[i].get("start_ms", 0) < chunks[i - 1].get("start_ms", 0):
            report.add_issue(ChunkIssue(
                chunk_index=i, audio_file=chunks[i].get("audio_file", "?"),
                issue="NON_MONOTONIC", detail="start_ms decreased from previous chunk"
            ))

    return chunks


def validate_jsonl(jsonl_path: Path, report: VideoReport) -> list:
    lines = []
    try:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError as e:
                    report.add_issue(ChunkIssue(
                        chunk_index=line_num, audio_file="?",
                        issue="JSONL_PARSE_ERROR", detail=str(e)
                    ))
                    continue

                if "audio" not in entry or "text" not in entry:
                    report.add_issue(ChunkIssue(
                        chunk_index=line_num,
                        audio_file=entry.get("audio", "?"),
                        issue="JSONL_MISSING_FIELDS",
                        detail=f"Keys present: {list(entry.keys())}"
                    ))
                lines.append(entry)
    except Exception as e:
        report.add_issue(ChunkIssue(
            chunk_index=-1, audio_file="",
            issue="JSONL_READ_ERROR", detail=str(e)
        ))
    return lines


def check_audio_files(chunks: list, video_dir: Path, report: VideoReport):
    audio_dir = video_dir / "audio"
    for i, chunk in enumerate(chunks):
        audio_file = chunk.get("audio_file", "")
        wav_path = audio_dir / audio_file
        if not wav_path.exists():
            report.add_issue(ChunkIssue(
                chunk_index=i, audio_file=audio_file,
                issue="MISSING_AUDIO", detail=f"{wav_path} not found"
            ))
        elif wav_path.stat().st_size < MIN_WAV_SIZE_BYTES:
            report.add_issue(ChunkIssue(
                chunk_index=i, audio_file=audio_file,
                issue="TINY_AUDIO",
                detail=f"Only {wav_path.stat().st_size} bytes (likely silence/corrupt)"
            ))


def check_text_quality(chunks: list, report: VideoReport):
    for i, chunk in enumerate(chunks):
        text = chunk.get("text", "")

        if len(text.strip()) < MIN_TEXT_LENGTH:
            report.add_issue(ChunkIssue(
                chunk_index=i, audio_file=chunk.get("audio_file", "?"),
                issue="SHORT_TEXT",
                detail=f"Only {len(text.strip())} chars: '{text.strip()[:50]}'"
            ))
            continue

        words = text.replace(SPEAKER_MARKER, "").split()
        if len(words) < 3:
            report.add_issue(ChunkIssue(
                chunk_index=i, audio_file=chunk.get("audio_file", "?"),
                issue="FEW_WORDS",
                detail=f"Only {len(words)} words after removing '>>' markers"
            ))

        marker_count = text.count(SPEAKER_MARKER)
        word_count = len(words)
        if word_count > 0 and marker_count / word_count > 0.3:
            report.add_issue(ChunkIssue(
                chunk_index=i, audio_file=chunk.get("audio_file", "?"),
                issue="HIGH_MARKER_RATIO",
                detail=f"{marker_count} markers vs {word_count} words — possibly fragmented/noisy"
            ))

        music_indicators = ["[музика]", "[music]", "♪", "♫", "[аплауз]", "[applause]"]
        text_lower = text.lower()
        for indicator in music_indicators:
            if indicator in text_lower:
                report.add_issue(ChunkIssue(
                    chunk_index=i, audio_file=chunk.get("audio_file", "?"),
                    issue="MUSIC_MARKER", detail=f"Contains '{indicator}'"
                ))
                break


def check_count_consistency(metadata_chunks: list, jsonl_lines: list, report: VideoReport):
    if len(metadata_chunks) != len(jsonl_lines):
        report.add_issue(ChunkIssue(
            chunk_index=-1, audio_file="",
            issue="COUNT_MISMATCH",
            detail=f"metadata has {len(metadata_chunks)} chunks, JSONL has {len(jsonl_lines)} lines"
        ))


def validate_video_dir(video_dir: Path) -> Optional[VideoReport]:
    metadata_path = video_dir / "metadata.json"
    jsonl_path = video_dir / "training_data.jsonl"

    has_metadata = metadata_path.exists()
    has_jsonl = jsonl_path.exists()

    if not has_metadata and not has_jsonl:
        return None

    report = VideoReport(path=str(video_dir.relative_to(CORPUS_ROOT)))

    if not has_metadata:
        report.add_issue(ChunkIssue(
            chunk_index=-1, audio_file="",
            issue="MISSING_FILE", detail="metadata.json not found"
        ))
    if not has_jsonl:
        report.add_issue(ChunkIssue(
            chunk_index=-1, audio_file="",
            issue="MISSING_FILE", detail="training_data.jsonl not found"
        ))

    metadata_chunks = []
    if has_metadata:
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            metadata_chunks = validate_metadata_structure(metadata, report, video_dir)
        except json.JSONDecodeError as e:
            report.add_issue(ChunkIssue(
                chunk_index=-1, audio_file="",
                issue="METADATA_PARSE_ERROR", detail=str(e)
            ))

    jsonl_lines = []
    if has_jsonl:
        jsonl_lines = validate_jsonl(jsonl_path, report)

    if metadata_chunks and jsonl_lines:
        check_count_consistency(metadata_chunks, jsonl_lines, report)

    chunks_to_check = metadata_chunks if metadata_chunks else jsonl_lines
    if chunks_to_check:
        check_audio_files(chunks_to_check, video_dir, report)
        check_text_quality(chunks_to_check, report)

    return report


def find_all_video_dirs(root: Path) -> list[Path]:
    dirs = []
    for dirpath, dirnames, filenames in os.walk(root):
        if "metadata.json" in filenames or "training_data.jsonl" in filenames:
            dirs.append(Path(dirpath))
    return sorted(dirs)


def main():
    if not CORPUS_ROOT.exists():
        print(f"ERROR: Corpus root not found: {CORPUS_ROOT}")
        sys.exit(1)

    print(f"Scanning: {CORPUS_ROOT}")
    video_dirs = find_all_video_dirs(CORPUS_ROOT)
    print(f"Found {len(video_dirs)} video directories\n")

    reports: list[VideoReport] = []
    for video_dir in video_dirs:
        report = validate_video_dir(video_dir)
        if report:
            reports.append(report)

    passed = [r for r in reports if r.passed]
    failed = [r for r in reports if not r.passed]

    total_chunks = sum(r.total_chunks for r in reports)
    total_issues = sum(len(r.issues) for r in reports)

    print("=" * 60)
    print(f"VALIDATION SUMMARY")
    print("=" * 60)
    print(f"  Videos scanned:  {len(reports)}")
    print(f"  Total chunks:    {total_chunks}")
    print(f"  Passed:          {len(passed)}")
    print(f"  Failed:          {len(failed)}")
    print(f"  Total issues:    {total_issues}")
    print()

    issue_types: dict[str, int] = {}
    for r in reports:
        for issue in r.issues:
            issue_types[issue.issue] = issue_types.get(issue.issue, 0) + 1

    if issue_types:
        print("Issues by type:")
        for issue_type, count in sorted(issue_types.items(), key=lambda x: -x[1]):
            print(f"  {issue_type}: {count}")
        print()

    if failed:
        print("-" * 60)
        print("FAILED VIDEOS:")
        print("-" * 60)
        for r in failed:
            print(f"\n  {r.path} [{r.govor}] ({r.total_chunks} chunks)")
            for issue in r.issues:
                chunk_str = f"chunk {issue.chunk_index}" if issue.chunk_index >= 0 else "file-level"
                print(f"    [{issue.issue}] {chunk_str}: {issue.detail}")

    REPORTS_DIR.mkdir(exist_ok=True)
    output_path = REPORTS_DIR / "validation_report.json"
    output_data = {
        "summary": {
            "videos_scanned": len(reports),
            "total_chunks": total_chunks,
            "passed": len(passed),
            "failed": len(failed),
            "total_issues": total_issues,
            "issue_types": issue_types,
        },
        "reports": [
            {
                "path": r.path,
                "govor": r.govor,
                "total_chunks": r.total_chunks,
                "passed": r.passed,
                "issues": [asdict(i) for i in r.issues],
            }
            for r in reports
        ],
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\nFull report saved to: {output_path}")
    sys.exit(0 if not failed else 1)


if __name__ == "__main__":
    main()
