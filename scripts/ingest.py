"""Build the Chroma pattern index.

    python scripts/ingest.py                 # seed corpus only
    python scripts/ingest.py --csv path.csv  # seed corpus + a labeled CSV

The seed corpus always loads, so the demo never depends on a dataset download
succeeding. A CSV is additive and optional; it needs a text column and a label
column (auto-detected from common names).
"""

import argparse
import csv
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phishguard import config  # noqa: E402
from phishguard.vectorstore import get_client, get_collection, load_seed_patterns  # noqa: E402

TEXT_COLUMNS = ["text", "body", "email_text", "message", "content", "email", "text_combined"]
LABEL_COLUMNS = ["label", "class", "type", "is_phishing", "target", "spam"]
PHISH_VALUES = {"1", "phishing", "phish", "spam", "malicious", "fraud", "true", "yes"}


def detect_column(header: list[str], candidates: list[str]) -> str | None:
    lowered = {h.lower().strip(): h for h in header}
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]
    return None


def chunk_text(text: str, max_chars: int = 600) -> list[str]:
    """Split on sentence boundaries, packing into chunks of roughly max_chars."""
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return [text] if text else []
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks, current = [], ""
    for sentence in sentences:
        if len(current) + len(sentence) + 1 > max_chars and current:
            chunks.append(current.strip())
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        chunks.append(current.strip())
    return [c for c in chunks if len(c) > 40]


def load_csv(path: Path, limit: int) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            print(f"  ! {path.name} has no header row; skipping.")
            return []
        text_column = detect_column(reader.fieldnames, TEXT_COLUMNS)
        label_column = detect_column(reader.fieldnames, LABEL_COLUMNS)
        if not text_column:
            print(f"  ! No text column found in {path.name}. Looked for: {TEXT_COLUMNS}")
            print(f"    Columns present: {reader.fieldnames}")
            return []
        print(f"  text column = '{text_column}', label column = '{label_column or 'n/a'}'")

        for index, row in enumerate(reader):
            if len(rows) >= limit:
                break
            body = (row.get(text_column) or "").strip()
            if len(body) < 60:
                continue
            raw_label = (row.get(label_column) or "").strip().lower() if label_column else ""
            label = "phishing" if raw_label in PHISH_VALUES else "legitimate"
            for part, chunk in enumerate(chunk_text(body)):
                rows.append(
                    {
                        "id": f"CSV-{index:06d}-{part}",
                        "text": chunk,
                        "label": label,
                        "pattern": f"dataset_{label}",
                        "source": path.name,
                    }
                )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the PhishGuard Chroma index.")
    parser.add_argument("--csv", type=Path, help="Optional labeled phishing CSV to add.")
    parser.add_argument("--limit", type=int, default=2000, help="Max chunks from the CSV.")
    parser.add_argument("--reset", action="store_true", help="Drop the existing collection first.")
    args = parser.parse_args()

    config.ensure_dirs()

    if args.reset:
        try:
            get_client().delete_collection(config.CHROMA_COLLECTION)
            print("Dropped existing collection.")
        except Exception:
            pass

    records: list[dict] = []

    seed = load_seed_patterns()
    for row in seed:
        records.append(
            {
                "id": row["id"],
                "text": row["text"],
                "label": row["label"],
                "pattern": row["pattern"],
                "source": "seed_patterns.jsonl",
            }
        )
    print(f"Seed corpus: {len(seed)} chunks from {config.SEED_PATTERNS.name}")

    if args.csv:
        if args.csv.exists():
            print(f"Loading CSV: {args.csv}")
            csv_rows = load_csv(args.csv, args.limit)
            records.extend(csv_rows)
            print(f"  added {len(csv_rows)} chunks")
        else:
            print(f"  ! {args.csv} not found — continuing with the seed corpus only.")

    if not records:
        print("Nothing to ingest. Is data/seed_patterns.jsonl present?")
        return 1

    collection = get_collection(create=True)
    if collection is None:
        print("Could not open the Chroma collection.")
        return 1

    print("\nEmbedding (first run downloads all-MiniLM-L6-v2, ~80MB)…")
    batch = 128
    for start in range(0, len(records), batch):
        part = records[start : start + batch]
        try:
            collection.upsert(
                ids=[r["id"] for r in part],
                documents=[r["text"] for r in part],
                metadatas=[
                    {"label": r["label"], "pattern": r["pattern"], "source": r["source"]}
                    for r in part
                ],
            )
        except Exception as exc:
            print(f"\n  ! Embedding failed: {type(exc).__name__}: {exc}")
            print(
                "\n  The embedding model could not be downloaded or loaded. The app still\n"
                "  runs: the Retriever node reports retrieval as UNAVAILABLE, which lowers\n"
                "  confidence and pushes borderline cases to human review rather than\n"
                "  silently scoring them as clean.\n"
                "  Re-run this script once the network is available."
            )
            return 1
        print(f"  embedded {min(start + batch, len(records))}/{len(records)}")

    phishing = sum(1 for r in records if r["label"] == "phishing")
    print(
        f"\nDone. {collection.count()} chunks in '{config.CHROMA_COLLECTION}' "
        f"({phishing} phishing / {len(records) - phishing} legitimate)."
    )
    print(f"Index path: {config.CHROMA_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
