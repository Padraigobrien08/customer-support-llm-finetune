#!/usr/bin/env python3
"""
Build frontend thread JSON from training data JSONL.

Reads data/splits/train.jsonl (or provided source) and emits a compact
thread list for the React UI.
"""

import argparse
import json
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def make_title(messages: list[dict[str, Any]]) -> str:
    # Use the first user message as a title anchor
    for msg in messages:
        if msg.get("role") == "user":
            text = msg.get("content", "").strip()
            return text[:48] + ("..." if len(text) > 48 else "")
    return "Conversation"

def normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # Drop system messages and ensure the first message is from the user
    filtered = [m for m in messages if m.get("role") in {"user", "assistant"}]
    while filtered and filtered[0].get("role") != "user":
        filtered.pop(0)
    return filtered


def build_threads(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    threads: list[dict[str, Any]] = []

    for row in rows:
        messages = row.get("messages") or []
        messages = normalize_messages(messages)
        if not messages:
            continue

        metadata = row.get("metadata") or {}
        thread_id = metadata.get("test_case_id") or row.get("id") or f"thread-{len(threads) + 1}"
        thread = {
            "id": thread_id,
            "title": make_title(messages),
            "messages": [
                {
                    "id": f"{thread_id}-m{idx + 1}",
                    "role": msg.get("role", "user"),
                    "content": msg.get("content", ""),
                    "timestamp": ""
                }
                for idx, msg in enumerate(messages)
            ]
        }
        threads.append(thread)
        if len(threads) >= limit:
            break

    return threads


def main() -> None:
    parser = argparse.ArgumentParser(description="Build frontend threads JSON from training data")
    parser.add_argument(
        "--source",
        type=str,
        default="data/splits/train.jsonl",
        help="Path to training JSONL (default: data/splits/train.jsonl)"
    )
    parser.add_argument(
        "--out",
        type=str,
        default="frontend/src/data/training_threads.json",
        help="Output JSON path (default: frontend/src/data/training_threads.json)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Max threads to export (default: 20)"
    )
    args = parser.parse_args()

    source_path = Path(args.source)
    out_path = Path(args.out)

    if not source_path.exists():
        raise SystemExit(f"Source file not found: {source_path}")

    rows = load_jsonl(source_path)
    threads = build_threads(rows, args.limit)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(threads, f, indent=2, ensure_ascii=False)

    print(f"✓ Wrote {len(threads)} threads to {out_path}")


if __name__ == "__main__":
    main()
