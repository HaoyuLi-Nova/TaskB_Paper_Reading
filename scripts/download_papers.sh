#!/usr/bin/env bash
# Re-download arXiv TeX sources for Task B papers.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PAPERS_DIR="$ROOT/papers"
MANIFEST="$PAPERS_DIR/papers_manifest.txt"

if [[ ! -f "$MANIFEST" ]]; then
  echo "Missing manifest: $MANIFEST" >&2
  exit 1
fi

while IFS='|' read -r dir arxiv title; do
  [[ "$dir" =~ ^# ]] && continue
  [[ -z "$dir" ]] && continue
  echo "=== $title ($arxiv) ==="
  out_dir="$PAPERS_DIR/$dir"
  mkdir -p "$out_dir/extracted"
  curl -fsSL -o "$out_dir/source.tar.gz" "https://arxiv.org/e-print/${arxiv}"
  rm -rf "$out_dir/extracted"/*
  tar -xzf "$out_dir/source.tar.gz" -C "$out_dir/extracted"
  echo "OK: $out_dir/extracted"
done < "$MANIFEST"

echo "All done."
