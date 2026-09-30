#!/usr/bin/env python3
"""Download papers listed in papers.yaml (PDF by default; optional TeX source).

Usage
-----
  python scripts/download_catalog.py --track base_models
  python scripts/download_catalog.py --slug ldm
  python scripts/download_catalog.py --all
  python scripts/download_catalog.py --all --with-tex
  python scripts/download_catalog.py --all --dry-run

Notes
-----
- Skips track ``foundations`` by default (already local under arxiv/foundations/).
  Pass ``--include-foundations`` to download/refresh those too.
- PDF → arxiv/<track>/<slug>/paper.pdf
- TeX → arxiv/<track>/<slug>/source.tar.gz (+ extracted/ if unzip succeeds)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    import yaml
except ImportError:
    print("Please: pip install pyyaml", file=sys.stderr)
    sys.exit(1)

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "papers.yaml"
UA = "PaperReadingBot/1.0 (research; +https://arxiv.org)"


def load_catalog() -> dict:
    return yaml.safe_load(CATALOG.read_text(encoding="utf-8"))


def load_papers() -> list[dict]:
    return load_catalog().get("papers", [])


def known_tracks() -> list[str]:
    tracks = load_catalog().get("tracks") or {}
    return sorted(tracks.keys()) if tracks else []


def select(
    papers: list[dict],
    track: str | None,
    slug: str | None,
    all_: bool,
    include_foundations: bool,
) -> list[dict]:
    out = []
    for p in papers:
        t = p.get("track")
        if t == "foundations" and not include_foundations:
            # still allow explicit --slug / --track foundations
            if not (slug == p.get("slug") or track == "foundations"):
                continue
        if slug and p.get("slug") != slug:
            continue
        if track and t != track:
            continue
        if not (all_ or track or slug):
            continue
        out.append(p)
    return out


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    cmd = [
        "curl",
        "-4",
        "-sS",
        "-L",
        "-C",
        "-",
        "--connect-timeout",
        "20",
        "--max-time",
        "0",
        "-A",
        UA,
        "-o",
        str(tmp),
        url,
    ]
    try:
        subprocess.run(cmd, check=True)
        if not tmp.exists() or tmp.stat().st_size == 0:
            raise RuntimeError(f"empty download: {url}")
        tmp.replace(dest)
    except (subprocess.CalledProcessError, FileNotFoundError):
        last_err: Exception | None = None
        for attempt in range(3):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": UA})
                with urllib.request.urlopen(req, timeout=180) as resp:
                    dest.write_bytes(resp.read())
                return
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                last_err = e
                time.sleep(2 * (attempt + 1))
        raise last_err or RuntimeError(f"download failed: {url}")


def try_extract_tex(tarball: Path, out_dir: Path) -> bool:
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tarball, "r:gz") as tf:
            tf.extractall(out_dir)
        return True
    except (tarfile.TarError, OSError) as e:
        print(f"  [warn] TeX extract failed: {e}")
        return False


def process_one(p: dict, with_tex: bool, dry_run: bool, force: bool, tex_only: bool = False) -> None:
    track, slug, arxiv = p["track"], p["slug"], str(p["arxiv"])
    base = ROOT / "arxiv" / track / slug
    pdf_path = base / "paper.pdf"
    print(f"\n[{track}/{slug}] arXiv:{arxiv} — {p['title'][:60]}", flush=True)

    if dry_run:
        print(f"  would download → {pdf_path}", flush=True)
        return

    if not tex_only:
        if pdf_path.exists() and not force:
            print(f"  skip PDF (exists): {pdf_path.relative_to(ROOT)}", flush=True)
        else:
            url = f"https://arxiv.org/pdf/{arxiv}.pdf"
            print(f"  downloading PDF {url}", flush=True)
            try:
                download(url, pdf_path)
                print(f"  OK {pdf_path.relative_to(ROOT)}", flush=True)
            except Exception as e:
                print(f"  [error] PDF {e}: {url}", flush=True)
                if pdf_path.exists() and pdf_path.stat().st_size == 0:
                    pdf_path.unlink()

    if with_tex or tex_only:
        tex_path = base / "source.tar.gz"
        extracted = base / "extracted"
        if tex_path.exists() and not force:
            print(f"  skip TeX (exists): {tex_path.relative_to(ROOT)}", flush=True)
            if not extracted.exists() and try_extract_tex(tex_path, extracted):
                print(f"  OK extracted → {extracted.relative_to(ROOT)}", flush=True)
        else:
            url = f"https://arxiv.org/e-print/{arxiv}"
            print(f"  downloading TeX {url}", flush=True)
            try:
                download(url, tex_path)
                if try_extract_tex(tex_path, extracted):
                    print(f"  OK extracted → {extracted.relative_to(ROOT)}", flush=True)
                else:
                    print(f"  saved tarball only: {tex_path.relative_to(ROOT)}", flush=True)
            except Exception as e:
                print(f"  [warn] TeX unavailable ({e}), PDF-only is fine", flush=True)
                if tex_path.exists() and tex_path.stat().st_size == 0:
                    tex_path.unlink()


def main() -> None:
    tracks = known_tracks()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--track",
        choices=tracks if tracks else None,
        help="track name from papers.yaml",
    )
    ap.add_argument("--slug", help="single paper slug, e.g. ldm")
    ap.add_argument("--all", action="store_true", help="all papers except foundations (unless flagged)")
    ap.add_argument(
        "--include-foundations",
        action="store_true",
        help="include foundations track when using --all",
    )
    ap.add_argument("--with-tex", action="store_true", help="also fetch arXiv e-print source")
    ap.add_argument("--tex-only", action="store_true", help="fetch e-print source only (skip PDF)")
    ap.add_argument("--jobs", type=int, default=1, help="parallel downloads (default 1)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="re-download even if file exists")
    args = ap.parse_args()

    if not (args.all or args.track or args.slug):
        ap.error("specify --all, --track, or --slug")

    papers = select(
        load_papers(),
        args.track,
        args.slug,
        args.all,
        include_foundations=args.include_foundations,
    )
    if not papers:
        print("No papers matched.", file=sys.stderr)
        sys.exit(1)

    print(f"Catalog: {CATALOG.relative_to(ROOT)}", flush=True)
    print(f"Selected: {len(papers)} papers", flush=True)
    kwargs = dict(
        with_tex=args.with_tex,
        dry_run=args.dry_run,
        force=args.force,
        tex_only=args.tex_only,
    )
    if args.jobs <= 1:
        for p in papers:
            process_one(p, **kwargs)
    else:
        with ThreadPoolExecutor(max_workers=args.jobs) as ex:
            futs = [ex.submit(process_one, p, **kwargs) for p in papers]
            for fut in as_completed(futs):
                fut.result()
    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
