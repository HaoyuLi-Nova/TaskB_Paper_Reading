#!/usr/bin/env python3
"""Convert every figure under ``extracted/`` to PNG in place (PyMuPDF + Pillow).

Scans only ``**/extracted/**``. Never touches top-level ``paper.pdf`` /
``source.tar.gz``. Writes PNGs next to sources (same directory); originals
are kept unless ``--remove-source``.

Formats
-------
  PDF  → PNG via **PyMuPDF** (auto DPI: embedded rasters 1:1, vectors 600)
  JPG / JPEG / WEBP / BMP / TIFF / GIF → PNG via **Pillow**
  EPS / PS → PNG via **Pillow** (needs Ghostscript) or ``gs`` fallback

Non-figure PDFs (LaTeX templates / style docs) are skipped by name heuristics.

Usage
-----
  # All extracted/ trees under arxiv/ (recommended)
  python scripts/figures_to_png.py
  python scripts/figures_to_png.py arxiv/audio_driven --force

  # One paper or one file
  python scripts/figures_to_png.py arxiv/audio_driven/omnihuman/extracted
  python scripts/figures_to_png.py arxiv/base_models/ldm/extracted/img/foo.pdf

  # Force fixed PDF DPI instead of auto
  python scripts/figures_to_png.py arxiv/audio_driven --dpi 300
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image, ImageChops

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SEARCH_ROOT = ROOT / "arxiv"

# Raster / vector containers we convert → PNG (never convert .png → .png)
RASTER_EXTS = {".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".gif"}
VECTOR_EXTS = {".eps", ".ps"}
PDF_EXTS = {".pdf"}
SOURCE_EXTS = PDF_EXTS | RASTER_EXTS | VECTOR_EXTS

DEFAULT_FALLBACK_DPI = 600
DEFAULT_EPS_DPI = 300
CONTENT_PAD_PT = 2.0
WHITE_DIFF_THRESH = 5
# Cap pixmap size so MuPDF does not raise FzErrorLimit ("Overly large image")
# when one tiny-placed high-res embed inflates whole-page zoom.
MAX_PIXMAP_SIDE = 8192
MAX_PIXMAP_PIXELS = 64_000_000  # ~64 MP ≈ 192 MB RGB
MIN_EMBED_PLACEMENT_PT = 8.0  # ignore thin/tiny placements for native DPI
MAX_EMBED_ASPECT = 12.0  # ignore extreme strip-like image placements

# Top-level assets that are never figures
SKIP_NAMES = {"paper.pdf", "source.tar.gz"}

# LaTeX / publisher template docs that land in extracted/ but are not paper figures
SKIP_NAME_RES = (
    re.compile(r"^llncs", re.I),
    re.compile(r"^template", re.I),
    re.compile(r"^sample[-_]?(paper|sigconf|article)", re.I),
    re.compile(r"^acmart", re.I),
    re.compile(r"^ieeeconf", re.I),
    re.compile(r"^svg[-_]?inkscape", re.I),
)


def page_content_clip(
    page: fitz.Page,
    pad_pt: float = CONTENT_PAD_PT,
    white_diff_thresh: int = WHITE_DIFF_THRESH,
) -> fitz.Rect:
    """Tight clip around non-white page content (TeX MediaBox often full-page)."""
    probe = page.get_pixmap(matrix=fitz.Identity, alpha=False)
    img = Image.frombytes("RGB", (probe.width, probe.height), probe.samples)
    white = Image.new("RGB", img.size, (255, 255, 255))
    diff = ImageChops.difference(img, white).convert("L")
    mask = diff.point(lambda p: 255 if p > white_diff_thresh else 0)
    bbox = mask.getbbox()
    if bbox is None:
        return fitz.Rect(page.rect)

    left, top, right, bottom = bbox
    clip = fitz.Rect(left, top, right, bottom)
    clip.x0 -= pad_pt
    clip.y0 -= pad_pt
    clip.x1 += pad_pt
    clip.y1 += pad_pt
    return clip & page.rect


def native_render_zoom(page: fitz.Page, clip: fitz.Rect | None = None) -> float | None:
    """Zoom so embedded rasters map ≥1 device pixel per source pixel.

    Ignores tiny / extreme-aspect placements (common in comparison strips)
    so one banner does not inflate whole-page DPI into MuPDF limits.
    """
    zooms: list[float] = []
    try:
        infos = page.get_image_info(xrefs=True)
    except Exception:  # noqa: BLE001
        infos = []

    clip_rect = fitz.Rect(clip) if clip is not None else None
    for info in infos:
        bbox = fitz.Rect(info["bbox"])
        if clip_rect is not None:
            bbox = bbox & clip_rect
            if bbox.is_empty or bbox.is_infinite:
                continue
        if bbox.width < MIN_EMBED_PLACEMENT_PT or bbox.height < MIN_EMBED_PLACEMENT_PT:
            continue
        aspect = max(bbox.width / bbox.height, bbox.height / bbox.width)
        if aspect > MAX_EMBED_ASPECT:
            continue
        pw = float(info["width"])
        ph = float(info["height"])
        full = fitz.Rect(info["bbox"])
        if full.width > 0 and bbox.width > 0 and bbox.width < full.width:
            pw *= bbox.width / full.width
        if full.height > 0 and bbox.height > 0 and bbox.height < full.height:
            ph *= bbox.height / full.height
        if bbox.width > 0 and pw > 0:
            zooms.append(pw / bbox.width)
        if bbox.height > 0 and ph > 0:
            zooms.append(ph / bbox.height)

    if not zooms:
        return None
    return max(zooms)


def clamp_zoom(zoom: float, clip: fitz.Rect) -> float:
    """Shrink zoom so rendered pixmap stays within MuPDF / memory limits."""
    if clip.width <= 0 or clip.height <= 0:
        return zoom
    z = zoom
    z = min(z, MAX_PIXMAP_SIDE / clip.width, MAX_PIXMAP_SIDE / clip.height)
    area = clip.width * clip.height
    if area > 0:
        z = min(z, (MAX_PIXMAP_PIXELS / area) ** 0.5)
    return max(z, 1.0)  # at least 72 dpi


def resolve_zoom(
    page: fitz.Page,
    dpi: int | None,
    fallback_dpi: int,
    clip: fitz.Rect | None = None,
) -> tuple[float, str]:
    if dpi is not None:
        z = dpi / 72.0
        reason = f"fixed {dpi} dpi"
    else:
        native = native_render_zoom(page, clip=clip)
        if native is not None:
            z, reason = native, f"native ~{native * 72:.0f} dpi"
        else:
            z, reason = fallback_dpi / 72.0, f"vector fallback {fallback_dpi} dpi"

    if clip is not None:
        capped = clamp_zoom(z, clip)
        if capped < z * 0.999:
            reason = f"{reason} → capped ~{capped * 72:.0f} dpi"
            z = capped
    return z, reason


def pdf_to_png(
    pdf_path: Path,
    dpi: int | None = None,
    fallback_dpi: int = DEFAULT_FALLBACK_DPI,
    force: bool = False,
) -> list[Path]:
    pdf_path = pdf_path.resolve()
    out_dir = pdf_path.parent
    stem = pdf_path.stem
    written: list[Path] = []

    with fitz.open(pdf_path) as doc:
        page_count = doc.page_count
        for page_index in range(page_count):
            out_path = (
                out_dir / f"{stem}.png"
                if page_count == 1
                else out_dir / f"{stem}-{page_index + 1}.png"
            )
            if out_path.exists() and not force:
                written.append(out_path)
                continue
            page = doc.load_page(page_index)
            clip = page_content_clip(page)
            zoom, _reason = resolve_zoom(
                page, dpi=dpi, fallback_dpi=fallback_dpi, clip=clip
            )
            pix = None
            for _attempt in range(6):
                try:
                    pix = page.get_pixmap(
                        matrix=fitz.Matrix(zoom, zoom), clip=clip, alpha=False
                    )
                    break
                except Exception as exc:  # noqa: BLE001 — FzErrorLimit / soft limit
                    msg = str(exc).lower()
                    if "overly large" not in msg and "limit" not in msg:
                        raise
                    zoom = clamp_zoom(zoom * 0.5, clip)
            if pix is None:
                raise RuntimeError(f"could not render within pixmap limits: {pdf_path}")
            pix.save(str(out_path))
            written.append(out_path)
    return written


def _save_pil_as_png(im: Image.Image, out_path: Path) -> None:
    if im.mode in ("RGBA", "LA"):
        im = im.convert("RGBA")
    elif im.mode == "P":
        im = im.convert("RGBA" if "transparency" in im.info else "RGB")
    else:
        im = im.convert("RGB")
    im.save(out_path, format="PNG")


def raster_to_png(src: Path, force: bool = False) -> list[Path]:
    """Lossless container convert; keep original pixel dimensions."""
    src = src.resolve()
    out_path = src.with_suffix(".png")
    if out_path.exists() and not force:
        return [out_path]
    with Image.open(src) as im:
        _save_pil_as_png(im, out_path)
    return [out_path]


def eps_to_png_ghostscript(src: Path, out_path: Path, dpi: int) -> None:
    gs = shutil.which("gs")
    if not gs:
        raise RuntimeError("Ghostscript (gs) not found; required for EPS/PS")
    cmd = [
        gs,
        "-dSAFER",
        "-dBATCH",
        "-dNOPAUSE",
        "-dEPSCrop",
        f"-r{dpi}",
        "-sDEVICE=png16m",
        f"-sOutputFile={out_path}",
        str(src),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not out_path.exists():
        err = (proc.stderr or proc.stdout or "").strip()[-500:]
        raise RuntimeError(f"gs failed ({proc.returncode}): {err}")


def eps_to_png(src: Path, dpi: int = DEFAULT_EPS_DPI, force: bool = False) -> list[Path]:
    """EPS/PS → PNG. Prefer Pillow (uses gs); fall back to explicit gs."""
    src = src.resolve()
    out_path = src.with_suffix(".png")
    if out_path.exists() and not force:
        return [out_path]

    # Pillow EPS path (requires Ghostscript backend)
    try:
        with Image.open(src) as im:
            # load() triggers Ghostscript rasterization
            im.load()
            _save_pil_as_png(im, out_path)
        return [out_path]
    except Exception:  # noqa: BLE001
        pass

    eps_to_png_ghostscript(src, out_path, dpi=dpi)
    return [out_path]


def convert_one(
    path: Path,
    dpi: int | None,
    fallback_dpi: int,
    eps_dpi: int,
    force: bool,
) -> list[Path]:
    suf = path.suffix.lower()
    if suf in PDF_EXTS:
        return pdf_to_png(path, dpi=dpi, fallback_dpi=fallback_dpi, force=force)
    if suf in RASTER_EXTS:
        return raster_to_png(path, force=force)
    if suf in VECTOR_EXTS:
        return eps_to_png(path, dpi=eps_dpi, force=force)
    raise ValueError(f"Unsupported figure type: {path}")


def is_skipped_name(path: Path) -> bool:
    name = path.name.lower()
    if name in SKIP_NAMES:
        return True
    stem = path.stem
    return any(rx.search(stem) for rx in SKIP_NAME_RES)


def is_under_extracted(path: Path) -> bool:
    return "extracted" in path.parts


def is_figure_source(path: Path) -> bool:
    if not path.is_file() or path.suffix.lower() not in SOURCE_EXTS:
        return False
    if is_skipped_name(path):
        return False
    return is_under_extracted(path)


def figure_search_roots(root: Path) -> list[Path]:
    """Directories to scan: only ``extracted/`` trees."""
    if root.name == "extracted" or "extracted" in root.parts:
        return [root]
    return sorted(p for p in root.rglob("extracted") if p.is_dir())


def discover(root: Path) -> list[Path]:
    files: list[Path] = []
    for search in figure_search_roots(root):
        for path in sorted(search.rglob("*")):
            if is_figure_source(path):
                files.append(path)
    return files


def maybe_remove_source(src: Path, outputs: list[Path], remove: bool) -> None:
    if not remove:
        return
    # Never delete if conversion produced nothing usable
    if not outputs or not all(p.exists() for p in outputs):
        return
    if src.resolve() in {p.resolve() for p in outputs}:
        return
    src.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Figure files or dirs (default: arxiv/ — only extracted/ is scanned)",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_SEARCH_ROOT,
        help=f"Default scan root when no paths given (default: {DEFAULT_SEARCH_ROOT})",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=None,
        help="Force fixed PDF render DPI (default: auto native / vector fallback)",
    )
    parser.add_argument(
        "--fallback-dpi",
        type=int,
        default=DEFAULT_FALLBACK_DPI,
        help=f"DPI for vector-only PDFs when --dpi omitted (default: {DEFAULT_FALLBACK_DPI})",
    )
    parser.add_argument(
        "--eps-dpi",
        type=int,
        default=DEFAULT_EPS_DPI,
        help=f"DPI for EPS/PS via Ghostscript fallback (default: {DEFAULT_EPS_DPI})",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing PNG")
    parser.add_argument(
        "--remove-source",
        action="store_true",
        help="Delete source after successful PNG write (default: keep originals)",
    )
    args = parser.parse_args(argv)

    scan_root = args.root if args.root.is_absolute() else ROOT / args.root
    targets: list[Path] = []
    if args.paths:
        for p in args.paths:
            p = p if p.is_absolute() else ROOT / p
            if p.is_dir():
                targets.extend(discover(p))
            elif is_figure_source(p):
                targets.append(p)
            elif p.is_file() and is_skipped_name(p):
                print(f"Skip (non-figure / full paper): {p}", file=sys.stderr)
            elif p.is_file() and p.suffix.lower() in SOURCE_EXTS:
                print(f"Skip (outside extracted/): {p}", file=sys.stderr)
            else:
                print(f"Skip: {p}", file=sys.stderr)
    else:
        targets = discover(scan_root)

    # De-dupe while preserving order
    seen: set[Path] = set()
    uniq: list[Path] = []
    for t in targets:
        key = t.resolve()
        if key not in seen:
            seen.add(key)
            uniq.append(t)
    targets = uniq

    if not targets:
        print("No convertible figures found under extracted/.")
        return 1

    print(f"Found {len(targets)} figure source(s) under extracted/.\n")

    ok, skipped, fail = 0, 0, 0
    for src in targets:
        rel = src.relative_to(ROOT) if src.is_relative_to(ROOT) else src
        try:
            # Detect pure skip (existing PNG, no --force) for clearer stats
            will_skip = False
            if not args.force:
                suf = src.suffix.lower()
                if suf in PDF_EXTS:
                    with fitz.open(src) as doc:
                        n = doc.page_count
                    expected = (
                        [src.with_suffix(".png")]
                        if n == 1
                        else [src.parent / f"{src.stem}-{i}.png" for i in range(1, n + 1)]
                    )
                    will_skip = all(p.exists() for p in expected)
                else:
                    will_skip = src.with_suffix(".png").exists()

            outputs = convert_one(
                src,
                dpi=args.dpi,
                fallback_dpi=args.fallback_dpi,
                eps_dpi=args.eps_dpi,
                force=args.force,
            )
            maybe_remove_source(src, outputs, args.remove_source)

            if will_skip:
                skipped += 1
                tag = "SKIP"
            else:
                ok += 1
                tag = "OK"
            for out in outputs:
                out_rel = out.relative_to(ROOT) if out.is_relative_to(ROOT) else out
                extra = ""
                if out.exists():
                    with Image.open(out) as im:
                        extra = f" ({im.width}x{im.height})"
                print(f"{tag}  {rel} -> {out_rel}{extra}")
        except Exception as exc:  # noqa: BLE001
            fail += 1
            print(f"ERR {rel}: {exc}", file=sys.stderr)

    print(
        f"\nDone: {ok} converted, {skipped} already-png, "
        f"{fail} failed, {len(targets)} total."
    )
    return 0 if fail == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
