#!/usr/bin/env python
import argparse
import json
import shutil
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.pdf import (  # noqa: E402
    _get_cache_path,
    _get_cache_root,
    _get_pdf_title,
    _parse_pdf_with_mineru,
    _save_cached_document,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Parse one PDF with MinerU and write a Psi-RAG-compatible PDF cache JSON. "
            "Run this in a MinerU-only environment, then run Psi-RAG indexing in the "
            "main environment."
        )
    )
    parser.add_argument(
        "--pdf",
        required=True,
        help="Input PDF path.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help=(
            "Output cache JSON path. Defaults to <pdf parent>/.psirag_pdf_cache/<pdf stem>.json, "
            "matching Psi-RAG read_local_pdf=file."
        ),
    )
    parser.add_argument(
        "--assets-dir",
        default=None,
        help="Directory for copied image/table assets. Defaults to <out stem>_assets next to --out.",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Document title stored in cache. Defaults to the PDF stem.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing cache JSON and assets directory.",
    )
    parser.add_argument(
        "--print-summary",
        action="store_true",
        help="Print chunk type counts after writing the cache.",
    )
    return parser.parse_args()


def default_cache_path(pdf_path: Path) -> Path:
    cache_root = _get_cache_root(pdf_path, "file")
    return _get_cache_path(pdf_path, cache_root, None)


def summarize_chunks(chunks):
    counts = {}
    for chunk in chunks:
        if isinstance(chunk, dict):
            chunk_type = chunk.get("type") or chunk.get("modality") or "dict"
        else:
            chunk_type = "text"
        counts[chunk_type] = counts.get(chunk_type, 0) + 1
    return counts


def main():
    args = parse_args()
    pdf_path = Path(args.pdf).expanduser().resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(str(pdf_path))
    if not pdf_path.is_file() or pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f'"{pdf_path}" must be a PDF file.')

    out_path = Path(args.out).expanduser().resolve() if args.out else default_cache_path(pdf_path)
    assets_dir = (
        Path(args.assets_dir).expanduser().resolve()
        if args.assets_dir
        else out_path.parent / f"{out_path.stem}_assets"
    )

    if out_path.exists() and not args.force:
        raise FileExistsError(f'Cache already exists: "{out_path}". Use --force to overwrite.')

    if args.force:
        if out_path.exists():
            out_path.unlink()
        if assets_dir.exists():
            shutil.rmtree(assets_dir)

    title = args.title or _get_pdf_title(pdf_path, None)
    chunks = _parse_pdf_with_mineru(pdf_path, image_output_dir=assets_dir)
    document = {
        "title": title,
        "chunks": chunks,
    }
    _save_cached_document(out_path, document)

    print(f'Wrote cache: "{out_path}"')
    print(f'Wrote assets: "{assets_dir}"')
    print(f"Chunks: {len(chunks)}")
    if args.print_summary:
        print(json.dumps(summarize_chunks(chunks), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
