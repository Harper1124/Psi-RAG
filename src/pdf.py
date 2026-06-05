import json
import re
import shutil
import subprocess
import sys
import tarfile
import zipfile
import html

from tempfile import TemporaryDirectory
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm


def normalize_local_pdf_mode(value) -> str | None:
    valid_modes = (
        "file",
        "dir",
        "dir_recursive",
        "package",
        "package_recursive",
    )
    if value in (None, False, ""):
        return None
    if not isinstance(value, str):
        raise ValueError(
            '"read_local_pdf" must be one of: "file", "dir", "dir_recursive", '
            '"package", or "package_recursive".'
        )

    normalized_value = " ".join(value.strip().split())
    if not normalized_value:
        return None
    if normalized_value in valid_modes:
        return normalized_value

    raise ValueError(
        '"read_local_pdf" must be one of: "file", "dir", "dir_recursive", '
        '"package", or "package_recursive".'
    )


def _sanitize_dataset_token(value: str) -> str:
    token = re.sub(r"[^0-9a-zA-Z]+", "_", value).strip("_").lower()
    return token or "pdf"

def build_local_pdf_dataset_name(data_dir: str | Path, read_mode: str) -> str:
    read_mode = normalize_local_pdf_mode(read_mode)
    if read_mode is None:
        raise ValueError('"read_local_pdf" cannot be None when building a local PDF dataset name.')

    input_path = Path(str(data_dir)).expanduser()
    return _sanitize_dataset_token(input_path.name or "pdf")


def _validate_archive_path(output_root: Path, member_name: str) -> None:
    output_root = output_root.resolve()
    member_path = (output_root / member_name).resolve()
    try:
        member_path.relative_to(output_root)
    except ValueError as exc:
        raise ValueError(f'Unsafe package member "{member_name}".')


def _extract_package(package_path: Path, output_root: Path) -> None:
    if zipfile.is_zipfile(package_path):
        with zipfile.ZipFile(package_path) as archive:
            for member_name in archive.namelist():
                _validate_archive_path(output_root, member_name)
            archive.extractall(output_root)
        return

    if tarfile.is_tarfile(package_path):
        with tarfile.open(package_path) as archive:
            for member in archive.getmembers():
                _validate_archive_path(output_root, member.name)
                if member.issym() or member.islnk():
                    raise ValueError(f'Unsupported symlink entry in package: "{member.name}".')
            archive.extractall(output_root)
        return

    if package_path.suffix.lower() == ".7z":
        try:
            import py7zr
        except ImportError as exc:
            raise ImportError('Reading ".7z" packages requires "py7zr".') from exc

        with py7zr.SevenZipFile(package_path) as archive:
            for member_name in archive.getnames():
                _validate_archive_path(output_root, member_name)
            archive.extractall(output_root)
        return

    if package_path.suffix.lower() == ".rar":
        try:
            import rarfile
        except ImportError as exc:
            raise ImportError('Reading ".rar" packages requires "rarfile".') from exc

        with rarfile.RarFile(package_path) as archive:
            for member_name in archive.namelist():
                _validate_archive_path(output_root, member_name)
            archive.extractall(output_root)
        return

    raise ValueError(
        f'Unsupported package format "{package_path.suffix}". '
        'Supported packages include zip, tar, 7z, and rar families.'
    )


def _collect_pdf_paths(root: Path, recursive: bool) -> List[Path]:
    iterator = root.rglob("*") if recursive else root.glob("*")
    return sorted(
        path for path in iterator
        if path.is_file() and path.suffix.lower() == ".pdf"
        and ".psirag_pdf_cache" not in path.parts
    )


def _temporary_directory(prefix: str, dir_path: Path) -> TemporaryDirectory:
    try:
        return TemporaryDirectory(prefix=prefix, dir=dir_path)
    except (FileNotFoundError, PermissionError):
        return TemporaryDirectory(prefix=prefix)


def _get_mineru_command() -> List[str]:
    if shutil.which("mineru") is not None:
        return ["mineru"]

    try:
        __import__("mineru")
    except ImportError as exc:
        raise ImportError(
            'MinerU is required for "read_local_pdf". Install the official "mineru" package.'
        ) from exc

    return [sys.executable, "-m", "mineru.cli.client"]


def _get_cache_root(input_path: Path, read_mode: str) -> Path:
    if read_mode in ("dir", "dir_recursive"):
        return input_path / ".psirag_pdf_cache"
    if read_mode in ("package", "package_recursive"):
        return input_path.parent / ".psirag_pdf_cache" / _sanitize_dataset_token(input_path.stem)
    return input_path.parent / ".psirag_pdf_cache"


def _get_cache_path(pdf_path: Path, cache_root: Path, root_path: Path | None) -> Path:
    relative_path = Path(pdf_path.name) if root_path is None else pdf_path.relative_to(root_path)
    return (cache_root / relative_path).with_suffix(".json")


def _load_cached_document(cache_path: Path, title: str) -> Dict | None:
    if not cache_path.exists():
        return None

    try:
        cached_document = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    if not isinstance(cached_document, dict):
        return None
    if cached_document.get("title") != title:
        return None
    if not isinstance(cached_document.get("chunks"), list):
        return None
    cached_document["chunks"] = _postprocess_visual_chunks(cached_document["chunks"])
    return cached_document


def _save_cached_document(cache_path: Path, document: Dict) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _figure_id(caption: str) -> str:
    match = re.search(r"\b(?:Figure|Fig\.)\s*\.?\s*(\d+)", caption, flags=re.IGNORECASE)
    return match.group(1) if match else ""


def _is_figure_caption(caption: str) -> bool:
    return bool(re.match(r"^\s*(?:Figure|Fig\.)\s*\.?\s*\d+\b", caption, flags=re.IGNORECASE))


def _split_subfigure_caption(caption: str) -> Dict[str, str]:
    matches = list(re.finditer(r"\(([a-z])\)", caption, flags=re.IGNORECASE))
    if not matches:
        return {}

    parts = {}
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(caption)
        text = caption[start:end].strip(" .;:")
        if text:
            parts[match.group(1).lower()] = text
    return parts


def _chunk_sort_key(chunk: dict):
    bbox = chunk.get("bbox") or []
    if isinstance(bbox, list) and len(bbox) >= 2:
        return (chunk.get("page") if chunk.get("page") is not None else -1, bbox[1], bbox[0])
    return (chunk.get("page") if chunk.get("page") is not None else -1, 0, 0)


def _ensure_visual_text(chunk: dict, figure_caption: str = "", subfigure_label: str = "", subfigure_caption: str = "") -> None:
    caption = _normalize_text(chunk.get("caption", ""))
    ocr = _normalize_text(chunk.get("ocr", ""))
    text = _normalize_text(chunk.get("text", ""))
    figure_caption = _normalize_text(figure_caption)
    subfigure_caption = _normalize_text(subfigure_caption)

    if figure_caption and not caption:
        caption = figure_caption
        chunk["caption"] = caption
    if subfigure_label:
        chunk["subfigure"] = subfigure_label
    if subfigure_caption:
        chunk["subfigure_caption"] = subfigure_caption

    parts = []
    figure_no = _figure_id(figure_caption or caption)
    if figure_no and subfigure_label and subfigure_caption:
        parts.append(f"Figure {figure_no}({subfigure_label}): {subfigure_caption}")
    elif figure_caption:
        parts.append(f"Figure caption: {figure_caption}")
    if caption and caption not in parts:
        parts.append(f"Caption: {caption}")
    if ocr:
        parts.append(f"OCR: {ocr}")
    if text and text not in (caption, ocr):
        parts.append(text)
    if chunk.get("image_path") and not parts:
        parts.append("Image content is available at the attached image path.")
    chunk["text"] = "\n".join(dict.fromkeys(part for part in parts if part))


def _postprocess_visual_chunks(chunks: List[dict | str]) -> List[dict | str]:
    visual_chunks = [
        chunk for chunk in chunks
        if isinstance(chunk, dict)
        and str(chunk.get("modality") or chunk.get("type") or "").lower() in ("image", "chart", "figure")
        and chunk.get("image_path")
    ]
    if not visual_chunks:
        return chunks

    captions_by_page: Dict[object, List[str]] = {}
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        caption = _normalize_text(chunk.get("caption") or chunk.get("text") or "")
        if _is_figure_caption(caption):
            captions_by_page.setdefault(chunk.get("page"), []).append(caption)

    visuals_by_page: Dict[object, List[dict]] = {}
    for chunk in visual_chunks:
        visuals_by_page.setdefault(chunk.get("page"), []).append(chunk)

    for page, page_visuals in visuals_by_page.items():
        page_visuals.sort(key=_chunk_sort_key)
        figure_caption = captions_by_page.get(page, [""])[-1]
        subcaptions = _split_subfigure_caption(figure_caption)
        labels = sorted(subcaptions.keys())

        for index, chunk in enumerate(page_visuals):
            label = labels[index] if index < len(labels) else ""
            _ensure_visual_text(
                chunk,
                figure_caption=figure_caption,
                subfigure_label=label,
                subfigure_caption=subcaptions.get(label, ""),
            )

    return chunks


def _run_mineru(input_path: Path, output_root: Path) -> None:
    command = _get_mineru_command() + [
        "-p", str(input_path),
        "-o", str(output_root),
        "-b", "pipeline",
    ]
    process = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if process.returncode != 0:
        raise RuntimeError(
            f'MinerU failed to parse "{input_path}".\n'
            f"stdout:\n{process.stdout}\n"
            f"stderr:\n{process.stderr}"
        )


def _find_output_file(output_root: Path, pdf_stem: str, suffix: str) -> Path | None:
    output_paths = sorted(output_root.rglob(f"*{suffix}"))
    if not output_paths:
        return None

    for path in output_paths:
        if path.stem == pdf_stem:
            return path
        if path.name.startswith(f"{pdf_stem}_"):
            return path
    return output_paths[0]


def _normalize_text(text) -> str:
    if text is None:
        return ""
    if isinstance(text, list):
        text = " ".join(_flatten_text_list(text))
    elif not isinstance(text, str):
        text = str(text)
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _flatten_text_list(value) -> List[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        texts = []
        for item in value:
            texts.extend(_flatten_text_list(item))
        return texts
    return []


def _resolve_media_path(output_root: Path, raw_path) -> str:
    if raw_path is None:
        return ""
    if isinstance(raw_path, list):
        raw_path = raw_path[0] if raw_path else ""
    raw_path = str(raw_path).strip()
    if not raw_path:
        return ""
    path = Path(raw_path)
    if path.is_absolute():
        return str(path)
    candidates = [
        output_root / raw_path,
        output_root / raw_path.lstrip("./"),
        output_root / path.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    matches = sorted(output_root.rglob(path.name))
    return str(matches[0]) if matches else raw_path


def _copy_media_to_cache(image_path: str, image_output_dir: Path) -> str:
    if not image_path:
        return ""
    source = Path(image_path)
    if not source.exists():
        return image_path
    image_output_dir.mkdir(parents=True, exist_ok=True)
    target = image_output_dir / source.name
    suffix_index = 1
    while target.exists() and source.resolve() != target.resolve():
        target = image_output_dir / f"{source.stem}_{suffix_index}{source.suffix}"
        suffix_index += 1
    if not target.exists():
        shutil.copy2(source, target)
    return str(target)


def _extract_page(content_item: dict):
    for key in ("page", "page_idx", "page_no", "page_number"):
        if key in content_item:
            return content_item[key]
    return None


def _text_chunk(content_item: dict, text: str, item_type: str | None = None) -> dict | None:
    if not text:
        return None
    return {
        "type": item_type or content_item.get("type") or "text",
        "modality": "text",
        "text": text,
        "page": _extract_page(content_item),
    }


def _collect_item_chunk(
    content_item: dict,
    output_root: Path,
    image_output_dir: Path | None = None,
) -> dict | None:
    item_type = content_item.get("type")

    if item_type in ("header", "footer", "page_number", "aside_text", "page_footnote", "seal"):
        return None

    if item_type == "text":
        text = _normalize_text(content_item.get("text", ""))
        text_level = content_item.get("text_level", 0)
        if text and isinstance(text_level, int) and text_level > 0:
            text = f'{"#" * text_level} {text}'
        return _text_chunk(content_item, text, item_type)

    if item_type in ("list", "equation"):
        parts = [_normalize_text(content_item.get("text", ""))]
        if item_type == "list":
            parts.extend(
                _normalize_text(item)
                for item in _flatten_text_list(content_item.get("list_items"))
            )
        if item_type == "equation":
            parts.extend(
                _normalize_text(content_item.get(field, ""))
                for field in ("latex", "html")
            )
        return _text_chunk(content_item, "\n".join(part for part in parts if part), item_type)

    if item_type == "code":
        parts = [
            _normalize_text(content_item.get("code_caption", "")),
            _normalize_text(content_item.get("code_body", "")),
            _normalize_text(content_item.get("code_footnote", "")),
        ]
        return _text_chunk(content_item, "\n".join(part for part in parts if part), item_type)

    if item_type == "table":
        caption_parts = []
        for field in ("table_caption",):
            value = content_item.get(field, "")
            if isinstance(value, list):
                value = " ".join(value)
            value = _normalize_text(value)
            if value:
                caption_parts.append(value)

        body_parts = []
        for field in ("table_body", "table_footnote", "html"):
            value = content_item.get(field, "")
            if isinstance(value, list):
                value = " ".join(value)
            value = _normalize_text(value)
            if value:
                body_parts.append(value)

        image_path = ""
        for field in ("table_img_path", "table_image_path", "image_path", "img_path", "path", "image"):
            image_path = _resolve_media_path(output_root, content_item.get(field))
            if image_path:
                break
        if image_path and image_output_dir is not None:
            image_path = _copy_media_to_cache(image_path, image_output_dir)

        table_caption = "\n".join(caption_parts)
        table_body = "\n".join(body_parts)
        text = "\n".join(part for part in (table_caption, table_body) if part)
        if not any((text, image_path)):
            return None
        return {
            "type": "table",
            "modality": "table",
            "text": text,
            "table_caption": table_caption,
            "table_body": table_body,
            "html": _normalize_text(content_item.get("html", "")),
            "image_path": image_path,
            "page": _extract_page(content_item),
            "bbox": content_item.get("bbox"),
        }

    if item_type in ("image", "chart"):
        caption_parts = []
        for field in ("image_caption", "chart_caption"):
            value = content_item.get(field, "")
            if isinstance(value, list):
                value = " ".join(value)
            value = _normalize_text(value)
            if value:
                caption_parts.append(value)

        ocr_parts = []
        for field in ("ocr", "image_text", "text", "image_footnote", "chart_footnote"):
            value = content_item.get(field, "")
            if isinstance(value, list):
                value = " ".join(value)
            value = _normalize_text(value)
            if value:
                ocr_parts.append(value)

        image_path = ""
        for field in ("image_path", "img_path", "path", "image"):
            image_path = _resolve_media_path(output_root, content_item.get(field))
            if image_path:
                break
        if image_path and image_output_dir is not None:
            image_path = _copy_media_to_cache(image_path, image_output_dir)

        caption = "\n".join(caption_parts)
        ocr = "\n".join(ocr_parts)
        text = "\n".join(part for part in (caption, ocr) if part)
        if not any((caption, ocr, image_path)):
            return None
        return {
            "type": item_type,
            "modality": "image",
            "caption": caption,
            "ocr": ocr,
            "text": text,
            "image_path": image_path,
            "page": _extract_page(content_item),
            "bbox": content_item.get("bbox"),
        }

    parts = [
        _normalize_text(content_item.get("text", "")),
        _normalize_text(content_item.get("content", "")),
        _normalize_text(content_item.get("latex", "")),
        _normalize_text(content_item.get("html", "")),
    ]
    return _text_chunk(content_item, "\n".join(part for part in parts if part), item_type)


def _load_mineru_content(
    output_root: Path,
    pdf_stem: str,
    image_output_dir: Path | None = None,
) -> List[dict | str]:
    content_list_path = _find_output_file(output_root, pdf_stem, "_content_list.json")
    if content_list_path is not None:
        content_list = json.loads(content_list_path.read_text(encoding="utf-8"))
        chunks = []
        for item in content_list:
            chunk = _collect_item_chunk(item, output_root, image_output_dir)
            if chunk:
                chunks.append(chunk)
        if chunks:
            return _postprocess_visual_chunks(chunks)

    markdown_path = _find_output_file(output_root, pdf_stem, ".md")
    if markdown_path is None:
        raise ValueError(f'MinerU did not produce readable output for "{pdf_stem}".')

    markdown_text = markdown_path.read_text(encoding="utf-8").strip()
    return [
        chunk.strip()
        for chunk in re.split(r"\n\s*\n", markdown_text)
        if chunk.strip()
    ]


def _parse_pdf_with_mineru(
    pdf_path: Path,
    image_output_dir: Path | None = None,
) -> List[dict | str]:
    with _temporary_directory(prefix=".psirag_mineru_", dir_path=pdf_path.parent) as output_dir:
        output_root = Path(output_dir)
        mineru_pdf_path = output_root / "document.pdf"
        shutil.copy2(pdf_path, mineru_pdf_path)
        _run_mineru(mineru_pdf_path, output_root)
        return _load_mineru_content(output_root, mineru_pdf_path.stem, image_output_dir)


def _parse_pdf_batch_with_mineru(
    pdf_paths: List[Path],
    image_output_dirs: Dict[Path, Path] | None = None,
) -> Dict[Path, List[dict | str]]:
    if len(pdf_paths) == 1:
        image_output_dir = image_output_dirs.get(pdf_paths[0]) if image_output_dirs else None
        return {pdf_paths[0]: _parse_pdf_with_mineru(pdf_paths[0], image_output_dir)}

    with _temporary_directory(prefix=".psirag_mineru_input_", dir_path=pdf_paths[0].parent) as input_dir, \
         _temporary_directory(prefix=".psirag_mineru_output_", dir_path=pdf_paths[0].parent) as output_dir:
        input_root = Path(input_dir)
        output_root = Path(output_dir)
        pdf_name_map = {}
        for i, pdf_path in enumerate(pdf_paths):
            mineru_pdf_path = input_root / f"doc_{i}.pdf"
            shutil.copy2(pdf_path, mineru_pdf_path)
            pdf_name_map[pdf_path] = mineru_pdf_path.stem

        _run_mineru(input_root, output_root)
        return {
            pdf_path: _load_mineru_content(
                output_root,
                pdf_stem,
                image_output_dirs.get(pdf_path) if image_output_dirs else None,
            )
            for pdf_path, pdf_stem in pdf_name_map.items()
        }


def _get_pdf_title(pdf_path: Path, root_path: Path | None) -> str:
    if root_path is None:
        return pdf_path.stem

    relative_path = pdf_path.relative_to(root_path)
    if relative_path.suffix.lower() == ".pdf":
        relative_path = relative_path.with_suffix("")
    return relative_path.as_posix()


def _parse_pdf_paths(
    pdf_paths: List[Path],
    root_path: Path | None,
    cache_root: Path,
    batch_size: int,
) -> List[dict]:
    documents = {}
    uncached_pdfs = []

    for pdf_path in pdf_paths:
        title = _get_pdf_title(pdf_path, root_path)
        cache_path = _get_cache_path(pdf_path, cache_root, root_path)
        cached_document = _load_cached_document(cache_path, title)
        if cached_document is not None:
            documents[pdf_path] = cached_document
        else:
            uncached_pdfs.append((pdf_path, title, cache_path))

    if documents:
        tqdm.write(f"Loaded {len(documents)} cached pdf(s) from \"{cache_root}\".")

    if uncached_pdfs:
        batch_total = (len(uncached_pdfs) - 1) // batch_size + 1
        bar = tqdm(total=batch_total, desc="parsing local pdf")
        for i in range(0, len(uncached_pdfs), batch_size):
            batch = uncached_pdfs[i : i + batch_size]
            image_output_dirs = {
                pdf_path: cache_path.parent / f"{cache_path.stem}_assets"
                for pdf_path, _, cache_path in batch
            }
            parsed_batch = _parse_pdf_batch_with_mineru(
                [pdf_path for pdf_path, _, _ in batch],
                image_output_dirs,
            )
            for pdf_path, title, cache_path in batch:
                document = {
                    "title": title,
                    "chunks": parsed_batch[pdf_path],
                }
                _save_cached_document(cache_path, document)
                documents[pdf_path] = document
            bar.update(1)
        bar.close()

    return [documents[pdf_path] for pdf_path in pdf_paths]


def load_local_pdf_data(data_dir: str | Path, read_mode: str, pdf_batch_size: int = 1) -> List[dict]:

    if pdf_batch_size < 1:
        raise ValueError('"local_pdf_batch_size" must be a positive integer.')
    
    read_mode = normalize_local_pdf_mode(read_mode)
    if read_mode is None:
        raise ValueError('"read_local_pdf" cannot be None when loading local PDFs.')

    input_path = Path(str(data_dir)).expanduser()
    if not input_path.exists():
        raise FileNotFoundError(str(input_path))
    cache_root = _get_cache_root(input_path, read_mode)
    cache_root.mkdir(parents=True, exist_ok=True)

    if read_mode == "file":
        if not input_path.is_file() or input_path.suffix.lower() != ".pdf":
            raise ValueError(f'"{input_path}" must be a single PDF file.')
        return _parse_pdf_paths([input_path], None, cache_root, pdf_batch_size)

    if read_mode in ("dir", "dir_recursive"):
        if not input_path.is_dir():
            raise ValueError(f'"{input_path}" is not a directory.')
        pdf_paths = _collect_pdf_paths(input_path, recursive=read_mode.endswith("recursive"))
        if not pdf_paths:
            raise FileNotFoundError(f'No PDF files found under "{input_path}".')
        return _parse_pdf_paths(pdf_paths, input_path, cache_root, pdf_batch_size)

    if read_mode in ("package", "package_recursive"):
        if not input_path.is_file():
            raise ValueError(f'"{input_path}" is not a package file.')
        with _temporary_directory(prefix=".psirag_pdf_package_", dir_path=input_path.parent) as output_dir:
            output_root = Path(output_dir)
            _extract_package(input_path, output_root)
            pdf_paths = _collect_pdf_paths(output_root, recursive=read_mode.endswith("recursive"))
            if not pdf_paths:
                raise FileNotFoundError(f'No PDF files found inside package "{input_path}".')
            return _parse_pdf_paths(pdf_paths, output_root, cache_root, pdf_batch_size)

    raise ValueError(f'Unsupported read_local_pdf mode "{read_mode}".')


def prepare_local_pdf_dataset(data_dir: str | Path, read_mode: str) -> str:
    read_mode = normalize_local_pdf_mode(read_mode)
    if read_mode is None:
        raise ValueError('"read_local_pdf" cannot be None when preparing a local PDF dataset.')
    return build_local_pdf_dataset_name(data_dir, read_mode)
