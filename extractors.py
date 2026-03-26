# -*- coding: utf-8 -*-
"""PDF text extraction helpers."""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

MIN_TEXT_THRESHOLD = 200
OCR_TRIGGER_TEXT_THRESHOLD = 80
DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 100
PYPDF_WORKER_TIMEOUT_SEC = 20
FITZ_TEXT_WORKER_TIMEOUT_SEC = 25
OCR_WORKER_TIMEOUT_SEC = 25
OCR_MAX_PAGES = 2
OCR_RENDER_SCALE = 1.25

TITLE_MAX_CHARS = 220
AUTHOR_MAX_CHARS = 220
AFFILIATION_MAX_CHARS = 320
KEYWORDS_MAX_CHARS = 260
ABSTRACT_MAX_CHARS = 900
INTRO_MAX_CHARS = 1200
BODY_FALLBACK_MAX_CHARS = 1200


def _new_extract_meta() -> Dict[str, object]:
    return {
        "extract_source": "empty",
        "raw_text_len": 0,
        "content_len": 0,
        "ocr_used": False,
        "extract_timeout": False,
    }


def _normalize_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\x00", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def _extract_text_layer_with_pypdf(path: Path, max_pages: int) -> str:
    """Pure-Python extraction first, so bad native pages do not kill the app."""
    try:
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        n = min(len(reader.pages), max_pages)
        parts = []
        for i in range(n):
            page_text = reader.pages[i].extract_text()
            if page_text:
                parts.append(page_text)
        return _normalize_text("\n".join(parts))
    except Exception:
        return ""


def _fitz_extract_text_layer(path: Path, max_pages: int) -> str:
    import fitz

    doc = fitz.open(str(path))
    try:
        parts = []
        n = min(len(doc), max_pages)
        for i in range(n):
            parts.append(doc[i].get_text())
        return _normalize_text("".join(part for part in parts if part))
    finally:
        doc.close()


def _fitz_extract_ocr(path: Path, max_pages: int) -> str:
    import fitz
    import pytesseract
    from PIL import Image

    doc = fitz.open(str(path))
    try:
        parts = []
        n = min(len(doc), max_pages, OCR_MAX_PAGES)
        for i in range(n):
            page = doc[i]
            pix = page.get_pixmap(matrix=fitz.Matrix(OCR_RENDER_SCALE, OCR_RENDER_SCALE))
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            parts.append(pytesseract.image_to_string(img, lang="eng+chi_sim"))
        return _normalize_text("".join(part for part in parts if part))
    finally:
        doc.close()


def run_pdf_worker_cli(argv: List[str]) -> int:
    """Run risky PyMuPDF work in an isolated child process."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("mode", choices=["pypdf-text", "fitz-text", "fitz-ocr"])
    parser.add_argument("file_path")
    parser.add_argument("max_pages", type=int)
    args = parser.parse_args(argv)

    try:
        path = Path(args.file_path)
        if args.mode == "pypdf-text":
            text = _extract_text_layer_with_pypdf(path, args.max_pages)
        elif args.mode == "fitz-text":
            text = _fitz_extract_text_layer(path, args.max_pages)
        else:
            text = _fitz_extract_ocr(path, args.max_pages)
        print(json.dumps({"ok": True, "text": text}, ensure_ascii=False))
        return 0
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1


def _pdf_worker_command(mode: str, path: Path, max_pages: int) -> List[str]:
    if getattr(sys, "frozen", False):
        return [sys.executable, "__pdf_worker__", mode, str(path), str(max_pages)]

    main_script = Path(__file__).with_name("main.py")
    return [sys.executable, str(main_script), "__pdf_worker__", mode, str(path), str(max_pages)]


def _run_pdf_worker(mode: str, path: Path, max_pages: int, timeout_sec: int) -> Tuple[str, bool]:
    cmd = _pdf_worker_command(mode, path, max_pages)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_sec,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except subprocess.TimeoutExpired:
        return "", True
    except Exception:
        return "", False

    if result.returncode != 0 or not result.stdout.strip():
        return "", False

    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        return "", False

    if not payload.get("ok"):
        return "", False

    return _normalize_text(payload.get("text") or ""), False


def _extract_text_layer(path: Path, max_pages: int) -> Tuple[str, str, bool]:
    text, timed_out = _run_pdf_worker("pypdf-text", path, max_pages, PYPDF_WORKER_TIMEOUT_SEC)
    if text:
        return text, "pypdf", timed_out
    fitz_text, fitz_timed_out = _run_pdf_worker("fitz-text", path, max_pages, FITZ_TEXT_WORKER_TIMEOUT_SEC)
    if fitz_text:
        return fitz_text, "fitz-text", timed_out or fitz_timed_out
    return "", "empty", timed_out or fitz_timed_out


def _extract_ocr_fallback(path: Path, max_pages: int) -> Tuple[str, bool]:
    return _run_pdf_worker("fitz-ocr", path, min(max_pages, OCR_MAX_PAGES), OCR_WORKER_TIMEOUT_SEC)


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[str]:
    if not text or chunk_size <= 0:
        return []

    text = text.strip()
    if len(text) <= chunk_size:
        return [text] if text else []

    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        if end >= len(text):
            chunk = text[start:].strip()
            if chunk:
                chunks.append(chunk)
            break

        segment = text[start:end]
        for sep in ("\n\n", "\n", "。", ".", " ", ""):
            idx = segment.rfind(sep)
            if idx > chunk_size // 2:
                end = start + idx + (len(sep) if sep else 0)
                break

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - min(overlap, chunk_size - 1)

    return chunks


def merge_chunks_for_llm(
    chunks: List[str],
    max_chars: int,
    separator: str = "\n\n",
) -> str:
    if not chunks:
        return ""

    merged = separator.join(chunks)
    if len(merged) <= max_chars:
        return merged

    truncated = merged[:max_chars]
    for sep in ("\n", "。", ".", " "):
        last = truncated.rfind(sep)
        if last > max_chars // 2:
            truncated = truncated[: last + 1]
            break
    return truncated.strip()


def extract_pdf_text(path: str, max_pages: int = 10) -> Tuple[str, Dict[str, object]]:
    p = Path(path)
    if not p.exists():
        return "", _new_extract_meta()

    raw, source, timed_out = _extract_text_layer(p, max_pages)
    meta = _new_extract_meta()
    meta["extract_source"] = source
    meta["raw_text_len"] = len(raw)
    meta["extract_timeout"] = bool(timed_out)
    if len(raw) >= MIN_TEXT_THRESHOLD:
        return _normalize_text(raw), meta
    if len(raw) >= OCR_TRIGGER_TEXT_THRESHOLD:
        return _normalize_text(raw), meta

    ocr, ocr_timed_out = _extract_ocr_fallback(p, max_pages)
    meta["ocr_used"] = True
    meta["extract_timeout"] = bool(meta["extract_timeout"] or ocr_timed_out)
    best = ocr if len(ocr) > len(raw) else raw
    meta["extract_source"] = "ocr" if len(ocr) > len(raw) and ocr else source
    meta["raw_text_len"] = len(best)
    return _normalize_text(best), meta


def _looks_like_title_noise(line: str, filename: str) -> bool:
    if not line:
        return True
    lower = line.lower().strip()
    filename_lower = filename.lower().strip()
    noise_phrases = (
        "full terms & conditions of access and use",
        "full terms and conditions of access and use",
        "downloaded by",
        "article views",
        "view related articles",
        "publisher",
        "published online",
        "rights reserved",
        "mathematics magazine",
        "taylor & francis",
        "informa uk limited",
    )
    if lower == filename_lower:
        return True
    return any(phrase in lower for phrase in noise_phrases)


def _find_section_span(
    text: str,
    start_markers: Tuple[str, ...],
    end_markers: Tuple[str, ...],
    *,
    fallback_offset: int = 0,
    search_window: int = 2400,
) -> Tuple[int, int]:
    text_lower = text.lower()
    start = -1
    for marker in start_markers:
        idx = text_lower.find(marker.lower())
        if idx != -1 and (start == -1 or idx < start):
            start = idx

    if start == -1:
        if fallback_offset <= 0:
            return -1, -1
        start = min(fallback_offset, len(text))
    else:
        line_end = text.find("\n", start)
        start = line_end + 1 if line_end != -1 else start

    search_region = text[start : start + search_window]
    region_lower = search_region.lower()
    end_in_region = len(search_region)
    for marker in end_markers:
        idx = region_lower.find(marker.lower())
        if idx != -1 and idx < end_in_region:
            end_in_region = idx

    return start, start + end_in_region


def _find_abstract_span(text: str) -> Tuple[int, int]:
    return _find_section_span(
        text,
        ("abstract", "摘要", "summary"),
        (
            "introduction",
            "1. introduction",
            "\n1 introduction",
            "keywords",
            "key words",
            "关键词",
            "\n1.\n",
            "\n1.\t",
        ),
        search_window=2200,
    )


def _truncate(s: str, max_chars: int) -> str:
    if not s or max_chars <= 0:
        return ""

    s = _normalize_text(s)
    if len(s) <= max_chars:
        return s

    truncated = s[:max_chars]
    for sep in ("\n", "。", ".", ";", " "):
        idx = truncated.rfind(sep)
        if idx > max_chars // 2:
            return truncated[: idx + 1].strip()
    return truncated.strip()


def _extract_keywords(text: str, max_chars: int = KEYWORDS_MAX_CHARS) -> str:
    head = text[:5000]
    patterns = (
        r"(?is)(?:^|\n)\s*(?:keywords?|key words|index terms?)\s*[:：]\s*(.+?)(?:\n|$)",
        r"(?is)(?:^|\n)\s*关键词\s*[:：]\s*(.+?)(?:\n|$)",
    )

    for pattern in patterns:
        match = re.search(pattern, head)
        if match:
            raw = match.group(1).strip()
            raw = re.split(r"(?:introduction|1\.\s|methods|results|背景|引言)", raw, maxsplit=1, flags=re.I)[0]
            return _truncate(raw, max_chars)

    marker_only_patterns = (
        r"(?is)(?:^|\n)\s*(?:keywords?|key words|index terms?)\s*[:：]?\s*\n(.+?)(?:\n|$)",
        r"(?is)(?:^|\n)\s*关键词\s*[:：]?\s*\n(.+?)(?:\n|$)",
    )
    for pattern in marker_only_patterns:
        match = re.search(pattern, head)
        if match:
            return _truncate(match.group(1), max_chars)

    return ""


def _extract_title_author_affiliation_abstract(full_text: str, filename: str) -> Tuple[str, str, str, str]:
    text = full_text.strip()
    lines = [line.strip() for line in text.split("\n") if line.strip()]

    title_lines = []
    for i, line in enumerate(lines[:6]):
        if (
            len(line) > 10
            and not line.lower().startswith(("http", "www."))
            and not _looks_like_title_noise(line, filename)
        ):
            title_lines.append(line)
            if i >= 1 or len(" ".join(title_lines)) > 110:
                break
    title = _truncate(" ".join(title_lines), TITLE_MAX_CHARS) if title_lines else ""

    abs_start, abs_end = _find_abstract_span(text)
    abstract = ""
    if abs_start >= 0 and abs_end > abs_start:
        abstract = _truncate(text[abs_start:abs_end], ABSTRACT_MAX_CHARS)

    block_before_abstract = text[:abs_start].strip() if abs_start > 0 else text[:1000].strip()
    for line in title_lines:
        block_before_abstract = block_before_abstract.replace(line, "", 1).strip()

    before_lines = [line for line in block_before_abstract.split("\n") if line.strip()]
    author_parts = []
    affiliation_parts = []
    affil_keywords = (
        "department",
        "university",
        "hospital",
        "school",
        "college",
        "institute",
        "laboratory",
        "lab ",
        "academy",
        "centre",
        "center",
        "学院",
        "大学",
        "系",
        "所",
        "医院",
        "实验室",
    )

    for line in before_lines[:20]:
        line_lower = line.lower()
        if any(keyword in line_lower for keyword in affil_keywords) or len(line) > 70:
            affiliation_parts.append(line)
        else:
            author_parts.append(line)

    author = _truncate("\n".join(author_parts), AUTHOR_MAX_CHARS)
    affiliation = _truncate("\n".join(affiliation_parts), AFFILIATION_MAX_CHARS)
    if not author and before_lines:
        author = _truncate("\n".join(before_lines[:5]), AUTHOR_MAX_CHARS)
    if not affiliation and before_lines and not author:
        affiliation = _truncate("\n".join(before_lines[:8]), AFFILIATION_MAX_CHARS)

    return title, author, affiliation, abstract


def _extract_introduction_excerpt(
    text: str,
    *,
    abstract_end: int,
    max_chars: int,
) -> str:
    intro_start, intro_end = _find_section_span(
        text,
        ("introduction", "1. introduction", "\n1 introduction", "引言", "background"),
        (
            "\n2.",
            "\n2 ",
            "materials and methods",
            "methodology",
            "\nmethods",
            "\nresults",
            "results and discussion",
            "related work",
            "experimental",
            "实验",
            "材料与方法",
        ),
        fallback_offset=max(abstract_end, 0),
        search_window=max_chars * 2,
    )

    if intro_start < 0 or intro_end <= intro_start:
        return ""

    excerpt = text[intro_start:intro_end]
    excerpt = re.sub(
        r"(?is)^(?:introduction|1\. introduction|1 introduction|引言|background)\s*[:：]?",
        "",
        excerpt,
    )
    return _truncate(excerpt, max_chars)


def _extract_body_fallback(text: str, start_offset: int, max_chars: int) -> str:
    if start_offset < 0:
        start_offset = 0
    excerpt = text[start_offset : start_offset + max_chars * 2]
    return _truncate(excerpt, max_chars)


def _assemble_parts(parts: List[Tuple[str, str]], max_chars: int) -> str:
    assembled: List[str] = []
    used = 0
    for label, content in parts:
        if not content:
            continue
        block = f"【{label}】\n{content.strip()}"
        extra = len(block) + (2 if assembled else 0)
        if used + extra <= max_chars:
            assembled.append(block)
            used += extra
            continue

        remaining = max_chars - used - (2 if assembled else 0)
        if remaining > len(label) + 6:
            content_budget = max(0, remaining - len(label) - 4)
            trimmed = _truncate(content, content_budget)
            if trimmed:
                assembled.append(f"【{label}】\n{trimmed}")
            break

    return "\n\n".join(assembled).strip()


def extract_title_abstract_body(
    file_path: str,
    abstract_max: int = 1200,
    body_pages_chars: int = 1200,
    max_chars_for_llm: int = 1500,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    author_section_chars: int = 800,
    **kwargs,
) -> Tuple[str, str, Dict[str, object]]:
    del chunk_size, chunk_overlap, kwargs

    path = Path(file_path)
    if path.suffix.lower() != ".pdf":
        return path.name, "", ""

    full_text, meta = extract_pdf_text(str(path), max_pages=5)
    if not full_text.strip():
        meta["content_len"] = 0
        return "", "", meta

    title, author, affiliation, abstract = _extract_title_author_affiliation_abstract(full_text, path.name)
    keywords = _extract_keywords(full_text)
    abs_start, abs_end = _find_abstract_span(full_text)
    intro_excerpt = _extract_introduction_excerpt(
        full_text,
        abstract_end=abs_end if abs_end > 0 else 0,
        max_chars=min(body_pages_chars, INTRO_MAX_CHARS),
    )

    body_fallback = ""
    if not abstract and not intro_excerpt:
        body_start = abs_end if abs_end > 0 else 0
        body_fallback = _extract_body_fallback(
            full_text,
            body_start,
            min(body_pages_chars, BODY_FALLBACK_MAX_CHARS),
        )

    author = _truncate(author, min(author_section_chars, AUTHOR_MAX_CHARS))
    affiliation = _truncate(affiliation, min(author_section_chars, AFFILIATION_MAX_CHARS))
    abstract = _truncate(abstract, min(abstract_max, ABSTRACT_MAX_CHARS))

    parts = [
        ("标题", title),
        ("关键词", keywords),
        ("摘要", abstract),
        ("引言片段", intro_excerpt),
        ("正文片段", body_fallback),
        ("作者", author),
        ("研究团队/机构", affiliation),
    ]

    content = _assemble_parts(parts, max_chars_for_llm)
    meta["raw_text_len"] = len(full_text)
    meta["content_len"] = len(content)
    return title, content, meta
