# -*- coding: utf-8 -*-
"""
PDF 提取器（RAG 式分块 + 整合，不存文件）：
1. 从 PDF 取全文（文本层优先，不足时 OCR 兜底）
2. 按 RAG 思路分块（固定长度 + 重叠）
3. 将碎片文本整合为一段，供大模型识别领域（不写入任何文件）
"""

from pathlib import Path
from typing import Tuple, List

# 文本过少则视为需 OCR
MIN_TEXT_THRESHOLD = 200

# 默认分块参数（与常见 RAG 配置一致：按字符近似等价于 ~300–500 token 的块）
DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 100

def _extract_text_layer(path: Path, max_pages: int) -> str:
    """文本层提取：优先 PyMuPDF，回退 pypdf。"""
    text = ""
    try:
        import fitz
        doc = fitz.open(str(path))
        n = min(len(doc), max_pages)
        for i in range(n):
            text += doc[i].get_text()
        doc.close()
        if text.strip():
            return text.strip()
    except Exception:
        pass

    try:
        from pypdf import PdfReader
        reader = PdfReader(path)
        n = min(len(reader.pages), max_pages)
        parts = [reader.pages[i].extract_text() for i in range(n) if reader.pages[i].extract_text()]
        if parts:
            return "\n".join(parts).strip()
    except Exception:
        pass
    return ""


def _extract_ocr_fallback(path: Path, max_pages: int) -> str:
    """OCR 兜底：用 PyMuPDF 渲染页面 + Tesseract 识别。"""
    try:
        import fitz
        import pytesseract
        from PIL import Image
        doc = fitz.open(str(path))
        text = ""
        n = min(len(doc), max_pages)
        for i in range(n):
            page = doc[i]
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            text += pytesseract.image_to_string(img, lang="eng+chi_sim")
        doc.close()
        return text.strip()
    except Exception:
        pass
    return ""


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[str]:
    """
    RAG 式分块：按字符数切分，块间带重叠，尽量在句/行边界切割。
    """
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
        # 在句号、换行或空格处截断，避免截断单词/中文
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
    """
    将分块文本整合为一段，总长度不超过 max_chars，供 LLM 使用。
    不写入任何文件。
    """
    if not chunks:
        return ""
    merged = separator.join(chunks)
    if len(merged) <= max_chars:
        return merged
    # 从开头截断到 max_chars，尽量在句末截断
    truncated = merged[:max_chars]
    for sep in ("\n", "。", ".", " "):
        last = truncated.rfind(sep)
        if last > max_chars // 2:
            truncated = truncated[: last + 1]
            break
    return truncated.strip()


def extract_pdf_text(path: str, max_pages: int = 10) -> str:
    """
    从 PDF 提取全文：先文本层，不足时 OCR 兜底。
    不写入任何中间文件。
    """
    p = Path(path)
    if not p.exists():
        return ""

    raw = _extract_text_layer(p, max_pages)
    if len(raw) >= MIN_TEXT_THRESHOLD:
        return raw
    ocr = _extract_ocr_fallback(p, max_pages)
    return ocr if ocr else raw


# 仅向 AI 提供四部分时的字符上限（控制 token）
TITLE_MAX_CHARS = 200
AUTHOR_MAX_CHARS = 200
AFFILIATION_MAX_CHARS = 300
ABSTRACT_MAX_CHARS = 600


def _find_abstract_span(text: str) -> Tuple[int, int]:
    """找到摘要段的起止位置（Abstract/摘要 到 Introduction/Keywords 等）。"""
    text_lower = text.lower()
    start = -1
    for mark in ("abstract", "摘要", "summary"):
        i = text_lower.find(mark)
        if i != -1 and (start == -1 or i < start):
            start = i
    if start == -1:
        return -1, -1
    line_end = text.find("\n", start)
    if line_end != -1:
        start = line_end + 1
    else:
        start = start + 8
    search_region = text[start : start + 2000]
    end_in_region = len(search_region)
    for mark in ("introduction", "1. introduction", "keywords", "key words", "索引", "1. ", "\n1.\t"):
        j = search_region.lower().find(mark)
        if j != -1 and j < end_in_region:
            end_in_region = j
    return start, start + end_in_region


def _truncate(s: str, max_chars: int) -> str:
    if not s or max_chars <= 0:
        return ""
    s = s.strip()
    if len(s) <= max_chars:
        return s
    t = s[:max_chars]
    for sep in ("\n", "。", ".", " "):
        idx = t.rfind(sep)
        if idx > max_chars // 2:
            return t[: idx + 1].strip()
    return t.strip()


def _extract_title_author_affiliation_abstract(full_text: str, filename: str) -> Tuple[str, str, str, str]:
    """从全文抽取：标题、作者、研究团队（机构）、摘要。"""
    text = full_text.strip()
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]

    title_lines = []
    for i, ln in enumerate(lines[:4]):
        if len(ln) > 10 and not ln.lower().startswith(("http", "www.")):
            title_lines.append(ln)
            if i >= 1 or len(ln) > 80:
                break
    title = _truncate(" ".join(title_lines), TITLE_MAX_CHARS) if title_lines else filename

    abs_start, abs_end = _find_abstract_span(text)
    abstract = ""
    if abs_start >= 0 and abs_end > abs_start:
        abstract = _truncate(text[abs_start:abs_end], ABSTRACT_MAX_CHARS)

    block_before_abstract = text[:abs_start].strip() if abs_start > 0 else text[:800].strip()
    for ln in title_lines:
        block_before_abstract = block_before_abstract.replace(ln, "", 1).strip()
    before_lines = [ln for ln in block_before_abstract.split("\n") if ln.strip()]
    author_parts = []
    affiliation_parts = []
    affil_keywords = ("department", "university", "hospital", "school", "college", "institute", "laboratory", "lab ", "学院", "大学", "系", "所", "医院", "实验室")
    for ln in before_lines[:20]:
        ln_lower = ln.lower()
        if any(kw in ln_lower for kw in affil_keywords) or len(ln) > 60:
            affiliation_parts.append(ln)
        else:
            author_parts.append(ln)
    author = _truncate("\n".join(author_parts), AUTHOR_MAX_CHARS)
    affiliation = _truncate("\n".join(affiliation_parts), AFFILIATION_MAX_CHARS)
    if not author and before_lines:
        author = _truncate("\n".join(before_lines[:5]), AUTHOR_MAX_CHARS)
    if not affiliation and before_lines and not author:
        affiliation = _truncate("\n".join(before_lines[:8]), AFFILIATION_MAX_CHARS)

    return title, author, affiliation, abstract


def extract_title_abstract_body(
    file_path: str,
    abstract_max: int = 1500,
    body_pages_chars: int = 2400,
    max_chars_for_llm: int = 1500,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    author_section_chars: int = 1200,
    **kwargs,
) -> Tuple[str, str, str]:
    """
    仅向 AI 提供：标题、作者、研究团队、摘要，以降低输入 token。
    返回 (文件名, 四部分整合文本, "")。
    """
    path = Path(file_path)
    if path.suffix.lower() != ".pdf":
        return path.name, "", ""

    full_text = extract_pdf_text(str(path), max_pages=5)
    if not full_text.strip():
        return path.name, "", ""

    title, author, affiliation, abstract = _extract_title_author_affiliation_abstract(
        full_text, path.name
    )
    parts = []
    if title:
        parts.append("【标题】\n" + title)
    if author:
        parts.append("【作者】\n" + author)
    if affiliation:
        parts.append("【研究团队/机构】\n" + affiliation)
    if abstract:
        parts.append("【摘要】\n" + abstract)
    content = "\n\n".join(parts)
    if len(content) > max_chars_for_llm:
        content = _truncate(content, max_chars_for_llm)
    return path.name, content.strip(), ""
