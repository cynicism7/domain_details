# -*- coding: utf-8 -*-
"""CSV read/write helpers for literature classification results."""

import csv
import queue
import threading
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple

BASE_HEADERS = ["file_path", "file_name", "domain_cn", "domain_en", "updated_at"]
DONE_SUFFIX = ".done"


def _normalize_stored_path(
    file_path: str,
    path_normalizer: Optional[Callable[[str], str]] = None,
) -> str:
    raw = (file_path or "").strip()
    if not raw:
        return ""
    if path_normalizer is not None:
        try:
            normalized = (path_normalizer(raw) or "").strip()
            if normalized:
                return normalized
        except Exception:
            pass
    return str(Path(raw).resolve())


def load_processed_paths(
    csv_path: str,
    path_normalizer: Optional[Callable[[str], str]] = None,
) -> Set[str]:
    """Load processed path ids from .done or from the CSV first column."""
    done_path = Path(csv_path + DONE_SUFFIX)
    if done_path.exists():
        paths = set()
        try:
            with open(done_path, "r", encoding="utf-8") as f:
                for line in f:
                    stored_path = _normalize_stored_path(line, path_normalizer)
                    if stored_path:
                        paths.add(stored_path)
            return paths
        except Exception:
            pass

    p = Path(csv_path)
    if not p.exists():
        return set()

    paths = set()
    try:
        with open(p, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if len(row) >= 1 and row[0].strip():
                    paths.add(_normalize_stored_path(row[0], path_normalizer))
        if paths:
            with open(done_path, "w", encoding="utf-8") as out:
                for stored_path in paths:
                    out.write(stored_path + "\n")
    except Exception:
        pass
    return paths


def _ensure_header(csv_path: str, headers: Optional[List[str]] = None) -> None:
    p = Path(csv_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        return
    with open(p, "w", newline="", encoding="utf-8-sig") as f:
        csv.writer(f).writerow(headers or BASE_HEADERS)


def _alt_csv_path(csv_path: str) -> str:
    p = Path(csv_path)
    return str(p.parent / (p.stem + "_alt" + p.suffix))


def normalize_csv_file_paths(
    csv_path: str,
    path_normalizer: Optional[Callable[[str], str]] = None,
) -> int:
    """Normalize first-column paths in place and rebuild the .done cache."""
    p = Path(csv_path)
    if not p.exists():
        return 0

    try:
        with open(p, "r", encoding="utf-8-sig", newline="") as f:
            rows = list(csv.reader(f))
    except Exception:
        return 0

    if not rows:
        return 0

    header = rows[0]
    normalized_rows: Dict[str, List[str]] = {}
    changed = False

    for row in rows[1:]:
        if not row:
            continue
        original_path = row[0] if len(row) >= 1 else ""
        stored_path = _normalize_stored_path(original_path, path_normalizer)
        if not stored_path:
            continue

        rewritten = list(row)
        rewritten[0] = stored_path
        if original_path != stored_path:
            changed = True
        if stored_path in normalized_rows:
            changed = True
            del normalized_rows[stored_path]
        normalized_rows[stored_path] = rewritten

    if not changed:
        return 0

    with open(p, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in normalized_rows.values():
            writer.writerow(row)

    with open(Path(csv_path + DONE_SUFFIX), "w", encoding="utf-8") as out:
        for stored_path in normalized_rows.keys():
            out.write(stored_path + "\n")

    return len(normalized_rows)


def append_row_sync(
    csv_path: str,
    file_path: str,
    file_name: str,
    domain_cn: str,
    domain_en: str,
    extra_fields: Optional[Dict[str, object]] = None,
    headers: Optional[List[str]] = None,
    effective_path_ref: Optional[list] = None,
    path_normalizer: Optional[Callable[[str], str]] = None,
) -> None:
    import logging as _log

    path_to_use = effective_path_ref[0] if effective_path_ref else csv_path
    p = Path(path_to_use)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        _ensure_header(path_to_use, headers=headers)

    stored_path = _normalize_stored_path(file_path, path_normalizer)
    fieldnames = headers or BASE_HEADERS
    row_map: Dict[str, object] = {
        "file_path": stored_path,
        "file_name": file_name or "",
        "domain_cn": domain_cn or "",
        "domain_en": domain_en or "",
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    for key, value in (extra_fields or {}).items():
        if key in fieldnames:
            row_map[key] = value if value is not None else ""
    row = [row_map.get(name, "") for name in fieldnames]

    try:
        with open(p, "a", newline="", encoding="utf-8-sig") as f:
            csv.writer(f).writerow(row)
        try:
            with open(p.parent / (p.name + DONE_SUFFIX), "a", encoding="utf-8") as done_file:
                done_file.write(stored_path + "\n")
        except Exception:
            pass
    except PermissionError:
        alt = _alt_csv_path(csv_path)
        if effective_path_ref is not None:
            effective_path_ref[0] = alt
            _log.getLogger(__name__).warning(
                "CSV is in use (for example by Excel); writing to fallback file: %s",
                alt,
            )
        _ensure_header(alt, headers=headers)
        with open(Path(alt), "a", newline="", encoding="utf-8-sig") as f:
            csv.writer(f).writerow(row)
        try:
            with open(Path(alt).parent / (Path(alt).name + DONE_SUFFIX), "a", encoding="utf-8") as done_file:
                done_file.write(stored_path + "\n")
        except Exception:
            pass


def _writer_loop(
    csv_path: str,
    q: queue.Queue,
    stop: threading.Event,
    effective_path_ref: list,
    headers: List[str],
    path_normalizer: Optional[Callable[[str], str]],
) -> None:
    while True:
        try:
            item = q.get(timeout=0.5)
            if item is None:
                break
            file_path, file_name, domain_cn, domain_en, extra_fields = item
            append_row_sync(
                csv_path,
                file_path,
                file_name,
                domain_cn,
                domain_en,
                extra_fields=extra_fields,
                headers=headers,
                effective_path_ref=effective_path_ref,
                path_normalizer=path_normalizer,
            )
        except queue.Empty:
            if stop.is_set():
                break
            continue
        except Exception as exc:
            import logging

            logging.getLogger(__name__).exception("CSV write failed: %s", exc)


class CsvWriterAsync:
    """Async CSV writer with de-duplication based on normalized stored paths."""

    def __init__(
        self,
        csv_path: str,
        headers: Optional[List[str]] = None,
        path_normalizer: Optional[Callable[[str], str]] = None,
    ):
        self.csv_path = csv_path
        self.headers = headers or list(BASE_HEADERS)
        self._path_normalizer = path_normalizer
        self._q = queue.Queue()
        self._stop = threading.Event()
        self._effective_path_ref = [csv_path]
        self._processed_paths = load_processed_paths(csv_path, path_normalizer=path_normalizer)
        self._queued_paths: Set[str] = set()
        self._lock = threading.Lock()
        _ensure_header(csv_path, headers=self.headers)
        self._thread = threading.Thread(
            target=_writer_loop,
            args=(
                csv_path,
                self._q,
                self._stop,
                self._effective_path_ref,
                self.headers,
                self._path_normalizer,
            ),
            daemon=True,
        )
        self._thread.start()

    def put(
        self,
        file_path: str,
        file_name: str,
        domain_cn: str,
        domain_en: str,
        extra_fields: Optional[Dict[str, object]] = None,
    ) -> None:
        stored_path = _normalize_stored_path(file_path, self._path_normalizer)
        with self._lock:
            if stored_path in self._processed_paths or stored_path in self._queued_paths:
                return
            self._queued_paths.add(stored_path)
        self._q.put((stored_path, file_name, domain_cn or "", domain_en or "", extra_fields or {}))

    def close(self) -> None:
        self._stop.set()
        self._q.put(None)
        self._thread.join(timeout=30)


def list_domains_from_csv(csv_path: str) -> List[str]:
    p = Path(csv_path)
    if not p.exists():
        return []
    domains = set()
    try:
        with open(p, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if len(row) >= 3 and row[2].strip():
                    domains.add(row[2].strip())
    except Exception:
        pass
    return sorted(domains)


def query_by_domain_from_csv(csv_path: str, domain: str) -> List[Tuple[str, str, str]]:
    p = Path(csv_path)
    if not p.exists():
        return []
    rows = []
    try:
        with open(p, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if len(row) >= 3 and (row[2] or "").strip() == domain.strip():
                    rows.append((str(row[0] or ""), str(row[1] or ""), str(row[2] or "")))
    except Exception:
        pass
    return rows
