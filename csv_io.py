# -*- coding: utf-8 -*-
"""将 文件路径/文件名 与 领域 对应关系写入 CSV，支持追加与断点续跑。"""

import csv
import queue
import threading
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple, Set

HEADERS = ["file_path", "file_name", "domain_cn", "domain_en", "updated_at"]
# 断点续跑用：仅存路径，一行一个，读取比解析整份 CSV 快得多
DONE_SUFFIX = ".done"


def load_processed_paths(csv_path: str) -> Set[str]:
    """从 .done 或 CSV 读取已处理路径集合；优先 .done（大文献量时启动更快）。"""
    done_path = Path(csv_path + DONE_SUFFIX)
    if done_path.exists():
        paths = set()
        try:
            with open(done_path, "r", encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if s:
                        paths.add(s)
            return paths
        except Exception:
            pass
    # 无 .done 时从 CSV 读一次，并生成 .done 供下次使用
    p = Path(csv_path)
    if not p.exists():
        return set()
    paths = set()
    try:
        with open(p, "r", encoding="utf-8-sig", newline="") as f:
            r = csv.reader(f)
            next(r, None)
            for row in r:
                if len(row) >= 1 and row[0].strip():
                    resolved = str(Path(row[0]).resolve())
                    paths.add(resolved)
        if paths:
            with open(done_path, "w", encoding="utf-8") as out:
                for x in paths:
                    out.write(x + "\n")
    except Exception:
        pass
    return paths


def _ensure_header(csv_path: str) -> None:
    """若文件不存在则创建并写入表头。"""
    p = Path(csv_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        return
    with open(p, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(HEADERS)


def _alt_csv_path(csv_path: str) -> str:
    """主文件被占用时的备用路径，如 literature_domains.csv -> literature_domains_alt.csv。"""
    p = Path(csv_path)
    return str(p.parent / (p.stem + "_alt" + p.suffix))


def append_row_sync(
    csv_path: str,
    file_path: str,
    file_name: str,
    domain_cn: str,
    domain_en: str,
    effective_path_ref: Optional[list] = None,
) -> None:
    """同步追加一行到 CSV 与 .done；若主文件被占用则自动改用 _alt 文件并继续。"""
    import logging as _log
    path_to_use = effective_path_ref[0] if effective_path_ref else csv_path
    p = Path(path_to_use)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        _ensure_header(path_to_use)
    resolved_path = str(Path(file_path).resolve())
    row = [resolved_path, file_name or "", domain_cn or "", domain_en or "", datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
    try:
        with open(p, "a", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(row)
        try:
            with open(p.parent / (p.name + DONE_SUFFIX), "a", encoding="utf-8") as d:
                d.write(resolved_path + "\n")
        except Exception:
            pass
    except PermissionError:
        alt = _alt_csv_path(csv_path)
        if effective_path_ref is not None:
            effective_path_ref[0] = alt
            _log.getLogger(__name__).warning(
                "主 CSV 被占用（如被 Excel 打开），已自动改用备用文件: %s", alt
            )
        _ensure_header(alt)
        with open(Path(alt), "a", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(row)
        try:
            with open(Path(alt).parent / (Path(alt).name + DONE_SUFFIX), "a", encoding="utf-8") as d:
                d.write(resolved_path + "\n")
        except Exception:
            pass


def _writer_loop(csv_path: str, q: queue.Queue, stop: threading.Event, effective_path_ref: list) -> None:
    """后台线程：从队列取结果并追加写入 CSV（被占用时自动写备用文件）。"""
    while True:
        try:
            item = q.get(timeout=0.5)
            if item is None:
                break
            file_path, file_name, domain_cn, domain_en = item
            append_row_sync(csv_path, file_path, file_name, domain_cn, domain_en, effective_path_ref)
        except queue.Empty:
            if stop.is_set():
                break
            continue
        except Exception as e:
            import logging
            logging.getLogger(__name__).exception("CSV 写入失败: %s", e)


class CsvWriterAsync:
    """异步写入 CSV：主线程入队，后台线程写入，不阻断识别；主文件被占用时自动写备用 _alt 文件。"""

    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self._q = queue.Queue()
        self._stop = threading.Event()
        self._effective_path_ref = [csv_path]
        self._processed_paths = load_processed_paths(csv_path)
        self._queued_paths: Set[str] = set()
        self._lock = threading.Lock()
        _ensure_header(csv_path)
        self._thread = threading.Thread(
            target=_writer_loop,
            args=(csv_path, self._q, self._stop, self._effective_path_ref),
            daemon=True,
        )
        self._thread.start()

    def put(self, file_path: str, file_name: str, domain_cn: str, domain_en: str) -> None:
        """将一条结果放入队列，由后台线程写入，不阻塞。"""
        resolved_path = str(Path(file_path).resolve())
        with self._lock:
            if resolved_path in self._processed_paths or resolved_path in self._queued_paths:
                return
            self._queued_paths.add(resolved_path)
        self._q.put((resolved_path, file_name, domain_cn or "", domain_en or ""))

    def close(self) -> None:
        """结束后台线程（等待队列写完）。"""
        self._stop.set()
        self._q.put(None)
        self._thread.join(timeout=30)


def list_domains_from_csv(csv_path: str) -> List[str]:
    """从 CSV 读取所有出现过的领域（domain_cn 去重）。"""
    p = Path(csv_path)
    if not p.exists():
        return []
    domains = set()
    try:
        with open(p, "r", encoding="utf-8-sig", newline="") as f:
            r = csv.reader(f)
            next(r, None)
            for row in r:
                if len(row) >= 3 and row[2].strip():
                    domains.add(row[2].strip())
    except Exception:
        pass
    return sorted(domains)


def query_by_domain_from_csv(csv_path: str, domain: str) -> List[Tuple[str, str, str]]:
    """按 domain_cn 筛选，返回 (file_path, file_name, domain_cn) 列表。"""
    p = Path(csv_path)
    if not p.exists():
        return []
    rows = []
    try:
        with open(p, "r", encoding="utf-8-sig", newline="") as f:
            r = csv.reader(f)
            next(r, None)
            for row in r:
                if len(row) >= 3 and (row[2] or "").strip() == domain.strip():
                    rows.append((str(row[0] or ""), str(row[1] or ""), str(row[2] or "")))
    except Exception:
        pass
    return rows
