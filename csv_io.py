# -*- coding: utf-8 -*-
"""将 文件路径/文件名 与 领域 对应关系写入 CSV，支持追加与断点续跑。"""

import csv
import queue
import threading
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Set

HEADERS = ["file_path", "file_name", "domain_cn", "domain_en", "updated_at"]


def load_processed_paths(csv_path: str) -> Set[str]:
    """从已有 CSV 中读取已处理的 file_path 集合，用于断点续跑。"""
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
                    paths.add(str(Path(row[0]).resolve()))
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


def append_row_sync(csv_path: str, file_path: str, file_name: str, domain_cn: str, domain_en: str) -> None:
    """同步追加一行到 CSV（供单线程或后台线程调用）。"""
    p = Path(csv_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        _ensure_header(csv_path)
    resolved_path = str(Path(file_path).resolve())
    row = [resolved_path, file_name or "", domain_cn or "", domain_en or "", datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
    with open(p, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(row)


def _writer_loop(csv_path: str, q: queue.Queue, stop: threading.Event) -> None:
    """后台线程：从队列取结果并追加写入 CSV。"""
    while True:
        try:
            item = q.get(timeout=0.5)
            if item is None:
                break
            file_path, file_name, domain_cn, domain_en = item
            append_row_sync(csv_path, file_path, file_name, domain_cn, domain_en)
        except queue.Empty:
            if stop.is_set():
                break
            continue
        except Exception as e:
            import logging
            logging.getLogger(__name__).exception("CSV 写入失败: %s", e)


class CsvWriterAsync:
    """异步写入 CSV：主线程入队，后台线程写入，不阻断识别。"""

    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self._q = queue.Queue()
        self._stop = threading.Event()
        _ensure_header(csv_path)
        self._thread = threading.Thread(
            target=_writer_loop,
            args=(csv_path, self._q, self._stop),
            daemon=True,
        )
        self._thread.start()

    def put(self, file_path: str, file_name: str, domain_cn: str, domain_en: str) -> None:
        """将一条结果放入队列，由后台线程写入，不阻塞。"""
        self._q.put((file_path, file_name, domain_cn or "", domain_en or ""))

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
