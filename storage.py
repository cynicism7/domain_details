# -*- coding: utf-8 -*-
"""将 文件路径/文件名 与 最小领域（domain_cn/domain_en）对应关系写入 SQLite 与 CSV。"""

import csv
import sqlite3
from pathlib import Path
from typing import List, Tuple, Optional


def ensure_db(db_path: str) -> sqlite3.Connection:
    """创建或连接数据库，并确保表存在，支持自动迁移。"""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    
    # 检查表是否存在
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='literature_domains'")
    table_exists = cursor.fetchone()
    
    if not table_exists:
        # 新建表
        conn.execute("""
            CREATE TABLE literature_domains (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT NOT NULL UNIQUE,
                file_name TEXT,
                domain_cn TEXT,
                domain_en TEXT,
                updated_at TEXT DEFAULT (datetime('now','localtime'))
            )
        """)
    else:
        # 检查是否需要迁移（是否存在 domain_cn 列）
        cursor = conn.execute("PRAGMA table_info(literature_domains)")
        columns = [column[1] for column in cursor.fetchall()]
        
        if "domain_cn" not in columns:
            print("\n[系统] 正在升级数据库结构以支持中英文双列...")
            # 简单处理：如果旧表存在 domain 列，将其迁往 domain_cn
            if "domain" in columns:
                conn.execute("ALTER TABLE literature_domains RENAME COLUMN domain TO domain_cn")
                conn.execute("ALTER TABLE literature_domains ADD COLUMN domain_en TEXT")
            else:
                # 如果结构差异太大，则添加缺失列
                conn.execute("ALTER TABLE literature_domains ADD COLUMN domain_cn TEXT")
                conn.execute("ALTER TABLE literature_domains ADD COLUMN domain_en TEXT")
            print("[系统] 数据库升级完成。")
            
    conn.commit()
    return conn


def upsert_domain(conn: sqlite3.Connection, file_path: str, domain_cn: str, domain_en: str) -> None:
    """插入或更新一条记录：路径 + 中文领域 + 英文领域。"""
    name = Path(file_path).name
    conn.execute(
        """
        INSERT INTO literature_domains (file_path, file_name, domain_cn, domain_en)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(file_path) DO UPDATE SET
            domain_cn = excluded.domain_cn,
            domain_en = excluded.domain_en,
            file_name = excluded.file_name,
            updated_at = datetime('now','localtime')
        """,
        (str(Path(file_path).resolve()), name, domain_cn, domain_en),
    )
    conn.commit()


def export_csv(db_path: str, csv_path: str) -> None:
    """从数据库导出到 CSV，包含中英文两列。"""
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT file_path, file_name, domain_cn, domain_en, updated_at FROM literature_domains ORDER BY domain_cn, file_name"
    ).fetchall()
    conn.close()

    target = Path(csv_path)
    try:
        with open(target, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["file_path", "file_name", "domain_cn", "domain_en", "updated_at"])
            w.writerows(rows)
    except PermissionError:
        alt = target.with_name(f"{target.stem}.new{target.suffix}")
        with open(alt, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["file_path", "file_name", "domain_cn", "domain_en", "updated_at"])
            w.writerows(rows)
        print(f"\n[警告] 无法写入 CSV（可能被 Excel 占用）：{target}\n已改为写入：{alt}")


def query_by_domain(db_path: str, domain: str) -> List[Tuple[str, str, str]]:
    """按领域筛选（按 domain_cn 匹配），返回 (file_path, file_name, domain_cn) 列表。"""
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT file_path, file_name, domain_cn FROM literature_domains WHERE domain_cn = ? ORDER BY file_name",
        (domain,),
    ).fetchall()
    conn.close()
    return rows


def list_domains(db_path: str) -> List[str]:
    """返回所有出现过的领域列表（按 domain_cn 去重）。"""
    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT DISTINCT domain_cn FROM literature_domains WHERE domain_cn IS NOT NULL ORDER BY domain_cn").fetchall()
    conn.close()
    return [r[0] for r in rows]
