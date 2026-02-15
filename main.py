# -*- coding: utf-8 -*-
"""
文献领域识别主程序：
扫描指定目录中的文献，用本地大模型识别领域，将「路径/文件名 + 领域」写入数据库与 CSV，便于后续筛选。
"""

import argparse
from pathlib import Path

from extractors import extract_title_abstract_body
from llm_client import identify_domain
from storage import ensure_db, upsert_domain, export_csv, list_domains, query_by_domain


def load_config(config_path: str = "config.yaml") -> dict:
    """加载 YAML 配置。"""
    try:
        import yaml
        p = Path(config_path)
        if not p.exists():
            return _default_config()
        with open(p, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or _default_config()
    except Exception:
        return _default_config()


def _default_config() -> dict:
    return {
        "literature_dirs": ["./papers"],
        "extensions": [".pdf", ".docx", ".doc", ".txt"],
        "llm": {
            "provider": "ollama",
            "model": "qwen2.5:7b",
            "api_base": "http://localhost:1234/v1",
            "api_key": "not-needed",
        },
        "max_chars_for_llm": 3000,
        "output": {
            "db_path": "./literature_domains.db",
            "export_csv": True,
            "csv_path": "./literature_domains.csv",
        },
    }


def collect_files(dirs: list, extensions: list) -> list:
    """收集所有符合扩展名的文献文件路径。"""
    collected = []
    for d in dirs:
        p = Path(d)
        if not p.exists():
            continue
        for f in p.rglob("*"):
            if f.is_file() and f.suffix.lower() in [e.lower() for e in extensions]:
                collected.append(str(f.resolve()))
    return sorted(set(collected))


def run_scan(config_path: str = "config.yaml", use_mock: bool = False) -> None:
    """根据配置扫描文献、识别领域并写入数据库。use_mock=True 时不调用大模型，用简单规则模拟领域（便于本地验证）。"""
    cfg = load_config(config_path)
    dirs = cfg.get("literature_dirs", ["./papers"])
    exts = cfg.get("extensions", [".pdf", ".docx", ".doc", ".txt"])
    llm_cfg = cfg.get("llm", {})
    max_chars = cfg.get("max_chars_for_llm", 800)
    out = cfg.get("output", {})
    db_path = out.get("db_path", "./literature_domains.db")
    do_csv = out.get("export_csv", True)
    csv_path = out.get("csv_path", "./literature_domains.csv")

    files = collect_files(dirs, exts)
    if not files:
        print("未找到任何文献文件，请检查 config.yaml 中的 literature_dirs 与 extensions。")
        return

    if use_mock:
        print("【模拟模式】未调用大模型，使用简单规则生成领域标签，仅用于验证流程。\n")

    conn = ensure_db(db_path)
    total = len(files)
    for i, fp in enumerate(files, 1):
        name = Path(fp).name
        print(f"[{i}/{total}] {name} ... ", end="", flush=True)
        # RAG 式：分块后整合成一段，不存文件，长度受 max_chars 限制
        title, content_for_llm, _ = extract_title_abstract_body(
            fp, max_chars_for_llm=max_chars
        )
        # 说明：LM Studio 等日志里的 "<Truncated in logs>" 只是显示截断，实际请求里送交的是完整内容
        n_chars = len(content_for_llm or "")
        print(f"(送交 {n_chars} 字) ", end="", flush=True)
        provider = "mock" if use_mock else llm_cfg.get("provider", "ollama")
        domain_cn, domain_en = identify_domain(
            title,
            content_for_llm,
            provider=provider,
            model=llm_cfg.get("model", "qwen2.5:7b"),
            api_base=llm_cfg.get("api_base", "http://localhost:1234/v1"),
            api_key=llm_cfg.get("api_key", "not-needed"),
            max_tokens=llm_cfg.get("max_tokens", 256),
            temperature=llm_cfg.get("temperature", 0.0),
            system_prompt=llm_cfg.get("system_prompt"),
        )
        upsert_domain(conn, fp, domain_cn, domain_en)
        print(f"{domain_cn} | {domain_en}")
    conn.close()

    if do_csv:
        export_csv(db_path, csv_path)
        print(f"已导出 CSV: {csv_path}")
    print(f"数据库: {db_path}")


def run_list_domains(config_path: str = "config.yaml") -> None:
    """列出数据库中所有出现过的领域。"""
    cfg = load_config(config_path)
    db_path = cfg.get("output", {}).get("db_path", "./literature_domains.db")
    if not Path(db_path).exists():
        print("数据库不存在，请先运行 scan。")
        return
    domains = list_domains(db_path)
    print("已记录的领域：")
    for d in domains:
        print(f"  - {d}")


def run_query(domain: str, config_path: str = "config.yaml") -> None:
    """按领域筛选，打印该领域下的所有文献路径。"""
    cfg = load_config(config_path)
    db_path = cfg.get("output", {}).get("db_path", "./literature_domains.db")
    if not Path(db_path).exists():
        print("数据库不存在，请先运行 scan。")
        return
    rows = query_by_domain(db_path, domain)
    if not rows:
        print(f"领域「{domain}」下没有文献。")
        return
    print(f"领域「{domain}」下的文献（共 {len(rows)} 条）：")
    for path, name, _ in rows:
        print(f"  {path}")


def main():
    parser = argparse.ArgumentParser(description="文献领域识别：用本地大模型打标签并记录到数据库/CSV")
    parser.add_argument("--config", "-c", default="config.yaml", help="配置文件路径")
    sub = parser.add_subparsers(dest="command", help="子命令")

    p_scan = sub.add_parser("scan", help="扫描文献目录，识别领域并写入数据库")
    p_scan.add_argument("--mock", "-m", action="store_true", help="模拟模式：不调用大模型，用简单规则生成领域，便于本地验证")
    sub.add_parser("domains", help="列出所有已记录的领域")
    p_query = sub.add_parser("filter", help="按领域筛选文献")
    p_query.add_argument("domain", help="领域名称，如：计算机科学")

    args = parser.parse_args()
    if args.command == "scan":
        run_scan(args.config, use_mock=getattr(args, "mock", False))
    elif args.command == "domains":
        run_list_domains(args.config)
    elif args.command == "filter":
        run_query(args.domain, args.config)
    else:
        parser.print_help()
        print("\n示例：")
        print("  python main.py scan              # 扫描并打标签")
        print("  python main.py domains           # 查看所有领域")
        print("  python main.py filter 计算机科学  # 筛选该领域文献")


if __name__ == "__main__":
    main()
