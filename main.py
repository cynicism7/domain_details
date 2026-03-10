# -*- coding: utf-8 -*-
"""
文献领域识别主程序：
扫描指定目录中的文献，用本地大模型识别领域，将「路径/文件名 + 领域」实时写入 CSV，支持断点续跑与运行日志。
"""

import argparse
import gc
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from extractors import extract_title_abstract_body
from llm_client import identify_domain, clear_llm_context
from csv_io import (
    load_processed_paths,
    CsvWriterAsync,
    list_domains_from_csv,
    query_by_domain_from_csv,
)


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
            "ollama_base": "http://localhost:11434",
        },
        "max_chars_for_llm": 3000,
        "max_prompt_chars": 4096,
        "clear_context_every_n": None,
        "output": {
            "csv_path": "./literature_domains.csv",
            "log_path": "./scan.log",
        },
        "concurrency": 1,
    }


def setup_logging(log_path: str) -> logging.Logger:
    """配置运行日志：同时输出到文件与控制台，报错带堆栈。"""
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("scan")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger


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


def _process_one_file(fp: str, job_config: dict, log: logging.Logger) -> tuple:
    """单篇文献：提取 + 领域识别，返回 (file_path, file_name, domain_cn, domain_en, 提取秒数, 模型秒数)。"""
    name = Path(fp).name
    t0 = time.perf_counter()
    title, content_for_llm, _ = extract_title_abstract_body(
        fp, max_chars_for_llm=job_config["max_chars"]
    )
    t_extract = time.perf_counter() - t0
    t1 = time.perf_counter()
    llm = job_config["llm_cfg"]
    domain_cn, domain_en = identify_domain(
        title,
        content_for_llm,
        provider=job_config["provider"],
        model=llm.get("model", "qwen2.5:7b"),
        api_base=llm.get("api_base", "http://localhost:1234/v1"),
        api_key=llm.get("api_key", "not-needed"),
        max_tokens=llm.get("max_tokens", 128),
        temperature=llm.get("temperature", 0.0),
        timeout=llm.get("timeout", 60),
        system_prompt=llm.get("system_prompt"),
        preferred_domains=job_config.get("preferred_domains"),
        max_prompt_chars=job_config["max_prompt_chars"],
        stream=job_config.get("stream", True),
        max_preferred_domains_in_prompt=job_config.get("max_preferred_domains_in_prompt"),
    )
    t_llm = time.perf_counter() - t1
    return fp, name, domain_cn, domain_en, t_extract, t_llm


def run_scan(config_path: str = "config.yaml", use_mock: bool = False) -> None:
    """根据配置扫描文献、识别领域并实时写入 CSV；支持断点续跑与运行日志。"""
    cfg = load_config(config_path)
    dirs = cfg.get("literature_dirs", ["./papers"])
    exts = cfg.get("extensions", [".pdf", ".docx", ".doc", ".txt"])
    llm_cfg = cfg.get("llm", {})
    max_chars = cfg.get("max_chars_for_llm", 800)
    max_prompt_chars = cfg.get("max_prompt_chars", 4096)
    clear_context_every_n = cfg.get("clear_context_every_n")
    out = cfg.get("output", {})
    csv_path = out.get("csv_path", "./literature_domains.csv")
    log_path = out.get("log_path", "./scan.log")
    concurrency = max(1, int(cfg.get("concurrency", 1)))

    log = setup_logging(log_path)
    log.info("日志文件: %s", log_path)

    all_files = collect_files(dirs, exts)
    if not all_files:
        log.warning("未找到任何文献文件，请检查 config.yaml 中的 literature_dirs 与 extensions。")
        return

    done_paths = load_processed_paths(csv_path)
    files = [f for f in all_files if str(Path(f).resolve()) not in done_paths]
    if not files:
        log.info("所有文献已处理完毕，无需继续。")
        return
    if len(done_paths) > 0:
        log.info("断点续跑: 已跳过 %d 篇，待处理 %d 篇。", len(done_paths), len(files))

    if use_mock:
        log.info("【模拟模式】未调用大模型，使用简单规则生成领域标签。")
    if concurrency > 1:
        log.info("并发数: %d", concurrency)

    job_config = {
        "max_chars": max_chars,
        "max_prompt_chars": max_prompt_chars,
        "provider": "mock" if use_mock else llm_cfg.get("provider", "ollama"),
        "llm_cfg": llm_cfg,
        "preferred_domains": cfg.get("preferred_domains"),
        "stream": bool(cfg.get("llm", {}).get("stream", True)),
        "max_preferred_domains_in_prompt": cfg.get("max_preferred_domains_in_prompt"),
    }

    writer = CsvWriterAsync(csv_path)
    total = len(files)
    try:
        if not use_mock and llm_cfg.get("warmup", False):
            try:
                log.info("模型预热中...")
                identify_domain(
                    "warmup",
                    "test",
                    provider=job_config["provider"],
                    model=llm_cfg.get("model", "qwen2.5:7b"),
                    api_base=llm_cfg.get("api_base", "http://localhost:1234/v1"),
                    api_key=llm_cfg.get("api_key", "not-needed"),
                    max_tokens=llm_cfg.get("max_tokens", 128),
                    temperature=llm_cfg.get("temperature", 0.0),
                    timeout=llm_cfg.get("timeout", 60),
                    system_prompt=llm_cfg.get("system_prompt"),
                    preferred_domains=cfg.get("preferred_domains"),
                    max_prompt_chars=max_prompt_chars,
                    stream=job_config.get("stream", True),
                    max_preferred_domains_in_prompt=cfg.get("max_preferred_domains_in_prompt"),
                )
                log.info("模型预热完成")
            except Exception as e:
                log.debug("预热请求忽略: %s", e)
        if concurrency <= 1:
            provider = job_config["provider"]
            for i, fp in enumerate(files, 1):
                name = Path(fp).name
                log.info("[%d/%d] %s ... ", i, total, name)
                try:
                    fp, name, domain_cn, domain_en, t_extract, t_llm = _process_one_file(fp, job_config, log)
                    writer.put(fp, name, domain_cn, domain_en)
                    log.info("%s | %s  (提取%.2fs 模型%.2fs)", domain_cn, domain_en, t_extract, t_llm)
                except Exception as e:
                    log.exception("处理失败 [%s]: %s", fp, e)
                    writer.put(fp, name, "处理失败", str(e)[:200])
                if i % 50 == 0:
                    gc.collect()
                if clear_context_every_n and i % clear_context_every_n == 0 and i > 0:
                    clear_llm_context(
                        provider=provider,
                        model=llm_cfg.get("model", "qwen2.5:7b"),
                        api_base=llm_cfg.get("api_base", "http://localhost:1234/v1"),
                        api_key=llm_cfg.get("api_key", "not-needed"),
                    )
        else:
            done = 0
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = {executor.submit(_process_one_file, fp, job_config, log): fp for fp in files}
                for fut in as_completed(futures):
                    fp = futures[fut]
                    name = Path(fp).name
                    try:
                        _, name, domain_cn, domain_en, t_extract, t_llm = fut.result()
                        writer.put(fp, name, domain_cn, domain_en)
                        done += 1
                        log.info("[%d/%d] %s -> %s | %s  (提取%.2fs 模型%.2fs)", done, total, name, domain_cn, domain_en, t_extract, t_llm)
                    except Exception as e:
                        log.exception("处理失败 [%s]: %s", fp, e)
                        writer.put(fp, name, "处理失败", str(e)[:200])
                        done += 1
                    if done % 50 == 0:
                        gc.collect()
            # 并发模式下不调用 clear_llm_context（卸载会影响所有 worker）
    finally:
        writer.close()

    log.info("本次完成 %d 篇，结果已写入: %s", len(files), csv_path)


def run_list_domains(config_path: str = "config.yaml") -> None:
    """列出 CSV 中所有出现过的领域。"""
    cfg = load_config(config_path)
    csv_path = cfg.get("output", {}).get("csv_path", "./literature_domains.csv")
    if not Path(csv_path).exists():
        print("CSV 不存在，请先运行 scan。")
        return
    domains = list_domains_from_csv(csv_path)
    print("已记录的领域：")
    for d in domains:
        print(f"  - {d}")


def run_query(domain: str, config_path: str = "config.yaml") -> None:
    """按领域筛选，打印该领域下的所有文献路径。"""
    cfg = load_config(config_path)
    csv_path = cfg.get("output", {}).get("csv_path", "./literature_domains.csv")
    if not Path(csv_path).exists():
        print("CSV 不存在，请先运行 scan。")
        return
    rows = query_by_domain_from_csv(csv_path, domain)
    if not rows:
        print(f"领域「{domain}」下没有文献。")
        return
    print(f"领域「{domain}」下的文献（共 {len(rows)} 条）：")
    for path, name, _ in rows:
        print(f"  {path}")


def main():
    parser = argparse.ArgumentParser(description="文献领域识别：用本地大模型打标签并实时写入 CSV，支持断点续跑")
    parser.add_argument("--config", "-c", default="config.yaml", help="配置文件路径")
    sub = parser.add_subparsers(dest="command", help="子命令")

    p_scan = sub.add_parser("scan", help="扫描文献目录，识别领域并写入 CSV（可断点续跑）")
    p_scan.add_argument("--mock", "-m", action="store_true", help="模拟模式：不调用大模型，用简单规则生成领域")
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
        print("  python main.py scan              # 扫描并打标签（断点续跑）")
        print("  python main.py domains           # 查看所有领域")
        print("  python main.py filter 计算机科学  # 筛选该领域文献")


if __name__ == "__main__":
    main()
