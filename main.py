# -*- coding: utf-8 -*-
"""
文献领域识别主程序：
扫描指定目录中的文献，用本地大模型识别领域，将「路径/文件名 + 领域」实时写入 CSV，支持断点续跑与运行日志。
"""

import argparse
import gc
import logging
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Dict, Optional

from extractors import extract_title_abstract_body, run_pdf_worker_cli
from llm_client import identify_domain, clear_llm_context
from csv_io import (
    load_processed_paths,
    CsvWriterAsync,
    list_domains_from_csv,
    normalize_csv_file_paths,
    query_by_domain_from_csv,
)

EXTRACTION_FAILURE_PRIMARY = "提取失败"
EXTRACTION_FAILURE_EN = "ExtractionFailure"
DEFAULT_MIN_CONTENT_CHARS = 120
DIAGNOSTIC_FIELD_ORDER = [
    "extract_source",
    "raw_text_len",
    "content_len",
    "ocr_used",
    "extract_timeout",
    "extract_seconds",
    "model_seconds",
]


def _runtime_search_roots() -> list:
    roots = []
    if getattr(sys, "frozen", False):
        roots.append(Path(sys.executable).resolve().parent)
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            roots.append(Path(meipass))
    roots.append(Path.cwd())
    return roots


def _resolve_existing_path(raw_path: str) -> Path:
    path_obj = Path(raw_path)
    if path_obj.is_absolute():
        return path_obj
    for root in _runtime_search_roots():
        candidate = (root / path_obj).resolve()
        if candidate.exists():
            return candidate
    return path_obj


def _normalize_cli_config_path(config_path: str) -> str:
    if not config_path or config_path == "config.yaml":
        return config_path
    path_obj = Path(config_path).expanduser()
    if path_obj.is_absolute():
        return str(path_obj)
    return str(path_obj.resolve())


def load_config(config_path: str = "config.yaml") -> dict:
    """加载 YAML 配置。"""
    try:
        import yaml

        p = _resolve_existing_path(config_path)
        if not p.exists():
            return _default_config()
        with open(p, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or _default_config()
    except Exception:
        return _default_config()


def _default_config() -> dict:
    return {
        "path_mapping": {
            "physical_root": "",
            "logical_root": "",
        },
        "literature_dirs": ["./papers"],
        "extensions": [".pdf"],
        "taxonomy_path": "./taxonomy.yaml",
        "llm": {
            "provider": "openai_api",
            "model": "qwen2.5-7b-instruct-q4_k_m",
            "api_base": "http://127.0.0.1:1234/v1",
            "api_key": "lm-studio",
            "max_tokens": 32,
            "temperature": 0.0,
            "timeout": 120,
            "stream": False,
            "warmup": True,
            "choice_constraints": False,
            "extra_body": {},
        },
        "max_chars_for_llm": 1200,
        "max_prompt_chars": 4096,
        "clear_context_every_n": 50,
        "output": {
            "csv_path": "./literature_domains.csv",
            "log_path": "./scan.log",
            "include_csv_diagnostics": False,
            "include_csv_timing": False,
            "log_include_diagnostics": True,
            "log_include_timing": True,
        },
        "concurrency": 1,
        "extract_concurrency": 1,
        "llm_concurrency": 1,
        "classification_retries": 0,
        "taxonomy_fast_path": True,
        "min_content_chars_for_classification": DEFAULT_MIN_CONTENT_CHARS,
    }


def load_taxonomy(taxonomy_path: str) -> Optional[dict]:
    """Load optional taxonomy YAML for controlled two-stage classification."""
    if not taxonomy_path:
        return None
    try:
        import yaml

        p = Path(taxonomy_path)
        if not p.exists():
            return None
        with open(p, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            return None
        return data
    except Exception:
        return None


def _resolve_config_relative_path(config_path: str, raw_path: str) -> str:
    """Resolve a relative path against the config file location."""
    if not raw_path:
        return raw_path
    path_obj = Path(raw_path)
    if path_obj.is_absolute():
        return str(path_obj)

    config_base = _resolve_existing_path(config_path)
    if config_base.exists():
        return str((config_base.parent / path_obj).resolve())

    for root in _runtime_search_roots():
        candidate = (root / path_obj).resolve()
        if candidate.exists():
            return str(candidate)

    return str(path_obj)


def _split_normalized_parts(raw_path: str) -> list:
    text = str(raw_path or "").strip().replace("\\", "/")
    return [part for part in text.split("/") if part not in ("", ".")]


def _normalize_logical_root(raw_root: str) -> str:
    return "/".join(_split_normalized_parts(raw_root))


def _join_logical_path(logical_root: str, relative_path: str = "") -> str:
    parts = _split_normalized_parts(logical_root)
    parts.extend(_split_normalized_parts(relative_path))
    return "/".join(parts)


def _extract_logical_path_from_raw(raw_path: str, logical_root: str) -> Optional[str]:
    root_parts = _split_normalized_parts(logical_root)
    if not root_parts:
        return None
    raw_parts = _split_normalized_parts(raw_path)
    if len(raw_parts) < len(root_parts):
        return None

    target = [part.casefold() for part in root_parts]
    for idx in range(len(raw_parts) - len(root_parts) + 1):
        window = [part.casefold() for part in raw_parts[idx : idx + len(root_parts)]]
        if window == target:
            suffix = raw_parts[idx + len(root_parts) :]
            return "/".join(root_parts + suffix)
    return None


def _load_path_mapping(config_path: str, cfg: dict) -> dict:
    mapping = cfg.get("path_mapping", {}) or {}
    if not isinstance(mapping, dict):
        mapping = {}

    physical_root = None
    raw_physical_root = mapping.get("physical_root", "")
    if raw_physical_root:
        physical_root = Path(_resolve_config_relative_path(config_path, raw_physical_root)).resolve()

    logical_root = _normalize_logical_root(mapping.get("logical_root", ""))
    if physical_root is not None and not logical_root:
        logical_root = physical_root.name

    return {
        "physical_root": physical_root,
        "logical_root": logical_root,
    }


def _resolve_literature_dir(config_path: str, raw_dir: str, path_mapping: dict) -> str:
    if not raw_dir:
        return raw_dir
    path_obj = Path(raw_dir).expanduser()
    if path_obj.is_absolute():
        return str(path_obj.resolve())

    physical_root = path_mapping.get("physical_root")
    if physical_root is not None:
        return str((physical_root / path_obj).resolve())

    return _resolve_config_relative_path(config_path, raw_dir)


def _stored_file_path(raw_path: str, path_mapping: dict) -> str:
    raw_text = str(raw_path or "").strip()
    if not raw_text:
        return ""

    logical_root = path_mapping.get("logical_root", "")
    logical_candidate = _extract_logical_path_from_raw(raw_text, logical_root)
    if logical_candidate:
        return logical_candidate

    path_obj = Path(raw_text).expanduser()
    physical_root = path_mapping.get("physical_root")
    if physical_root is not None:
        try:
            relative = path_obj.resolve().relative_to(physical_root)
        except Exception:
            pass
        else:
            return _join_logical_path(logical_root, relative.as_posix())

    if logical_root and not path_obj.is_absolute():
        return _join_logical_path(logical_root, raw_text)

    return str(path_obj.resolve())


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


def _normalize_output_flags(output_cfg: dict) -> dict:
    # Backward compatible defaults:
    # - diagnostics / timing are shown in log by default
    # - CSV stays in the original compact format unless explicitly enabled
    base_diag = bool(output_cfg.get("include_diagnostics", True))
    base_timing = bool(output_cfg.get("include_timing", True))
    return {
        "include_csv_diagnostics": bool(output_cfg.get("include_csv_diagnostics", False)),
        "include_csv_timing": bool(output_cfg.get("include_csv_timing", False)),
        "log_include_diagnostics": bool(output_cfg.get("log_include_diagnostics", base_diag)),
        "log_include_timing": bool(output_cfg.get("log_include_timing", base_timing)),
    }


def _build_csv_headers(output_flags: dict) -> list:
    headers = ["file_path", "file_name", "domain_cn", "domain_en", "updated_at"]
    if output_flags["include_csv_diagnostics"]:
        headers.extend(["extract_source", "raw_text_len", "content_len", "ocr_used", "extract_timeout"])
    if output_flags["include_csv_timing"]:
        headers.extend(["extract_seconds", "model_seconds"])
    return headers


def _build_row_extra_fields(result: dict, output_flags: dict) -> Dict[str, object]:
    meta = result.get("extract_meta") or {}
    extra: Dict[str, object] = {}
    if output_flags["include_csv_diagnostics"]:
        extra.update({
            "extract_source": meta.get("extract_source", ""),
            "raw_text_len": meta.get("raw_text_len", 0),
            "content_len": meta.get("content_len", 0),
            "ocr_used": bool(meta.get("ocr_used", False)),
            "extract_timeout": bool(meta.get("extract_timeout", False)),
        })
    if output_flags["include_csv_timing"]:
        extra.update({
            "extract_seconds": f"{result.get('t_extract', 0.0):.2f}",
            "model_seconds": f"{result.get('t_llm', 0.0):.2f}",
        })
    return extra


def _format_result_log(done: int, total: int, result: dict, output_flags: dict) -> str:
    line = f"[{done}/{total}] {result['file_name']} -> {result['domain_cn']} | {result['domain_en']}"
    if output_flags["log_include_timing"]:
        line += f"  (提取{result.get('t_extract', 0.0):.2f}s 模型{result.get('t_llm', 0.0):.2f}s)"
    if output_flags["log_include_diagnostics"]:
        meta = result.get("extract_meta") or {}
        line += (
            " [source={source} raw={raw} content={content} ocr={ocr} timeout={timeout}]"
        ).format(
            source=meta.get("extract_source", ""),
            raw=meta.get("raw_text_len", 0),
            content=meta.get("content_len", 0),
            ocr=bool(meta.get("ocr_used", False)),
            timeout=bool(meta.get("extract_timeout", False)),
        )
    return line


def _extraction_failure_result(extract_meta: dict) -> tuple:
    if extract_meta.get("extract_timeout"):
        return "提取失败/提取超时", f"{EXTRACTION_FAILURE_EN}/Timeout"
    if extract_meta.get("raw_text_len", 0) <= 0:
        if extract_meta.get("ocr_used"):
            return "提取失败/OCR无结果", f"{EXTRACTION_FAILURE_EN}/NoTextAfterOCR"
        return "提取失败/无有效文本", f"{EXTRACTION_FAILURE_EN}/NoText"
    return "提取失败/文本不足", f"{EXTRACTION_FAILURE_EN}/InsufficientContent"


def _should_mark_extract_failure(extract_meta: dict, min_content_chars: int) -> bool:
    if extract_meta.get("extract_timeout"):
        return True
    if int(extract_meta.get("raw_text_len", 0) or 0) <= 0:
        return True
    return int(extract_meta.get("content_len", 0) or 0) < max(1, min_content_chars)


def _build_llm_kwargs(job_config: dict) -> dict:
    llm = job_config["llm_cfg"]
    return {
        "provider": job_config["provider"],
        "model": llm.get("model", "qwen2.5:7b"),
        "api_base": llm.get("api_base", "http://localhost:1234/v1"),
        "api_key": llm.get("api_key", "not-needed"),
        "max_tokens": llm.get("max_tokens", 128),
        "temperature": llm.get("temperature", 0.0),
        "timeout": llm.get("timeout", 60),
        "system_prompt": llm.get("system_prompt"),
        "max_prompt_chars": job_config["max_prompt_chars"],
        "stream": job_config.get("stream", True),
        "taxonomy": job_config.get("taxonomy"),
        "retries": job_config.get("classification_retries", 0),
        "taxonomy_fast_path": job_config.get("taxonomy_fast_path", True),
        "choice_constraints": bool(llm.get("choice_constraints", False)),
        "extra_body": llm.get("extra_body") if isinstance(llm.get("extra_body"), dict) else None,
    }


def _extract_one_file(fp: str, job_config: dict) -> dict:
    name = Path(fp).name
    t0 = time.perf_counter()
    title, content_for_llm, extract_meta = extract_title_abstract_body(
        fp, max_chars_for_llm=job_config["max_chars"]
    )
    t_extract = time.perf_counter() - t0
    return {
        "file_path": fp,
        "file_name": name,
        "title": title or "",
        "content": content_for_llm or "",
        "extract_meta": extract_meta or {},
        "t_extract": t_extract,
    }


def _classify_extracted(item: dict, job_config: dict) -> dict:
    result = dict(item)
    extract_meta = result.get("extract_meta") or {}
    if _should_mark_extract_failure(extract_meta, job_config["min_content_chars_for_classification"]):
        domain_cn, domain_en = _extraction_failure_result(extract_meta)
        result.update({"domain_cn": domain_cn, "domain_en": domain_en, "t_llm": 0.0})
        return result

    if job_config["provider"] == "mock":
        domain_cn, domain_en = identify_domain(
            result.get("title", ""),
            result.get("content", ""),
            provider="mock",
            taxonomy=job_config.get("taxonomy"),
        )
        result.update({"domain_cn": domain_cn, "domain_en": domain_en, "t_llm": 0.0})
        return result

    t1 = time.perf_counter()
    domain_cn, domain_en = identify_domain(
        result.get("title", ""),
        result.get("content", ""),
        **_build_llm_kwargs(job_config),
    )
    result.update({
        "domain_cn": domain_cn,
        "domain_en": domain_en,
        "t_llm": time.perf_counter() - t1,
    })
    return result


def _failure_result(file_path: str, file_name: str, message: str, extract_meta: Optional[dict] = None, t_extract: float = 0.0) -> dict:
    return {
        "file_path": file_path,
        "file_name": file_name,
        "domain_cn": "处理失败",
        "domain_en": (message or "处理失败")[:200],
        "extract_meta": extract_meta or {
            "extract_source": "empty",
            "raw_text_len": 0,
            "content_len": 0,
            "ocr_used": False,
            "extract_timeout": False,
        },
        "t_extract": t_extract,
        "t_llm": 0.0,
    }


def run_scan(config_path: str = "config.yaml", use_mock: bool = False) -> None:
    """根据配置扫描文献、识别领域并实时写入 CSV；支持断点续跑与运行日志。"""
    cfg = load_config(config_path)
    path_mapping = _load_path_mapping(config_path, cfg)
    if path_mapping["physical_root"] is not None and not path_mapping["physical_root"].exists():
        raise FileNotFoundError(f'Configured physical_root does not exist: {path_mapping["physical_root"]}')
    dirs = [
        _resolve_literature_dir(config_path, raw_dir, path_mapping)
        for raw_dir in cfg.get("literature_dirs", ["./papers"])
    ]
    exts = cfg.get("extensions", [".pdf"])
    llm_cfg = cfg.get("llm", {})
    max_chars = cfg.get("max_chars_for_llm", 800)
    max_prompt_chars = cfg.get("max_prompt_chars", 4096)
    clear_context_every_n = cfg.get("clear_context_every_n")
    taxonomy_path = _resolve_config_relative_path(
        config_path,
        cfg.get("taxonomy_path", "./taxonomy.yaml"),
    )
    out = cfg.get("output", {})
    output_flags = _normalize_output_flags(out)
    csv_path = _resolve_config_relative_path(config_path, out.get("csv_path", "./literature_domains.csv"))
    log_path = _resolve_config_relative_path(config_path, out.get("log_path", "./scan.log"))
    concurrency = max(1, int(cfg.get("concurrency", 1)))
    extract_concurrency = max(1, int(cfg.get("extract_concurrency", concurrency)))
    llm_concurrency = max(1, int(cfg.get("llm_concurrency", concurrency)))
    classification_retries = max(0, int(cfg.get("classification_retries", 0)))
    taxonomy_fast_path = bool(cfg.get("taxonomy_fast_path", True))
    min_content_chars = max(1, int(cfg.get("min_content_chars_for_classification", DEFAULT_MIN_CONTENT_CHARS)))
    taxonomy = load_taxonomy(taxonomy_path)

    log = setup_logging(log_path)
    if path_mapping["physical_root"] is not None:
        log.info("physical root: %s", path_mapping["physical_root"])
    if path_mapping["logical_root"]:
        log.info("logical root: %s", path_mapping["logical_root"])
        normalized_count = normalize_csv_file_paths(
            csv_path,
            path_normalizer=lambda raw_path: _stored_file_path(raw_path, path_mapping),
        )
        if normalized_count > 0:
            log.info("normalized existing CSV paths: %d", normalized_count)
    log.info("日志文件: %s", log_path)
    if taxonomy and isinstance(taxonomy.get("level1"), dict):
        log.info("已加载 taxonomy: %s（一级领域 %d 个）", taxonomy_path, len(taxonomy["level1"]))
    else:
        log.warning("未加载有效 taxonomy: %s；当前版本将统一回退为“未分类”，请检查 taxonomy.yaml。", taxonomy_path)

    all_files = collect_files(dirs, exts)
    if not all_files:
        log.warning("未找到任何文献文件，请检查 config.yaml 中的 literature_dirs 与 extensions。")
        return

    done_paths = load_processed_paths(
        csv_path,
        path_normalizer=lambda raw_path: _stored_file_path(raw_path, path_mapping),
    )
    files = [f for f in all_files if _stored_file_path(f, path_mapping) not in done_paths]
    if not files:
        log.info("所有文献已处理完毕，无需继续。")
        return
    if len(done_paths) > 0:
        log.info("断点续跑: 已跳过 %d 篇，待处理 %d 篇。", len(done_paths), len(files))

    if use_mock:
        log.info("【模拟模式】未调用大模型，使用简单规则生成领域标签。")
    log.info("提取并发: %d | 模型并发: %d", extract_concurrency, llm_concurrency)

    if not use_mock:
        log.info(
            "classification retries: %d | taxonomy fast path: %s | choice constraints: %s | min content chars: %d",
            classification_retries,
            taxonomy_fast_path,
            bool(llm_cfg.get("choice_constraints", False)),
            min_content_chars,
        )

    job_config = {
        "max_chars": max_chars,
        "max_prompt_chars": max_prompt_chars,
        "provider": "mock" if use_mock else llm_cfg.get("provider", "ollama"),
        "llm_cfg": llm_cfg,
        "stream": bool(cfg.get("llm", {}).get("stream", True)),
        "taxonomy": taxonomy,
        "classification_retries": classification_retries,
        "taxonomy_fast_path": taxonomy_fast_path,
        "min_content_chars_for_classification": min_content_chars,
    }

    writer = CsvWriterAsync(
        csv_path,
        headers=_build_csv_headers(output_flags),
        path_normalizer=lambda raw_path: _stored_file_path(raw_path, path_mapping),
    )
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
                    max_prompt_chars=max_prompt_chars,
                    stream=job_config.get("stream", True),
                    taxonomy=taxonomy,
                    retries=classification_retries,
                    taxonomy_fast_path=taxonomy_fast_path,
                    choice_constraints=bool(llm_cfg.get("choice_constraints", False)),
                    extra_body=llm_cfg.get("extra_body") if isinstance(llm_cfg.get("extra_body"), dict) else None,
                )
                log.info("模型预热完成")
            except Exception as e:
                log.debug("预热请求忽略: %s", e)
        file_queue: "queue.Queue[str]" = queue.Queue()
        classify_queue: "queue.Queue[Optional[dict]]" = queue.Queue(maxsize=max(4, llm_concurrency * 2))
        for fp in files:
            file_queue.put(fp)

        done_lock = threading.Lock()
        done_count = {"value": 0}

        def emit_result(result: dict) -> None:
            writer.put(
                result["file_path"],
                result["file_name"],
                result["domain_cn"],
                result["domain_en"],
                extra_fields=_build_row_extra_fields(result, output_flags),
            )
            with done_lock:
                done_count["value"] += 1
                current = done_count["value"]
            log.info(_format_result_log(current, total, result, output_flags))
            if current % 50 == 0:
                gc.collect()

        def extractor_worker() -> None:
            while True:
                try:
                    fp = file_queue.get_nowait()
                except queue.Empty:
                    return
                try:
                    classify_queue.put(_extract_one_file(fp, job_config))
                except Exception as exc:
                    name = Path(fp).name
                    log.exception("提取失败 [%s]: %s", fp, exc)
                    emit_result(_failure_result(fp, name, str(exc)))
                finally:
                    file_queue.task_done()

        def classifier_worker() -> None:
            while True:
                item = classify_queue.get()
                if item is None:
                    classify_queue.task_done()
                    break
                try:
                    emit_result(_classify_extracted(item, job_config))
                except Exception as exc:
                    log.exception("处理失败 [%s]: %s", item.get("file_path"), exc)
                    emit_result(
                        _failure_result(
                            item.get("file_path", ""),
                            item.get("file_name", ""),
                            str(exc),
                            extract_meta=item.get("extract_meta"),
                            t_extract=float(item.get("t_extract", 0.0)),
                        )
                    )
                finally:
                    classify_queue.task_done()

        extract_threads = [
            threading.Thread(target=extractor_worker, name=f"extract-{i+1}", daemon=True)
            for i in range(extract_concurrency)
        ]
        classify_threads = [
            threading.Thread(target=classifier_worker, name=f"classify-{i+1}", daemon=True)
            for i in range(llm_concurrency)
        ]

        for thread in classify_threads:
            thread.start()
        for thread in extract_threads:
            thread.start()

        for thread in extract_threads:
            thread.join()

        for _ in classify_threads:
            classify_queue.put(None)

        for thread in classify_threads:
            thread.join()

        if clear_context_every_n and not use_mock:
            clear_llm_context(
                provider=job_config["provider"],
                model=llm_cfg.get("model", "qwen2.5:7b"),
                api_base=llm_cfg.get("api_base", "http://localhost:1234/v1"),
                api_key=llm_cfg.get("api_key", "not-needed"),
            )
    finally:
        writer.close()

    log.info("本次完成 %d 篇，结果已写入: %s", len(files), csv_path)


def run_list_domains(config_path: str = "config.yaml") -> None:
    """列出 CSV 中所有出现过的领域。"""
    cfg = load_config(config_path)
    path_mapping = _load_path_mapping(config_path, cfg)
    csv_path = _resolve_config_relative_path(
        config_path,
        cfg.get("output", {}).get("csv_path", "./literature_domains.csv"),
    )
    if not Path(csv_path).exists():
        print("CSV 不存在，请先运行 scan。")
        return
    if path_mapping["logical_root"]:
        normalize_csv_file_paths(
            csv_path,
            path_normalizer=lambda raw_path: _stored_file_path(raw_path, path_mapping),
        )
    domains = list_domains_from_csv(csv_path)
    print("已记录的领域：")
    for d in domains:
        print(f"  - {d}")


def run_query(domain: str, config_path: str = "config.yaml") -> None:
    """按领域筛选，打印该领域下的所有文献路径。"""
    cfg = load_config(config_path)
    path_mapping = _load_path_mapping(config_path, cfg)
    csv_path = _resolve_config_relative_path(
        config_path,
        cfg.get("output", {}).get("csv_path", "./literature_domains.csv"),
    )
    if not Path(csv_path).exists():
        print("CSV 不存在，请先运行 scan。")
        return
    if path_mapping["logical_root"]:
        normalize_csv_file_paths(
            csv_path,
            path_normalizer=lambda raw_path: _stored_file_path(raw_path, path_mapping),
        )
    rows = query_by_domain_from_csv(csv_path, domain)
    if not rows:
        print(f"领域「{domain}」下没有文献。")
        return
    print(f"领域「{domain}」下的文献（共 {len(rows)} 条）：")
    for path, name, _ in rows:
        print(f"  {path}")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "__pdf_worker__":
        raise SystemExit(run_pdf_worker_cli(sys.argv[2:]))

    parser = argparse.ArgumentParser(description="文献领域识别：用本地大模型打标签并实时写入 CSV，支持断点续跑")
    parser.add_argument("--config", "-c", default="config.yaml", help="配置文件路径")
    sub = parser.add_subparsers(dest="command", help="子命令")

    p_scan = sub.add_parser("scan", help="扫描文献目录，识别领域并写入 CSV（可断点续跑）")
    p_scan.add_argument("--mock", "-m", action="store_true", help="模拟模式：不调用大模型，用简单规则生成领域")
    sub.add_parser("domains", help="列出所有已记录的领域")
    p_query = sub.add_parser("filter", help="按领域筛选文献")
    p_query.add_argument("domain", help="领域名称，如：计算机科学")

    args = parser.parse_args()
    config_path = _normalize_cli_config_path(args.config)
    if args.command == "scan":
        run_scan(config_path, use_mock=getattr(args, "mock", False))
    elif args.command == "domains":
        run_list_domains(config_path)
    elif args.command == "filter":
        run_query(args.domain, config_path)
    else:
        parser.print_help()
        print("\n示例：")
        print("  python main.py scan              # 扫描并打标签（断点续跑）")
        print("  python main.py domains           # 查看所有领域")
        print("  python main.py filter 土木工程/岩土工程  # 筛选该领域文献")


if __name__ == "__main__":
    main()
