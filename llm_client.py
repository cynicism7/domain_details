# -*- coding: utf-8 -*-
"""Local LLM helpers for domain classification."""

import json
import re
import threading
from typing import Dict, List, Optional, Tuple

# Streamed responses can stop early once a valid field JSON appears.
_FIELD_JSON_PATTERN = re.compile(r'\{\s*"field"\s*:\s*"[^"]*"\s*\}')
_OPENAI_CLIENTS = threading.local()


def _normalize_domain(raw: str) -> str:
    """Normalize raw model output into a single short label."""
    if not raw or not isinstance(raw, str):
        return "未分类"

    text = raw.strip()
    for sep in ("\n", "，", "。", ",", "."):
        if sep in text:
            text = text.split(sep)[0].strip()
    text = re.sub(r"^(领域|学科|类别|一级领域|二级领域|领域：|学科：|类别：)\s*", "", text, flags=re.I)
    text = text.strip('"\' \t')
    return text if text else "未分类"


def ask_ollama(prompt: str, model: str = "qwen2.5:7b", timeout: int = 60, stream: bool = False) -> str:
    """Request a completion from a local Ollama server."""
    if stream:
        try:
            import requests

            response = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": model, "prompt": prompt, "stream": True},
                timeout=timeout,
                stream=True,
            )
            response.raise_for_status()
            buf = ""
            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue
                try:
                    part = json.loads(line)
                except ValueError:
                    continue
                buf += part.get("response") or ""
                if _FIELD_JSON_PATTERN.search(buf):
                    return buf.strip()
            return buf.strip()
        except Exception as exc:
            return f"[Ollama 请求失败: {exc}]"

    try:
        import ollama
    except ImportError:
        try:
            import requests

            response = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": model, "prompt": prompt, "stream": False},
                timeout=timeout,
            )
            response.raise_for_status()
            return (response.json().get("response") or "").strip()
        except Exception as exc:
            return f"[Ollama 请求失败: {exc}]"

    try:
        payload = ollama.generate(model=model, prompt=prompt)
        return (payload.get("response") or "").strip()
    except Exception as exc:
        return f"[Ollama 请求失败: {exc}]"


def clear_llm_context(
    provider: str,
    model: str = "qwen2.5:7b",
    api_base: str = "http://localhost:1234/v1",
    api_key: str = "not-needed",
) -> None:
    """Release local model state where possible."""
    if provider == "mock":
        return

    if provider == "openai_api":
        try:
            import requests

            base = api_base.rstrip("/").replace("/v1", "")
            url = f"{base}/api/v1/models/unload"
            headers = {"Content-Type": "application/json"}
            if api_key and api_key != "not-needed":
                headers["Authorization"] = f"Bearer {api_key}"
            response = requests.post(
                url,
                json={"instance_id": model},
                headers=headers,
                timeout=30,
            )
            response.raise_for_status()
        except Exception:
            pass
        return

    import gc

    gc.collect()


def ask_openai_api(
    prompt: str,
    model: str = "local-model",
    api_base: str = "http://localhost:1234/v1",
    api_key: str = "not-needed",
    timeout: int = 60,
    max_tokens: int = 128,
    temperature: float = 0.0,
    system_prompt: Optional[str] = None,
    stream: bool = False,
    extra_body: Optional[dict] = None,
) -> str:
    """Request a chat completion from an OpenAI-compatible local API."""
    try:
        client = _get_openai_client(api_base, api_key)
    except ImportError:
        return "[未安装 openai 包]"

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    request_kwargs = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "timeout": timeout,
    }
    if isinstance(extra_body, dict) and extra_body:
        request_kwargs["extra_body"] = extra_body

    try:
        if stream:
            response = client.chat.completions.create(
                stream=True,
                **request_kwargs,
            )
            buf = ""
            for chunk in response:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if getattr(delta, "content", None):
                    buf += delta.content
                    if _FIELD_JSON_PATTERN.search(buf):
                        return buf.strip()
            return buf.strip()

        response = client.chat.completions.create(
            **request_kwargs,
        )
        message = response.choices[0].message.content if response.choices else ""
        return (message or "").strip()
    except Exception as exc:
        err_msg = str(exc).strip() or repr(exc)
        status_code = getattr(exc, "status_code", None) or getattr(getattr(exc, "response", None), "status_code", None)
        if status_code is not None:
            err_msg = f"HTTP {status_code} - {err_msg}"
        if status_code == 502:
            err_msg += " （502 常见原因：config 里 llm.model 与 LM Studio 当前加载的模型名不一致，或模型未加载。请在 LM Studio 中确认已加载模型，且「开发」/ Local Server 里显示的 model 与 config 中 model 完全一致）"
        return f"[API 请求失败: {err_msg}]"


DEFAULT_SYSTEM_PROMPT = """你是文献领域分类器。
你只需要做标签选择，不需要输出推理过程。
禁止使用 <think> 或任何思考标签，不要输出解释。
始终只输出一行 JSON：{"field": "标签名"}。"""

DEFAULT_CHOICE_SYSTEM_PROMPT = """You are a paper domain classifier.
Choose exactly one label from the provided candidates.
Do not output reasoning, JSON, <think>, or any extra text.
Return only the chosen label."""


def _get_openai_client(api_base: str, api_key: str):
    from openai import OpenAI

    cache = getattr(_OPENAI_CLIENTS, "by_base", None)
    if cache is None:
        cache = {}
        _OPENAI_CLIENTS.by_base = cache

    cache_key = (api_base.rstrip("/"), api_key)
    client = cache.get(cache_key)
    if client is None:
        client = OpenAI(base_url=api_base, api_key=api_key)
        cache[cache_key] = client
    return client


DEFAULT_MAX_PROMPT_CHARS = 4096


def _truncate_for_context(text: str, max_chars: int) -> str:
    if not text or max_chars <= 0:
        return ""
    text = text.strip()
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    for sep in ("\n", "。", ".", " ", "，", ",", ";"):
        last = truncated.rfind(sep)
        if last > max_chars // 2:
            return truncated[: last + 1].strip()
    return truncated.strip()


def _normalize_key(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"[\s\-_/:：,，。;；|（）()【】\[\]<>]+", "", text.lower())


def _contains_cjk(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text or ""))


def _dedupe(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for item in items:
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _taxonomy_level1_map(taxonomy: Optional[dict]) -> Dict[str, dict]:
    if not isinstance(taxonomy, dict):
        return {}
    level1 = taxonomy.get("level1")
    return level1 if isinstance(level1, dict) else {}


def _taxonomy_default_label(taxonomy: Optional[dict]) -> str:
    if isinstance(taxonomy, dict):
        label = taxonomy.get("default_label")
        if isinstance(label, str) and label.strip():
            return label.strip()
    return "未分类"


def _taxonomy_default_secondary_label(taxonomy: Optional[dict]) -> str:
    if isinstance(taxonomy, dict):
        label = taxonomy.get("default_secondary_label")
        if isinstance(label, str) and label.strip():
            return label.strip()
    return "其他"


def _taxonomy_global_aliases(taxonomy: Optional[dict]) -> Dict[str, str]:
    aliases = taxonomy.get("global_aliases") if isinstance(taxonomy, dict) else None
    if not isinstance(aliases, dict):
        return {}
    return {str(key): str(value) for key, value in aliases.items() if str(key).strip() and str(value).strip()}


def _taxonomy_secondary_aliases(taxonomy: Optional[dict], primary: str) -> Dict[str, str]:
    if not isinstance(taxonomy, dict):
        return {}
    secondary_aliases = taxonomy.get("secondary_aliases")
    if not isinstance(secondary_aliases, dict):
        return {}
    aliases = secondary_aliases.get(primary)
    if not isinstance(aliases, dict):
        return {}
    return {str(key): str(value) for key, value in aliases.items() if str(key).strip() and str(value).strip()}


def _taxonomy_primary_aliases(taxonomy: Optional[dict]) -> Dict[str, str]:
    alias_map: Dict[str, str] = {}
    level1 = _taxonomy_level1_map(taxonomy)
    for primary, node in level1.items():
        alias_map[primary] = primary
        children = node.get("children") if isinstance(node, dict) else []
        for child in children or []:
            alias_map[str(child)] = primary
        for alias, target in _taxonomy_secondary_aliases(taxonomy, primary).items():
            alias_map[alias] = primary
            alias_map[target] = primary
    for alias, target in _taxonomy_global_aliases(taxonomy).items():
        primary = _find_primary_for_taxonomy_target(target, taxonomy)
        if primary:
            alias_map[alias] = primary
            alias_map[target] = primary
    return alias_map


def _taxonomy_level1_candidates(taxonomy: Optional[dict]) -> List[str]:
    candidates = list(_taxonomy_level1_map(taxonomy).keys())
    default_label = _taxonomy_default_label(taxonomy)
    if default_label not in candidates:
        candidates.append(default_label)
    return candidates


def _taxonomy_level2_candidates(taxonomy: Optional[dict], primary: str) -> List[str]:
    level1 = _taxonomy_level1_map(taxonomy)
    node = level1.get(primary, {})
    children = node.get("children") if isinstance(node, dict) else []
    if not isinstance(children, list):
        children = []
    candidates = [str(item).strip() for item in children if str(item).strip()]
    fallback = _taxonomy_default_secondary_label(taxonomy)
    if fallback not in candidates:
        candidates.append(fallback)
    return _dedupe(candidates)


def _find_primary_for_taxonomy_target(target: str, taxonomy: Optional[dict]) -> Optional[str]:
    if not target:
        return None
    target_norm = _normalize_key(target)
    level1 = _taxonomy_level1_map(taxonomy)
    for primary, node in level1.items():
        if _normalize_key(primary) == target_norm:
            return primary
        children = node.get("children") if isinstance(node, dict) else []
        for child in children or []:
            if _normalize_key(str(child)) == target_norm:
                return primary
    return None


def _resolve_candidate(raw: str, candidates: List[str], alias_map: Optional[Dict[str, str]] = None) -> Optional[str]:
    if not raw:
        return None

    candidate_map = {_normalize_key(candidate): candidate for candidate in candidates}
    alias_lookup = {
        _normalize_key(alias): target
        for alias, target in (alias_map or {}).items()
        if _normalize_key(alias) and _normalize_key(target) in candidate_map
    }

    variants = [_normalize_domain(raw)] + [part.strip() for part in re.split(r"[|/＞>]+", raw) if part.strip()]
    for variant in variants:
        norm = _normalize_key(variant)
        if not norm:
            continue
        if norm in candidate_map:
            return candidate_map[norm]
        if norm in alias_lookup:
            return candidate_map[_normalize_key(alias_lookup[norm])]
        matches = [candidate for key, candidate in candidate_map.items() if key and (key in norm or norm in key)]
        matches = _dedupe(matches)
        if len(matches) == 1:
            return matches[0]

    return None


def _parse_field_value(raw: str) -> Optional[str]:
    if not raw or not isinstance(raw, str):
        return None

    work = raw.split("</think>")[-1].strip() if "</think>" in raw else raw.strip()
    match = re.search(r'\{\s*"field"\s*:\s*"([^"]+)"\s*\}', work)
    if match:
        return _normalize_domain(match.group(1))

    try:
        for match in re.finditer(r'\{[^{}]*"field"[^{}]*\}', work):
            try:
                data = json.loads(match.group(0))
            except json.JSONDecodeError:
                continue
            field = data.get("field")
            if isinstance(field, str) and field.strip():
                return _normalize_domain(field)
    except Exception:
        pass

    field_line = _normalize_domain(work)
    if field_line and field_line != "未分类":
        return field_line
    return None


def _call_model(
    prompt: str,
    *,
    provider: str,
    model: str,
    api_base: str,
    api_key: str,
    max_tokens: int,
    temperature: float,
    timeout: int,
    system_prompt: Optional[str],
    stream: bool,
    extra_body: Optional[dict] = None,
) -> str:
    if provider == "openai_api":
        return ask_openai_api(
            prompt,
            model=model,
            api_base=api_base,
            api_key=api_key,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout,
            system_prompt=system_prompt,
            stream=stream,
            extra_body=extra_body,
        )
    return ask_ollama(prompt, model=model, timeout=timeout, stream=stream)


def _build_prompt(prefix: str, content: str, system_prompt: str, max_prompt_chars: int) -> str:
    max_content = max(0, max_prompt_chars - len(system_prompt) - len(prefix))
    return prefix + _truncate_for_context(content, max_content)


def _merge_extra_body(base: Optional[dict], override: Optional[dict]) -> Optional[dict]:
    merged = dict(base) if isinstance(base, dict) else {}
    if isinstance(override, dict):
        merged.update(override)
    return merged or None


def _build_level1_prompt(
    title: str,
    content: str,
    candidates: List[str],
    system_prompt: str,
    max_prompt_chars: int,
    default_label: str,
    response_mode: str = "json",
) -> str:
    if response_mode == "label":
        prefix = """Classify the paper into exactly one primary domain from the candidate list.
Return only the label text and nothing else.
Candidates:
%s
If uncertain, return %s.

[File]
%s

[Paper]
""" % ("\n".join(candidates), json.dumps(default_label, ensure_ascii=False), title)
        return _build_prompt(prefix, content, system_prompt, max_prompt_chars)

    prefix = """判断下面文献的一级领域。
只能从以下候选中选择一项，不允许自造标签：
%s
如果无法判断，输出“%s”。
直接输出一行 JSON，不要 <think>、不要解释：
{"field": "一级领域"}

【文件名】%s

【文献信息】
""" % ("、".join(candidates), default_label, title)
    return _build_prompt(prefix, content, system_prompt, max_prompt_chars)


def _build_level2_prompt(
    title: str,
    content: str,
    primary: str,
    candidates: List[str],
    system_prompt: str,
    max_prompt_chars: int,
    default_secondary: str,
    response_mode: str = "json",
) -> str:
    if response_mode == "label":
        prefix = """The paper is already classified into the primary domain %s.
Choose exactly one secondary domain from the candidate list.
Return only the label text and nothing else.
Candidates:
%s
If uncertain, return %s.

[File]
%s

[Paper]
""" % (
            json.dumps(primary, ensure_ascii=False),
            "\n".join(candidates),
            json.dumps(default_secondary, ensure_ascii=False),
            title,
        )
        return _build_prompt(prefix, content, system_prompt, max_prompt_chars)

    prefix = """已知该文献的一级领域是“%s”。
现在只在以下二级领域中选择一项，不允许自造标签：
%s
如果无法判断，输出“%s”。
直接输出一行 JSON，不要 <think>、不要解释：
{"field": "二级领域"}

【文件名】%s

【文献信息】
""" % (primary, "、".join(candidates), default_secondary, title)
    return _build_prompt(prefix, content, system_prompt, max_prompt_chars)


def _score_candidates_from_text(text: str, candidates: List[str], alias_map: Optional[Dict[str, str]] = None) -> Optional[str]:
    raw_lower = (text or "").lower()
    norm_text = _normalize_key(text)
    if not norm_text:
        return None

    scores = {candidate: 0 for candidate in candidates}
    for candidate in candidates:
        candidate_norm = _normalize_key(candidate)
        if candidate_norm and candidate_norm in norm_text:
            scores[candidate] += 6

    for alias, target in (alias_map or {}).items():
        alias_norm = _normalize_key(alias)
        if not alias_norm or target not in scores:
            continue
        if _contains_cjk(alias):
            if alias_norm in norm_text:
                scores[target] += 5
            continue
        alias_word = re.sub(r"[^a-z0-9]+", "", alias.lower())
        if len(alias_word) < 4:
            if re.search(rf"(?<![a-z0-9]){re.escape(alias.lower())}(?![a-z0-9])", raw_lower):
                scores[target] += 5
            continue
        if alias.lower() in raw_lower or alias_norm in norm_text:
            scores[target] += 5

    best_score = max(scores.values()) if scores else 0
    if best_score <= 0:
        return None
    best = [candidate for candidate, score in scores.items() if score == best_score]
    return best[0] if len(best) == 1 else None


def _guess_primary_from_taxonomy(text: str, taxonomy: Optional[dict]) -> Optional[str]:
    candidates = _taxonomy_level1_candidates(taxonomy)
    default_label = _taxonomy_default_label(taxonomy)
    candidates = [candidate for candidate in candidates if candidate != default_label]
    return _score_candidates_from_text(text, candidates, _taxonomy_primary_aliases(taxonomy))


def _guess_secondary_from_taxonomy(text: str, taxonomy: Optional[dict], primary: str) -> Optional[str]:
    candidates = _taxonomy_level2_candidates(taxonomy, primary)
    fallback = _taxonomy_default_secondary_label(taxonomy)
    candidates = [candidate for candidate in candidates if candidate != fallback]
    alias_map = _taxonomy_secondary_aliases(taxonomy, primary)
    for alias, target in _taxonomy_global_aliases(taxonomy).items():
        if _find_primary_for_taxonomy_target(target, taxonomy) == primary:
            alias_map[alias] = target
    return _score_candidates_from_text(text, candidates, alias_map)


def _compose_domain_label(primary: str, secondary: str, taxonomy: Optional[dict]) -> str:
    fallback = _taxonomy_default_secondary_label(taxonomy)
    if not primary or primary == _taxonomy_default_label(taxonomy):
        return _taxonomy_default_label(taxonomy)
    if not secondary:
        return primary
    if secondary == fallback:
        return f"{primary}/{fallback}"
    return f"{primary}/{secondary}"


def _uncategorized_result(taxonomy: Optional[dict]) -> Tuple[str, str]:
    return _taxonomy_default_label(taxonomy), "Uncategorized"


def _resolve_with_retries(
    prompt: str,
    candidates: List[str],
    alias_map: Optional[Dict[str, str]],
    call_fn,
    retries: int,
    extra_body: Optional[dict] = None,
) -> Optional[str]:
    for _ in range(max(0, retries) + 1):
        raw = call_fn(prompt, extra_body=extra_body)
        field = _parse_field_value(raw)
        if not field:
            continue
        resolved = _resolve_candidate(field, candidates, alias_map)
        if resolved:
            return resolved
    return None


def _taxonomy_identify(
    title: str,
    content: str,
    *,
    provider: str,
    model: str,
    api_base: str,
    api_key: str,
    max_tokens: int,
    temperature: float,
    timeout: int,
    system_prompt: str,
    max_prompt_chars: int,
    stream: bool,
    taxonomy: dict,
    retries: int,
    taxonomy_fast_path: bool,
    choice_constraints: bool,
    extra_body: Optional[dict],
) -> Tuple[str, str]:
    default_label = _taxonomy_default_label(taxonomy)
    default_secondary = _taxonomy_default_secondary_label(taxonomy)
    level1_candidates = _taxonomy_level1_candidates(taxonomy)
    combined_text = title + "\n" + content
    use_choice_constraints = provider == "openai_api" and choice_constraints

    title_primary = _guess_primary_from_taxonomy(title, taxonomy) if taxonomy_fast_path else None
    if title_primary and title_primary != default_label:
        title_secondary = _guess_secondary_from_taxonomy(title, taxonomy, title_primary)
        if title_secondary:
            label = _compose_domain_label(title_primary, title_secondary, taxonomy)
            return label, label

    prompt_level1 = _build_level1_prompt(
        title,
        content,
        level1_candidates,
        system_prompt,
        max_prompt_chars,
        default_label,
        response_mode="label" if use_choice_constraints else "json",
    )

    def _call(prompt: str, extra_body: Optional[dict] = None) -> str:
        return _call_model(
            prompt,
            provider=provider,
            model=model,
            api_base=api_base,
            api_key=api_key,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout,
            system_prompt=system_prompt,
            stream=stream,
            extra_body=extra_body,
        )

    level1_extra_body = _merge_extra_body(
        extra_body,
        {"structured_outputs": {"choice": level1_candidates}} if use_choice_constraints else None,
    )

    primary = title_primary
    if not primary:
        primary = _resolve_with_retries(
            prompt_level1,
            level1_candidates,
            _taxonomy_primary_aliases(taxonomy),
            _call,
            retries,
            extra_body=level1_extra_body,
        )

    if not primary:
        primary = _guess_primary_from_taxonomy(combined_text, taxonomy) or default_label

    if primary == default_label:
        return default_label, "Uncategorized"

    level2_candidates = _taxonomy_level2_candidates(taxonomy, primary)
    prompt_level2 = _build_level2_prompt(
        title,
        content,
        primary,
        level2_candidates,
        system_prompt,
        max_prompt_chars,
        default_secondary,
        response_mode="label" if use_choice_constraints else "json",
    )

    level2_extra_body = _merge_extra_body(
        extra_body,
        {"structured_outputs": {"choice": level2_candidates}} if use_choice_constraints else None,
    )

    secondary = _guess_secondary_from_taxonomy(title, taxonomy, primary) if taxonomy_fast_path else None
    if not secondary:
        secondary = _resolve_with_retries(
            prompt_level2,
            level2_candidates,
            _taxonomy_secondary_aliases(taxonomy, primary),
            _call,
            retries,
            extra_body=level2_extra_body,
        )

    if not secondary:
        secondary = _guess_secondary_from_taxonomy(combined_text, taxonomy, primary) or default_secondary

    label = _compose_domain_label(primary, secondary, taxonomy)
    return label, label


def identify_domain(
    title: str,
    full_text: str,
    *,
    provider: str = "ollama",
    model: str = "qwen2.5:7b",
    api_base: str = "http://localhost:1234/v1",
    api_key: str = "not-needed",
    max_tokens: int = 128,
    temperature: float = 0.0,
    timeout: int = 60,
    system_prompt: Optional[str] = None,
    max_prompt_chars: Optional[int] = None,
    stream: bool = False,
    taxonomy: Optional[dict] = None,
    retries: int = 1,
    taxonomy_fast_path: bool = True,
    choice_constraints: bool = False,
    extra_body: Optional[dict] = None,
) -> Tuple[str, str]:
    """
    Identify the best-fitting domain label for a paper.
    When taxonomy is provided, classification becomes a controlled two-stage process.
    """
    title_s = (title or "Unknown").strip()
    content_s = (full_text or "No Content Detected").strip()
    if system_prompt is not None:
        sys_msg = system_prompt
    elif provider == "openai_api" and choice_constraints:
        sys_msg = DEFAULT_CHOICE_SYSTEM_PROMPT
    else:
        sys_msg = DEFAULT_SYSTEM_PROMPT
    cap = max_prompt_chars if max_prompt_chars is not None else DEFAULT_MAX_PROMPT_CHARS
    content_s = _truncate_for_context(content_s, max(0, cap - len(sys_msg) - 512))

    if not _taxonomy_level1_map(taxonomy):
        return _uncategorized_result(taxonomy)

    if provider == "mock":
        return _identify_domain_mock(title_s, "", content_s, taxonomy=taxonomy)

    return _taxonomy_identify(
        title_s,
        content_s,
        provider=provider,
        model=model,
        api_base=api_base,
        api_key=api_key,
        max_tokens=max_tokens,
        temperature=temperature,
        timeout=timeout,
        system_prompt=sys_msg,
        max_prompt_chars=cap,
        stream=stream,
        taxonomy=taxonomy,
        retries=retries,
        taxonomy_fast_path=taxonomy_fast_path,
        choice_constraints=choice_constraints,
        extra_body=extra_body,
    )


def _identify_domain_mock(
    title: str,
    abstract: str,
    body: str,
    *,
    taxonomy: Optional[dict] = None,
) -> Tuple[str, str]:
    """
    Deterministic mock classifier for local workflow verification.
    """
    text = f"{title} {abstract} {body}".lower()

    if not _taxonomy_level1_map(taxonomy):
        return _uncategorized_result(taxonomy)

    primary = _guess_primary_from_taxonomy(text, taxonomy)
    if not primary:
        keyword_domains = [
            (["computer", "computing", "algorithm", "machine learning", "deep learning", "neural network", "software"], "计算机科学"),
            (["bioinformatics", "genome", "proteome"], "生物信息学"),
            (["medical", "medicine", "hospital", "clinical", "tumor", "cancer"], "医学"),
            (["biology", "cell", "gene", "genetic", "life science"], "生物学"),
            (["chemistry", "molecule", "molecular"], "化学"),
            (["physics", "quantum"], "物理学"),
            (["material", "nanomaterial", "polymer"], "材料科学"),
            (["agriculture", "crop"], "农学"),
            (["econom", "finance"], "经济学"),
            (["geology", "geological", "slope", "landslide", "rock"], "地质学"),
            (["civil", "bridge", "tunnel", "geotechnical"], "土木工程"),
        ]
        valid_primary = set(_taxonomy_level1_map(taxonomy).keys())
        for keywords, candidate in keyword_domains:
            if candidate in valid_primary and any(keyword in text for keyword in keywords):
                primary = candidate
                break
    if not primary:
        return _uncategorized_result(taxonomy)
    secondary = _guess_secondary_from_taxonomy(text, taxonomy, primary) or _taxonomy_default_secondary_label(taxonomy)
    label = _compose_domain_label(primary, secondary, taxonomy)
    return label, label
