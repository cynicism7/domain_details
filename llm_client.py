# -*- coding: utf-8 -*-
"""本地大模型调用：支持 Ollama 与 OpenAI 兼容 API（如 LM Studio）。"""

import re
from typing import Optional, Tuple

# 流式响应中用于尽早截断：一旦看到完整 JSON 就返回
_FIELD_JSON_PATTERN = re.compile(r'\{\s*"field"\s*:\s*"[^"]*"\s*\}')


def _normalize_domain(raw: str) -> str:
    """从模型输出中提取单一领域标签，去除多余符号和换行。"""
    if not raw or not isinstance(raw, str):
        return "未分类"
    s = raw.strip()
    # 取第一行或第一个逗号/句号前
    for sep in ("\n", "，", "。", ",", "."):
        if sep in s:
            s = s.split(sep)[0].strip()
    # 去掉常见前缀和引号
    s = re.sub(r"^(领域|学科|类别|领域：|学科：|类别：)\s*", "", s, flags=re.I)
    s = s.strip('"\' \t')
    return s if s else "未分类"


def ask_ollama(prompt: str, model: str = "qwen2.5:7b", timeout: int = 60, stream: bool = False) -> str:
    """通过 Ollama 本地 API 请求，返回模型回复文本。stream=True 时流式接收，收到完整 JSON 即返回。"""
    if stream:
        try:
            import requests
            r = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": model, "prompt": prompt, "stream": True},
                timeout=timeout,
                stream=True,
            )
            r.raise_for_status()
            buf = ""
            for line in r.iter_lines(decode_unicode=True):
                if not line:
                    continue
                try:
                    import json as _json
                    part = _json.loads(line)
                    buf += (part.get("response") or "")
                    if _FIELD_JSON_PATTERN.search(buf):
                        return buf.strip()
                except (ValueError, KeyError):
                    continue
            return buf.strip()
        except Exception as e:
            return f"[Ollama 请求失败: {e}]"
    try:
        import ollama
    except ImportError:
        try:
            import requests
            r = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": model, "prompt": prompt, "stream": False},
                timeout=timeout,
            )
            r.raise_for_status()
            return (r.json().get("response") or "").strip()
        except Exception as e:
            return f"[Ollama 请求失败: {e}]"
    try:
        resp = ollama.generate(model=model, prompt=prompt)
        return (resp.get("response") or "").strip()
    except Exception as e:
        return f"[Ollama 请求失败: {e}]"


def clear_llm_context(
    provider: str,
    model: str = "qwen2.5:7b",
    api_base: str = "http://localhost:1234/v1",
    api_key: str = "not-needed",
) -> None:
    """
    定期清理本地模型上下文/显存，避免大批量跑时内存累积。领域判断无需对话历史，可安全调用。
    - LM Studio（openai_api）：调用 LM Studio 原生接口 POST /api/v1/models/unload 卸载当前模型，释放显存；
      下次请求时会由 LM Studio 自动重新加载，从而清空上下文。
    - 其他 provider（如 ollama）：仅做进程内 gc。
    """
    if provider == "mock":
        return
    if provider == "openai_api":
        try:
            import requests
            # 从 OpenAI 兼容 base（如 http://127.0.0.1:1234/v1）得到 host，拼 LM Studio 原生 unload 地址
            base = api_base.rstrip("/").replace("/v1", "")
            url = f"{base}/api/v1/models/unload"
            headers = {"Content-Type": "application/json"}
            if api_key and api_key != "not-needed":
                headers["Authorization"] = f"Bearer {api_key}"
            r = requests.post(
                url,
                json={"instance_id": model},
                headers=headers,
                timeout=30,
            )
            r.raise_for_status()
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
) -> str:
    """通过 OpenAI 兼容 API（如 LM Studio）请求。stream=True 时收到完整 JSON 即返回，可略缩短等待。"""
    try:
        from openai import OpenAI
    except ImportError:
        return "[未安装 openai 包]"
    client = OpenAI(base_url=api_base, api_key=api_key)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    try:
        if stream:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=timeout,
                stream=True,
            )
            buf = ""
            for chunk in resp:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if getattr(delta, "content", None):
                    buf += delta.content
                    if _FIELD_JSON_PATTERN.search(buf):
                        return buf.strip()
            return buf.strip()
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout,
        )
        msg = resp.choices[0].message.content if resp.choices else ""
        return (msg or "").strip()
    except Exception as e:
        err_msg = str(e).strip() or repr(e)
        status_code = getattr(e, "status_code", None) or getattr(getattr(e, "response", None), "status_code", None)
        if status_code is not None:
            err_msg = f"HTTP {status_code} - {err_msg}"
        return f"[API 请求失败: {err_msg}]"


import json


# 默认系统提示：约束思考型模型少用 <think>、直接输出 JSON，提升速度并避免被截断
DEFAULT_SYSTEM_PROMPT = """你是文献领域分类器。本任务只需从给定内容判断一个学科名称，无需推理过程。
禁止使用 <think> 或任何思考标签，不要输出解释，直接只输出一行 JSON：{"field": "学科名称"}。"""

# 单次请求上下文总长度上限（系统提示 + 用户提示），防止本地模型上下文过长导致内存越界
DEFAULT_MAX_PROMPT_CHARS = 4096

# 基础领域列表：优先让 AI 从此列表中选择；若不贴近再自行生成
PREFERRED_DOMAINS = [
    "细胞生物学", "分子生物学", "免疫学", "肿瘤学", "癌症生物学", "干细胞生物学", "发育生物学",
    "药理学", "毒理学", "再生医学", "组织工程", "疫苗学", "病毒学", "生物制药", "生物技术",
    "体外受精", "生殖生物学", "培养肉", "合成生物学", "微生物学", "植物生物学", "神经科学",
    "内分泌学", "代谢研究", "流行病学", "公共卫生",
]


def _truncate_for_context(s: str, max_chars: int) -> str:
    """将文本截断到 max_chars 以内，尽量在句末截断，避免单次请求上下文过长导致内存越界。"""
    if not s or max_chars <= 0:
        return ""
    s = s.strip()
    if len(s) <= max_chars:
        return s
    truncated = s[:max_chars]
    for sep in ("\n", "。", ".", " ", "，", ","):
        last = truncated.rfind(sep)
        if last > max_chars // 2:
            return truncated[: last + 1].strip()
    return truncated.strip()


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
    preferred_domains: Optional[list] = None,
    max_prompt_chars: Optional[int] = None,
    stream: bool = False,
    max_preferred_domains_in_prompt: Optional[int] = None,
) -> tuple[str, str]:
    """
    调用本地大模型识别文献最接近的最小领域，返回 (domain_cn, domain_en)。
    未分类或解析失败时会自动重试一次。
    stream=True 时流式接收，收到完整 JSON 即返回，可略缩短等待。
    max_preferred_domains_in_prompt 限制提示中领域数量，减少 token 以提速。
    """
    domains_list = preferred_domains if preferred_domains is not None else PREFERRED_DOMAINS
    if max_preferred_domains_in_prompt is not None and max_preferred_domains_in_prompt > 0:
        domains_list = domains_list[: max_preferred_domains_in_prompt]
    domains_str = "、".join(domains_list)
    title_s = (title or "Unknown").strip()
    content_s = (full_text or "No Content Detected").strip()

    # 固定前缀长度（不含文献正文），用于计算正文可用的最大长度
    prompt_prefix_tpl = """判断下面文献的最接近最小领域（学科）。
请优先从以下领域中选择最贴近的一项：%s
若以上均不贴近，再自行给出一个学科名称。只输出一个中文名。
直接输出一行 JSON，不要 <think>、不要解释：
{"field": "学科名称"}

【文件名】%s

【标题、作者、机构、摘要】
"""
    prompt_prefix = prompt_prefix_tpl % (domains_str, title_s)
    sys_msg = system_prompt if system_prompt is not None else DEFAULT_SYSTEM_PROMPT
    cap = max_prompt_chars if max_prompt_chars is not None else DEFAULT_MAX_PROMPT_CHARS
    # 总上下文 = 系统提示 + 前缀 + 正文，限制在 cap 内
    max_content = max(0, cap - len(sys_msg) - len(prompt_prefix))
    content_s = _truncate_for_context(content_s, max_content)

    prompt = prompt_prefix + content_s

    if provider == "mock":
        cn, en = _identify_domain_mock(title or "", "", full_text or "")
        return cn, en

    def _call() -> str:
        if provider == "openai_api":
            return ask_openai_api(
                prompt, model=model, api_base=api_base, api_key=api_key,
                max_tokens=max_tokens, temperature=temperature,
                timeout=timeout, system_prompt=sys_msg, stream=stream,
            )
        return ask_ollama(prompt, model=model, timeout=timeout, stream=stream)

    def _normalize_minimal_domain(domain_cn: str, domain_en: str) -> Tuple[str, str]:
        """清洗并校验最小领域标签，空或含 <think>/</think> 则返回未分类。"""
        cn = (domain_cn or "").strip()
        en = (domain_en or "").strip()
        cn = _normalize_domain(cn) if cn else ""
        if not en and cn:
            en = cn
        if not cn or cn == "未分类":
            return "未分类", en if en else "Uncategorized"
        if "<think>" in cn or "</think>" in cn:
            return "未分类", "Uncategorized"
        return cn, en if en else cn

    # 约定格式：{"field": "学科名称"}
    _FIELD_JSON_RE = re.compile(r'\{\s*"field"\s*:\s*"([^"]+)"\s*\}')

    def _is_think_only(raw: str) -> bool:
        """判断是否为「仅思考」输出：以 <think> 开头且未在 </think> 后给出 {"field": "..."}。"""
        if not raw or not isinstance(raw, str):
            return False
        s = raw.strip()
        if not s.lower().startswith("<think>"):
            return False
        if "</think>" not in s:
            return True
        after = s.split("</think>", 1)[-1].strip()
        if not after:
            return True
        if _FIELD_JSON_RE.search(after):
            return False
        if '"field"' in after and '"' in after:
            return False
        return True

    def _parse_field_format(text: str) -> Optional[Tuple[str, str]]:
        """从 {"field": "学科名称"} 格式中解析领域。"""
        m = _FIELD_JSON_RE.search(text)
        if m:
            cn = (m.group(1) or "").strip().strip("。，,、")
            if cn and "<think>" not in cn and "</think>" not in cn and len(cn) <= 50:
                return _normalize_minimal_domain(cn, cn)
        try:
            for m in re.finditer(r'\{[^{}]*"field"[^{}]*\}', text):
                try:
                    data = json.loads(m.group(0))
                    if "field" in data and isinstance(data["field"], str):
                        cn = (data["field"] or "").strip()
                        if cn and "<think>" not in cn and "</think>" not in cn:
                            return _normalize_minimal_domain(cn, cn)
                except (json.JSONDecodeError, ValueError):
                    continue
        except (json.JSONDecodeError, ValueError, KeyError):
            pass
        return None

    def _parse(raw: str) -> Optional[Tuple[str, str]]:
        if not raw:
            return None
        if _is_think_only(raw):
            return None
        work = raw.split("</think>")[-1].strip() if "</think>" in raw else raw.strip()
        # 1）优先检测 {"field": "学科名称"}
        result = _parse_field_format(work)
        if result is not None:
            return result
        try:
            for m in re.finditer(r'\{[^{}]*"domain_cn"[^{}]*"domain_en"[^{}]*\}', work):
                try:
                    data = json.loads(m.group(0))
                    if "domain_cn" in data and "domain_en" in data:
                        return _normalize_minimal_domain(
                            data.get("domain_cn", ""), data.get("domain_en", "")
                        )
                except (json.JSONDecodeError, ValueError):
                    continue
            match = re.search(r"\{.*\}", work, re.DOTALL)
            if match:
                data = json.loads(match.group(0))
                if "field" in data:
                    cn = (data.get("field") or "").strip()
                    if cn:
                        return _normalize_minimal_domain(cn, cn)
                return _normalize_minimal_domain(
                    data.get("domain_cn", ""), data.get("domain_en", "")
                )
        except (json.JSONDecodeError, ValueError, KeyError):
            pass
        if "|" in work:
            parts = work.split("|")
            if len(parts) >= 2:
                return _normalize_minimal_domain(parts[0].strip(), parts[1].strip())
        return None

    raw = _call()
    result = _parse(raw)
    # 解析成功（含「未分类」）直接返回，避免无谓重试
    if result is not None:
        return result
    # 仅解析失败时重试一次
    raw2 = _call()
    result = _parse(raw2)
    if result is not None:
        return result
    s = (raw2 or raw or "").strip()
    cn = _normalize_domain(s)
    if cn and cn != "未分类" and "<think>" not in cn and "</think>" not in cn:
        return cn, cn
    return "未分类", "Uncategorized"


def _identify_domain_mock(title: str, abstract: str, body: str) -> Tuple[str, str]:
    """
    模拟最小领域：根据标题/摘要/正文关键词返回最接近的最小领域（中英文），用于本地验证流程。
    """
    text = (title + " " + abstract + " " + body).lower()
    # 按优先级匹配，返回第一个匹配到的最小领域
    keyword_domains = [
        (["computer", "computing", "algorithm", "机器学习", "深度学习", "神经网络", "软件", "计算机"], "计算机科学", "Computer Science"),
        (["bioinformatics", "生物信息", "基因组", "genome", "蛋白组"], "生物信息学", "Bioinformatics"),
        (["medical", "medicine", "hospital", "临床", "肿瘤", "癌症", "医学"], "医学", "Medicine"),
        (["biology", "生物", "细胞", "cell", "基因", "生命科学"], "生命科学", "Life Science"),
        (["chemistry", "化学", "分子"], "化学", "Chemistry"),
        (["physics", "物理", "量子"], "物理学", "Physics"),
        (["material", "材料", "纳米"], "材料科学", "Materials Science"),
        (["agriculture", "农学", "作物"], "农学", "Agriculture"),
        (["econom", "经济", "金融"], "经济学", "Economics"),
    ]
    for keywords, cn, en in keyword_domains:
        if any(kw in text for kw in keywords):
            return cn, en
    return "未分类", "Uncategorized"
