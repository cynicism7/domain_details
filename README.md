# 文献领域识别（最小领域）

批量扫描本地 PDF 文献，提取文本（含**作者与机构信息**）后送交本地大模型，识别每条文献**最接近的最小领域**（最具体、最细分的学科或方向），结果写入 SQLite 并导出 CSV，便于按领域筛选与统计。

## 功能概览

- **扫描**：递归扫描配置目录下的 PDF（可配置扩展名）
- **提取**：优先从 PDF 文本层取文（PyMuPDF / pypdf），不足时使用 Tesseract OCR 兜底；按 RAG 思路分块后整合，**显式包含「作者与机构信息」**（标题、作者、单位、期刊等）
- **分类**：调用本地大模型（LM Studio / Ollama，OpenAI 兼容 API）判断**最接近的最小领域**，返回一个最细分的学科/方向（如计算机科学、生物信息学、医学、材料科学等），中英文各一
- **输出**：写入 `literature_domains.db`，并导出 `literature_domains.csv`；支持按领域筛选与查看所有已记录领域

## 环境要求

- Python 3.9+
- 本地大模型服务：**LM Studio**（推荐，OpenAI 兼容 API）或 **Ollama**

## 安装依赖

```bash
pip install -r requirements.txt
```

- **PDF 文本**：`pymupdf`、`pypdf`（取文本层）
- **OCR 兜底**（可选）：若 PDF 无文本层或文字过少，需安装 `pytesseract` 并安装 [Tesseract](https://github.com/tesseract-ocr/tesseract) 本体
- **大模型调用**：`openai`（LM Studio）、`ollama`（Ollama）

建议在 venv 中安装以免与其它项目冲突。

## 配置（config.yaml）

将项目根目录下的 `config.yaml` 放在与可执行文件同目录（或通过 `--config` 指定路径）。打包时已默认把 `config.yaml` 打进程序目录，首次运行前可按需修改该文件。

```yaml
literature_dirs:
  - "./papers"
extensions:
  - ".pdf"

llm:
  provider: "openai_api"        # 或 ollama
  model: "Qwen2.5-VL-7B-Instruct"
  api_base: "http://127.0.0.1:1234/v1"
  api_key: "lm-studio"
  max_tokens: 512
  temperature: 0.0

max_chars_for_llm: 800          # 送交模型的字符上限，减小可提速
max_prompt_chars: 4096          # 单次请求总上下文上限，防本地模型内存越界
clear_context_every_n: 50       # 每 N 篇后清理 LM 上下文（LM Studio 会卸载再加载），大批量时建议开启

output:
  db_path: "./literature_domains.db"
  export_csv: True
  csv_path: "./literature_domains.csv"
```

| 项 | 说明 |
|----|------|
| `literature_dirs` | 文献根目录列表，会递归扫描子目录 |
| `extensions` | 参与扫描的扩展名，通常只保留 `.pdf` |
| `llm.provider` | `openai_api`（LM Studio）或 `ollama` |
| `llm.model` | 与 LM Studio / Ollama 中加载的模型名一致（LM Studio 定期清理时用此作 instance_id） |
| `llm.api_base` | LM Studio 的 API 地址，常见 `http://127.0.0.1:1234/v1` |
| `llm.max_tokens` | 生成 token 上限；思考型模型建议 512，否则可 256 |
| `llm.temperature` | 建议 0 以稳定输出 JSON |
| `max_chars_for_llm` | 送交的文献内容最大字符数，适当减小可加快推理 |
| `max_prompt_chars` | 单次请求「系统提示+用户提示」总字符数上限，避免上下文过长导致本地模型内存越界 |
| `clear_context_every_n` | 每处理 N 篇文献后清理 LM 上下文（LM Studio 调用卸载接口再自动加载），不设或 0 表示不清理 |

## 使用方法

### 扫描并打标签

```bash
python main.py scan
```

会逐个处理 PDF，打印「送交 N 字」及识别出的最小领域（中英文），并写入数据库与 CSV。

### 模拟模式（不调用大模型）

```bash
python main.py scan --mock
```

按关键词规则模拟最小领域（如计算机科学、生物信息学、医学等），用于验证流程。

### 查看已记录的领域

```bash
python main.py domains
```

### 按领域筛选文献

```bash
python main.py filter 计算机科学
```

会列出所有被标为该最小领域的文献路径。领域名以 `domains` 子命令列出的为准（如计算机科学、生物信息学、医学等）。

### 指定配置文件路径

```bash
python main.py --config /path/to/config.yaml scan
```

## 大批量运行与内存

- **单次请求长度**：程序会按 `max_prompt_chars` 对单次请求的总上下文（系统提示+用户提示）做硬截断，避免本地模型因上下文过长而内存越界。
- **定期清理上下文**：领域判断不依赖对话历史。使用 LM Studio 时，可设置 `clear_context_every_n`（如 50），每处理 N 篇后程序会调用 LM Studio 的「卸载模型」接口，释放显存并清空上下文，下次请求时 LM Studio 会自动重新加载模型，从而可一次性跑完大量文献而不会因上下文累积退出。

## 打包（含 config）

在项目根目录执行（需已安装 PyInstaller，且当前目录存在 `config.yaml`）：

```bash
pip install pyinstaller
pyinstaller 文献领域识别.spec
```

打包完成后，可执行文件与内嵌的 `config.yaml` 位于 `dist/文献领域识别/`。运行前可在该目录下直接修改 `config.yaml` 调整文献目录、模型和 API 等。若希望使用外部配置，可将 `config.yaml` 放在与 exe 同目录并编辑，程序会优先读取同目录下的 `config.yaml`（或通过命令行 `--config` 指定路径）。

## 输出说明

- **SQLite**（`output.db_path`）：表内为 `file_path`、`file_name`、`domain_cn`、`domain_en`、`updated_at`
- **CSV**（`output.csv_path`）：同上字段，便于 Excel 打开

领域取值为 AI 返回的**最小领域**（中英文），如计算机科学/Computer Science、生物信息学/Bioinformatics 等；无法识别时为「未分类」/Uncategorized。

## 常见问题
**Q: 识别速度慢？**  
A: ① 在 LM Studio 里把 **GPU 卸载层数** 调到显存允许的最大值（影响最大）。② 本程序已缩短提示语、`max_chars_for_llm` 默认 800，仅送标题/作者/研究团队/摘要。③ **推荐**：本任务为简单分类，换用**非思考型**小模型（如 Qwen2.5-7B、Qwen2.5-3B 等）通常更快、更稳，且不会把 token 耗在 <think> 里。

**Q: 思考型模型（如 DeepSeek-R1、带 <think> 的 Qwen）输出被截断或总是「未分类」？**  
A: 思考型模型会在 <think> 里消耗大量 token，512 可能不够导致后面的 `{"field":"学科名"}` 被截断。可尝试：① 程序已自动加上**系统提示**，约束模型「禁止 <think>、直接输出 JSON」，部分模型会遵守从而少用 token。② 将 `llm.max_tokens` 调大（如 1024 或 2048），保证有足够空间输出完整 JSON。③ 若仍不稳定，建议换用**非思考型**小模型做分类，速度与成功率都更好。

**Q: 结果有时是「未分类」？**  
A: 已通过 `temperature=0`、解析失败时自动重试一次减少该情况。若仍出现，请确认 `max_tokens` 足够，且模型能稳定输出一行 `{"field": "学科名称"}`。

**Q: LM Studio 请求失败 / 模型不存在？**  
A: 确认 `config.yaml` 中 `llm.model`、`llm.api_base` 与 LM Studio 中一致，且 Server 已启动。

**Q: 大批量跑文献时程序或 LM Studio 内存越界/退出？**  
A: ① 设置 `max_prompt_chars`（如 4096 或更小）限制单次请求长度。② 设置 `clear_context_every_n`（如 50），每 N 篇后让程序调用 LM Studio 卸载模型再加载，清空显存与上下文。③ 适当减小 `max_chars_for_llm` 以降低单篇送交字数。
