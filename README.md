# 文献领域识别

批量扫描本地 PDF 文献，提取标题、关键词、摘要和引言片段，调用本地或局域网中的大模型服务做基于 `taxonomy.yaml` 的两级领域分类，结果实时写入 CSV。当前版本保留了 PDF 原生库崩溃隔离兜底，遇到异常 PDF 时不会把主程序一起带崩。

## 功能概览

- 仅面向 PDF 流程，默认扫描 `config.yaml` 中配置的目录
- 提取优先级：标题、关键词、摘要、引言片段，必要时补正文片段
- 分类方式：先判一级领域，再在对应一级下判二级领域
- 输出方式：实时写入 `literature_domains.csv`，支持断点续跑
- 支持 OpenAI 兼容接口，可对接 LM Studio、vLLM 等服务
- 支持受限标签选择，适合固定 taxonomy 的分类场景

## 目录中的关键文件

- `main.py`：程序入口
- `extractors.py`：PDF 文本提取与 PDF 崩溃隔离
- `llm_client.py`：模型调用与两级分类逻辑
- `config.yaml`：默认配置
- `config.vllm.example.yaml`：Linux + vLLM 示例配置
- `taxonomy.yaml`：一级/二级领域和别名表
- `VLLM_LINUX.md`：vLLM 部署说明

## 配置说明

将 `config.yaml` 和 `taxonomy.yaml` 放在与可执行文件同目录即可。程序会优先读取可执行文件同目录下的这两份文件；若通过 `--config` 指定其他配置文件，则相对路径也会按该配置文件所在目录解析。

示例：

```yaml
literature_dirs:
  - "./papers"

extensions:
  - ".pdf"

llm:
  provider: "openai_api"
  model: "paper-domain"
  api_base: "http://127.0.0.1:8000/v1"
  api_key: "EMPTY"
  max_tokens: 8
  temperature: 0.0
  timeout: 120
  stream: false
  warmup: true
  choice_constraints: true
  extra_body:
    top_k: 1
    # 如果使用 Qwen3，可增加：
    # chat_template_kwargs:
    #   enable_thinking: false

taxonomy_path: "./taxonomy.yaml"
max_chars_for_llm: 1200
max_prompt_chars: 4096
clear_context_every_n: 0

concurrency: 12
llm_concurrency: 12
classification_retries: 1
taxonomy_fast_path: true

output:
  csv_path: "./literature_domains.csv"
  log_path: "./scan.log"
```

## taxonomy.yaml 说明

当前版本的分类完全由 `taxonomy.yaml` 驱动，建议直接维护这份文件：

- `level1`：一级领域
- `children`：该一级下的二级领域
- `global_aliases`：跨领域通用别名
- `secondary_aliases`：某个一级领域内部的二级别名
- `default_label`：无法判断时的一级兜底标签
- `default_secondary_label`：无法判断时的二级兜底标签

## 使用方法

扫描并分类：

```bash
python main.py scan
```

模拟模式：

```bash
python main.py scan --mock
```

查看已记录的领域：

```bash
python main.py domains
```

按领域筛选：

```bash
python main.py filter 土木工程/岩土工程
```

指定配置文件：

```bash
python main.py --config /path/to/config.yaml scan
```

## Linux 本机部署：本地 vLLM + OCR 兜底

下面这套流程适用于“模型就在 Linux 本机运行，同时程序需要保留 OCR 兜底”。

### 1. 系统准备

如果你希望尽量少手敲命令，仓库里已经提供了 Linux 脚本：

- `scripts/linux/install_system_deps.sh`：安装系统级依赖
- `scripts/linux/setup_python_env.sh`：安装指定 Python 版本、创建虚拟环境、安装 Python 依赖
- `scripts/linux/download_model.sh`：把模型下载到本地目录
- `scripts/linux/start_vllm.sh`：启动本机 vLLM
- `scripts/linux/build_linux.sh`：在 Linux 上打包程序

首次拉到 Linux 后，如果脚本没有执行权限，可以先执行：

```bash
chmod +x scripts/linux/*.sh
```

建议先确认这台 Linux 机器已经具备：

- NVIDIA 驱动
- 可用的 CUDA 环境
- Python 3.10+
- 可联网下载 Hugging Face 模型，或者已经准备好离线模型目录

如果你希望 OCR 兜底生效，还需要安装系统级 Tesseract：

```bash
sudo apt update
sudo apt install -y python3 python3-venv tesseract-ocr tesseract-ocr-chi-sim tesseract-ocr-eng
```

或者直接执行：

```bash
bash scripts/linux/install_system_deps.sh
```

说明：

- 这个项目里的 OCR 是自动兜底，不需要在 `config.yaml` 里单独开启
- 程序会优先走文本层提取；当 PDF 文本层不足时，再尝试 OCR
- 只要系统里有 `tesseract`，并且 Python 环境里安装了 `pytesseract` 和 `Pillow`，OCR 兜底就能工作

### 2. 创建 Python 环境并安装依赖

如果你是从源码运行项目，可以直接用脚本准备环境：

```bash
bash scripts/linux/setup_python_env.sh
source .venv-linux/bin/activate
```

如果你想手动执行，也可以：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip uv
python -m pip install -r requirements-linux.txt
```

说明：

- `openai` 用于访问 vLLM 的 OpenAI 兼容接口
- `pypdf`、`pymupdf`、`pytesseract`、`pillow` 用于 PDF 提取和 OCR 兜底
- 我已经在仓库里添加了 [requirements-linux.txt](/e:/Python%20Program/domain_details/requirements-linux.txt)
- 如果你的机器在安装 `vllm` 时遇到 CUDA / torch 兼容问题，优先改用 `uv pip install vllm --torch-backend=auto`

如果你不是从源码运行，而是分发 Linux 下重新打包后的可执行文件，那么通常不需要再安装这些 Python 包，但 `tesseract-ocr` 这类系统级依赖仍然建议保留。

### 3. 下载模型到 Linux 本地

如果模型要在本机 vLLM 上运行，模型权重需要在这台 Linux 机器上可用。推荐直接用脚本下载到固定目录：

```bash
MODEL_ID=Qwen/Qwen2.5-7B-Instruct \
MODEL_DIR=/data/models/Qwen2.5-7B-Instruct \
bash scripts/linux/download_model.sh
```

如果你想手动执行，也可以：

```bash
hf auth login
hf download Qwen/Qwen2.5-7B-Instruct --local-dir /data/models/Qwen2.5-7B-Instruct
```

如果是私有模型，先登录 Hugging Face；如果是离线环境，建议先在可联网机器下载，再拷贝到目标服务器。

你也可以不手动下载，直接在第一次启动 vLLM 时让它自动拉取模型，但生产环境更推荐提前下载，路径更清晰，也更方便排查问题。

### 4. 启动本机 vLLM

本地目录模型示例：

```bash
MODEL=/data/models/Qwen2.5-7B-Instruct \
SERVED_MODEL_NAME=paper-domain \
bash scripts/linux/start_vllm.sh
```

等价的手动命令如下：

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve /data/models/Qwen2.5-7B-Instruct \
  --served-model-name paper-domain \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype auto \
  --gpu-memory-utilization 0.92 \
  --max-num-seqs 32 \
  --max-num-batched-tokens 8192 \
  --enable-prefix-caching \
  --generation-config vllm
```

如果你想让 vLLM 自动下载模型，也可以这样启动：

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen2.5-7B-Instruct \
  --served-model-name paper-domain \
  --host 0.0.0.0 \
  --port 8000
```

如果使用 Qwen3，建议关闭 thinking：

```bash
vllm serve /data/models/Qwen3-8B \
  --served-model-name paper-domain \
  --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"enable_thinking": false}'
```

### 5. 配置本项目连接本机 vLLM

推荐直接复制 `config.vllm.example.yaml`，或者按下面这样设置：

```yaml
literature_dirs:
  - "./papers"

extensions:
  - ".pdf"

llm:
  provider: "openai_api"
  model: "paper-domain"
  api_base: "http://127.0.0.1:8000/v1"
  api_key: "EMPTY"
  max_tokens: 8
  temperature: 0.0
  timeout: 120
  stream: false
  warmup: true
  choice_constraints: true
  extra_body:
    top_k: 1
    # 如果使用 Qwen3，可增加：
    # chat_template_kwargs:
    #   enable_thinking: false

taxonomy_path: "./taxonomy.yaml"
max_chars_for_llm: 1200
max_prompt_chars: 4096
clear_context_every_n: 0

concurrency: 12
llm_concurrency: 12
classification_retries: 1
taxonomy_fast_path: true

output:
  csv_path: "./literature_domains.csv"
  log_path: "./scan.log"
```

说明：

- `llm.model` 必须和 vLLM 的 `--served-model-name` 一致
- `api_base` 指向本机 vLLM 的 OpenAI 兼容服务地址
- `choice_constraints: true` 会把一级/二级标签作为受限候选传给 vLLM，更适合 taxonomy 分类
- `stream: false` 更适合这种短标签输出

### 6. 运行本项目

```bash
source .venv-linux/bin/activate
python main.py --config config.vllm.example.yaml scan
```

如果你使用的是自己修改过的配置文件：

```bash
python main.py --config /path/to/config.yaml scan
```

### 7. OCR 兜底如何生效

OCR 兜底在当前项目里是自动逻辑，不需要配置单独的开关：

- 程序先尝试读取 PDF 文本层
- 如果文本层提取不足，会自动进入 OCR 兜底流程
- OCR 兜底依赖系统中的 `tesseract` 可执行程序

因此对你来说，真正需要做的是：

1. Linux 上安装 `tesseract-ocr`
2. Python 环境中安装 `pytesseract` 和 `Pillow`
3. 正常运行程序即可

### 8. 推荐模型

- 16GB 显存：`Qwen/Qwen2.5-3B-Instruct`
- 24GB 显存：`Qwen/Qwen2.5-7B-Instruct`
- 48GB 以上：`Qwen/Qwen3-8B` 或 `Qwen/Qwen3-14B`

对于当前这种“短文本分类 + 固定标签输出”的任务，一般优先从 `Qwen2.5-3B/7B-Instruct` 开始，吞吐更高，已经够用。

## 打包与分发

### Windows 分发

安装 PyInstaller 后执行：

```bash
pyinstaller 文献领域识别.spec
```

建议分发时保证以下文件位于同一目录：

- `文献领域识别.exe`
- `config.yaml`
- `taxonomy.yaml`

### Linux 分发

这里有一个非常重要的点：

- `Windows 的 .exe 不能直接当作 Linux 程序分发`
- 如果要给 Linux 用，必须在 Linux 上重新打包成 Linux 可执行文件
- PyInstaller 不是“打一次到处跑”，通常需要在哪个平台运行，就在哪个平台构建
- 如果你手头只有 Windows，比较现实的替代方案是：在 Linux 机器、WSL2 或 Linux 容器里构建 Linux 版本

也就是说：

- 给 Windows 用户：分发 `.exe`
- 给 Linux 用户：在 Linux 上执行 `pyinstaller 文献领域识别.spec`，分发 Linux 产物

如果你想自动化 Linux 打包，直接执行：

```bash
bash scripts/linux/build_linux.sh
```

打包完成后，分发整个 `dist_linux/<应用目录>/`，不要只拿其中单个可执行文件。

## Linux 上还需不需要额外安装库

分两种情况：

### 情况 1：你只分发“客户端程序”，模型服务在另一台机器

这时 Linux 客户端通常不需要再安装 vLLM，也不需要下载模型权重。  
但如果你希望 OCR 兜底可用，仍建议安装系统级 Tesseract：

```bash
sudo apt update
sudo apt install -y tesseract-ocr tesseract-ocr-chi-sim tesseract-ocr-eng
```

原因是程序里虽然打包了 `pytesseract` Python 库，但它底层仍依赖系统中的 `tesseract` 可执行程序。

### 情况 2：Linux 机器既运行本程序，也运行本地 vLLM

如果你采用上面“Linux 本机部署：本地 vLLM + OCR 兜底”的方案，这时除了本程序本身，还需要：

- NVIDIA 驱动
- 与 vLLM 兼容的 CUDA 环境
- `vllm`
- 本地模型权重
- `tesseract-ocr`
- Python 环境中的 `openai`、`pypdf`、`pymupdf`、`pytesseract`、`pillow`、`pyyaml`、`requests`

## 模型是否需要下载到 Linux 本地

如果 vLLM 就跑在这台 Linux 机器上，那么模型权重需要在这台机器上可用。通常有两种方式：

### 方式 1：让 vLLM 首次启动时自动下载

直接运行：

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct --served-model-name paper-domain
```

如果这台机器能访问 Hugging Face，vLLM 会通过 Hugging Face 生态自动拉取模型到本地缓存目录。

### 方式 2：提前手动下载到本地目录

更推荐在生产环境这么做，路径可控，也更方便离线部署：

```bash
pip install -U "huggingface_hub[cli]"
hf auth login
hf download Qwen/Qwen2.5-7B-Instruct --local-dir /data/models/Qwen2.5-7B-Instruct
```

然后这样启动：

```bash
vllm serve /data/models/Qwen2.5-7B-Instruct --served-model-name paper-domain
```

如果是私有模型，需要先登录 Hugging Face；如果是离线环境，建议在能联网的机器上先下载好，再拷贝到目标服务器。

## 常见问题

`Q: 程序打包成 exe 以后，Linux 能直接运行吗？`

`A:` 不能。Windows 的 `.exe` 不是 Linux 原生可执行文件。你需要在 Linux 上重新打包。

`Q: 打包后是不是完全不需要装任何东西？`

`A:` 不是。PyInstaller 主要帮你打包 Python 运行时和大部分 Python 依赖，但系统级依赖仍可能需要单独安装，比如 `tesseract-ocr`、NVIDIA 驱动、CUDA 相关环境。

`Q: 结果全是“未分类”？`

`A:` 先检查 `taxonomy.yaml` 是否存在、路径是否正确，以及 `taxonomy.yaml` 的 `level1` 结构是否有效。当前版本未加载有效 taxonomy 时会统一回退为 `未分类`。

`Q: 某些 PDF 会直接让程序退出？`

`A:` 当前版本已经把高风险的 PyMuPDF 提取放到隔离子进程中，异常 PDF 不应再导致主进程直接退出；如果仍有问题，请保留样本 PDF 继续排查。

## 参考链接

- vLLM OpenAI 兼容服务：https://docs.vllm.ai/en/stable/serving/openai_compatible_server/
- vLLM Structured Outputs：https://docs.vllm.ai/en/stable/features/structured_outputs.html
- vLLM Prefix Caching：https://docs.vllm.ai/en/stable/design/prefix_caching/
- Hugging Face 下载模型：https://huggingface.co/docs/huggingface_hub/en/guides/download
- Hugging Face CLI：https://huggingface.co/docs/huggingface_hub/en/package_reference/cli
