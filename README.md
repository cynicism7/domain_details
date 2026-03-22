# 文献领域识别

批量扫描本地 PDF 文献，提取标题、关键词、摘要和引言片段，调用本地大模型做基于 `taxonomy.yaml` 的两级领域分类，结果实时写入 CSV。当前版本保留了 PDF 原生库崩溃隔离兜底，遇到异常 PDF 时不会把主程序一起带崩。

## 当前版本特性

- 仅面向 PDF 流程，默认扫描 `config.yaml` 中配置的目录
- 提取优先级：标题、关键词、摘要、引言片段，必要时补正文片段
- 分类方式：先判一级领域，再在对应一级下判二级领域
- 输出方式：实时写入 `literature_domains.csv`，支持断点续跑
- 可编辑配置：`config.yaml`
- 可编辑领域体系：`taxonomy.yaml`

## 目录中的关键文件

- `main.py`：程序入口
- `extractors.py`：PDF 文本提取与 PDF 崩溃隔离
- `llm_client.py`：本地模型调用与两级分类
- `config.yaml`：用户配置
- `taxonomy.yaml`：一级/二级领域和别名表

## 配置说明

将 `config.yaml` 和 `taxonomy.yaml` 放在与可执行文件同目录即可。程序会优先读取 exe 同目录下的这两份文件；若通过 `--config` 指定其他配置文件，则相对路径也会按该配置文件所在目录解析。

示例：

```yaml
literature_dirs:
  - "./papers"

extensions:
  - ".pdf"

llm:
  provider: "openai_api"
  model: "Qwen2.5-3B-Instruct"
  api_base: "http://127.0.0.1:1234/v1"
  api_key: "lm-studio"
  max_tokens: 64
  temperature: 0.0
  timeout: 60
  stream: true
  warmup: false

taxonomy_path: "./taxonomy.yaml"
max_chars_for_llm: 2000
max_prompt_chars: 4096
clear_context_every_n: 50
concurrency: 1

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

## 打包

安装 PyInstaller 后执行：

```bash
pyinstaller 文献领域识别.spec
```

建议分发时保证以下文件位于同一目录：

- `文献领域识别.exe`
- `config.yaml`
- `taxonomy.yaml`

这样用户可以直接修改模型配置和领域列表，而无需重新打包。

## 常见问题

`Q: 识别速度慢？`

`A:` 优先使用非推理模型，并尽量使用本地服务中已加载的 3B 或 7B 量化模型。当前默认配置为 `Qwen2.5-3B-Instruct`，更适合速度和精度平衡。

`Q: 结果全是“未分类”？`

`A:` 先检查 `taxonomy.yaml` 是否存在、路径是否正确，以及 `taxonomy.yaml` 的 `level1` 结构是否有效。当前版本未加载有效 taxonomy 时会统一回退为 `未分类`。

`Q: 某些 PDF 会直接让程序退出？`

`A:` 当前版本已经把高风险的 PyMuPDF 提取放到隔离子进程中，异常 PDF 不应再导致主进程直接退出；如果仍有问题，请保留样本 PDF 继续排查。
