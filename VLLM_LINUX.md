# Linux + vLLM notes

## 1. Start vLLM

Example for a single GPU server:

```bash
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-7B-Instruct \
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

Recommended starting points:

- 16 GB VRAM: `Qwen/Qwen2.5-3B-Instruct`
- 24 GB VRAM: `Qwen/Qwen2.5-7B-Instruct`
- 48 GB+ VRAM: `Qwen/Qwen3-14B`

If your main goal is throughput, start from 3B or 7B first. This project only needs short label outputs, so smaller instruct models are usually enough.

If you use Qwen3, disable thinking for this workload. You can do it server-side:

```bash
vllm serve Qwen/Qwen3-8B \
  --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"enable_thinking": false}'
```

## 2. Run this project

Use the example config in this repo:

```bash
python main.py --config config.vllm.example.yaml scan
```

The `llm.model` value in the config must match `--served-model-name`.

## 3. Why this works well with vLLM

- `concurrency` and `llm_concurrency` let the client keep multiple requests in flight.
- vLLM will continuously batch those concurrent requests on the server side.
- `stream: false` reduces per-request overhead for short classification outputs.
- `choice_constraints: true` enables constrained label selection through `structured_outputs.choice`.
- `--enable-prefix-caching` helps because many requests share the same taxonomy prompt prefix.

## 4. Tuning order

Tune in this order:

1. Keep `temperature: 0.0` and `max_tokens: 8`.
2. Increase `llm_concurrency` until GPU utilization is stable.
3. If latency spikes or OOM happens, lower `--max-num-seqs` or `llm_concurrency`.
4. If throughput is still low, switch to a smaller model before changing the prompt.
