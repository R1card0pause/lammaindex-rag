# LlamaIndex RAG Benchmark

This project benchmarks PDF ingestion for a LlamaIndex RAG service.

The runtime path uses online services only:

- PDF parsing/OCR: MinerU API
- Embedding: SiliconFlow
- LLM for query answering: SiliconFlow

No local MinerU model or OpenAI SDK is required.

## Setup

```bash
git clone https://github.com/R1card0pause/lammaindex-rag.git
cd lammaindex-rag

conda create -y -n lammaindex-rag \
  -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
conda activate lammaindex-rag
conda install -y python=3.10 pip \
  -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main

python -m pip install -U pip
python -m pip install -r requirements.txt \
  -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## Environment

```bash
mkdir -p ~/.config/lammarag_benchmark
cat > ~/.config/lammarag_benchmark/env.sh <<'EOF'
export LAZYLLM_SILICONFLOW_API_KEY='your_siliconflow_key'
export MINERU_API_TOKEN='your_mineru_token'
export MINERU_MODEL_SOURCE='modelscope'
EOF
chmod 600 ~/.config/lammarag_benchmark/env.sh
source ~/.config/lammarag_benchmark/env.sh
```
❗修改这里的 your_siliconflow_key 和 your_mineru_token 

## Smoke Test

Run the smallest PDF first:

```bash
conda activate lammaindex-rag
source ~/.config/lammarag_benchmark/env.sh

export INPUT_DIR='/mnt/lustre/share_data/zhangyc/afs_space/datasets/data/quantum/rawdata/城市设计-详规处/01-原文文件'
export RUN_DIR="runs/smoke_$(date +%Y%m%d_%H%M%S)"

python ingest_benchmark.py \
  --input-dir "$INPUT_DIR" \
  --work-dir "$RUN_DIR" \
  --smallest-first \
  --max-files 1 \
  --mineru-api-model vlm \
  --mineru-api-disable-formula \
  --mineru-api-disable-table
```

## Full Ingestion

```bash
export RUN_DIR="runs/full_$(date +%Y%m%d_%H%M%S)"

python ingest_benchmark.py \
  --input-dir "$INPUT_DIR" \
  --work-dir "$RUN_DIR" \
  --mineru-api-model vlm \
  --mineru-api-disable-formula \
  --mineru-api-disable-table \
  --mineru-api-poll-interval 30 \
  --mineru-api-timeout 14400 \
  --mineru-api-batch-size 50 \
  --mineru-api-upload-workers 16 \
  --mineru-api-max-size-mb 190
```

Timing results are written to:

```text
$RUN_DIR/events.jsonl
```

## RAG Service

```bash
export RAG_STORAGE_DIR="$RUN_DIR/storage"
export RAG_MODEL_CONFIG="$(pwd)/models.yaml"

uvicorn rag_service:app --host 127.0.0.1 --port 18080
```

```bash
curl http://127.0.0.1:18080/health

curl -X POST http://127.0.0.1:18080/query \
  -H "Content-Type: application/json" \
  -d '{"question":"上海市城乡规划条例适用于哪些活动？","top_k":3}'
```
