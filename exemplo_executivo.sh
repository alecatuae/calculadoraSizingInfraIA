#!/bin/bash
# exemplo_executivo.sh - Exemplos de uso do relatório executivo

echo "=========================================="
echo "EXEMPLO 1: Relatório Executivo Básico"
echo "=========================================="

python3 sizing.py \
  --model opt-oss-120b \
  --server dgx300 \
  --storage profile_default \
  --concurrency 1000 \
  --effective-context 131072 \
  --kv-precision fp8 \
  --executive-report \
  --output-markdown-file reports/exec_basic.md

echo ""
echo "✅ Relatório salvo em: reports/exec_basic.md"
echo ""

echo "=========================================="
echo "EXEMPLO 2: Cenário de Alta Carga"
echo "=========================================="

python3 sizing.py \
  --model opt-oss-120b \
  --server dgx300 \
  --storage profile_default \
  --concurrency 5000 \
  --effective-context 131072 \
  --kv-precision fp8 \
  --peak-headroom-ratio 0.30 \
  --executive-report \
  --output-markdown-file reports/exec_high_load.md

echo ""
echo "✅ Relatório salvo em: reports/exec_high_load.md"
echo ""

echo "=========================================="
echo "EXEMPLO 3: Análise Comparativa FP8 vs FP16"
echo "=========================================="

# FP8
python3 sizing.py \
  --model opt-oss-120b \
  --server dgx300 \
  --storage profile_default \
  --concurrency 1000 \
  --effective-context 131072 \
  --kv-precision fp8 \
  --executive-report \
  --output-markdown-file reports/exec_fp8.md

# FP16
python3 sizing.py \
  --model opt-oss-120b \
  --server dgx300 \
  --storage profile_default \
  --concurrency 1000 \
  --effective-context 131072 \
  --kv-precision fp16 \
  --executive-report \
  --output-markdown-file reports/exec_fp16.md

echo ""
echo "✅ Relatórios comparativos salvos:"
echo "   - reports/exec_fp8.md"
echo "   - reports/exec_fp16.md"
echo ""

echo "=========================================="
echo "EXEMPLO 4: Modelo Menor (20B)"
echo "=========================================="

python3 sizing.py \
  --model opt-oss-20b \
  --server dgx200 \
  --storage profile_default \
  --concurrency 500 \
  --effective-context 131072 \
  --kv-precision fp8 \
  --executive-report \
  --output-markdown-file reports/exec_20b.md

echo ""
echo "✅ Relatório salvo em: reports/exec_20b.md"
echo ""

echo "=========================================="
echo "TODOS OS EXEMPLOS CONCLUÍDOS"
echo "=========================================="
echo ""
echo "Diretório de relatórios: reports/"
echo ""
echo "Para apresentação à diretoria, use:"
echo "  reports/exec_basic.md      - Caso base (1k sessões, fp8)"
echo "  reports/exec_high_load.md  - Alta carga (5k sessões)"
echo "  reports/exec_fp8.md        - Análise de precisão (FP8)"
echo "  reports/exec_fp16.md       - Análise de precisão (FP16)"
echo "  reports/exec_20b.md        - Modelo menor (opt-oss-20b)"
echo ""
