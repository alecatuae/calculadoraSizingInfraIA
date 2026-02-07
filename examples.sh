#!/bin/bash
# examples.sh - Exemplos práticos de uso do sistema de dimensionamento

echo "================================================================================"
echo "EXEMPLOS PRÁTICOS - SISTEMA DE DIMENSIONAMENTO LLM"
echo "================================================================================"
echo ""

# ==============================================================================
# EXEMPLO 1: Cenário de Produção - Alta Disponibilidade
# ==============================================================================
echo "EXEMPLO 1: Cenário de Produção - opt-oss-120b + DGX B300 + N+1 HA"
echo "--------------------------------------------------------------------------------"
python3 sizing.py \
  --model opt-oss-120b \
  --server dgx300 \
  --storage profile_default \
  --concurrency 1000 \
  --effective-context 131072 \
  --kv-precision fp8 \
  --kv-budget-ratio 0.70 \
  --runtime-overhead-gib 120 \
  --peak-headroom-ratio 0.20 \
  --ha n+1

echo ""
read -p "Pressione ENTER para continuar..."
echo ""

# ==============================================================================
# EXEMPLO 2: Cenário Econômico - Sem HA
# ==============================================================================
echo "EXEMPLO 2: Cenário Econômico - opt-oss-20b + DGX H200 + Sem HA"
echo "--------------------------------------------------------------------------------"
python3 sizing.py \
  --model opt-oss-20b \
  --server dgx200 \
  --storage profile_default \
  --concurrency 1000 \
  --effective-context 32768 \
  --kv-precision fp8 \
  --kv-budget-ratio 0.70 \
  --runtime-overhead-gib 80 \
  --peak-headroom-ratio 0.20 \
  --ha none

echo ""
read -p "Pressione ENTER para continuar..."
echo ""

# ==============================================================================
# EXEMPLO 3: Comparação FP8 vs FP16
# ==============================================================================
echo "EXEMPLO 3A: Comparação - opt-oss-20b com FP8"
echo "--------------------------------------------------------------------------------"
python3 sizing.py \
  --model opt-oss-20b \
  --server dgx200 \
  --storage profile_default \
  --concurrency 500 \
  --effective-context 65536 \
  --kv-precision fp8 \
  --json-only | python3 -c "import sys, json; d=json.load(sys.stdin); print(f\"KV/sessão: {d['results']['kv_per_session_gib']} GiB, Sessões/nó: {d['results']['sessions_per_node']}, Nós: {d['results']['nodes_final']}\")"

echo ""
echo "EXEMPLO 3B: Comparação - opt-oss-20b com FP16 (dobro de memória)"
echo "--------------------------------------------------------------------------------"
python3 sizing.py \
  --model opt-oss-20b \
  --server dgx200 \
  --storage profile_default \
  --concurrency 500 \
  --effective-context 65536 \
  --kv-precision fp16 \
  --json-only | python3 -c "import sys, json; d=json.load(sys.stdin); print(f\"KV/sessão: {d['results']['kv_per_session_gib']} GiB, Sessões/nó: {d['results']['sessions_per_node']}, Nós: {d['results']['nodes_final']}\")"

echo ""
read -p "Pressione ENTER para continuar..."
echo ""

# ==============================================================================
# EXEMPLO 4: Contextos Extremos
# ==============================================================================
echo "EXEMPLO 4A: Contexto Pequeno (4k) - Máxima Eficiência"
echo "--------------------------------------------------------------------------------"
python3 sizing.py \
  --model opt-oss-20b \
  --server dgx200 \
  --storage profile_default \
  --concurrency 2000 \
  --effective-context 4096 \
  --kv-precision fp8 \
  --json-only | python3 -c "import sys, json; d=json.load(sys.stdin); print(f\"KV/sessão: {d['results']['kv_per_session_gib']} GiB, Sessões/nó: {d['results']['sessions_per_node']}, Nós: {d['results']['nodes_final']}\")"

echo ""
echo "EXEMPLO 4B: Contexto Grande (128k) - Alto Uso de Memória"
echo "--------------------------------------------------------------------------------"
python3 sizing.py \
  --model opt-oss-20b \
  --server dgx200 \
  --storage profile_default \
  --concurrency 500 \
  --effective-context 131072 \
  --kv-precision fp8 \
  --json-only | python3 -c "import sys, json; d=json.load(sys.stdin); print(f\"KV/sessão: {d['results']['kv_per_session_gib']} GiB, Sessões/nó: {d['results']['sessions_per_node']}, Nós: {d['results']['nodes_final']}\")"

echo ""
read -p "Pressione ENTER para continuar..."
echo ""

# ==============================================================================
# EXEMPLO 5: Perfis de Storage Diferentes
# ==============================================================================
echo "EXEMPLO 5A: Storage NVMe Local (melhor performance)"
echo "--------------------------------------------------------------------------------"
python3 sizing.py \
  --model opt-oss-120b \
  --server dgx300 \
  --storage profile_default \
  --concurrency 500 \
  --effective-context 131072 \
  --kv-precision fp8 \
  --json-only | python3 -c "import sys, json; d=json.load(sys.stdin); print(f\"Storage: {d['storage']['type']}, Throughput: {d['storage']['throughput_read_gbps']} GB/s, Latência P99: {d['storage']['latency_read_ms_p99']} ms\")"

echo ""
echo "EXEMPLO 5B: Storage de Rede SSD"
echo "--------------------------------------------------------------------------------"
python3 sizing.py \
  --model opt-oss-120b \
  --server dgx300 \
  --storage profile_network_ssd \
  --concurrency 500 \
  --effective-context 131072 \
  --kv-precision fp8 \
  --json-only | python3 -c "import sys, json; d=json.load(sys.stdin); print(f\"Storage: {d['storage']['type']}, Throughput: {d['storage']['throughput_read_gbps']} GB/s, Latência P99: {d['storage']['latency_read_ms_p99']} ms\")"

echo ""
echo "EXEMPLO 5C: Cloud Block Storage"
echo "--------------------------------------------------------------------------------"
python3 sizing.py \
  --model opt-oss-120b \
  --server dgx300 \
  --storage profile_cloud_premium \
  --concurrency 500 \
  --effective-context 131072 \
  --kv-precision fp8 \
  --json-only | python3 -c "import sys, json; d=json.load(sys.stdin); print(f\"Storage: {d['storage']['type']}, Throughput: {d['storage']['throughput_read_gbps']} GB/s, Latência P99: {d['storage']['latency_read_ms_p99']} ms\")"

echo ""
read -p "Pressione ENTER para continuar..."
echo ""

# ==============================================================================
# EXEMPLO 6: Escalabilidade - Alta Concorrência
# ==============================================================================
echo "EXEMPLO 6: Alta Concorrência - 5000 sessões simultâneas"
echo "--------------------------------------------------------------------------------"
python3 sizing.py \
  --model opt-oss-120b \
  --server dgx300 \
  --storage profile_default \
  --concurrency 5000 \
  --effective-context 131072 \
  --kv-precision fp8 \
  --peak-headroom-ratio 0.30 \
  --ha n+1

echo ""
echo "================================================================================"
echo "EXEMPLOS CONCLUÍDOS!"
echo "================================================================================"
echo ""
echo "Para mais informações, consulte:"
echo "  - README.md (documentação completa)"
echo "  - python3 sizing.py --help (ajuda da CLI)"
echo "  - python3 test_sizing.py (bateria de testes)"
echo ""
