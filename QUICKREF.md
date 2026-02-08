# GUIA DE REFERÊNCIA RÁPIDA - v2.0
# Sistema de Dimensionamento de Inferência LLM

## 🚀 Quick Start (v2.0 - 3 Cenários Automáticos)

```bash
# Uso básico - gera MÍNIMO, RECOMENDADO e IDEAL automaticamente
python3 sizing.py \
  --model opt-oss-120b \
  --server dgx300 \
  --storage profile_default \
  --concurrency 1000 \
  --effective-context 131072

# Output:
#   MÍNIMO: 2 nós (sem HA, sem headroom)
#   RECOMENDADO: 3 nós (N+1, 20% headroom) ← USAR EM PRODUÇÃO
#   IDEAL: 5 nós (N+2, 30% headroom, budget 65%)

# Ver apenas JSON (sem relatório)
python3 sizing.py ... --json-only

# Ver apenas Markdown (sem JSON)
python3 sizing.py ... --markdown-only

# Salvar relatório em Markdown
python3 sizing.py ... --output-markdown-file report.md

# Salvar JSON em arquivo
python3 sizing.py ... --output-json-file results.json

# Salvar ambos
python3 sizing.py ... --output-markdown-file report.md --output-json-file results.json
```

## 🎯 Os 3 Cenários (v2.0)

| Cenário | Headroom | HA | Budget KV | Uso |
|---------|----------|----|-----------|----|
| **MÍNIMO** | 0% | none | 70% | PoC, dev, teste |
| **RECOMENDADO** ✅ | 20% | N+1 | 70% | **PRODUÇÃO** |
| **IDEAL** | ≥30% | N+2 | ≤65% | Missão crítica |

**Regra de Ouro:** Para produção, use **CENÁRIO RECOMENDADO** ✅

## 📋 Modelos Disponíveis

| Modelo | Camadas | KV Heads | Max Context | Padrão Atenção |
|--------|---------|----------|-------------|----------------|
| `opt-oss-120b` | 36 | 8 | 131k | hybrid (18 full + 18 sliding) |
| `opt-oss-20b` | 24 | 8 | 131k | hybrid (12 full + 12 sliding) |

## 🖥️ Servidores Disponíveis

| Servidor | GPUs | HBM/GPU | HBM Total | NVLink |
|----------|------|---------|-----------|--------|
| `dgx300` | 8 | 288 GB | 2304 GB (2.3 TB) | 14.4 TB/s |
| `dgx200` | 8 | 141 GB | 1128 GB (1.1 TB) | - |

## 💾 Perfis de Storage

| Perfil | Tipo | IOPS R/W | Throughput | Latência P99 |
|--------|------|----------|------------|--------------|
| `profile_default` | NVMe local | 1M / 800k | 28 / 25 GB/s | 0.15 / 0.20 ms |
| `profile_network_ssd` | Network SSD | 500k / 300k | 12 / 10 GB/s | 2.0 / 3.5 ms |
| `profile_cloud_premium` | Cloud Block | 160k / 120k | 4 / 4 GB/s | 5.0 / 6.0 ms |

## ⚙️ Parâmetros Comuns

```bash
# Precisão KV (impacto direto na memória)
--kv-precision fp8     # 1 byte/elem (recomendado)
--kv-precision fp16    # 2 bytes/elem (dobro da memória)

# Budget de HBM
--kv-budget-ratio 0.70      # 70% para KV cache (padrão)
--runtime-overhead-gib 120  # 120 GiB para modelo + ativações

# Headroom e HA (afetam RECOMENDADO e IDEAL)
--peak-headroom-ratio 0.20  # 20% headroom para picos (padrão)

# Arquivos
--models-file models.json
--servers-file servers.json
--storage-file storage.json
```

## 📊 Interpretando Resultados (v2.0)

### Saída em Texto (4 Seções)

```
┌─────────────────────────────────────────────────────┐
│ SEÇÃO 1: ENTRADAS                                   │
│   • Modelo, Servidor, Storage, NFRs                 │
├─────────────────────────────────────────────────────┤
│ SEÇÃO 2: DICIONÁRIO DE PARÂMETROS                   │
│   • 12+ parâmetros explicados                       │
│   • O que é / Por que importa / Erros comuns        │
├─────────────────────────────────────────────────────┤
│ SEÇÃO 3: RESULTADOS POR CENÁRIO                     │
│   • MÍNIMO: 2 nós                                   │
│   • RECOMENDADO: 3 nós (N+1) ✅                     │
│   • IDEAL: 5 nós (N+2)                              │
│   • Cada resultado com RACIONAL detalhado           │
├─────────────────────────────────────────────────────┤
│ SEÇÃO 4: ALERTAS E RISCOS                           │
│   • Validações automáticas                          │
└─────────────────────────────────────────────────────┘
```

### Saída em Markdown (--markdown-only)

```markdown
# Relatório de Dimensionamento de Inferência LLM

## 📋 Seção 1: Entradas
### Modelo
- **Nome:** opt-oss-120b
- **Camadas:** 36
...

## 🎯 Seção 3: Resultados por Cenário

### Comparação Rápida
| Métrica | MÍNIMO | RECOMENDADO | IDEAL |
|---------|--------|-------------|-------|
| **Nós Finais** | 2 | 3 ✅ | 5 |

### 🟢 Cenário: RECOMENDADO
- Nós Finais: **3**
- HA: N+1
- Headroom: 20%
...
```

### JSON Estruturado (v2.0)

```json
{
  "inputs": {...},
  "parameter_dictionary": {...},  // Novo na v2.0
  "scenarios": {
    "minimum": {
      "results": {
        "kv_per_session_gib": 2.25,
        "nodes_final": 2
      },
      "rationale": {              // Novo na v2.0
        "kv_per_session_gib": {
          "formula": "...",
          "inputs": {...},
          "explanation": "..."
        }
      }
    },
    "recommended": {...},         // Novo na v2.0
    "ideal": {...}                // Novo na v2.0
  },
  "alerts": [...]
}
```

## 🎯 Cenários Típicos (v2.0)

### Produção com HA (Recomendado)
```bash
python3 sizing.py --model opt-oss-120b --server dgx300 \
  --storage profile_default --concurrency 1000 \
  --effective-context 131072 --kv-precision fp8

# Output:
#   MÍNIMO: 2 nós
#   RECOMENDADO: 3 nós (2 + N+1) ✅ USAR ESTE
#   IDEAL: 5 nós (3 + N+2)
```

### Modelo Menor, Contexto Menor
```bash
python3 sizing.py --model opt-oss-20b --server dgx200 \
  --storage profile_default --concurrency 1000 \
  --effective-context 32768 --kv-precision fp8

# Output:
#   MÍNIMO: 1 nó
#   RECOMENDADO: 2 nós (1 + N+1) ✅
#   IDEAL: 3 nós (1 + N+2)
```

### Alta Precisão (FP16 vs FP8)
```bash
# FP8 (1 byte/elem)
python3 sizing.py --model opt-oss-120b --server dgx300 \
  --storage profile_default --concurrency 1000 \
  --effective-context 131072 --kv-precision fp8
# RECOMENDADO: 3 nós

# FP16 (2 bytes/elem) - dobra memória
python3 sizing.py --model opt-oss-120b --server dgx300 \
  --storage profile_default --concurrency 1000 \
  --effective-context 131072 --kv-precision fp16
# RECOMENDADO: 5 nós (+67% custo)
```

## 🧮 Fórmulas (v2.0)

### KV Cache por Sessão (com Racional)
```
Para cada camada:
  • Full attention: seq = effective_context
  • Sliding attention: seq = sliding_window
  • Hybrid: mix de full e sliding

KV_bytes = 2 × sum(seq_por_camada) × num_kv_heads × head_dim × bytes_per_elem
KV_gib = KV_bytes / (1024^3)
```

### Sessões por Nó
```
HBM_total_gib = total_hbm_gb × (10^9 / 2^30)
Budget_KV = (HBM_total_gib - runtime_overhead_gib) × kv_budget_ratio
Sessões_per_node = floor(Budget_KV / KV_per_session_gib)
```

### Nós por Cenário
```
Nodes_capacity = ceil(concurrency / sessions_per_node)
Nodes_with_headroom = ceil(concurrency × (1 + headroom) / sessions_per_node)
Nodes_final = Nodes_with_headroom + ha_extra_nodes

Onde:
  • MÍNIMO: headroom=0%, ha_extra_nodes=0
  • RECOMENDADO: headroom=20%, ha_extra_nodes=1 (N+1)
  • IDEAL: headroom=30%+, ha_extra_nodes=2 (N+2), budget≤65%
```

## 🆕 Racional de Cálculo (v2.0)

Cada resultado inclui:

```
▸ Kv Per Session Gib: 2.25 GiB

  Racional:
    Fórmula:
      Hybrid attention: 18 full + 18 sliding
      Full: 2 × 131072 × 8 × 64 × 1 × 18
      Sliding: 2 × 128 × 8 × 64 × 1 × 18
    Inputs:
      • model: opt-oss-120b
      • num_layers: 36
      • effective_context: 131072
    Interpretação:
      KV cache armazena tensores Key e Value...
      Total de 2.25 GiB por sessão ativa.
```

## 📚 Dicionário de Parâmetros (v2.0)

Cada parâmetro tem:
- **O que é:** Definição técnica
- **Origem:** Modelo / Runtime / NFR
- **Importância:** Impacto no sizing
- **Erro comum:** O que evitar

Exemplo:

```
【num_key_value_heads】
  O que é: Número de KV heads (GQA pode ter menos que query heads)
  Importância: 8 KV heads vs 32 = 4x menos memória
  Erro comum: Usar num_attention_heads → superestimação 4-8x
```

Ver relatório completo ou JSON para dicionário de 12+ parâmetros.

## ⚠️ Alertas Automatizados (v2.0)

| Condição | Alerta |
|----------|--------|
| `effective_context > max_position_embeddings` | Clamp automático + aviso |
| `kv_precision = fp16/bf16` | Aviso: dobra memória vs fp8 |
| `effective_context > 128k` | Alerta: pressiona I/O de storage |
| `kv_budget_ratio > 0.75` | Risco de fragmentação |
| `runtime_overhead_gib < 50` | Provavelmente subestimado |
| `sessions_per_node = 0` | ERRO: não cabe nem 1 sessão |

## 🔍 Debugging

```bash
# Ver ajuda completa
python3 sizing.py --help

# Validar JSONs
python3 -m json.tool models.json > /dev/null && echo "✓ models.json OK"
python3 -m json.tool servers.json > /dev/null && echo "✓ servers.json OK"
python3 -m json.tool storage.json > /dev/null && echo "✓ storage.json OK"

# Executar testes (8 testes)
python3 test_sizing.py
# Esperado: ✅ 8 testes passados (100%)

# Gerar relatório em Markdown para análise
python3 sizing.py ... --markdown-only > report.md

# Ver apenas um cenário específico no JSON
python3 sizing.py ... --json-only | python3 -c \
  "import sys,json; d=json.load(sys.stdin); \
   print(f'RECOMENDADO: {d[\"scenarios\"][\"recommended\"][\"results\"][\"nodes_final\"]} nós')"

# Gerar relatório completo com múltiplos formatos
python3 sizing.py ... \
  --output-markdown-file report.md \
  --output-json-file results.json
```

## 🎯 Decisão Rápida: Qual Cenário?

| Seu Contexto | Cenário |
|--------------|---------|
| PoC / Alpha | MÍNIMO |
| Beta / Produção | **RECOMENDADO** ✅ |
| Missão Crítica | IDEAL |
| SLA < 99% | MÍNIMO |
| SLA 99-99.9% | **RECOMENDADO** ✅ |
| SLA > 99.9% | IDEAL |
| Budget Limitado | MÍNIMO ou RECOMENDADO |
| Budget Flexível | IDEAL |

## 📖 Documentação Completa

- **README_v2.md** - Documentação completa v2.0
- **SCENARIO_GUIDE.md** - Guia detalhado de decisão entre cenários
- **VERSION_2.0_SUMMARY.txt** - Sumário visual da v2.0
- **USE_CASES.md** - 5 casos de uso reais
- **FLOWCHART.md** - Fluxogramas e diagramas

## 🆚 v1.0 → v2.0

| Feature | v1.0 | v2.0 |
|---------|------|------|
| Cálculo KV cache | ✅ | ✅ |
| Dimensionamento nós | ✅ | ✅ |
| Racional de cálculo | ❌ | ✅ |
| Dicionário parâmetros | ❌ | ✅ |
| 3 cenários | ❌ | ✅ |
| Alertas avançados | Básico | ✅ |
| JSON com rationale | ❌ | ✅ |

## 💡 Dicas Rápidas

### FP8 vs FP16
```bash
# FP8 (recomendado): 1 byte/elem, ~mínima perda de qualidade
--kv-precision fp8

# FP16: 2 bytes/elem, dobra memória e custos
--kv-precision fp16  # Use apenas se fp8 não atender qualidade
```

### Budget KV
```bash
# Conservador (mais estável, menos fragmentação)
--kv-budget-ratio 0.65

# Balanceado (padrão, recomendado)
--kv-budget-ratio 0.70

# Agressivo (máxima utilização, risco de fragmentação)
--kv-budget-ratio 0.75  # Não recomendado > 0.75
```

### Headroom
```bash
# Tráfego estável
--peak-headroom-ratio 0.10  # 10%

# Tráfego moderado (padrão)
--peak-headroom-ratio 0.20  # 20%

# Tráfego variável
--peak-headroom-ratio 0.30  # 30%
```

## 📞 Suporte

Para adicionar modelos, servidores ou perfis de storage:
1. Edite o respectivo JSON
2. Siga o formato existente
3. Valide: `python3 -m json.tool <file>.json`

---

**Versão:** 2.0  
**Data:** 2026-02-08  
**Python:** 3.8+  
**Status:** ✅ Produção Ready
