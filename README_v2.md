# 📊 Calculadora de Sizing de Inferência LLM - Versão 2.0

## Sistema Avançado de Dimensionamento com Racional de Cálculo

Sistema profissional de dimensionamento de infraestrutura para inferência de Large Language Models (LLMs) em GPUs NVIDIA DGX-class, com foco em **capacity planning** e **SRE** (SLO, p95/p99, HA e headroom).

### 🆕 Novidades da Versão 2.0

- ✅ **Racional de Cálculo Detalhado**: Cada resultado inclui fórmula, inputs e explicação
- ✅ **Dicionário de Parâmetros**: Explicação completa de cada parâmetro usado
- ✅ **3 Cenários Obrigatórios**: MÍNIMO, RECOMENDADO e IDEAL
- ✅ **Alertas e Riscos Automatizados**: Validações operacionais
- ✅ **JSON Estruturado**: Saída completa com rationale para integração

---

## 🎯 Os 3 Cenários

### 1. MÍNIMO (Bare Minimum)
- **Objetivo**: Atender requisitos no limite, sem folga
- **Características**:
  - `peak_headroom_ratio = 0%` (sem headroom)
  - `ha_mode = none` (sem redundância)
  - `kv_budget_ratio = configurado` (default 70%)
- **Uso**: Estimativa de custo mínimo, PoC, ambientes de teste

### 2. RECOMENDADO (Production Ready)
- **Objetivo**: Produção com HA e headroom para picos
- **Características**:
  - `peak_headroom_ratio = configurado` (default 20%)
  - `ha_mode = n+1` (tolera falha de 1 nó)
  - `kv_budget_ratio = configurado` (default 70%)
- **Uso**: **Recomendado para produção**, SLA 99.9%+

### 3. IDEAL (Enterprise Grade)
- **Objetivo**: Máxima disponibilidade e performance
- **Características**:
  - `peak_headroom_ratio = max(configurado, 30%)` (mínimo 30%)
  - `ha_mode = n+2` (tolera falha de 2 nós)
  - `kv_budget_ratio = min(configurado, 65%)` (mais conservador)
- **Uso**: Produção crítica, SLA 99.99%+, cargas imprevisíveis

---

## 📁 Estrutura do Projeto

```
calculadoraSizingInfraIA/
├── sizing.py          # Script principal v2.0 (~1200 linhas)
├── models.json        # 2 modelos LLM (120B, 20B)
├── servers.json       # 2 servidores DGX (B300, H200)
├── storage.json       # 3 perfis de storage
├── test_sizing.py     # Testes automatizados
├── examples.sh        # Exemplos práticos
└── docs/              # Documentação completa
    ├── README.md
    ├── QUICKREF.md
    ├── USE_CASES.md
    ├── FLOWCHART.md
    └── PROJECT_SUMMARY.md
```

---

## 🚀 Quick Start

### Uso Básico (3 Cenários Automáticos)

```bash
python3 sizing.py \
  --model opt-oss-120b \
  --server dgx300 \
  --storage profile_default \
  --concurrency 1000 \
  --effective-context 131072

# Output:
# MÍNIMO: 2 nós (sem HA, sem headroom)
# RECOMENDADO: 3 nós (N+1, 20% headroom)  ← PRODUÇÃO
# IDEAL: 5 nós (N+2, 30% headroom, 65% budget)
```

### Parâmetros Principais

| Parâmetro | Descrição | Default | Exemplo |
|-----------|-----------|---------|---------|
| `--model` | Nome do modelo | - | `opt-oss-120b` |
| `--server` | Nome do servidor | - | `dgx300` |
| `--storage` | Perfil de storage | - | `profile_default` |
| `--concurrency` | Sessões simultâneas | - | `1000` |
| `--effective-context` | Tamanho do contexto (tokens) | - | `131072` |
| `--kv-precision` | Precisão KV cache | `fp8` | `fp8`, `fp16` |
| `--kv-budget-ratio` | % HBM para KV | `0.70` | `0.65-0.75` |
| `--runtime-overhead-gib` | Overhead (GiB) | `120` | `80-150` |
| `--peak-headroom-ratio` | Headroom para picos | `0.20` | `0.10-0.40` |

---

## 📊 Formato de Saída

### Relatório em Texto (stdout)

```
====================================================================================================
RELATÓRIO DE DIMENSIONAMENTO AVANÇADO DE INFERÊNCIA LLM
====================================================================================================

┌──────────────────────────────────────────────────────────────────────────────────┐
│ SEÇÃO 1: ENTRADAS (Modelo / Servidor / Storage / NFR)                            │
└──────────────────────────────────────────────────────────────────────────────────┘

MODELO: opt-oss-120b, 36 camadas, 8 KV heads, hybrid attention...

┌──────────────────────────────────────────────────────────────────────────────────┐
│ SEÇÃO 2: DICIONÁRIO DE PARÂMETROS (Explicação e Importância)                     │
└──────────────────────────────────────────────────────────────────────────────────┘

【num_layers】
  O que é: Número total de camadas do transformer...
  Origem: Parâmetro fixo da arquitetura...
  Importância: Impacta linearmente o tamanho do KV cache...
  Erro comum: Confundir num_layers com num_hidden_layers...

┌──────────────────────────────────────────────────────────────────────────────────┐
│ SEÇÃO 3: RESULTADOS POR CENÁRIO (MÍNIMO / RECOMENDADO / IDEAL)                   │
└──────────────────────────────────────────────────────────────────────────────────┘

====================================================================================================
CENÁRIO: MÍNIMO
====================================================================================================
  • Peak Headroom: 0%
  • HA Mode: none
  • KV Budget Ratio: 70%

▸ Kv Per Session Gib: 2.25 GiB

  Racional:
    Fórmula:
      Hybrid attention: 18 full + 18 sliding
      Full: 2 × 131072 × 8 × 64 × 1 × 18
      Sliding: 2 × 128 × 8 × 64 × 1 × 18
      total = full + sliding
    Inputs:
      • model: opt-oss-120b
      • num_layers: 36
      • num_kv_heads: 8
      • effective_context: 131072
      • kv_precision: fp8
      • bytes_per_element: 1
    Interpretação:
      KV cache armazena tensores Key e Value de todas as camadas para o contexto
      da sessão. Modelo com attention_pattern='hybrid' usa contexto efetivo
      diferente por camada. Total de 2.25 GiB por sessão ativa.

▸ Nodes Final: 2 nós

  Racional:
    Fórmula:
      nodes_final = nodes_with_headroom + ha_extra_nodes
    Inputs:
      • nodes_with_headroom: 2
      • ha_extra_nodes: 0
      • ha_mode: none
    Interpretação:
      Sem HA: qualquer falha de nó causa degradação imediata.

[... RECOMENDADO e IDEAL seguem o mesmo formato ...]

┌──────────────────────────────────────────────────────────────────────────────────┐
│ SEÇÃO 4: ALERTAS E RISCOS OPERACIONAIS                                           │
└──────────────────────────────────────────────────────────────────────────────────┘

[1] ALERTA: Contexto longo (131,072 tokens) aumenta TTFT...
[2] AVISO: kv_precision=fp16 usa 2 bytes/elemento. Considere fp8...
```

### JSON Estruturado (stdout final)

```json
{
  "inputs": {
    "model": {...},
    "server": {...},
    "storage": {...},
    "nfr": {...}
  },
  "parameter_dictionary": {
    "num_layers": {
      "description": "...",
      "source": "...",
      "importance": "...",
      "common_errors": "..."
    }
  },
  "scenarios": {
    "minimum": {
      "name": "MÍNIMO",
      "configuration": {...},
      "results": {
        "kv_per_session_gib": 2.25,
        "nodes_final": 2,
        ...
      },
      "rationale": {
        "kv_per_session_gib": {
          "formula": "...",
          "inputs": {...},
          "explanation": "..."
        }
      },
      "warnings": [...]
    },
    "recommended": {...},
    "ideal": {...}
  },
  "alerts": [...]
}
```

---

## 📝 Exemplos Práticos

### Exemplo 1: Produção com FP8

```bash
python3 sizing.py \
  --model opt-oss-120b \
  --server dgx300 \
  --storage profile_default \
  --concurrency 1000 \
  --effective-context 131072 \
  --kv-precision fp8
```

**Resultado:**
- **MÍNIMO:** 2 nós
- **RECOMENDADO:** 3 nós (N+1) ← **Usar este**
- **IDEAL:** 5 nós (N+2)

### Exemplo 2: Alta Precisão com FP16

```bash
python3 sizing.py \
  --model opt-oss-120b \
  --server dgx300 \
  --storage profile_default \
  --concurrency 1000 \
  --effective-context 131072 \
  --kv-precision fp16
```

**Resultado:**
- **MÍNIMO:** 4 nós (fp16 dobra memória)
- **RECOMENDADO:** 5 nós (N+1)
- **IDEAL:** 7 nós (N+2)

**Insight:** FP16 vs FP8 aumenta custos em ~67% (3→5 nós no cenário recomendado).

### Exemplo 3: Modelo Menor, Contexto Menor

```bash
python3 sizing.py \
  --model opt-oss-20b \
  --server dgx200 \
  --storage profile_default \
  --concurrency 1000 \
  --effective-context 32768 \
  --kv-precision fp8
```

**Resultado:**
- **MÍNIMO:** 1 nó
- **RECOMENDADO:** 2 nós (N+1)
- **IDEAL:** 3 nós (N+2)

---

## 🧮 Metodologia de Cálculo

### KV Cache por Sessão

```
Para cada camada:
  - Full attention: seq = effective_context
  - Sliding attention: seq = sliding_window
  - Hybrid: mix de full e sliding

KV_bytes = 2 × sum(seq_por_camada) × num_kv_heads × head_dim × bytes_per_elem
KV_gib = KV_bytes / (1024^3)
```

### Sessões por Nó

```
HBM_total_gib = total_hbm_gb × (10^9 / 2^30)
Budget_KV_gib = (HBM_total_gib - runtime_overhead_gib) × kv_budget_ratio
Sessions_per_node = floor(Budget_KV_gib / KV_per_session_gib)
```

### Nós por Cenário

```
Nodes_capacity = ceil(concurrency / sessions_per_node)
Nodes_with_headroom = ceil(concurrency × (1 + headroom) / sessions_per_node)
Nodes_final = Nodes_with_headroom + ha_extra_nodes
```

**Onde:**
- **MÍNIMO:** headroom=0%, ha_extra_nodes=0
- **RECOMENDADO:** headroom=20% (configurável), ha_extra_nodes=1
- **IDEAL:** headroom=max(30%, configurado), ha_extra_nodes=2, budget_ratio=min(65%, configurado)

---

## 📚 Dicionário de Parâmetros (Resumo)

<details>
<summary><b>num_layers</b> (Modelo)</summary>

- **O que é:** Número de camadas do transformer
- **Impacto:** Linear no KV cache (36 camadas = 1.5x mais memória que 24)
- **Erro comum:** Confundir com num_hidden_layers ou contar só encoder/decoder
</details>

<details>
<summary><b>num_key_value_heads</b> (Modelo)</summary>

- **O que é:** Número de KV heads (GQA pode ter menos que query heads)
- **Impacto:** Direto no KV cache (8 heads vs 32 = 4x menos memória)
- **Erro comum:** Usar num_attention_heads causando superestimação de 4-8x
</details>

<details>
<summary><b>effective_context</b> (NFR)</summary>

- **O que é:** Tamanho de contexto que sua aplicação usará
- **Impacto:** Quadrático para full attention, linear para sliding
- **Erro comum:** Usar max_position_embeddings sem necessidade
</details>

<details>
<summary><b>kv_precision</b> (Runtime)</summary>

- **O que é:** Precisão numérica (fp8=1 byte, fp16=2 bytes)
- **Impacto:** 2x na memória (fp16 vs fp8)
- **Erro comum:** Usar fp16 por default sem validar se fp8 atende
</details>

<details>
<summary><b>kv_budget_ratio</b> (Tuning)</summary>

- **O que é:** % da HBM alocada para KV cache
- **Impacto:** Quanto maior, mais sessões/nó, mas mais risco de fragmentação
- **Erro comum:** Alocar 100% ignorando overhead do modelo
</details>

Veja relatório completo ou JSON para dicionário detalhado de **todos** os parâmetros.

---

## ⚠️ Alertas Automatizados

O sistema gera avisos automáticos para:

| Condição | Alerta |
|----------|--------|
| `effective_context > max_position_embeddings` | Clamp automático + aviso |
| `kv_precision = fp16/bf16` | Aviso que dobra memória vs fp8 |
| `effective_context > 128k` | Alerta de pressão em I/O de storage |
| `kv_budget_ratio > 0.75` | Risco de fragmentação/instabilidade |
| `runtime_overhead_gib < 50` | Provavelmente subestimado |
| `sessions_per_node = 0` | Erro crítico: não cabe nem 1 sessão |

---

## 🎯 Recomendações por Use Case

### Startup / PoC
- **Cenário:** MÍNIMO
- **Motivo:** Custo mínimo, sem HA
- **Risco:** Qualquer falha causa downtime

### Produção (SLA 99.9%)
- **Cenário:** RECOMENDADO ✅
- **Motivo:** N+1 + headroom balanceado
- **TCO:** Ideal para maioria dos casos

### Missão Crítica (SLA 99.99%)
- **Cenário:** IDEAL
- **Motivo:** N+2, headroom 30%+, budget conservador
- **TCO:** +40-60% vs RECOMENDADO, mas máxima resiliência

---

## 🔧 Opções de CLI

```bash
# Obrigatórios
--model MODEL                    # Nome do modelo (models.json)
--server SERVER                  # Nome do servidor (servers.json)
--storage STORAGE                # Perfil de storage (storage.json)
--concurrency N                  # Sessões simultâneas
--effective-context N            # Contexto em tokens

# Opcionais
--kv-precision {fp8,fp16,bf16,int8}  # Default: fp8
--kv-budget-ratio RATIO              # Default: 0.70
--runtime-overhead-gib GIB           # Default: 120
--peak-headroom-ratio RATIO          # Default: 0.20

# Arquivos
--models-file FILE               # Default: models.json
--servers-file FILE              # Default: servers.json
--storage-file FILE              # Default: storage.json

# Output
--output-json-file FILE          # Salvar JSON em arquivo
--json-only                      # Apenas JSON, sem relatório texto
--verbose                        # Mais detalhes
```

---

## 📊 Casos de Uso Validados

1. **Startup SaaS** (1k concurrent, 32k context): 1→2→3 nós
2. **Enterprise** (500 concurrent, 131k context, fp16, N+1): 4 nós
3. **API Provider** (5k concurrent, 131k context, N+1): 12 nós
4. **Pesquisa** (50 concurrent, 131k context, fp16): 1 nó
5. **Cloud Serverless** (2k concurrent/região, 32k, N+1): 3 nós/região

Veja `USE_CASES.md` para análise detalhada.

---

## 🧪 Testes

```bash
# Executar bateria de testes
python3 test_sizing.py

# Output esperado:
# ✅ 8 testes passados (100%)
```

---

## 📄 Requisitos

- **Python:** 3.8+
- **Dependências:** Nenhuma (stdlib only)
- **Plataforma:** Linux, macOS, Windows

---

## 🆚 Comparação v1.0 → v2.0

| Feature | v1.0 | v2.0 |
|---------|------|------|
| Cálculo de KV cache | ✅ | ✅ |
| Dimensionamento de nós | ✅ | ✅ |
| Racional de cálculo | ❌ | ✅ |
| Dicionário de parâmetros | ❌ | ✅ |
| 3 cenários (MIN/REC/IDEAL) | ❌ | ✅ |
| Alertas automatizados | Básico | ✅ Avançado |
| JSON com rationale | ❌ | ✅ |
| Explicação operacional | ❌ | ✅ |

---

## 📞 Suporte

Para adicionar modelos, servidores ou perfis de storage:
1. Edite o respectivo arquivo JSON
2. Siga o formato existente
3. Execute `python3 -m json.tool <file>.json` para validar

---

## 📖 Documentação Adicional

- **QUICKREF.md** - Referência rápida de comandos
- **USE_CASES.md** - 5 casos de uso reais detalhados
- **FLOWCHART.md** - Fluxogramas e diagramas
- **PROJECT_SUMMARY.md** - Sumário técnico completo

---

**Versão:** 2.0  
**Data:** 2026-02-08  
**Autor:** Sistema de Sizing de Infraestrutura IA  
**Python:** 3.8+  
**Licença:** Interno
