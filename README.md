# Calculadora de Sizing de Inferência LLM em GPU NVIDIA

Sistema de dimensionamento de infraestrutura para inferência de Large Language Models (LLMs) em GPUs NVIDIA DGX-class.

## 📋 Descrição

Este projeto calcula o dimensionamento baseado em **memória (KV cache)** para inferência de LLMs, considerando:

- KV cache por sessão e total
- Budget de HBM por nó
- Número de sessões simultâneas por nó
- Número de nós necessários (com headroom e HA)
- Perfis de storage para cold-start e checkpoints

## 🗂️ Estrutura do Projeto

```
calculadoraSizingInfraIA/
├── sizing.py          # Script principal de dimensionamento
├── models.json        # Tabela de modelos LLM e parâmetros
├── servers.json       # Tabela de servidores GPU (DGX)
├── storage.json       # Perfis de storage e métricas I/O
└── README.md          # Este arquivo
```

## 📦 Requisitos

- Python 3.8+
- Somente bibliotecas padrão (stdlib)

## 🚀 Uso

### Sintaxe Básica

```bash
python3 sizing.py \
  --model <nome_modelo> \
  --server <nome_servidor> \
  --storage <perfil_storage> \
  --concurrency <num_sessoes> \
  --effective-context <tamanho_contexto> \
  [opções adicionais]
```

### Parâmetros Obrigatórios

| Parâmetro | Descrição |
|-----------|-----------|
| `--model` | Nome do modelo (ex: `opt-oss-120b`, `opt-oss-20b`) |
| `--server` | Nome do servidor (ex: `dgx300`, `dgx200`) |
| `--storage` | Perfil de storage (ex: `profile_default`) |
| `--concurrency` | Número de sessões simultâneas alvo |
| `--effective-context` | Tamanho do contexto efetivo em tokens |

### Parâmetros Opcionais

| Parâmetro | Padrão | Descrição |
|-----------|--------|-----------|
| `--kv-precision` | `fp8` | Precisão do KV cache: `fp8`, `fp16`, `bf16`, `int8` |
| `--kv-budget-ratio` | `0.70` | Fração da HBM alocada para KV (0.0-1.0) |
| `--runtime-overhead-gib` | `120` | Overhead de runtime em GiB (modelo + ativações) |
| `--peak-headroom-ratio` | `0.20` | Headroom para picos de tráfego (0.20 = 20%) |
| `--ha` | `none` | Modo HA: `none` ou `n+1` |
| `--json-only` | - | Imprimir apenas JSON (sem relatório texto) |

## 📝 Exemplos

### Exemplo 1: Modelo 120B + DGX B300 + N+1 HA

```bash
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
```

**Resultado:**
- KV por sessão: 2.25 GiB
- KV total: 2.20 TiB
- Sessões por nó: 613
- Nós finais: **3** (2 + N+1)

### Exemplo 2: Modelo 20B + DGX H200 + Sem HA

```bash
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
```

**Resultado:**
- KV por sessão: 0.38 GiB
- KV total: 0.37 TiB
- Sessões por nó: 1740
- Nós finais: **1**

### Exemplo 3: Alta Precisão (FP16) vs FP8

```bash
# Com FP8 (1 byte por elemento)
python3 sizing.py --model opt-oss-20b --server dgx200 \
  --storage profile_default --concurrency 500 \
  --effective-context 65536 --kv-precision fp8

# Com FP16 (2 bytes por elemento - dobra a memória)
python3 sizing.py --model opt-oss-20b --server dgx200 \
  --storage profile_default --concurrency 500 \
  --effective-context 65536 --kv-precision fp16
```

## 🧮 Metodologia de Cálculo

### 1. KV Cache por Sessão

A fórmula base por camada é:

```
KV_size = 2 × seq_length × num_kv_heads × head_dim × bytes_per_element
```

Onde:
- `2` = key + value tensors
- `seq_length` depende do padrão de atenção
- `bytes_per_element`: fp8/int8=1, fp16/bf16=2

#### Padrões de Atenção

1. **Full Attention**: Todas as camadas usam `effective_context`
2. **Sliding Window**: Todas as camadas usam `sliding_window`
3. **Hybrid**: Camadas full usam `effective_context`, camadas sliding usam `sliding_window`

### 2. Sessões por Nó

```
Budget_KV = (Total_HBM_GiB × kv_budget_ratio) - runtime_overhead_gib
Sessões_por_nó = floor(Budget_KV / KV_per_session)
```

### 3. Nós Necessários

```
Nós_mínimos = ceil(concurrency / sessões_por_nó)
Nós_com_headroom = ceil(concurrency × (1 + peak_headroom_ratio) / sessões_por_nó)
Nós_finais = Nós_com_headroom + (1 se HA=n+1, senão 0)
```

## 📊 Arquivos de Dados

### models.json

Define modelos LLM com parâmetros de arquitetura:

```json
{
  "models": [
    {
      "name": "opt-oss-120b",
      "num_layers": 36,
      "num_key_value_heads": 8,
      "head_dim": 64,
      "max_position_embeddings": 131072,
      "attention_pattern": "hybrid",
      "hybrid_full_layers": 18,
      "hybrid_sliding_layers": 18,
      "sliding_window": 128,
      "default_kv_precision": "fp8"
    }
  ]
}
```

### servers.json

Define servidores GPU com especificações de hardware:

```json
{
  "servers": [
    {
      "name": "dgx300",
      "gpus": 8,
      "hbm_per_gpu_gb": 288,
      "total_hbm_gb": 2304,
      "nvlink_bandwidth_tbps": 14.4,
      "system_memory_tb": 2
    }
  ]
}
```

### storage.json

Define perfis de storage com métricas de I/O:

```json
{
  "profiles": [
    {
      "name": "profile_default",
      "type": "nvme_local",
      "iops_read": 1000000,
      "iops_write": 800000,
      "throughput_read_gbps": 28,
      "throughput_write_gbps": 25,
      "latency_read_ms_p50": 0.08,
      "latency_read_ms_p99": 0.15
    }
  ]
}
```

## ⚠️ Validações e Avisos

O sistema gera avisos automáticos para:

- **Context overflow**: Se `effective_context > max_position_embeddings`, clamp e avisa
- **Precisão alta**: Se usar fp16/bf16, avisa que memória dobra vs fp8
- **Storage I/O**: Contextos longos (>128k) podem pressionar I/O no cold-start
- **Capacidade zero**: Se budget HBM insuficiente para uma sessão

## 📤 Saídas

### Relatório em Texto

Imprime no stdout um relatório detalhado com:
- Parâmetros do modelo, servidor e storage
- Resultados do dimensionamento
- Avisos e alertas

### JSON

Saída estruturada em JSON para integração com outras ferramentas:

```json
{
  "model": {...},
  "server": {...},
  "storage": {...},
  "parameters": {...},
  "results": {
    "kv_per_session_gib": 2.25,
    "kv_total_tib": 2.2,
    "sessions_per_node": 613,
    "nodes_minimum": 2,
    "nodes_with_headroom": 2,
    "nodes_final": 3
  },
  "warnings": [...]
}
```

## 🔧 Customização

### Adicionar Novo Modelo

Edite `models.json` e adicione entrada com todos os campos necessários.

### Adicionar Novo Servidor

Edite `servers.json` e adicione entrada com especificações de HBM.

### Adicionar Perfil de Storage

Edite `storage.json` e adicione perfil com métricas de I/O.

## 📈 Casos de Uso

1. **Planejamento de Capacidade**: Dimensionar cluster para tráfego alvo
2. **Análise de TCO**: Comparar diferentes configurações de hardware
3. **Sizing de PoC**: Validar requisitos antes de procurement
4. **Otimização**: Avaliar impacto de fp8 vs fp16, contexto, etc.

## 🎯 Limitações e Considerações

- **Foco em memória**: Cálculo baseado em KV cache (não considera latência, throughput)
- **Storage passivo**: Perfis de storage são informativos, não dimensionam storage automaticamente
- **Modelo simplificado**: Não considera fragmentação de memória, variações de batching, etc.
- **HBM como limite**: Assume que HBM é o bottleneck (geralmente verdadeiro para inferência de LLMs)

## 📄 Licença

Este projeto foi criado para uso interno de engenharia de infraestrutura.

## 👤 Autor

Sistema de Sizing de Infraestrutura IA
Data: 2026-02-07
