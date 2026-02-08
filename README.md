# Calculadora de Sizing de Inferência LLM

Sistema profissional de dimensionamento de infraestrutura para inferência de Large Language Models (LLMs) em GPUs NVIDIA DGX-class, com foco em capacity planning, resiliência operacional e otimização de custo.

---

## Visão Geral

### O Problema

Dimensionar infraestrutura para inferência de LLMs é fundamentalmente diferente de treinar modelos. Durante a inferência, o principal gargalo não é compute (FLOPs), mas **memória de GPU (HBM)**, especialmente para armazenar o **KV cache** — estruturas de dados que mantêm o contexto conversacional.

Um erro comum é dimensionar baseado apenas no tamanho do modelo (parâmetros). Na prática, para modelos modernos com contextos longos (32k–200k tokens), a memória necessária para KV cache pode **exceder em 5–10x a memória dos pesos do modelo**.

### Para Quem Este Projeto Foi Feito

- **Arquitetos de Infraestrutura**: Planejamento de capacidade e CapEx
- **Engenheiros SRE/Platform**: Definição de SLOs, HA e headroom
- **Líderes de FinOps**: Análise de custo por sessão e TCO
- **CTOs/Diretoria**: Decisões de investimento baseadas em cenários de risco

### O Que o Projeto Resolve

Este projeto calcula quantos **nós DGX** são necessários para sustentar uma carga de inferência, considerando:

- Concorrência alvo (sessões simultâneas)
- Tamanho do contexto efetivo
- Precisão do KV cache (fp8, fp16, bf16, int8)
- Tolerância a falhas (HA: none, N+1, N+2)
- Headroom para picos de tráfego

E avalia **3 cenários** automaticamente:
1. **MÍNIMO**: Atende no limite, sem folga (risco alto)
2. **RECOMENDADO**: Produção com HA e headroom (risco médio)
3. **IDEAL**: Máxima resiliência e estabilidade (risco baixo)

---

## Conceitos-Chave

### O Que é KV Cache?

Durante a geração de texto, transformers mantêm tensores **Key** e **Value** para cada token processado, em cada camada de atenção. Esses tensores formam o **KV cache**, permitindo que o modelo "lembre" o contexto sem recomputar tudo a cada token.

**Características operacionais:**
- Cresce linearmente com o tamanho do contexto (tokens)
- Cresce linearmente com o número de camadas do modelo
- Persiste em HBM durante toda a sessão
- **Não** pode ser offloaded para CPU sem degradar latência drasticamente

### Por Que Contexto e Concorrência Dominam o Custo

**Exemplo prático:**
- Modelo: opt-oss-120b (36 camadas, 8 KV heads, fp8)
- Contexto: 128k tokens
- **KV por sessão**: ~2.25 GiB

Para **1000 sessões simultâneas**:
- KV total: 2.25 TiB
- Servidor DGX B300: 2.3 TiB HBM total
- **Budget efetivo**: ~70% HBM → ~1.4 TiB usável para KV por nó
- **Resultado**: 2 nós (mínimo), 3 nós (com N+1)

Se o contexto dobrar para 256k tokens:
- KV por sessão dobra (~4.5 GiB)
- Nós necessários **dobram**

### Diferença Entre Pesos do Modelo e Memória Viva (KV)

| Aspecto | Pesos do Modelo | KV Cache |
|---------|----------------|----------|
| **Tamanho** | Fixo (ex.: 120B param = ~240 GB fp16) | Variável (contexto × concorrência) |
| **Escala com** | Arquitetura do modelo | Carga de inferência |
| **Reuso** | Compartilhado entre sessões | 1 cópia por sessão |
| **Impacto no sizing** | Overhead fixo (~80–150 GiB) | Principal limitador de capacidade |

**Implicação prática:** Aumentar concorrência de 100 para 1000 sessões (10x) **não** aumenta a memória de pesos (permanece constante), mas aumenta KV cache em 10x.

---

## Arquitetura da Solução

### sizing.py (Script Principal)

Engine de cálculo que:
1. Lê configurações de modelos, servidores e storage (JSONs)
2. Recebe parâmetros de NFR e runtime via CLI
3. Calcula KV cache por sessão baseado na arquitetura do modelo
4. Dimensiona número de nós necessários para 3 cenários
5. Gera relatório texto + JSON estruturado com racional de cálculo

**Características técnicas:**
- Python 3.8+ (stdlib only, zero dependências externas)
- ~1700 linhas, funções puras para cálculos core
- CLI via argparse, extensível

### models.json (Parâmetros de Modelos LLM)

Define características arquiteturais **fixas** de cada modelo:

```json
{
  "name": "opt-oss-120b",
  "num_layers": 36,
  "num_key_value_heads": 8,
  "head_dim": 64,
  "max_position_embeddings": 131072,
  "attention_pattern": "hybrid",
  "default_kv_precision": "fp8"
}
```

**Campos críticos:**
- `num_layers`: Impacta linearmente o tamanho do KV
- `num_key_value_heads`: Define número de heads de atenção (GQA/MQA)
- `attention_pattern`: full (contexto completo), sliding (janela), hybrid (misto)
- `max_position_embeddings`: Limite máximo de contexto do modelo

### servers.json (Hardware de Inferência)

Define especificações de servidores DGX:

```json
{
  "name": "dgx300",
  "gpus": 8,
  "hbm_per_gpu_gb": 288,
  "total_hbm_gb": 2304,
  "nvlink_bandwidth_tbps": 14.4
}
```

**Campos críticos:**
- `total_hbm_gb`: Memória total de GPU (determinante da capacidade)
- `gpus`: Número de GPUs (informativo)
- `nvlink_bandwidth_tbps`: Opcional, para análise de throughput

### storage.json (Perfis de I/O)

Define características de storage para validações:

```json
{
  "name": "profile_default",
  "type": "nvme_local",
  "iops_read": 1000000,
  "throughput_read_gbps": 28,
  "latency_read_ms_p99": 0.15
}
```

**Uso:** Gera alertas se contexto longo puder pressionar I/O (prefill, cold-start). **Não** é usado no cálculo de KV cache (que reside em HBM).

### O Que é Fixo vs Variável

| Parâmetro | Tipo | Origem | Exemplo |
|-----------|------|--------|---------|
| `num_layers` | Fixo | Arquitetura do modelo | 36 |
| `attention_pattern` | Fixo | Arquitetura do modelo | hybrid |
| `total_hbm_gb` | Fixo | Hardware do servidor | 2304 GB |
| `concurrency` | Variável | NFR do produto | 1000 |
| `effective_context` | Variável | NFR do produto | 128k |
| `kv_precision` | Variável | Configuração de runtime | fp8 |
| `peak_headroom_ratio` | Variável | Política de resiliência | 20% |

---

## Metodologia de Cálculo

### Visão Geral do Processo

1. **KV por Sessão (GiB)**
   - Calcula memória necessária para armazenar Key e Value de uma única sessão
   - Depende de: contexto efetivo, arquitetura do modelo, precisão (fp8/fp16)

2. **KV Total (TiB)**
   - Multiplica KV por sessão pela concorrência alvo
   - Representa demanda agregada do cluster

3. **Budget de HBM por Nó (GiB)**
   - Subtrai overhead fixo (modelo, ativações) da HBM total
   - Aplica fator de budget (ex.: 70%) para evitar fragmentação
   - Define quanto de HBM está disponível para KV cache

4. **Sessões por Nó**
   - Divide budget de KV pela memória de KV por sessão
   - Determina capacidade efetiva de cada nó

5. **Nós Necessários**
   - Calcula nós para atender concorrência
   - Aplica headroom para picos
   - Adiciona nós extras para HA (N+1, N+2)

### Atenção Pattern e Seu Impacto

**Full Attention:**
- Todas as camadas atendem ao contexto completo
- KV cresce linearmente com `effective_context`
- Exemplo: GPT-3, LLaMA (camadas iniciais)

**Sliding Window Attention:**
- Camadas atendem apenas a uma janela fixa (ex.: 128 tokens)
- KV **não** cresce com contexto além da janela
- Reduz drasticamente memória para contextos longos
- Exemplo: Mistral, algumas camadas de modelos híbridos

**Hybrid Attention:**
- Mistura de full e sliding por camada
- Exemplo: 18 camadas full + 18 sliding
- Balanceia qualidade e eficiência de memória

### Budget de HBM e Overhead

**Overhead típico (por nó):**
- Pesos do modelo: 80–150 GiB (dependendo do modelo e quantização)
- Ativações de computação: 10–30 GiB
- Buffers de runtime: 10–20 GiB
- **Total conservador**: 120 GiB

**Budget ratio típico:**
- 70%: Padrão balanceado
- 65%: Conservador (cenário IDEAL), reduz risco de fragmentação
- 75%: Agressivo, pode causar instabilidade em runtime

**Cálculo:**
```
HBM_total = 2304 GB × (10^9 / 2^30) = 2145.8 GiB
HBM_disponivel = 2145.8 - 120 = 2025.8 GiB
KV_budget = 2025.8 × 0.70 = 1418.0 GiB
```

### Racional Operacional

**Por que não usar 100% da HBM?**
- Fragmentação de memória ao longo do tempo
- Variação no tamanho real de contexto por sessão
- Buffers para operações temporárias (ex.: beam search)

**Por que headroom para picos?**
- Tráfego raramente é constante
- Eventos (lançamentos, promoções) causam spikes
- Manutenções planejadas reduzem capacidade temporariamente

**Por que N+1 ou N+2?**
- Hardware falha (GPUs, NVLink, alimentação)
- Manutenção preventiva exige rotação de nós
- N+1: Tolera 1 falha sem degradação
- N+2: Tolera 2 falhas ou 1 falha durante manutenção

---

## Cenários Avaliados

O script **sempre** calcula 3 cenários automaticamente. Isso permite avaliar trade-offs entre custo, risco e resiliência.

### 1. MÍNIMO (Bare Minimum)

**Objetivo:** Atender requisitos no limite absoluto

**Configuração:**
- `peak_headroom_ratio = 0%` (sem folga para picos)
- `ha_mode = none` (sem redundância)
- `kv_budget_ratio = configurado` (default 70%)

**Característica operacional:**
- Máxima eficiência de capital (menor número de nós)
- **Risco alto**: Falha de hardware causa indisponibilidade imediata
- Picos de tráfego causam throttling ou recusa de conexões
- Manutenção planejada exige downtime

**Uso típico:**
- PoC, alpha, ambientes de desenvolvimento
- Estimativa de custo mínimo absoluto
- Workloads com tráfego estável e previsível

### 2. RECOMENDADO (Production Ready)

**Objetivo:** Operação estável em produção com resiliência

**Configuração:**
- `peak_headroom_ratio = 20%` (configurável)
- `ha_mode = n+1` (tolera 1 falha)
- `kv_budget_ratio = configurado` (default 70%)

**Característica operacional:**
- Equilíbrio entre custo e resiliência
- **Risco médio**: Sistema tolera 1 falha de nó sem perda de capacidade crítica
- Absorve picos de até 20% acima da carga nominal
- Permite manutenção rotativa (rolling updates)

**Uso típico:**
- **Produção padrão** (SLA 99.9%)
- APIs comerciais, SaaS, enterprise
- Workloads com variabilidade moderada

### 3. IDEAL (Enterprise Grade)

**Objetivo:** Máxima disponibilidade e performance

**Configuração:**
- `peak_headroom_ratio = max(configurado, 30%)` (mínimo 30%)
- `ha_mode = n+2` (tolera 2 falhas)
- `kv_budget_ratio = min(configurado, 65%)` (mais conservador)

**Característica operacional:**
- Máxima resiliência operacional
- **Risco baixo**: Sistema tolera 2 falhas simultâneas de nós
- Budget conservador (65%) reduz risco de fragmentação
- Headroom mínimo de 30% para picos e imprevistos

**Uso típico:**
- Produção crítica (SLA 99.99%+)
- Financeiro, healthcare, governo
- Workloads com alta imprevisibilidade
- Ambientes com histórico de falhas múltiplas

### Comparação Rápida

| Critério | Mínimo | Recomendado | Ideal |
|----------|--------|-------------|-------|
| **CapEx** | Baseline | +30–50% | +80–150% |
| **Tolerância a falhas** | 0 nós | 1 nó | 2 nós |
| **Headroom** | 0% | 20% | 30%+ |
| **Risco de indisponibilidade** | Alto | Médio | Baixo |
| **SLA típico** | < 99% | 99.9% | 99.99%+ |

---

## Saídas do Script

### 1. Relatório em Texto (stdout)

Formato estruturado em 4 seções:

**SEÇÃO 1: Entradas**
- Parâmetros do modelo (lidos de models.json)
- Parâmetros do servidor (lidos de servers.json)
- Parâmetros de storage (lidos de storage.json)
- NFRs configurados (concorrência, contexto, precisão, etc.)

**SEÇÃO 2: Dicionário de Parâmetros**
- Explicação detalhada de cada parâmetro usado
- Origem (modelo, hardware, NFR, runtime)
- Importância para o sizing
- Erros comuns

**SEÇÃO 3: Resultados por Cenário**

Para cada cenário (MÍNIMO, RECOMENDADO, IDEAL):
- KV per session (GiB)
- KV total (TiB)
- Budget de HBM por nó (GiB)
- Sessões por nó
- Nós necessários (capacidade, com headroom, final com HA)

E para cada resultado, um **Racional** explicando:
- Fórmula usada
- Inputs do cálculo
- Interpretação operacional

**SEÇÃO 4: Alertas e Riscos**
- Validações automáticas (ex.: contexto excede max, precisão fp16 dobra memória)
- Impactos operacionais
- Recomendações

### 2. JSON Estruturado (stdout final)

```json
{
  "inputs": {
    "model": {...},
    "server": {...},
    "storage": {...},
    "nfr": {
      "concurrency": 1000,
      "effective_context": 131072,
      "kv_precision": "fp8",
      ...
    }
  },
  "parameter_dictionary": {
    "num_layers": {
      "description": "...",
      "source": "...",
      "importance": "...",
      "common_errors": "..."
    },
    ...
  },
  "scenarios": {
    "minimum": {
      "name": "MÍNIMO",
      "configuration": {...},
      "results": {
        "kv_per_session_gib": 2.25,
        "kv_total_tib": 2.2,
        "nodes_final": 2,
        ...
      },
      "rationale": {
        "kv_per_session_gib": {
          "formula": "...",
          "inputs": {...},
          "explanation": "..."
        },
        ...
      },
      "warnings": [...]
    },
    "recommended": {...},
    "ideal": {...}
  },
  "alerts": [...]
}
```

**Uso do JSON:**
- Integração com pipelines de IaC (Terraform, Ansible)
- Dashboards de capacity planning
- Análise programática de cenários
- Export para planilhas (FinOps)

### 3. Relatório Executivo (Opcional)

Com flag `--executive-report`, gera relatório especializado para diretoria:

- Sumário executivo (1 página)
- Cenários apresentados primeiro (tabela comparativa)
- Linguagem estratégica (não técnica)
- Foco em capacidade, risco, custo e decisão
- Recomendação final clara e acionável

**Uso:** Apresentações para comitê de investimento, CFO, CTO.

---

## Como Interpretar os Resultados

### Campos-Chave para Capacity Planning

**`nodes_final` (por cenário)**
- Número de nós DGX a provisionar
- Multiplicar por custo unitário do servidor para CapEx
- Comparar MÍNIMO vs RECOMENDADO vs IDEAL para análise de custo-benefício

**`sessions_per_node`**
- Capacidade efetiva de cada nó
- Se = 0, **erro crítico**: não cabe nem 1 sessão
  - Ações: reduzir contexto, usar fp8, aumentar overhead, ou servidor maior

**`kv_per_session_gib`**
- Memória por sessão ativa
- Dobra se usar fp16 em vez de fp8
- Cresce linearmente com contexto

### Alertas Críticos

**"effective_context excede max_position_embeddings"**
- Contexto solicitado maior que limite do modelo
- Script clampará automaticamente, mas indica configuração errada

**"kv_precision=fp16/bf16 usa 2x memória"**
- Considerar fp8 ou int8 (qualidade equivalente na maioria dos casos)
- Impacto direto: dobro de nós necessários

**"kv_budget_ratio > 0.75"**
- Alocação agressiva de HBM aumenta risco de instabilidade
- Reduzir para 0.70 ou menos

**"Não cabe nem 1 sessão por nó"**
- **Erro fatal de dimensionamento**
- Ajustar: contexto, precisão, overhead, ou usar servidor maior

### Sinais de Subdimensionamento

- `sessions_per_node` muito baixo (< 50): contexto muito longo ou precisão ineficiente
- `nodes_final` muito alto (> 20): revisar NFRs ou considerar modelo menor
- Diferença pequena entre MÍNIMO e RECOMENDADO (< 20%): carga leve, considerar otimizações

---

## Limitações Conhecidas

### O Que o Script NÃO Calcula

**Latência e Throughput:**
- Não estima tokens/s, TTFT (Time To First Token), ou TBT (Time Between Tokens)
- Não considera FLOPs ou utilização de compute
- **Por quê:** Latência depende de implementação (vLLM, TRT-LLM), kernels, batching dinâmico

**Network e I/O:**
- Não dimensiona bandwidth de rede entre nós
- Não calcula IOPS necessário para checkpoint/restore
- **Por quê:** Storage profile é usado apenas para alertas, não sizing

**Custos Operacionais:**
- Não calcula TCO (energia, cooling, manutenção)
- Não estima custo por sessão ou por token
- **Por quê:** Custos variam por região, fornecedor, contrato

**Batching e Otimizações:**
- Assume sessões independentes (1 sessão = 1 KV cache)
- Não considera continuous batching, PagedAttention, ou técnicas de compressão
- **Por quê:** Ganhos dependem de implementação específica

### Premissas Assumidas

1. **KV cache permanece em HBM durante toda a sessão**
   - Offload para CPU não é considerado (degradaria latência)

2. **Overhead fixo por nó (default: 120 GiB)**
   - Válido para modelos 20B–120B quantizados
   - Ajustar `--runtime-overhead-gib` se necessário

3. **Sessões têm contexto uniforme**
   - Na prática, varia por usuário
   - Budget deve acomodar percentil alto (P95/P99)

4. **Budget ratio conservador (70%)**
   - Evita fragmentação de memória ao longo do tempo
   - Valores >75% aumentam risco operacional

5. **Servidor opera com todas as GPUs funcionais**
   - Falhas parciais (1–2 GPUs) reduzem capacidade
   - HA (N+1/N+2) mitiga, mas não elimina completamente

### Dependência de Precisão dos Dados de Entrada

**Impacto de erros nos JSONs:**

| Parâmetro Errado | Impacto |
|------------------|---------|
| `num_layers` (incorreto) | KV calculado errado, sizing inválido |
| `total_hbm_gb` (incorreto) | Capacidade superestimada ou subestimada |
| `max_position_embeddings` (incorreto) | Validação de contexto falha |
| `attention_pattern` (incorreto) | KV pode ser 2–5x maior que o real |

**Recomendação:** Sempre validar parâmetros contra documentação oficial do modelo e especificações do hardware.

---

## Público-Alvo e Casos de Uso

### 1. Planejamento de Capacidade Anual

**Contexto:** Estimar crescimento de infraestrutura para os próximos 12 meses.

**Como usar:**
- Rodar sizing para projeções Q1, Q2, Q3, Q4 (concorrência crescente)
- Comparar `nodes_final` por trimestre
- Planejar procurement escalonado

**Exemplo:**
```bash
# Q1: 1k sessões → 3 nós
python3 sizing.py --concurrency 1000 ...

# Q4: 5k sessões → 12 nós
python3 sizing.py --concurrency 5000 ...

# Procurement: 3 nós agora, +3 em Q2, +3 em Q3, +3 em Q4
```

### 2. Avaliação de Investimento (CapEx)

**Contexto:** CFO pede justificativa para compra de nós DGX.

**Como usar:**
- Gerar relatório executivo (`--executive-report`)
- Mostrar diferença entre MÍNIMO, RECOMENDADO, IDEAL
- Apresentar CapEx relativo (+30%, +80%) e risco operacional

**Exemplo:**
```bash
python3 sizing.py ... --executive-report --output-markdown-file proposal.md

# proposal.md contém:
# - Sumário executivo para CFO
# - Tabela comparativa de cenários
# - Recomendação: RECOMENDADO (N+1, SLA 99.9%)
```

### 3. Comparação de Arquiteturas

**Contexto:** Decidir entre DGX B300 vs H200 vs cloud.

**Como usar:**
- Rodar sizing para cada servidor
- Comparar `nodes_final` e `sessions_per_node`
- Calcular TCO: `nodes_final × custo_unitário × 3 anos`

**Exemplo:**
```bash
# DGX B300: 3 nós × $500k = $1.5M
python3 sizing.py --server dgx300 ...

# DGX H200: 5 nós × $300k = $1.5M (mesma capacidade, custo similar)
python3 sizing.py --server dgx200 ...
```

### 4. Discussão com Fornecedores

**Contexto:** Negociar contrato com NVIDIA, AWS, Azure.

**Como usar:**
- Apresentar cálculos de sizing como baseline técnico
- Validar se proposta do fornecedor atende NFRs
- Usar JSON para comparar múltiplas propostas

**Exemplo:**
```bash
# Gerar JSON para cada proposta
python3 sizing.py ... --output-json-file proposta_a.json
python3 sizing.py ... --output-json-file proposta_b.json

# Comparar nodes_final, sessions_per_node, alertas
```

### 5. Resposta a Incidentes de Capacidade

**Contexto:** Sistema atingiu limite, filas de espera crescendo.

**Como usar:**
- Rodar sizing com carga atual
- Identificar se está em cenário MÍNIMO (sem folga)
- Mostrar necessidade de escala para RECOMENDADO

**Exemplo:**
```bash
# Diagnóstico: operando com 2 nós (MÍNIMO), picos causam degradação
python3 sizing.py --concurrency 1000 ...

# Output mostra:
# MÍNIMO: 2 nós (você está aqui) → Risco: Alto
# RECOMENDADO: 3 nós → Adicionar 1 nó resolve picos
```

---

## Instalação e Requisitos

### Pré-requisitos

- Python 3.8 ou superior
- Nenhuma dependência externa (usa apenas stdlib)

### Instalação

Nenhuma instalação necessária. Basta clonar o repositório:

```bash
git clone <repo>
cd calculadoraSizingInfraIA
```

### Estrutura de Arquivos

```
calculadoraSizingInfraIA/
├── README.md          # Este arquivo (documentação completa)
├── QUICKSTART.md      # Guia de uso rápido
├── sizing.py          # Script principal
├── models.json        # Parâmetros de modelos LLM
├── servers.json       # Especificações de servidores DGX
└── storage.json       # Perfis de storage (para alertas)
```

---

## Contribuindo e Extensões

### Adicionar Novo Modelo

Editar `models.json`:

```json
{
  "name": "seu-modelo-200b",
  "num_layers": 48,
  "num_key_value_heads": 16,
  "head_dim": 128,
  "max_position_embeddings": 200000,
  "attention_pattern": "full",
  "default_kv_precision": "fp8",
  "notes": "Seu modelo customizado"
}
```

### Adicionar Novo Servidor

Editar `servers.json`:

```json
{
  "name": "seu-servidor",
  "gpus": 8,
  "hbm_per_gpu_gb": 320,
  "total_hbm_gb": 2560,
  "notes": "Especificações do seu servidor"
}
```

### Validar JSONs

```bash
python3 -m json.tool models.json
python3 -m json.tool servers.json
python3 -m json.tool storage.json
```

---

## Licença e Autoria

Este projeto foi desenvolvido como ferramenta interna de sizing de infraestrutura para inferência de LLMs, com foco em capacity planning, resiliência operacional e otimização de custo.

**Versão:** 2.0  
**Data:** 2026-02-08  
**Linguagem:** Python 3.8+ (stdlib only)
