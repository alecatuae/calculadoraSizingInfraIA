# Calculadora de Sizing de Infraestrutura para Inferência

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
- **✨ NOVO (v3.0):** Dimensionamento completo de **storage** (volumetria, IOPS, throughput)

E avalia **3 cenários** automaticamente:
1. **MÍNIMO**: Atende no limite, sem folga (risco alto)
2. **RECOMENDADO**: Produção com HA e headroom (risco médio)
3. **IDEAL**: Máxima resiliência e estabilidade (risco baixo)

**Storage** é dimensionado por cenário, considerando:
- Pesos do modelo (checkpoints, shards, versionamento)
- Cache de runtime (engine compilado, artefatos)
- Logs, métricas e auditoria (retenção variável)
- Dados operacionais (configurações, metadados)

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

### Por Que Storage é Crítico (Não Apenas "Onde o Modelo Fica")

Embora o KV cache resida em HBM (memória GPU), **storage** é um recurso operacional crítico para:

#### 1. Operação Contínua
- **Pesos do modelo** (checkpoints/shards): Necessários para startup, restart, scale-out
- **Cache de runtime** (engine compilado TensorRT-LLM/NIM): Reduz tempo de inicialização de ~10min para ~30s
- **Logs e métricas**: Essenciais para debugging, auditoria, conformidade

#### 2. Resiliência e Tempo de Recuperação
- **Restart de nós**: Storage subdimensionado aumenta tempo de recuperação de minutos para horas
- **Scale-out**: IOPS insuficientes criam gargalo ao adicionar nós simultaneamente
- **Versionamento**: Rollback rápido requer múltiplas versões de checkpoints

#### 3. Governança e Conformidade
- **Retenção de logs**: Auditoria e troubleshooting exigem retenção adequada (7–90 dias)
- **Métricas de inferência**: SLO tracking, billing, capacity planning
- **Traces distribuídos**: Diagnóstico de latência e comportamento anômalo

**Dimensionamento por Cenário:**
- **MÍNIMO**: Apenas operação steady-state (retenção 7 dias, sem margem para picos)
- **RECOMENDADO**: Suporta picos e restart de 25% dos nós (retenção 30 dias)
- **IDEAL**: Máxima resiliência, falhas em cascata, retenção estendida (90 dias)

**Exemplo Prático (opt-oss-120b, 1000 sessões, 3 nós):**
- Storage RECOMENDADO: **~7.8 TB** (2.5 TB modelo + 3 TB cache + 1.8 TB logs + 0.5 TB ops)
- IOPS pico: **187,500 leitura** (restart de 25% dos nós) / **3,000 escrita** (flush de logs)
- Throughput pico: **6.9 GB/s leitura** (modelo < 60s) / **0.7 GB/s escrita**

---

## Arquitetura da Solução

### main.py + /sizing/ (Arquitetura Modular)

**✨ NOVO (v2.0):** Projeto refatorado para arquitetura modular!

**main.py** orquestra o fluxo completo:
1. Parse CLI (sizing/cli.py)
2. Carrega configurações (sizing/config_loader.py)
3. Calcula KV cache (sizing/calc_kv.py)
4. Calcula VRAM real (sizing/calc_vram.py)
5. Calcula storage (sizing/calc_storage.py) **← NOVO v3.0**
6. Avalia 3 cenários (sizing/calc_scenarios.py)
7. Gera relatórios (sizing/report_full.py, sizing/report_exec.py)
8. Salva arquivos (sizing/writer.py)

**Características técnicas:**
- Python 3.8+ (stdlib only, zero dependências externas)
- Módulos especializados (~200 linhas cada)
- Funções puras para cálculos core
- CLI via argparse, extensível
- Fácil manutenção e testes

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

### storage.json (Perfis de Storage)

Define características completas de storage para dimensionamento e validações:

```json
{
  "name": "profile_default",
  "type": "nvme_local",
  "capacity_total_tb": 61.44,
  "usable_capacity_tb": 56.0,
  "iops_read_max": 1000000,
  "iops_write_max": 800000,
  "throughput_read_gbps": 28,
  "throughput_write_gbps": 25,
  "latency_read_ms_p50": 0.08,
  "latency_read_ms_p99": 0.15,
  "latency_write_ms_p50": 0.10,
  "latency_write_ms_p99": 0.20
}
```

**✨ NOVO (v3.0):** Storage agora é dimensionado ativamente:
- **Volumetria calculada**: Pesos, cache, logs, dados operacionais
- **IOPS por cenário**: Steady-state vs. pico (restart, scale-out)
- **Throughput por cenário**: Otimizado para tempo de recuperação < 60s
- **Alertas automáticos**: Se requisitos excedem capacidade do perfil

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

## Arquitetura Data-Driven e Schemas

### Princípios Fundamentais

Este projeto segue uma **arquitetura data-driven**, onde todos os valores usados nos cálculos vêm exclusivamente dos arquivos JSON:

- `models.json` → parâmetros arquiteturais de LLMs
- `servers.json` → especificações de hardware de servidores GPU
- `storage.json` → perfis de storage (IOPS, throughput, block size)

**Nenhum valor hardcoded** no código. Isso permite:
- ✅ Adicionar novos modelos/servidores/storages sem editar código
- ✅ Evolução contínua via incremento de JSON
- ✅ Validação automática de schemas e constraints
- ✅ Governança e auditoria

---

### Como Adicionar Novos Modelos/Servidores/Storages

#### A) Onde ficam os arquivos

Todos os arquivos JSON estão na raiz do projeto:
```
calculadoraSizingInfraIA/
├── models.json     # Modelos de LLM
├── servers.json    # Servidores GPU
├── storage.json    # Perfis de storage
├── main.py
└── sizing/
```

#### B) Passos para adicionar um novo item

1. **Copie um item existente** do arquivo JSON relevante
2. **Altere o `name`** (deve ser único, case-insensitive)
3. **Preencha os campos obrigatórios** (veja schemas abaixo)
4. **Execute validação:**
   ```bash
   python3 main.py --validate-only
   ```
5. **Se válido, execute um sizing de teste:**
   ```bash
   python3 main.py --model <seu-modelo> --server <seu-servidor> --storage profile_default --concurrency 100 --effective-context 32768
   ```

---

### Schema Completo: `models.json`

| Campo | Tipo | Obrigatório? | Descrição | Unidade/Enum | Exemplo |
|-------|------|--------------|-----------|--------------|---------|
| `name` | str | ✅ Sim | Nome único do modelo | - | `"opt-oss-120b"` |
| `num_layers` | int | ✅ Sim | Número total de camadas do transformer | layers | `96` |
| `num_key_value_heads` | int | ✅ Sim | Número de cabeças KV (GQA/MQA/MHA) | heads | `8` |
| `head_dim` | int | ✅ Sim | Dimensão de cada cabeça de atenção | dims | `128` |
| `max_position_embeddings` | int | ✅ Sim | Contexto máximo suportado pelo modelo | tokens | `131072` |
| `attention_pattern` | str | ✅ Sim | Padrão de atenção | enum: `full` \| `sliding` \| `hybrid` | `"full"` |
| `hybrid_full_layers` | int | ⚠️ Se `hybrid` | Número de camadas com atenção full (hybrid) | layers | `48` |
| `hybrid_sliding_layers` | int | ⚠️ Se `hybrid` | Número de camadas com atenção sliding (hybrid) | layers | `48` |
| `sliding_window` | int | ⚠️ Se `sliding`/`hybrid` | Tamanho da janela de atenção sliding | tokens | `4096` |
| `default_kv_precision` | str | ✅ Sim | Precisão padrão do KV cache | enum: `fp16` \| `bf16` \| `fp8` \| `int8` | `"fp8"` |
| `total_params_b` | float\|null | ❌ Não | Parâmetros totais (bilhões) | B | `120.5` |
| `active_params_b` | float\|null | ❌ Não | Parâmetros ativos (MoE) | B | `13.0` |
| `weights_memory_gib_fp16` | float\|null | ❌ Não | Memória dos pesos em FP16 | GiB | `224.4` |
| `weights_memory_gib_bf16` | float\|null | ❌ Não | Memória dos pesos em BF16 | GiB | `224.4` |
| `weights_memory_gib_fp8` | float\|null | ❌ Não | Memória dos pesos em FP8 | GiB | `112.2` |
| `weights_memory_gib_int8` | float\|null | ❌ Não | Memória dos pesos em INT8 | GiB | `112.2` |
| `weights_memory_gib_int4` | float\|null | ❌ Não | Memória dos pesos em INT4 | GiB | `56.1` |
| `default_weights_precision` | str | ❌ Não | Precisão padrão dos pesos | enum: `fp16` \| `bf16` \| `fp8` \| `int8` \| `int4` | `"fp8"` |
| `model_artifact_size_gib` | float\|null | ❌ Não | Tamanho do artefato para warmup/storage | GiB | `230.0` |
| `notes` | str | ❌ Não | Notas e observações | - | `"Modelo open-source..."` |

**Constraints:**
- Todos os valores numéricos devem ser > 0
- Se `attention_pattern = "hybrid"`: `hybrid_full_layers + hybrid_sliding_layers` deve ser igual a `num_layers`
- Se `attention_pattern = "sliding"` ou `"hybrid"`: `sliding_window` é obrigatório

---

### Schema Completo: `servers.json` (Estrutura Hierárquica)

**✨ NOVO (v2.0):** `servers.json` usa estrutura **hierárquica (nested)** para organizar componentes logicamente.

**Documentação completa:** [`servers.schema.md`](servers.schema.md)

#### Estrutura Nested

```json
{
  "servers": [
    {
      "name": "dgx-b300",
      "manufacturer": "NVIDIA",
      "form_factor": "Rackmount",
      "rack_units_u": 10,
      
      "cpu": { ... },
      "system_memory": { ... },
      "gpu": { ... },        // Obrigatório
      "power": { ... },      // Obrigatório
      "thermal": { ... },
      "cooling": { ... },
      "storage": { ... },
      "networking": { ... },
      "software": { ... },
      "physical": { ... },
      
      "notes": "...",
      "source": [ ... ]
    }
  ]
}
```

#### Campos Obrigatórios (Mínimo)

| Campo | Tipo | Descrição | Usado no Cálculo? |
|-------|------|-----------|-------------------|
| `name` | string | Nome único do servidor | ✅ Identificação |
| `rack_units_u` | integer | Espaço em rack (U) | ✅ **Rack total** |
| `gpu.count` | integer | Número de GPUs por nó | ✅ **HBM total e paralelismo** |
| `gpu.model` | string | Modelo da GPU | ✅ Identificação |
| `gpu.hbm_per_gpu_gb` | float | HBM por GPU (GB) | ✅ **Capacidade crítica** |
| `power.power_kw_max` | float | Consumo máximo (kW) | ✅ **Energia total** |

#### Campos Opcionais Importantes

| Campo | Tipo | Descrição |
|-------|------|-----------|
| `gpu.total_hbm_gb` | float | HBM total (validado automaticamente) |
| `thermal.heat_output_btu_hr_max` | float | Dissipação térmica (BTU/hr) |
| `source` | array[string] | Links de documentação oficial |

**Constraints:**
- `rack_units_u > 0`
- `gpu.count > 0`
- `gpu.hbm_per_gpu_gb > 0`
- `power.power_kw_max > 0`
- Se `gpu.total_hbm_gb` presente: validação automática de consistência (tolerância 1%)

#### Exemplo de Adição de Servidor

**Passo 1:** Edite `servers.json` e adicione:

```json
{
  "name": "dgx-h200",
  "rack_units_u": 10,
  "gpu": {
    "count": 8,
    "model": "NVIDIA H200",
    "hbm_per_gpu_gb": 141.0,
    "total_hbm_gb": 1128.0
  },
  "power": {
    "power_kw_max": 10.2
  },
  "thermal": {
    "heat_output_btu_hr_max": 34800.0
  },
  "notes": "DGX H200 com 8x H200 (141GB HBM3 cada)",
  "source": [
    "https://docs.nvidia.com/dgx/dgxh200-user-guide/"
  ]
}
```

**Passo 2:** Validar:
```bash
python3 main.py --validate-only
```

**Passo 3:** Testar:
```bash
python3 main.py --model opt-oss-120b --server dgx-h200 --storage profile_default --concurrency 1000 --effective-context 131072
```

**Checklist:**
- [ ] Nome único
- [ ] Seções `gpu` e `power` preenchidas
- [ ] Campos obrigatórios: `gpu.count`, `gpu.hbm_per_gpu_gb`, `power.power_kw_max`
- [ ] Valores > 0
- [ ] Se `gpu.total_hbm_gb`: consistência validada
- [ ] `python3 main.py --validate-only` → ✅ OK

**Documentação detalhada:** Consulte [`servers.schema.md`](servers.schema.md) para schema completo com todos os campos, seções opcionais e exemplos

---

### Schema Completo: `storage.json`

| Campo | Tipo | Obrigatório? | Descrição | Unidade/Enum | Exemplo |
|-------|------|--------------|-----------|--------------|---------|
| `name` | str | ✅ Sim | Nome único do perfil de storage | - | `"profile_default"` |
| `type` | str | ✅ Sim | Tipo de storage | - | `"nvme_local"` |
| `capacity_total_tb` | float | ✅ Sim | Capacidade total bruta | TB | `61.44` |
| `usable_capacity_tb` | float | ✅ Sim | Capacidade utilizável | TB | `56.0` |
| `iops_read_max` | int | ✅ Sim | IOPS máximo de leitura | IOPS | `1000000` |
| `iops_write_max` | int | ✅ Sim | IOPS máximo de escrita | IOPS | `800000` |
| `throughput_read_mbps` | float | ✅ Sim | Throughput máximo de leitura | MB/s (decimal) | `3500.0` |
| `throughput_write_mbps` | float | ✅ Sim | Throughput máximo de escrita | MB/s (decimal) | `3125.0` |
| `block_size_kb_read` | float | ✅ Sim | Tamanho de bloco típico leitura | KB | `3.584` |
| `block_size_kb_write` | float | ✅ Sim | Tamanho de bloco típico escrita | KB | `4.0` |
| `latency_read_ms_p50` | float\|null | ❌ Não | Latência leitura (percentil 50) | ms | `0.08` |
| `latency_read_ms_p99` | float\|null | ❌ Não | Latência leitura (percentil 99) | ms | `0.15` |
| `latency_write_ms_p50` | float\|null | ❌ Não | Latência escrita (percentil 50) | ms | `0.10` |
| `latency_write_ms_p99` | float\|null | ❌ Não | Latência escrita (percentil 99) | ms | `0.20` |
| `rack_units_u` | int | ❌ Não | Espaço ocupado em rack | U | `2` |
| `power_kw` | float | ❌ Não | Consumo elétrico | kW | `0.5` |
| `notes` | str | ❌ Não | Notas e observações | - | `"Perfil padrão..."` |

**Constraints:**
- Todos os valores numéricos devem ser > 0
- `usable_capacity_tb` ≤ `capacity_total_tb`
- **CRÍTICO:** `Throughput(MB/s) = (IOPS × BlockSize(KB)) / 1024`
  - Se divergência > 25%: **ERRO (bloqueia relatório)**
  - Se divergência 10-25%: **WARNING**
  - Se divergência < 10%: **OK**

---

### Validação Automática de Storage (Física)

O script valida automaticamente a **consistência física** entre IOPS, Throughput e Block Size usando a fórmula:

```
Throughput(MB/s) = (IOPS × BlockSize(KB)) / 1024
```

**Exemplo de validação OK:**
```json
{
  "iops_read_max": 1000000,
  "block_size_kb_read": 3.584,
  "throughput_read_mbps": 3500.0
}
```
Cálculo: `(1000000 × 3.584) / 1024 = 3500.0` ✅

**Exemplo de erro (divergência > 25%):**
```json
{
  "iops_read_max": 100000,
  "block_size_kb_read": 4.0,
  "throughput_read_mbps": 5000.0  ❌ ERRADO (deveria ser ~390 MB/s)
}
```

---

### Exemplo Completo: Adicionar Novo Modelo

**Passo 1:** Edite `models.json` e adicione:

```json
{
  "name": "llama-4-70b",
  "num_layers": 80,
  "num_key_value_heads": 8,
  "head_dim": 128,
  "max_position_embeddings": 131072,
  "attention_pattern": "full",
  "default_kv_precision": "fp8",
  "total_params_b": 70.0,
  "weights_memory_gib_fp16": 130.2,
  "weights_memory_gib_fp8": 65.1,
  "default_weights_precision": "fp8",
  "model_artifact_size_gib": 140.0,
  "notes": "LLaMA 4 70B com suporte a 128K context"
}
```

**Passo 2:** Validar:
```bash
python3 main.py --validate-only
```

**Passo 3:** Testar sizing:
```bash
python3 main.py \
  --model llama-4-70b \
  --server dgx-b300 \
  --storage profile_default \
  --concurrency 500 \
  --effective-context 65536 \
  --kv-precision fp8
```

---

### Exemplo Completo: Adicionar Novo Servidor

**Passo 1:** Edite `servers.json` e adicione:

```json
{
  "name": "dgx-h200",
  "gpus": 8,
  "hbm_per_gpu_gb": 141.0,
  "rack_units_u": 10,
  "power_kw_max": 10.2,
  "heat_output_btu_hr_max": 34800.0,
  "notes": "NVIDIA DGX H200 com 8x H200 (141GB HBM3 cada)"
}
```

**Passo 2:** Validar:
```bash
python3 main.py --validate-only
```

**Passo 3:** Testar sizing:
```bash
python3 main.py \
  --model opt-oss-120b \
  --server dgx-h200 \
  --storage profile_default \
  --concurrency 1000 \
  --effective-context 131072
```

---

### Exemplo Completo: Adicionar Novo Storage

**Passo 1:** Edite `storage.json` e adicione:

```json
{
  "name": "profile_enterprise_ssd",
  "type": "enterprise_ssd_array",
  "capacity_total_tb": 200.0,
  "usable_capacity_tb": 180.0,
  "iops_read_max": 750000,
  "iops_write_max": 600000,
  "throughput_read_mbps": 2400.0,
  "throughput_write_mbps": 2000.0,
  "block_size_kb_read": 3.2,
  "block_size_kb_write": 3.413,
  "latency_read_ms_p50": 0.12,
  "latency_read_ms_p99": 0.25,
  "latency_write_ms_p50": 0.15,
  "latency_write_ms_p99": 0.30,
  "rack_units_u": 4,
  "power_kw": 1.5,
  "notes": "Array SSD enterprise com 24x SSD NVMe em JBOD"
}
```

**IMPORTANTE:** Validar consistência física:
- Read: `(750000 × 3.2) / 1024 = 2343.75` ≈ `2400.0` ✅ (2.4% divergência)
- Write: `(600000 × 3.413) / 1024 = 2000.0` ✅ (0% divergência)

**Passo 2:** Validar:
```bash
python3 main.py --validate-only
```

---

### Checklist Rápido

Antes de commitar novos itens:

- [ ] Nome é único (case-insensitive)
- [ ] Todos os campos obrigatórios preenchidos
- [ ] Unidades estão corretas (GiB vs GB, MB/s, etc.)
- [ ] Enums estão com valores válidos
- [ ] Para `hybrid`: `hybrid_full_layers + hybrid_sliding_layers = num_layers`
- [ ] Para `storage`: IOPS/Throughput/BlockSize são fisicamente consistentes (< 10% divergência)
- [ ] Rodar `python3 main.py --validate-only` → ✅ OK
- [ ] Rodar um sizing simples de teste → relatórios gerados

---

### Comando de Validação

Para validar todos os arquivos JSON sem executar sizing:

```bash
python3 main.py --validate-only
```

**O que é validado:**
- ✅ Schema de todos os modelos, servidores e storages
- ✅ Campos obrigatórios presentes
- ✅ Tipos corretos
- ✅ Valores em enums válidos
- ✅ Constraints (ex.: soma de layers, valores > 0)
- ✅ Nomes únicos
- ✅ Consistência física de storage (IOPS/Throughput/BlockSize)

**Saída esperada (se tudo OK):**
```
====================================================================================================
VALIDAÇÃO DE STORAGE (Consistência Física IOPS/Throughput/BlockSize)
====================================================================================================
[... tabelas de validação ...]

====================================================================================================
VALIDAÇÃO DE SCHEMAS E CONSTRAINTS
====================================================================================================

✅ Todos os arquivos de configuração são válidos.
====================================================================================================
```

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

### 1. Resumo Executivo no Terminal (stdout)

Saída resumida para validação rápida e decisão inicial:

**Formato da Tabela:**

```
================================================================================
RESUMO EXECUTIVO - SIZING DE INFRAESTRUTURA PARA INFERÊNCIA
================================================================================

Modelo:              opt-oss-120b
Servidor:            dgx-b300
Contexto Efetivo:    131,072 tokens
Concorrência Alvo:   1,000 sessões simultâneas
Precisão KV Cache:   FP8

--------------------------------------------------------------------------------
Cenário          Nós     kW      Rack    Storage (TB)   Sessões/Nó  KV/Sessão (GiB)
------------------------------------------------------------------------------------------------------------------------
MÍNIMO             2    29.0      20          4.2              629             2.25
RECOMENDADO        3    43.5      30          7.8              629             2.25
IDEAL              5    72.5      50         15.6              584             2.25
------------------------------------------------------------------------------------------------------------------------

✓ Cenário RECOMENDADO (3 nós, 43.5 kW, 30U, 7.8 TB storage) atende os requisitos com 
  tolerância a falhas (N+1).

================================================================================
📄 Relatórios completos salvos em:
   • Texto:  relatorios/sizing_<model>_<server>_<timestamp>.txt
   • JSON:   relatorios/sizing_<model>_<server>_<timestamp>.json
   • Executivo: relatorios/executive_<model>_<server>_<timestamp>.md
```

**Inclui:**
- **Energia (kW)**: Consumo elétrico total por cenário (impacto em PDU/UPS)
- **Rack (U)**: Espaço físico em rack necessário (densidade de datacenter)
- **Status final**: Validação de viabilidade operacional

### 2. Relatório Completo em Texto (relatorios/*.txt)

Artefato formal detalhado em 4 seções:

**SEÇÃO 1: Entradas**
- Parâmetros do modelo (lidos de models.json)
- Parâmetros do servidor (lidos de servers.json, **incluindo energia e rack**)
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
- **Energia total (kW)** e consumo anual (MWh)
- **Espaço em rack (U)** e equivalente em racks padrão (42U)
- **Dissipação térmica (BTU/hr)** e tons de refrigeração
- **✨ NOVO v3.0: Storage por cenário**
  - Volumetria total (TB): modelo + cache + logs + operacional
  - IOPS (pico e steady-state): leitura e escrita
  - Throughput (pico e steady-state): leitura e escrita
  - Alertas se requisitos excedem capacidade do perfil

E para cada resultado, um **Racional** explicando:
- Fórmula usada
- Inputs do cálculo
- Interpretação operacional
- **Impacto físico no datacenter**

**SEÇÃO 4: Alertas e Riscos**
- Validações automáticas (ex.: contexto excede max, precisão fp16 dobra memória)
- Impactos operacionais
- Recomendações
- Alertas sobre capacidade elétrica e densidade de rack

### 3. JSON Estruturado (relatorios/*.json)

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

### 4. Relatório Executivo (relatorios/executive_*.md)

Com flag `--executive-report`, gera relatório especializado para diretoria em Markdown:

**Estrutura Executiva Obrigatória:**

1. **Sumário Executivo**: Problema, modelo, carga, impacto em servidores/energia/datacenter
2. **Cenários Avaliados**: Tabela comparativa (Mínimo/Recomendado/Ideal) com objetivos e riscos
3. **Informações do Modelo**: Perfil técnico simplificado
4. **Consumo Unitário**: KV/sessão, % HBM por sessão, energia estimada por sessão
5. **Consumo Agregado**: Total de KV, energia (kW + MWh/ano), rack (U), térmica (BTU/hr)
6. **Resultados por Cenário**: Tabelas individuais com **energia**, **rack** e significado operacional
7. **Racional de Cálculo**: Tabela com fórmulas, parâmetros, suposições e significado operacional (incluindo energia e rack)
8. **Comparação Executiva**: Tabela comparativa incluindo CapEx relativo, energia relativa
9. **Recomendação Final**: Decisão clara com justificativa baseada em estabilidade, energia, datacenter e risco
10. **Dicionário de Parâmetros**: Tabela com parâmetros físicos (power_kw_max, rack_units_u)

**Foco Executivo:**
- Linguagem estratégica (não acadêmica)
- Todas as métricas em tabelas
- **Impacto físico explícito**: Energia (kW, MWh/ano), Rack (U, racks), Térmica (BTU/hr, tons)
- Decisão baseada em custo implícito, densidade e resiliência
- Consumo unitário vs agregado claramente separado

**Uso:** 
- Apresentações para comitê de investimento, CFO, CTO
- Decisões de datacenter (capacidade elétrica, densidade de rack, cooling)
- Análise de TCO (incluindo OpEx elétrico)

---

## Como Interpretar os Resultados

### Campos-Chave para Capacity Planning

**`nodes_final` (por cenário)**
- Número de nós DGX a provisionar
- Multiplicar por custo unitário do servidor para CapEx
- Comparar MÍNIMO vs RECOMENDADO vs IDEAL para análise de custo-benefício

**`total_power_kw` (por cenário)**
- Consumo elétrico total contínuo
- Dimensiona PDU, UPS, contrato de energia
- Considerar PUE (~1.4x) para cooling: total_facility_kw = total_power_kw × PUE
- Multiplicar por 8.76 para obter MWh/ano (OpEx elétrico)

**`total_rack_u` (por cenário)**
- Espaço físico em rack necessário
- Dividir por 42 para obter número de racks padrão
- Adicionar ~20% para switches, PDUs, ventilação
- Define densidade de implantação e viabilidade física

**`total_heat_btu_hr` (por cenário, opcional)**
- Dissipação térmica total
- Dividir por 12,000 para obter tons de refrigeração
- Dimensiona capacidade de HVAC e COP do datacenter

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
