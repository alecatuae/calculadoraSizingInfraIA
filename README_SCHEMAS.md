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

### Schema Completo: `servers.json`

| Campo | Tipo | Obrigatório? | Descrição | Unidade/Enum | Exemplo |
|-------|------|--------------|-----------|--------------|---------|
| `name` | str | ✅ Sim | Nome único do servidor | - | `"dgx-b300"` |
| `gpus` | int | ✅ Sim | Número de GPUs por nó | count | `8` |
| `hbm_per_gpu_gb` | float | ✅ Sim | Memória HBM por GPU | GB (decimal) | `192.0` |
| `rack_units_u` | int | ✅ Sim | Espaço ocupado em rack | U | `10` |
| `power_kw_max` | float | ✅ Sim | Consumo elétrico máximo | kW | `14.5` |
| `heat_output_btu_hr_max` | float\|null | ❌ Não | Dissipação térmica máxima | BTU/hr | `49500.0` |
| `notes` | str | ❌ Não | Notas e observações | - | `"NVIDIA DGX B300..."` |

**Constraints:**
- Todos os valores numéricos devem ser > 0

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

