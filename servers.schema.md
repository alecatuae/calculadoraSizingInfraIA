# Schema de `servers.json` (Estrutura Hierárquica)

## Visão Geral

O arquivo `servers.json` define as especificações de servidores GPU usados para inferência de LLMs. Utiliza uma **estrutura hierárquica (nested)** para organizar logicamente os componentes do servidor.

---

## Estrutura do Arquivo

```json
{
  "servers": [
    {
      "name": "...",
      "manufacturer": "...",
      "form_factor": "...",
      "rack_units_u": 10,
      
      "cpu": { ... },
      "system_memory": { ... },
      "gpu": { ... },
      "power": { ... },
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

---

## Schema Completo de Campos

### Campos de Nível Raiz (Root Level)

| Campo | Tipo | Obrigatório? | Unidade | Descrição | Usado no Cálculo? |
|-------|------|--------------|---------|-----------|-------------------|
| `name` | string | ✅ **Sim** | - | Nome único do servidor (usado para seleção via `--server`) | ✅ Sim (identificação) |
| `manufacturer` | string | ❌ Não | - | Fabricante do servidor | ❌ Não (metadado) |
| `form_factor` | string | ❌ Não | - | Tipo de forma física (ex.: "Rackmount") | ❌ Não (metadado) |
| `rack_units_u` | integer | ✅ **Sim** | U | Espaço ocupado em rack (altura) | ✅ **Sim** (cálculo de rack total) |
| `notes` | string | ❌ Não | - | Notas e observações sobre o servidor | ❌ Não (metadado) |
| `source` | array[string] | ⚠️ Recomendado | - | Links de documentação oficial (rastreabilidade) | ❌ Não (metadado) |

---

### Seção: `cpu` (Opcional)

| Campo | Tipo | Obrigatório? | Unidade | Descrição | Usado no Cálculo? |
|-------|------|--------------|---------|-----------|-------------------|
| `cpu.model` | string | ❌ Não | - | Modelo do processador | ❌ Não |
| `cpu.cores_total` | integer | ❌ Não | cores | Total de cores físicos | ❌ Não |
| `cpu.threads_total` | integer | ❌ Não | threads | Total de threads (com HT/SMT) | ❌ Não |
| `cpu.base_frequency_ghz` | float | ❌ Não | GHz | Frequência base do clock | ❌ Não |
| `cpu.max_boost_frequency_ghz` | float | ❌ Não | GHz | Frequência máxima (boost) | ❌ Não |

---

### Seção: `system_memory` (Opcional)

| Campo | Tipo | Obrigatório? | Unidade | Descrição | Usado no Cálculo? |
|-------|------|--------------|---------|-----------|-------------------|
| `system_memory.capacity_total_tb` | float | ❌ Não | TB | Capacidade total de RAM do sistema | ❌ Não |
| `system_memory.type` | string | ❌ Não | - | Tipo de memória (ex.: "DDR5") | ❌ Não |
| `system_memory.speed_mhz` | integer | ❌ Não | MHz | Velocidade da memória | ❌ Não |

---

### Seção: `gpu` (Obrigatória)

| Campo | Tipo | Obrigatório? | Unidade | Descrição | Usado no Cálculo? |
|-------|------|--------------|---------|-----------|-------------------|
| `gpu.count` | integer | ✅ **Sim** | count | Número de GPUs por nó | ✅ **Sim** (cálculo de HBM total e paralelismo) |
| `gpu.model` | string | ✅ **Sim** | - | Modelo da GPU | ✅ **Sim** (identificação no relatório) |
| `gpu.hbm_per_gpu_gb` | float | ✅ **Sim** | GB (decimal) | Memória HBM por GPU | ✅ **Sim** (cálculo crítico de capacidade) |
| `gpu.total_hbm_gb` | float | ⚠️ Opcional (validado) | GB (decimal) | HBM total (count × hbm_per_gpu_gb) | ⚠️ Validado (consistência automática) |
| `gpu.nvlink_bandwidth_tbps_total` | float | ❌ Não | TB/s | Largura de banda total NVLink | ❌ Não |
| `gpu.nvlink_generation` | string | ❌ Não | - | Geração do NVLink/NVSwitch | ❌ Não |

**Validação automática:**
- Se `gpu.total_hbm_gb` estiver presente, será validado contra `gpu.count × gpu.hbm_per_gpu_gb`
- Divergência > 1%: **warning** + correção automática para valor derivado

---

### Seção: `power` (Obrigatória)

| Campo | Tipo | Obrigatório? | Unidade | Descrição | Usado no Cálculo? |
|-------|------|--------------|---------|-----------|-------------------|
| `power.power_kw_max` | float | ✅ **Sim** | kW | Consumo elétrico máximo | ✅ **Sim** (cálculo de kW total do cluster) |
| `power.power_supplies.count` | integer | ❌ Não | count | Número de fontes | ❌ Não |
| `power.power_supplies.rating_each_watts` | integer | ❌ Não | W | Potência de cada fonte | ❌ Não |
| `power.power_supplies.redundancy` | string | ❌ Não | - | Tipo de redundância (ex.: "N+1") | ❌ Não |
| `power.input_voltage` | array[string] | ❌ Não | - | Voltagens de entrada suportadas | ❌ Não |
| `power.max_current.208v_3phase_amps` | integer | ❌ Não | A | Corrente máxima em 208V trifásico | ❌ Não |
| `power.max_current.480v_3phase_amps` | integer | ❌ Não | A | Corrente máxima em 480V trifásico | ❌ Não |

---

### Seção: `thermal` (Opcional)

| Campo | Tipo | Obrigatório? | Unidade | Descrição | Usado no Cálculo? |
|-------|------|--------------|---------|-----------|-------------------|
| `thermal.heat_output_btu_hr_max` | float | ⚠️ Opcional | BTU/hr | Dissipação térmica máxima | ⚠️ Usado se presente (cálculo de refrigeração) |
| `thermal.ambient_temp_operating_c_min` | integer | ❌ Não | °C | Temperatura ambiente mínima operacional | ❌ Não |
| `thermal.ambient_temp_operating_c_max` | integer | ❌ Não | °C | Temperatura ambiente máxima operacional | ❌ Não |

---

### Seções Adicionais (Todas Opcionais)

**`cooling`:** Especificações de refrigeração  
**`storage`:** Storage interno do servidor  
**`networking`:** Interfaces de rede  
**`software`:** Software e OS suportados  
**`physical`:** Dimensões e peso  

---

## Campos Obrigatórios para Cálculos

Para que um servidor seja utilizável no sizing, os seguintes campos são **OBRIGATÓRIOS**:

### Mínimo Absoluto

```json
{
  "name": "nome-unico",
  "rack_units_u": 10,
  "gpu": {
    "count": 8,
    "model": "GPU Model",
    "hbm_per_gpu_gb": 192.0
  },
  "power": {
    "power_kw_max": 14.5
  }
}
```

### Campos Usados nos Cálculos

| Componente | Campo | Uso no Cálculo |
|------------|-------|----------------|
| **GPU** | `gpu.count` | Paralelismo de tensor (TP), número de réplicas |
| **GPU** | `gpu.hbm_per_gpu_gb` | Budget total de HBM disponível |
| **GPU** | `gpu.total_hbm_gb` | Derivado automaticamente se ausente |
| **Power** | `power.power_kw_max` | Consumo energético total do cluster |
| **Rack** | `rack_units_u` | Espaço físico total no datacenter |
| **Thermal** | `thermal.heat_output_btu_hr_max` | Carga térmica (se disponível) |

---

## Regras de Validação (Constraints)

### 1. Valores Positivos

- `rack_units_u > 0`
- `gpu.count > 0`
- `gpu.hbm_per_gpu_gb > 0`
- `power.power_kw_max > 0`

### 2. Consistência de HBM Total

Se `gpu.total_hbm_gb` for informado:

```
Esperado = gpu.count × gpu.hbm_per_gpu_gb
Divergência = |total_hbm_gb - Esperado| / Esperado

Se Divergência > 1%:
  → Emitir WARNING
  → Usar valor derivado como "correto"
```

### 3. Seções Obrigatórias

- `gpu` (objeto nested) deve existir
- `power` (objeto nested) deve existir

### 4. Nomes Únicos

- Não pode haver servidores duplicados (case-insensitive)
- A validação falhará se houver conflito de nomes

---

## Exemplo: Servidor Mínimo Válido

```json
{
  "servers": [
    {
      "name": "servidor-minimal",
      "rack_units_u": 8,
      "gpu": {
        "count": 4,
        "model": "GPU Generic",
        "hbm_per_gpu_gb": 80.0
      },
      "power": {
        "power_kw_max": 8.0
      },
      "notes": "Servidor mínimo para validação"
    }
  ]
}
```

---

## Exemplo: Servidor Completo (DGX B300)

```json
{
  "servers": [
    {
      "name": "dgx-b300",
      "manufacturer": "NVIDIA",
      "form_factor": "Rackmount",
      "rack_units_u": 10,
      
      "cpu": {
        "model": "Dual AMD EPYC 9754",
        "cores_total": 256,
        "threads_total": 512,
        "base_frequency_ghz": 2.25,
        "max_boost_frequency_ghz": 3.1
      },
      
      "system_memory": {
        "capacity_total_tb": 2.0,
        "type": "DDR5",
        "speed_mhz": 4800
      },
      
      "gpu": {
        "count": 8,
        "model": "NVIDIA B300 Blackwell Ultra",
        "hbm_per_gpu_gb": 288,
        "total_hbm_gb": 2304,
        "nvlink_bandwidth_tbps_total": 14.4,
        "nvlink_generation": "5th Generation NVSwitch"
      },
      
      "power": {
        "power_kw_max": 14.5,
        "power_supplies": {
          "count": 6,
          "rating_each_watts": 3000,
          "redundancy": "N+1"
        },
        "input_voltage": ["200-240VAC", "380-480VAC"],
        "max_current": {
          "208v_3phase_amps": 52,
          "480v_3phase_amps": 23
        }
      },
      
      "thermal": {
        "heat_output_btu_hr_max": 49476.054,
        "ambient_temp_operating_c_min": 5,
        "ambient_temp_operating_c_max": 30
      },
      
      "notes": "DGX B300: 8x Blackwell Ultra GPUs",
      "source": [
        "https://docs.nvidia.com/dgx/dgxb300-user-guide/"
      ]
    }
  ]
}
```

---

## Como Adicionar Novo Servidor

### Passo 1: Copiar Servidor Existente

Edite `servers.json` e copie um servidor existente (ex.: `dgx-b300`).

### Passo 2: Definir Nome Único

```json
{
  "name": "dgx-h200",
  ...
}
```

### Passo 3: Preencher Campos Obrigatórios

Mínimo necessário:
- `name`
- `rack_units_u`
- `gpu.count`
- `gpu.model`
- `gpu.hbm_per_gpu_gb`
- `power.power_kw_max`

### Passo 4: Validar

```bash
python3 main.py --validate-only
```

**Saída esperada:**
```
✅ Todos os arquivos de configuração são válidos.
```

### Passo 5: Testar Sizing

```bash
python3 main.py \
  --model opt-oss-120b \
  --server dgx-h200 \
  --storage profile_default \
  --concurrency 1000 \
  --effective-context 131072
```

---

## Checklist de Adição

- [ ] Nome é único (não duplica servidor existente)
- [ ] Campos obrigatórios preenchidos (`gpu`, `power`, `rack_units_u`)
- [ ] `gpu.count > 0`
- [ ] `gpu.hbm_per_gpu_gb > 0`
- [ ] `power.power_kw_max > 0`
- [ ] `rack_units_u > 0`
- [ ] Se `gpu.total_hbm_gb` presente: validar consistência (count × hbm_per_gpu)
- [ ] Rodar `python3 main.py --validate-only` → ✅ OK
- [ ] Testar um sizing simples

---

## Unidades e Convenções

| Campo | Unidade | Base | Observação |
|-------|---------|------|------------|
| `hbm_per_gpu_gb` | GB | Decimal (10^9) | Padrão do fabricante |
| `total_hbm_gb` | GB | Decimal (10^9) | Derivado ou validado |
| `power_kw_max` | kW | - | Consumo máximo especificado |
| `rack_units_u` | U | - | Altura em rack (1U = 1.75 polegadas) |
| `heat_output_btu_hr_max` | BTU/hr | - | Dissipação térmica máxima |
| `capacity_total_tb` | TB | Decimal (10^12) | Memória do sistema |

---

## Rastreabilidade e Governança

### Campo `source` (Recomendado)

Sempre incluir links para documentação oficial:

```json
"source": [
  "https://docs.nvidia.com/dgx/dgxb300-user-guide/introduction-to-dgxb300.html",
  "https://www.nvidia.com/en-us/data-center/dgx-b300/"
]
```

**Benefícios:**
- Rastreabilidade de valores
- Auditoria facilitada
- Verificação independente de specs

---

## Evolução do Schema

Este schema suporta evolução incremental:

✅ **Novos campos opcionais** podem ser adicionados sem quebrar compatibilidade  
✅ **Novas seções** (ex.: `accelerators`, `interconnect`) podem ser incluídas  
✅ **Servidor mínimo** continua funcionando mesmo com schema expandido  

Campos obrigatórios (`name`, `rack_units_u`, `gpu`, `power`) **não devem ser removidos**.

---

**Versão do Schema:** 2.0 (Estrutura Hierárquica)  
**Data:** 2026-02-12  
**Compatibilidade:** Python 3.10+, stdlib only

