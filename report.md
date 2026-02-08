# Relatório de Dimensionamento de Inferência LLM

**Sistema de Sizing com Racional de Cálculo e Análise de Cenários**

**Data:** 2026-02-08 01:02:55

---

## 📋 Seção 1: Entradas

### Modelo

- **Nome:** opt-oss-120b
- **Camadas:** 36
- **KV Heads:** 8
- **Head Dim:** 64
- **Max Position Embeddings:** 131,072
- **Padrão de Atenção:** hybrid
  - Full Layers: 18
  - Sliding Layers: 18
  - Sliding Window: 128
- **Precisão KV Padrão:** fp8

### Servidor

- **Nome:** dgx300
- **GPUs:** 8
- **HBM por GPU:** 288 GB
- **HBM Total:** 2304 GB (2145.8 GiB)
- **NVLink Bandwidth:** 14.4 TB/s

### Storage

- **Perfil:** profile_default
- **Tipo:** nvme_local
- **IOPS:** 1,000,000 read / 800,000 write
- **Throughput:** 28 GB/s read / 25 GB/s write
- **Latência P99:** 0.15 ms read / 0.2 ms write

### NFR (Non-Functional Requirements)

- **Concorrência Alvo:** 1,000 sessões simultâneas
- **Contexto Efetivo:** 131,072 tokens
- **Precisão KV:** fp16

---

## 📚 Seção 2: Dicionário de Parâmetros

Principais parâmetros utilizados no dimensionamento:

### `num_layers`

**O que é:** Número total de camadas (layers) do transformer no modelo LLM. Cada camada possui seu próprio conjunto de tensores Key e Value no KV cache.

**Importância:** Impacta linearmente o tamanho do KV cache. Modelos com mais camadas (ex: 36 vs 24) consomem proporcionalmente mais memória GPU para armazenar o histórico de atenção.

**Erro comum:** Erro comum: Confundir num_layers com num_hidden_layers ou contar apenas encoder/decoder. Deve ser o total de camadas que mantêm KV cache.

### `num_key_value_heads`

**O que é:** Número de cabeças (heads) de atenção para Key e Value. Em GQA (Grouped Query Attention), este valor pode ser menor que o número de query heads.

**Importância:** Impacta diretamente o tamanho do KV cache. Menos KV heads = menos memória. GQA com 8 KV heads vs 32 representa redução de 4x na memória de KV.

**Erro comum:** Erro comum: Usar num_attention_heads (query heads) em vez de num_key_value_heads. Em GQA esses valores são diferentes e isso causa superestimação de 4-8x na memória.

### `effective_context`

**O que é:** Tamanho de contexto (em tokens) que sua aplicação efetivamente usará em runtime. Diferente de max_position_embeddings (limite do modelo).

**Importância:** Impacta diretamente o tamanho do KV cache por sessão. Contexto maior = mais memória = menos sessões por nó. Definir incorretamente causa over/under-provisioning.

**Erro comum:** Erro comum: Usar max_position_embeddings como effective_context. Isso superestima memória se aplicação usa contextos menores, ou causa problemas se excede o limite do modelo.

### `kv_precision`

**O que é:** Precisão numérica usada para armazenar tensores Key e Value: fp8/int8 (1 byte/elemento) ou fp16/bf16 (2 bytes/elemento).

**Importância:** Impacta diretamente (2x) o tamanho do KV cache. fp16 vs fp8 dobra a memória necessária e reduz pela metade o número de sessões por nó.

**Erro comum:** Erro comum: Usar fp16 por default sem testar fp8. Muitos casos fp8 tem qualidade equivalente, mas fp16 dobra o custo de infraestrutura desnecessariamente.

### `kv_budget_ratio`

**O que é:** Fração da HBM total alocada para KV cache (ex: 0.70 = 70%). O restante é para modelo, ativações, overhead de runtime.

**Importância:** Define quantas sessões cabem por nó. Budget muito alto (>0.80) causa fragmentação e instabilidade. Budget muito baixo (<0.50) desperdiça HBM.

**Erro comum:** Erro comum: Alocar 100% da HBM para KV cache, ignorando overhead do modelo, ativações, e buffers do runtime. Isso causa OOM (Out of Memory) em produção.

### `ha_mode`

**O que é:** Modo de alta disponibilidade: 'none' (sem redundância), 'n+1' (tolera falha de 1 nó), 'n+2' (tolera 2 nós).

**Importância:** Define quantos nós extras alocar para redundância. N+1 garante que falha de 1 nó não quebra SLA. Sem HA, falha de nó causa degradação imediata.

**Erro comum:** Erro comum: Não ter HA (none) em produção com SLA > 99%. Falha de hardware é inevitável. Outro erro: N+2 quando N+1 já atende, desperdiçando capex.

> ℹ️ Veja JSON para dicionário completo de todos os parâmetros

---

## 🎯 Seção 3: Resultados por Cenário

### Comparação Rápida

| Métrica | MÍNIMO | RECOMENDADO | IDEAL |
|---------|--------|-------------|-------|
| **Headroom** | 0% | 20% | 30% |
| **HA** | none | n+1 | n+2 |
| **Budget KV** | 70% | 70% | 65% |
| **KV/Sessão** | 4.50 GiB | 4.50 GiB | 4.50 GiB |
| **Sessões/Nó** | 314 | 314 | 292 |
| **Nós Finais** | **4** | **5** ✅ | **7** |

> ✅ **RECOMENDADO** é o cenário ideal para produção

### 🔴 Cenário: MÍNIMO

**Configuração:**

- Peak Headroom: 0%
- HA Mode: none
- KV Budget Ratio: 70%

**Resultados:**

- KV por Sessão: 4.50 GiB
- KV Total: 4.40 TiB
- HBM Total: 2145.8 GiB
- KV Budget: 1418.0 GiB
- Sessões por Nó: 314
- Nós (Capacidade): 4
- Nós (com Headroom): 4
- **Nós Finais**: **4**

<details>
<summary><b>📊 Racional: Nós Finais</b></summary>

**Fórmula:**

```
nodes_final = nodes_with_headroom + ha_extra_nodes
```

**Interpretação:**

Adicionando 0 nó(s) para alta disponibilidade, total final é 4 nós. Sem HA: qualquer falha de nó causa degradação imediata.

</details>

**⚠️ Avisos:**

1. AVISO: kv_precision=fp16 usa 2 bytes/elemento. Considere fp8 (1 byte) para reduzir memória pela metade com mínima perda de qualidade.
2. ALERTA: Contexto longo (131,072 tokens) aumenta TTFT (Time To First Token) e pressiona I/O de storage durante prefill. Storage: profile_default (28 GB/s read, P99=0.15 ms).

---

### 🟢 Cenário: RECOMENDADO

**Configuração:**

- Peak Headroom: 20%
- HA Mode: n+1
- KV Budget Ratio: 70%

**Resultados:**

- KV por Sessão: 4.50 GiB
- KV Total: 4.40 TiB
- HBM Total: 2145.8 GiB
- KV Budget: 1418.0 GiB
- Sessões por Nó: 314
- Nós (Capacidade): 4
- Nós (com Headroom): 4
- **Nós Finais**: **5**

<details>
<summary><b>📊 Racional: Nós Finais</b></summary>

**Fórmula:**

```
nodes_final = nodes_with_headroom + ha_extra_nodes
```

**Interpretação:**

Adicionando 1 nó(s) para alta disponibilidade, total final é 5 nós. Com N+1: sistema tolera falha de 1 nó mantendo SLO.

</details>

**⚠️ Avisos:**

1. AVISO: kv_precision=fp16 usa 2 bytes/elemento. Considere fp8 (1 byte) para reduzir memória pela metade com mínima perda de qualidade.
2. ALERTA: Contexto longo (131,072 tokens) aumenta TTFT (Time To First Token) e pressiona I/O de storage durante prefill. Storage: profile_default (28 GB/s read, P99=0.15 ms).

---

### 🔵 Cenário: IDEAL

**Configuração:**

- Peak Headroom: 30%
- HA Mode: n+2
- KV Budget Ratio: 65%

**Resultados:**

- KV por Sessão: 4.50 GiB
- KV Total: 4.40 TiB
- HBM Total: 2145.8 GiB
- KV Budget: 1316.7 GiB
- Sessões por Nó: 292
- Nós (Capacidade): 4
- Nós (com Headroom): 5
- **Nós Finais**: **7**

<details>
<summary><b>📊 Racional: Nós Finais</b></summary>

**Fórmula:**

```
nodes_final = nodes_with_headroom + ha_extra_nodes
```

**Interpretação:**

Adicionando 2 nó(s) para alta disponibilidade, total final é 7 nós. Com N+2: sistema tolera falha de 2 nós mantendo SLO.

</details>

**⚠️ Avisos:**

1. AVISO: kv_precision=fp16 usa 2 bytes/elemento. Considere fp8 (1 byte) para reduzir memória pela metade com mínima perda de qualidade.
2. ALERTA: Contexto longo (131,072 tokens) aumenta TTFT (Time To First Token) e pressiona I/O de storage durante prefill. Storage: profile_default (28 GB/s read, P99=0.15 ms).

---

## ⚠️ Seção 4: Alertas e Riscos

1. ALERTA: Contexto longo (131,072 tokens) aumenta TTFT (Time To First Token) e pressiona I/O de storage durante prefill. Storage: profile_default (28 GB/s read, P99=0.15 ms).
2. AVISO: kv_precision=fp16 usa 2 bytes/elemento. Considere fp8 (1 byte) para reduzir memória pela metade com mínima perda de qualidade.

---

## 📝 Observações

- Este relatório foi gerado automaticamente pelo sistema de sizing v2.0
- Para análise completa, consulte também o JSON output
- Use o **CENÁRIO RECOMENDADO** para produção (N+1, balanceado)

---

*Gerado por: Sistema de Sizing de Infraestrutura IA*