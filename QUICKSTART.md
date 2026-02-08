# Quick Start - Sizing de Inferência LLM

Guia rápido para executar o dimensionamento de infraestrutura para inferência de LLMs.

---

## Pré-requisitos

- **Python 3.8+**
- **Nenhuma dependência externa** (usa apenas stdlib)

---

## Execução Básica

```bash
python3 sizing.py \
  --model opt-oss-120b \
  --server dgx300 \
  --storage profile_default \
  --concurrency 1000 \
  --effective-context 131072
```

**O que faz:** Calcula quantos nós DGX B300 são necessários para sustentar 1000 sessões simultâneas com contexto de 128k tokens, usando o modelo opt-oss-120b.

**Output no terminal:** Resumo executivo com tabela comparativa dos 3 cenários (MÍNIMO, RECOMENDADO, IDEAL).

**Relatórios completos:** Salvos automaticamente em `relatorios/` com timestamp:
- `sizing_<modelo>_<servidor>_<timestamp>.txt` (texto completo)
- `sizing_<modelo>_<servidor>_<timestamp>.json` (JSON estruturado)

---

## Interface de Saída

### No Terminal (Resumo Executivo)

```
================================================================================
RESUMO EXECUTIVO - SIZING DE INFERÊNCIA LLM
================================================================================

Modelo:              opt-oss-120b
Servidor:            dgx-b300
Contexto Efetivo:    131,072 tokens
Concorrência Alvo:   1,000 sessões simultâneas
Precisão KV Cache:   FP8

--------------------------------------------------------------------------------
Cenário          Nós DGX  Energia (kW)  Rack (U)  Sessões/Nó  KV/Sessão (GiB)
--------------------------------------------------------------------------------
MÍNIMO                 2          29.0        20         629             2.25
RECOMENDADO            3          43.5        30         629             2.25
IDEAL                  5          72.5        50         584             2.25
--------------------------------------------------------------------------------

✓ Cenário RECOMENDADO (3 nós, 43.5 kW, 30U) atende os requisitos com 
  tolerância a falhas (N+1).

================================================================================
📄 Relatórios completos salvos em:
   • Texto:  relatorios/sizing_opt-oss-120b_dgx-b300_<timestamp>.txt
   • JSON:   relatorios/sizing_opt-oss-120b_dgx-b300_<timestamp>.json
   • Executivo: relatorios/executive_opt-oss-120b_dgx-b300_<timestamp>.md
                (se usar --executive-report)
```

**O que mudou:**
- Agora exibe **Energia (kW)** e **Rack (U)** por cenário
- Essencial para decisões de datacenter (capacidade elétrica, densidade)
📄 Relatórios completos salvos em:
   • Texto:  relatorios/sizing_opt-oss-120b_dgx300_20260208_134031.txt
   • JSON:   relatorios/sizing_opt-oss-120b_dgx300_20260208_134031.json
```

### Nos Arquivos (Relatórios Completos)

Os arquivos em `relatorios/` contêm:
- ✅ Todas as entradas (modelo, servidor, storage, NFRs)
- ✅ Dicionário completo de parâmetros
- ✅ Resultados detalhados por cenário
- ✅ Racional de cálculo (fórmulas, inputs, explicações)
- ✅ Análise comparativa
- ✅ Alertas e riscos operacionais

**Para auditoria, revisão técnica ou apresentação executiva.**

---

## Exemplos Comuns

### 1. Modelo Grande (120B) + DGX B300

**Objetivo:** Dimensionar produção com alta concorrência e contexto longo.

```bash
python3 sizing.py \
  --model opt-oss-120b \
  --server dgx300 \
  --storage profile_default \
  --concurrency 1000 \
  --effective-context 131072 \
  --kv-precision fp8
```

**Resumo no terminal:**
- MÍNIMO: 2 nós (sem HA)
- RECOMENDADO: 3 nós (N+1, 20% headroom) ✓
- IDEAL: 5 nós (N+2, 30% headroom)

**Relatórios salvos em:** `relatorios/sizing_opt-oss-120b_dgx300_<timestamp>.{txt,json}`

---

### 2. Modelo Médio (20B) + DGX H200

**Objetivo:** Ambiente de menor escala ou staging.

```bash
python3 sizing.py \
  --model opt-oss-20b \
  --server dgx200 \
  --storage profile_default \
  --concurrency 500 \
  --effective-context 131072 \
  --kv-precision fp8
```

**Resumo no terminal:**
- MÍNIMO: 2 nós
- RECOMENDADO: 3 nós (N+1) ✓
- IDEAL: 4 nós (N+2)

---

### 3. Comparação FP8 vs FP16

**Objetivo:** Analisar impacto de precisão na memória e número de nós.

**FP8 (recomendado, 1 byte/elemento):**
```bash
python3 sizing.py \
  --model opt-oss-120b \
  --server dgx300 \
  --storage profile_default \
  --concurrency 1000 \
  --effective-context 131072 \
  --kv-precision fp8
```

**FP16 (dobro de memória, 2 bytes/elemento):**
```bash
python3 sizing.py \
  --model opt-oss-120b \
  --server dgx300 \
  --storage profile_default \
  --concurrency 1000 \
  --effective-context 131072 \
  --kv-precision fp16
```

**Interpretação:**
- FP16 dobra o KV por sessão
- Reduz sessões por nó em ~50%
- Aumenta número de nós necessários (ex: 3 → 5)
- Compare os relatórios salvos em `relatorios/` para análise detalhada

---

### 4. Gerar Relatório Executivo Adicional

**Objetivo:** Criar relatório formatado para apresentação a CFO/CTO/Diretoria (além dos relatórios padrão).

```bash
python3 sizing.py \
  --model opt-oss-120b \
  --server dgx-b300 \
  --storage profile_default \
  --concurrency 1000 \
  --effective-context 131072 \
  --executive-report
```

**Arquivos gerados:**
- `relatorios/sizing_<modelo>_<servidor>_<timestamp>.txt` (padrão)
- `relatorios/sizing_<modelo>_<servidor>_<timestamp>.json` (padrão)
- `relatorios/executive_<modelo>_<servidor>_<timestamp>.md` (executivo) ← Adicional

**O relatório executivo inclui:**
- Sumário executivo com impacto em servidores, energia e datacenter
- Consumo unitário por sessão (KV cache, % HBM, energia estimada)
- Consumo agregado total (KV, energia kW + MWh/ano, rack U, dissipação BTU/hr)
- Resultados detalhados por cenário com métricas de datacenter
- Comparação executiva (incluindo CapEx relativo, energia relativa)
- Recomendação baseada em estabilidade, energia, densidade e risco

---

## Interpretação Rápida

### Como Ler o Resumo no Terminal

**1. Tabela de Cenários**
- `Nós DGX`: Número de servidores necessários para cada cenário
- `Energia (kW)`: Consumo elétrico total contínuo → dimensiona PDU/UPS/contrato
- `Rack (U)`: Espaço físico necessário → densidade de datacenter (42U/rack padrão)
- `Sessões/Nó`: Capacidade de cada servidor
- `KV/Sessão (GiB)`: Memória consumida por cada sessão ativa
- `Sessões/Nó`: Capacidade efetiva de cada servidor
- `KV/Sessão (GiB)`: Memória GPU necessária por sessão ativa
- `Observação`: Classificação de risco/resiliência

**2. Status Final**
- ✓ Verde: Dimensionamento adequado
- ⚠️  Amarelo: Atenção necessária (revisar NFRs ou configuração)

**3. Localização dos Relatórios**
- Sempre em `relatorios/` com timestamp
- Arquivos nunca são sobrescritos

### Onde Olhar nos Relatórios Completos

**Para análise técnica detalhada:**
1. Abra o arquivo `.txt` em `relatorios/`
2. Leia a **Seção 3: Resultados por Cenário**
3. Consulte o **Racional de Cálculo** para entender as fórmulas

**Para integração programática:**
1. Abra o arquivo `.json` em `relatorios/`
2. Use `scenarios.recommended.results.nodes_final` para número de nós
3. Use `scenarios.recommended.results.sessions_per_node` para capacidade
4. Consulte `alerts` para avisos automatizados

---

### Sinais de Subdimensionamento

| Sinal | Significado | Ação |
|-------|-------------|------|
| `sessions_per_node = 0` | **Erro crítico:** Não cabe nem 1 sessão | Reduzir contexto, usar fp8, ou servidor maior |
| `nodes_final` muito alto (>20) | Carga excessiva ou configuração ineficiente | Revisar NFRs ou considerar modelo menor |
| Diferença pequena entre cenários (<10%) | Carga leve, sobre-provisionado | Considerar otimizações ou reduzir recursos |

---

### Sinais de Risco Operacional

| Campo | Valor de Alerta | Impacto |
|-------|----------------|---------|
| `kv_precision` | `fp16` ou `bf16` | Dobra consumo de memória, duplica custo |
| `kv_budget_ratio` | `> 0.75` | Risco de fragmentação e instabilidade |
| `runtime_overhead_gib` | `< 50` | Overhead subestimado, pode causar OOM |
| `peak_headroom_ratio` | `0%` (cenário MÍNIMO) | Sem tolerância a picos, degradação garantida |

---

## Parâmetros Principais (CLI)

| Parâmetro | Descrição | Default | Exemplo |
|-----------|-----------|---------|---------|
| `--model` | Nome do modelo (models.json) | - | `opt-oss-120b` |
| `--server` | Nome do servidor (servers.json) | - | `dgx300` |
| `--storage` | Perfil de storage (storage.json) | - | `profile_default` |
| `--concurrency` | Sessões simultâneas | - | `1000` |
| `--effective-context` | Tamanho do contexto (tokens) | - | `131072` |
| `--kv-precision` | Precisão KV cache | `fp8` | `fp8`, `fp16`, `bf16`, `int8` |
| `--kv-budget-ratio` | % HBM para KV | `0.70` | `0.65` (conservador), `0.75` (agressivo) |
| `--runtime-overhead-gib` | Overhead (GiB) | `120` | `80` (modelo pequeno), `150` (grande) |
| `--peak-headroom-ratio` | Headroom para picos | `0.20` | `0.10` (baixo), `0.30` (alto) |
| `--executive-report` | Gerar relatório executivo | - | (flag booleana) |
| `--output-json-file` | Salvar JSON em arquivo | - | `results.json` |
| `--output-markdown-file` | Salvar Markdown | - | `report.md` |

---

## Ajuda Completa

```bash
python3 sizing.py --help
```

---

## Próximos Passos

1. **Entender conceitos:** Ler `README.md` completo
2. **Validar resultados:** Comparar com benchmarks reais
3. **Customizar:** Adicionar seus modelos/servidores aos JSONs
4. **Integrar:** Usar JSON em pipelines de IaC

---

**Versão:** 2.0  
**Documentação completa:** `README.md`
