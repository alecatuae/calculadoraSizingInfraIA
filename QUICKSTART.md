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

**Output:** 3 cenários (MÍNIMO: 2 nós | RECOMENDADO: 3 nós | IDEAL: 5 nós) + relatório detalhado + JSON.

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

**Resultado esperado:**
- MÍNIMO: 2 nós (sem HA)
- RECOMENDADO: 3 nós (N+1, 20% headroom)
- IDEAL: 5 nós (N+2, 30% headroom)

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

**Resultado esperado:**
- MÍNIMO: 1 nó
- RECOMENDADO: 2 nós (N+1)
- IDEAL: 3 nós (N+2)

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

---

### 4. Gerar Relatório Executivo (para Diretoria)

**Objetivo:** Criar relatório formatado para apresentação a CFO/CTO.

```bash
python3 sizing.py \
  --model opt-oss-120b \
  --server dgx300 \
  --storage profile_default \
  --concurrency 1000 \
  --effective-context 131072 \
  --executive-report \
  --output-markdown-file relatorio_diretoria.md
```

**Output:** Arquivo Markdown com linguagem estratégica, tabelas comparativas, análise de CapEx e recomendação clara.

---

### 5. Salvar JSON para Análise Programática

**Objetivo:** Exportar dados para integração com pipelines de IaC ou dashboards.

```bash
python3 sizing.py \
  --model opt-oss-120b \
  --server dgx300 \
  --storage profile_default \
  --concurrency 1000 \
  --effective-context 131072 \
  --output-json-file sizing_results.json
```

**Uso do JSON:**
- Integração com Terraform/Ansible
- Dashboards de capacity planning
- Análise em planilhas (FinOps)

---

## Interpretação Rápida

### Onde Olhar Primeiro no Output

**1. Seção "CENÁRIO: RECOMENDADO"**
- `Nodes Final`: Número de nós DGX a provisionar para produção
- `Sessions Per Node`: Capacidade efetiva por nó
- `Kv Per Session Gib`: Memória necessária por sessão ativa

**2. Seção "ALERTAS E RISCOS"**
- Avisos críticos (ex: contexto excede limite, precisão ineficiente)
- Recomendações operacionais

**3. JSON Final**
- `scenarios.recommended.results.nodes_final`: Nós necessários
- `scenarios.recommended.results.sessions_per_node`: Capacidade por nó
- `alerts`: Lista de avisos automatizados

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
