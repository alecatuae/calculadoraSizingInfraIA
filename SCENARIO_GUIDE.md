# 📊 GUIA DE DECISÃO: Qual Cenário Escolher?

## Comparação dos 3 Cenários de Dimensionamento

Este guia ajuda você a escolher entre MÍNIMO, RECOMENDADO e IDEAL baseado em seu contexto operacional.

---

## 🎯 Visão Rápida

| Aspecto | MÍNIMO | RECOMENDADO | IDEAL |
|---------|---------|-------------|-------|
| **Objetivo** | Custo mínimo | Produção balanceada | Máxima resiliência |
| **Headroom** | 0% | 20% (configurável) | ≥30% |
| **HA** | Nenhum | N+1 | N+2 |
| **Budget KV** | 70% (configurável) | 70% (configurável) | ≤65% (conservador) |
| **SLA Típico** | < 99% | 99.9% | 99.99%+ |
| **Use Case** | PoC, Dev, Teste | **PRODUÇÃO** | Missão crítica |
| **Risco** | ⚠️ Alto | ✅ Balanceado | 🛡️ Mínimo |

---

## 📋 Detalhamento por Cenário

### 🔴 CENÁRIO MÍNIMO

#### Configuração
```
peak_headroom_ratio = 0.0      # Sem folga para picos
ha_mode = "none"               # Sem redundância
kv_budget_ratio = configurado  # Default 70%
```

#### Quando Usar
- ✅ **PoC (Proof of Concept):** Validar viabilidade técnica
- ✅ **Ambiente de Desenvolvimento:** Infra compartilhada, não crítica
- ✅ **Testes de Performance:** Baseline de capacidade
- ✅ **Estimativa de Custo:** "Quanto custa no mínimo?"

#### Quando NÃO Usar
- ❌ **Produção com usuários reais**
- ❌ **SLA > 95%**
- ❌ **Tráfego com variação**
- ❌ **Dados críticos de negócio**

#### Riscos
| Risco | Probabilidade | Impacto | Mitigação |
|-------|---------------|---------|-----------|
| **Falha de 1 nó = Downtime** | Alta (hardware falha) | Crítico | Nenhuma mitigação possível |
| **Pico de tráfego = Degradação** | Média | Alto | Rate limiting agressivo |
| **OOM por fragmentação** | Média (budget alto) | Alto | Monitorar HBM continuamente |

#### Exemplo Real

**Cenário:** Startup validando modelo de negócio

```bash
python3 sizing.py \
  --model opt-oss-20b \
  --server dgx200 \
  --storage profile_default \
  --concurrency 100 \
  --effective-context 8192 \
  --kv-precision fp8
```

**Resultado MÍNIMO:**
- **Nós:** 1
- **Capacidade:** 100 sessões exatas
- **Custo:** ~$300k (1 DGX H200)
- **Risco:** Se nó falhar, serviço para completamente

**Decisão:** OK para fase alpha com < 50 usuários teste. Migrar para RECOMENDADO antes de beta público.

---

### 🟢 CENÁRIO RECOMENDADO (✅ Produção)

#### Configuração
```
peak_headroom_ratio = configurado  # Default 20%
ha_mode = "n+1"                    # Tolera 1 falha
kv_budget_ratio = configurado      # Default 70%
```

#### Quando Usar
- ✅ **Produção com SLA 99.9%** (8.76h downtime/ano aceitável)
- ✅ **Tráfego com variação moderada** (picos até 50% acima da média)
- ✅ **Negócio em crescimento** (scale-up planejado)
- ✅ **TCO balanceado** (custo vs disponibilidade)

#### Vantagens
- 🛡️ **Tolera falha de 1 nó:** Manutenção ou hardware failure não causa downtime
- 📈 **Headroom para picos:** Aguenta Black Friday, marketing campaigns, viral spikes
- 💰 **TCO otimizado:** +33-50% vs MÍNIMO, mas com resiliência real
- ⚙️ **Deploy confiável:** Rolling updates sem degradação

#### Quando NÃO Usar
- ❌ SLA > 99.95% (< 4.38h downtime/ano)
- ❌ Tráfego extremamente variável (picos > 100%)
- ❌ Zero tolerância a degradação temporária

#### Exemplo Real

**Cenário:** SaaS B2B com 1k usuários simultâneos

```bash
python3 sizing.py \
  --model opt-oss-120b \
  --server dgx300 \
  --storage profile_default \
  --concurrency 1000 \
  --effective-context 131072 \
  --kv-precision fp8 \
  --peak-headroom-ratio 0.20
```

**Resultado RECOMENDADO:**
- **Nós:** 3 (2 capacidade + 1 HA)
- **Capacidade:** 1,200 sessões (20% headroom)
- **Tolerância:** Falha de 1 nó → 2 nós restantes suportam 1,226 sessões
- **Custo:** ~$900k (3 DGX B300)
- **SLA:** 99.9% atingível

**Análise:**
- Durante operação normal: 1,000 sessões / 3 nós = 333 sessões/nó (54% utilização)
- Durante falha de 1 nó: 1,000 sessões / 2 nós = 500 sessões/nó (81% utilização)
- Durante pico com 1 nó falhado: 1,200 sessões / 2 nós = 600 sessões/nó (98% utilização → ainda OK)

**Decisão:** ✅ **Ideal para produção**. TCO aceitável, resiliência adequada.

---

### 🔵 CENÁRIO IDEAL (Enterprise Grade)

#### Configuração
```
peak_headroom_ratio = max(configurado, 0.30)  # Mínimo 30%
ha_mode = "n+2"                               # Tolera 2 falhas
kv_budget_ratio = min(configurado, 0.65)      # Conservador (≤65%)
```

#### Quando Usar
- ✅ **SLA > 99.95%** (< 4.38h downtime/ano)
- ✅ **Missão crítica:** Financeiro, healthcare, infraestrutura
- ✅ **Tráfego imprevisível:** Picos > 100%, eventos não planejados
- ✅ **Manutenção frequente:** Rolling updates sem impacto zero
- ✅ **Compliance rigoroso:** Auditoria exige redundância dupla

#### Vantagens
- 🛡️🛡️ **Tolera 2 falhas simultâneas:** Raro, mas possível (rack failure, network partition)
- 📈📈 **Headroom generoso:** Picos extremos sem degradação
- 🧠 **Budget conservador:** Menos fragmentação de memória, mais estável
- 🔧 **Operação sem stress:** Manutenção planejada sem preocupação

#### Quando NÃO Usar
- ❌ **Budget limitado:** +40-60% custo vs RECOMENDADO
- ❌ **Tráfego estável e previsível**
- ❌ **SLA 99.9% é suficiente**
- ❌ **Over-engineering desnecessário**

#### Exemplo Real

**Cenário:** Plataforma financeira com compliance rigoroso

```bash
python3 sizing.py \
  --model opt-oss-120b \
  --server dgx300 \
  --storage profile_default \
  --concurrency 1000 \
  --effective-context 131072 \
  --kv-precision fp8 \
  --peak-headroom-ratio 0.40 \
  --kv-budget-ratio 0.60
```

**Resultado IDEAL:**
- **Nós:** 5 (2 capacidade + 2 HA + 1 headroom extra)
- **Capacidade:** 1,400 sessões (40% headroom efetivo)
- **Tolerância:** Falha de 2 nós → 3 nós restantes suportam 1,050 sessões (ainda aguenta picos menores)
- **Budget:** 60% (vs 70% default) → mais estável, menos fragmentação
- **Custo:** ~$1.5M (5 DGX B300)
- **SLA:** 99.99%+ atingível

**Análise:**
- Durante operação normal: 1,000 sessões / 5 nós = 200 sessões/nó (36% utilização)
- Durante falha de 1 nó: 1,000 / 4 nós = 250/nó (45% utilização)
- Durante falha de 2 nós: 1,000 / 3 nós = 333/nó (60% utilização)
- Durante pico com 2 nós falhados: 1,400 / 3 nós = 467/nó (84% utilização → confortável)

**Decisão:** ✅ **Justificado para missão crítica**. TCO alto, mas resiliência máxima.

---

## 🤔 Árvore de Decisão

```
Qual seu SLA alvo?
├─ < 99% → MÍNIMO (se não for produção)
├─ 99% - 99.9% → RECOMENDADO ✅
└─ > 99.9% → IDEAL

Qual seu budget?
├─ Limitado → MÍNIMO (risco) ou RECOMENDADO (balanceado)
├─ Moderado → RECOMENDADO ✅
└─ Flexível → IDEAL

Qual variação de tráfego?
├─ Estável (±10%) → RECOMENDADO com headroom 10%
├─ Moderado (±30%) → RECOMENDADO com headroom 20-30%
└─ Imprevisível (±100%+) → IDEAL com headroom 40%+

Qual criticidade?
├─ Não crítico (dev/test) → MÍNIMO
├─ Produção normal → RECOMENDADO ✅
└─ Missão crítica → IDEAL

Qual fase do produto?
├─ PoC / Alpha → MÍNIMO
├─ Beta / GA → RECOMENDADO ✅
└─ Enterprise / Compliance → IDEAL
```

---

## 💰 Análise de TCO (Total Cost of Ownership)

### Exemplo Comparativo: opt-oss-120b + dgx300 + 1k concurrent + 131k context + fp8

| Cenário | Nós | Custo HW | TCO 3 anos* | Downtime/ano | Custo/Sessão |
|---------|-----|----------|-------------|--------------|--------------|
| **MÍNIMO** | 2 | $600k | $1.2M | ~87h (99%) | $600 |
| **RECOMENDADO** | 3 | $900k | $1.8M | ~8.7h (99.9%) | $900 |
| **IDEAL** | 5 | $1.5M | $3.0M | ~52min (99.99%) | $1,500 |

\* TCO = Hardware + Energia + Datacenter + Operação (3 anos)

### ROI por Downtime Evitado

**Premissa:** Cada hora de downtime custa $10k (perda de receita + SLA penalties)

| Cenário | Downtime evitado vs MÍNIMO | Valor economizado/ano | ROI |
|---------|---------------------------|----------------------|-----|
| **RECOMENDADO** | ~78h | $780k | **+161%** |
| **IDEAL** | ~86h | $860k | **+43%** |

**Conclusão:** RECOMENDADO tem melhor ROI para maioria dos casos.

---

## 📊 Matriz de Decisão

| Fator | MÍNIMO | RECOMENDADO | IDEAL |
|-------|---------|-------------|-------|
| **Custo Inicial** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐ |
| **Resiliência** | ⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Flexibilidade** | ⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Simplicidade** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| **TCO 3 anos** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ |
| **Peace of Mind** | ⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

---

## 🎬 Casos de Uso Recomendados

### Use MÍNIMO se:
- ✅ Ambiente de desenvolvimento/teste
- ✅ PoC com < 1 mês de duração
- ✅ Orçamento extremamente limitado
- ✅ Consciente dos riscos e aceita downtime

### Use RECOMENDADO se: (✅ Maioria dos casos)
- ✅ Produção com usuários reais
- ✅ SLA 99% - 99.9%
- ✅ Budget moderado
- ✅ Crescimento planejado
- ✅ Tráfego com variação moderada

### Use IDEAL se:
- ✅ SLA > 99.95%
- ✅ Missão crítica (financeiro, healthcare)
- ✅ Compliance exige redundância dupla
- ✅ Tráfego imprevisível com picos extremos
- ✅ Zero tolerância a degradação

---

## 🔄 Migração Entre Cenários

### De MÍNIMO → RECOMENDADO
**Quando:** Antes de lançar beta público ou atingir 1k MAU

**Passos:**
1. Provisionar +1 nó (N+1)
2. Configurar load balancer com health checks
3. Testar failover (desligar 1 nó intencionalmente)
4. Ativar monitoramento de SLO
5. Migrar tráfego gradualmente

**Custo adicional:** +33-50%

### De RECOMENDADO → IDEAL
**Quando:** SLA < 99.9% se torna inaceitável, ou compliance exige N+2

**Passos:**
1. Provisionar +2 nós (N+2)
2. Reduzir kv_budget_ratio para 0.65
3. Aumentar peak_headroom_ratio para 0.30
4. Re-testar failover (desligar 2 nós)
5. Ajustar alertas de SLO

**Custo adicional:** +40-60% vs RECOMENDADO

---

## 📈 Recomendação Final

### Para 90% dos casos: **CENÁRIO RECOMENDADO** ✅

**Por quê?**
- ✅ Balanceamento ideal entre custo e resiliência
- ✅ Tolera falha de 1 nó (requisito mínimo para produção)
- ✅ Headroom para picos (evita surpresas)
- ✅ SLA 99.9% atingível (suficiente para maioria)
- ✅ TCO justificável (ROI claro vs MÍNIMO)

**Exceções:**
- PoC/Dev → MÍNIMO
- Missão crítica → IDEAL

---

**Versão:** 1.0  
**Data:** 2026-02-08
