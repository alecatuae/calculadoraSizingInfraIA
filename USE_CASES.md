# CASOS DE USO DETALHADOS
# Sistema de Dimensionamento de Inferência LLM

Este documento apresenta casos de uso reais e detalhados do sistema de dimensionamento.

---

## 📌 CASO 1: Startup SaaS - Assistente IA Conversacional

### Contexto
- Aplicação SaaS B2B de assistente IA
- Previsão: 10k usuários ativos simultâneos no pico
- Budget inicial limitado
- Precisa escalar com demanda

### Requisitos NFR
- Concorrência: 1.000 sessões simultâneas (fase 1)
- Contexto: 32k tokens (conversas de média duração)
- SLA: 99.9% (sem N+1 na fase inicial)
- Custo: Otimizar TCO

### Comando
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

### Resultado
- **KV por sessão:** 0.38 GiB
- **Sessões por nó:** 1.740
- **Nós necessários:** 1 nó DGX H200
- **Capacidade ociosa:** ~740 sessões (42% headroom)

### Recomendações
1. ✅ 1 nó DGX H200 suficiente para fase 1
2. ✅ FP8 ideal para custo-benefício
3. ✅ Storage NVMe local para cold-start rápido
4. ⚠️ Planejar N+1 quando atingir 1.200+ sessões
5. 📈 Escalar para 2 nós quando atingir 1.500+ sessões

### Custo Estimado (Referência)
- 1x DGX H200: ~$300k - $400k (CapEx)
- Sem N+1: Economia de $300k na fase inicial

---

## 📌 CASO 2: Empresa Enterprise - Análise de Documentos

### Contexto
- Sistema de análise de contratos e documentos legais
- Documentos longos (50-100 páginas)
- Processamento batch + consultas ad-hoc
- Criticidade alta (dados sensíveis)

### Requisitos NFR
- Concorrência: 500 análises simultâneas
- Contexto: 131k tokens (documentos longos)
- SLA: 99.99% com N+1
- Precisão: FP16 para análise precisa
- Storage: On-prem de alta performance

### Comando
```bash
python3 sizing.py \
  --model opt-oss-120b \
  --server dgx300 \
  --storage profile_default \
  --concurrency 500 \
  --effective-context 131072 \
  --kv-precision fp16 \
  --kv-budget-ratio 0.65 \
  --runtime-overhead-gib 150 \
  --peak-headroom-ratio 0.30 \
  --ha n+1
```

### Resultado
- **KV por sessão:** 4.50 GiB (fp16 = 2x fp8)
- **Sessões por nó:** 228
- **Nós necessários:** 3 + 1 (N+1) = **4 nós DGX B300**

### Recomendações
1. ✅ 4 nós DGX B300 com N+1 para HA
2. ⚠️ FP16 dobra memória - considerar validar se fp8 atende precisão
3. ✅ Storage NVMe local essencial para 131k tokens
4. 📊 Monitorar latência de prefill (131k = alto custo)
5. 💡 Considerar chunking de documentos > 100 páginas

### Análise TCO
- **Com FP8:** 2 nós + N+1 = 3 nós (~$900k)
- **Com FP16:** 3 nós + N+1 = 4 nós (~$1.2M)
- **Economia potencial com FP8:** ~$300k (25%)

---

## 📌 CASO 3: Provedor de API - Serviço Multi-Tenant

### Contexto
- API pública de inferência LLM (OpenAI-like)
- Múltiplos tenants com SLA diferenciados
- Tráfego variável (picos 2-3x normal)
- Precisa de elasticidade

### Requisitos NFR
- Concorrência: 5.000 sessões (carga normal)
- Contexto: 128k tokens
- SLA: 99.95% com N+1
- Headroom: 30% para picos de tráfego
- Storage: Híbrido (local + rede para checkpoints)

### Comando
```bash
python3 sizing.py \
  --model opt-oss-120b \
  --server dgx300 \
  --storage profile_default \
  --concurrency 5000 \
  --effective-context 131072 \
  --kv-precision fp8 \
  --kv-budget-ratio 0.70 \
  --runtime-overhead-gib 120 \
  --peak-headroom-ratio 0.30 \
  --ha n+1
```

### Resultado
- **KV por sessão:** 2.25 GiB
- **Sessões por nó:** 613
- **Nós mínimos:** 9 (capacidade pura)
- **Com headroom (30%):** 11 nós
- **Com N+1:** **12 nós DGX B300**

### Recomendações
1. ✅ 12 nós DGX B300 para 5k concurrent + picos + N+1
2. 📈 Implementar auto-scaling (adicionar nós sob demanda)
3. 💾 Storage híbrido:
   - NVMe local para KV cache e modelo ativo
   - Network SSD para checkpoints e modelo backups
4. 🔄 Load balancing entre nós com awareness de capacidade
5. 📊 Monitorar:
   - Utilização HBM por nó (alertar se > 85%)
   - Latência P99 (< 500ms ideal)
   - Taxa de throttling

### Arquitetura
```
                    ┌─────────────────┐
                    │  Load Balancer  │
                    └────────┬────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
    ┌───▼───┐           ┌───▼───┐           ┌───▼───┐
    │ Pod 1 │           │ Pod 2 │    ...    │ Pod 4 │
    │ 3 nós │           │ 3 nós │           │ 3 nós │
    └───────┘           └───────┘           └───────┘
        │                    │                    │
        └────────────────────┼────────────────────┘
                             │
                    ┌────────▼────────┐
                    │  Storage Pool   │
                    │  (NVMe + SSD)   │
                    └─────────────────┘
```

---

## 📌 CASO 4: Pesquisa Acadêmica - Fine-Tuning e Avaliação

### Contexto
- Lab de pesquisa em NLP
- Experimentos com diferentes configurações
- Budget limitado (uso compartilhado)
- Foco em qualidade, não em throughput

### Requisitos NFR
- Concorrência: 50 sessões simultâneas (pesquisadores)
- Contexto: 128k tokens (artigos científicos)
- Precisão: FP16 para experimentos
- Sem requisito de HA

### Comando
```bash
python3 sizing.py \
  --model opt-oss-20b \
  --server dgx200 \
  --storage profile_network_ssd \
  --concurrency 50 \
  --effective-context 131072 \
  --kv-precision fp16 \
  --kv-budget-ratio 0.75 \
  --runtime-overhead-gib 60 \
  --peak-headroom-ratio 0.10 \
  --ha none
```

### Resultado
- **KV por sessão:** 2.82 GiB (fp16 + full attention em metade das camadas)
- **Sessões por nó:** 227
- **Nós necessários:** **1 nó DGX H200**

### Recomendações
1. ✅ 1 nó DGX H200 suficiente (227 >> 50 sessões)
2. ✅ FP16 adequado para pesquisa
3. ✅ Storage de rede OK (não é crítico para pesquisa)
4. 💡 Shared allocation: tempo de GPU por pesquisador
5. 📊 Quotas recomendadas: 5 sessões/pesquisador

---

## 📌 CASO 5: Cloud Provider - Serviço Serverless

### Contexto
- Provider oferece "LLM as a Service" serverless
- Cold start crítico (< 5s)
- Auto-scaling agressivo
- Múltiplas regiões

### Requisitos NFR
- Concorrência: 2.000 sessões/região
- Contexto: Mix (4k-128k, média 32k)
- SLA: 99.9% por região
- Cold start: < 5s
- Storage: Ultra-rápido para loading

### Comando (dimensionamento por região)
```bash
python3 sizing.py \
  --model opt-oss-120b \
  --server dgx300 \
  --storage profile_default \
  --concurrency 2000 \
  --effective-context 32768 \
  --kv-precision fp8 \
  --kv-budget-ratio 0.70 \
  --runtime-overhead-gib 100 \
  --peak-headroom-ratio 0.40 \
  --ha n+1
```

### Resultado
- **KV por sessão:** 0.56 GiB (contexto médio menor)
- **Sessões por nó:** 2.399
- **Nós com headroom (40%):** 2
- **Com N+1:** **3 nós/região**

### Recomendações
1. ✅ 3 nós DGX B300 por região
2. 🌍 Multi-região:
   - US-East: 3 nós
   - US-West: 3 nós
   - EU: 3 nós
   - Total: 9 nós
3. 💾 Storage:
   - NVMe local obrigatório (cold start < 5s)
   - Cache de modelo em RAM (2TB system memory)
4. 🔄 Auto-scaling:
   - Scale-up: add nó quando utilização > 70%
   - Scale-down: remove nó quando < 30% por 10min
5. 📊 Métricas críticas:
   - Cold start latency (P99 < 5s)
   - Utilização HBM por nó
   - Taxa de scale events

---

## 📊 Comparação de Casos

| Caso | Modelo | Nós | Custo Aprox | Contexto | HA | Uso |
|------|--------|-----|-------------|----------|----|----|
| Startup SaaS | 20B | 1 | $300k | 32k | Não | Dev/Prod inicial |
| Enterprise | 120B | 4 | $1.2M | 131k | N+1 | Docs longos críticos |
| API Provider | 120B | 12 | $3.6M | 131k | N+1 | Alta escala |
| Pesquisa | 20B | 1 | $300k | 131k | Não | Experimentos |
| Cloud Serverless | 120B | 9 | $2.7M | 32k | N+1 | Multi-região |

---

## 🎯 Lições Aprendidas

### 1. Precisão KV
- **FP8 vs FP16:** Diferença de 2x em memória
- **Recomendação:** Sempre começar com FP8, validar qualidade
- **Quando usar FP16:** Apenas se FP8 não atender requisitos de qualidade

### 2. Contexto
- **4k-32k:** Sweet spot para maioria dos casos
- **128k-131k:** Requer cuidado com memória e I/O
- **Prefill:** Contextos > 100k pressionam I/O no cold start

### 3. HA (N+1)
- **Quando usar:** Produção crítica, SLA > 99.9%
- **Custo:** +1 nó (pode ser 10-50% de overhead)
- **Trade-off:** Custo vs disponibilidade

### 4. Storage
- **NVMe local:** Sempre que possível (cold start)
- **Network SSD:** OK para cargas não críticas
- **Cloud storage:** Evitar para inferência de produção

### 5. Headroom
- **10-20%:** Crescimento orgânico
- **30-40%:** Tráfego sazonal/variável
- **> 50%:** Over-provisioning (desperdício)

---

**Versão:** 1.0  
**Data:** 2026-02-07
