# RELATÓRIO EXECUTIVO
## Dimensionamento de Infraestrutura de Inferência LLM

**Data:** 08/02/2026
**Modelo Analisado:** opt-oss-120b
**Carga Projetada:** 1,000 sessões simultâneas

---

## 1. Sumário Executivo

Este relatório analisa a capacidade de infraestrutura necessária para operar o modelo **opt-oss-120b** em ambiente de produção, sustentando **1,000 sessões simultâneas** com contexto efetivo de **131,072 tokens**.

A arquitetura do modelo (36 camadas, 8 KV heads, padrão de atenção hybrid) define que o principal fator limitante da operação é a **memória de GPU (HBM)**, especificamente o armazenamento de **KV cache** para manutenção do contexto conversacional.

Foram avaliados três cenários de dimensionamento — **Mínimo**, **Recomendado** e **Ideal** — representando diferentes níveis de tolerância a falhas, capacidade para picos e investimento de capital. O servidor base considerado é o **dgx300** (8 GPUs, 2304 GB HBM total).

**Conclusão Principal:** Para sustentar 1,000 sessões simultâneas com contexto de 131,072 tokens, a infraestrutura passa a ser limitada por memória de GPU, exigindo **3 nós DGX** (cenário recomendado) para garantir estabilidade, tolerância a falhas e continuidade operacional.

---

## 2. Cenários Avaliados

A análise contempla três cenários de dimensionamento, cada um representando um trade-off distinto entre investimento e resiliência operacional:

| Cenário | Objetivo | Característica Principal | Risco Operacional |
|---------|----------|-------------------------|-------------------|
| **Mínimo** | Atender requisitos no limite | Sem tolerância a falhas ou picos | **Alto** — Falha de hardware causa indisponibilidade imediata |
| **Recomendado** | Operação estável em produção | Tolerância a picos (20%) e falha simples (N+1) | **Médio** — Degradação gerenciável em cenários adversos |
| **Ideal** | Operação resiliente | Alta tolerância a falhas (N+2) e folga (30%+) | **Baixo** — Sistema mantém SLA mesmo sob múltiplas falhas |

A avaliação de múltiplos cenários é essencial para que a diretoria possa calibrar o investimento em infraestrutura de acordo com o perfil de risco aceitável e os requisitos de SLA. O **cenário Mínimo** representa o menor capex possível, mas com risco operacional elevado. O **cenário Recomendado** equilibra custo e resiliência, sendo adequado para operações de produção. O **cenário Ideal** oferece máxima resiliência, indicado para cargas críticas ou ambientes com alta variabilidade.

---

## 3. Resultado Consolidado por Cenário

### Cenário MÍNIMO

| Métrica | Valor |
|---------|-------|
| **Modelo avaliado** | opt-oss-120b |
| **Servidor base** | dgx300 |
| **Contexto efetivo** | 131,072 tokens |
| **Concorrência alvo** | 1,000 sessões simultâneas |
| **KV cache por sessão** | 2.25 GiB |
| **KV total necessário** | 2.20 TiB (2252.2 GiB) |
| **Budget efetivo de HBM por nó** | 1418.0 GiB (de 2145.8 GiB totais) |
| **Sessões suportadas por nó** | 629 |
| **Nós DGX necessários** | **2** |
| **Arquitetura de HA** | Sem redundância |
| **Headroom para picos** | 0% |

**Significado Operacional:** Este cenário dimensiona a infraestrutura no limite da capacidade técnica. Com 2 nó(s), o sistema atende a carga nominal, mas **não possui tolerância a falhas**. Qualquer evento de manutenção ou falha de hardware resulta em indisponibilidade imediata. Picos de tráfego não planejados causarão degradação de performance ou throttling. Adequado apenas para ambientes de desenvolvimento ou PoCs com baixa criticidade.

---

### Cenário RECOMENDADO

| Métrica | Valor |
|---------|-------|
| **Modelo avaliado** | opt-oss-120b |
| **Servidor base** | dgx300 |
| **Contexto efetivo** | 131,072 tokens |
| **Concorrência alvo** | 1,000 sessões simultâneas |
| **KV cache por sessão** | 2.25 GiB |
| **KV total necessário** | 2.20 TiB (2252.2 GiB) |
| **Budget efetivo de HBM por nó** | 1418.0 GiB (de 2145.8 GiB totais) |
| **Sessões suportadas por nó** | 629 |
| **Nós DGX necessários** | **3** |
| **Arquitetura de HA** | N+1 |
| **Headroom para picos** | 20% |

**Significado Operacional:** Este cenário foi dimensionado para operação em produção. Com 3 nós (2 operacionais + 1 para HA), o sistema tolera a **falha de 1 nó** sem perda de capacidade crítica. O headroom de 20% garante absorção de picos de demanda sem degradação de experiência. Em caso de falha, os nós restantes mantêm 1258 sessões, suficiente para a carga nominal com margem. **Recomendado para produção com SLA de 99.9%**.

---

### Cenário IDEAL

| Métrica | Valor |
|---------|-------|
| **Modelo avaliado** | opt-oss-120b |
| **Servidor base** | dgx300 |
| **Contexto efetivo** | 131,072 tokens |
| **Concorrência alvo** | 1,000 sessões simultâneas |
| **KV cache por sessão** | 2.25 GiB |
| **KV total necessário** | 2.20 TiB (2252.2 GiB) |
| **Budget efetivo de HBM por nó** | 1316.7 GiB (de 2145.8 GiB totais) |
| **Sessões suportadas por nó** | 584 |
| **Nós DGX necessários** | **5** |
| **Arquitetura de HA** | N+2 |
| **Headroom para picos** | 30% |

**Significado Operacional:** Este cenário maximiza resiliência operacional. Com 5 nós (3 operacionais + 2 para HA), o sistema tolera **falhas simultâneas de até 2 nós**, cenário raro mas possível em eventos de rack ou rede. O headroom de 30% e budget conservador de KV (65% vs 70% padrão) garantem estabilidade mesmo sob múltiplas adversidades. Indicado para cargas de missão crítica (financeiro, healthcare) ou ambientes com alta imprevisibilidade de demanda.

---

## 4. Racional de Cálculo

O dimensionamento segue uma metodologia baseada em limitações de memória GPU (HBM). A tabela abaixo apresenta as fórmulas utilizadas, parâmetros de entrada, suposições aplicadas e o significado operacional de cada resultado.

| Resultado | Fórmula Utilizada | Parâmetros do Cálculo | Suposição Aplicada | Significado Operacional |
|-----------|-------------------|----------------------|-------------------|------------------------|
| **KV cache por sessão** | 2 × (full_layers × context + sliding_layers × window) × kv_heads × head_dim × bytes | Camadas: 36, KV heads: 8, Context: 131,072, Precisão: fp8 (1 byte/elem) | Padrão de atenção 'hybrid' determina seq_length por camada | Memória GPU consumida por cada sessão ativa. Subdimensionamento causa OOM (Out of Memory). |
| **KV total** | KV_per_session × concurrency | KV/sessão: 2.25 GiB, Concorrência: 1,000 | Carga simultânea define demanda agregada | Volume total de memória necessário no cluster. Determina número mínimo de nós. |
| **Budget HBM por nó** | (HBM_total - overhead) × budget_ratio | HBM: 2145.8 GiB, Overhead: 120 GiB, Ratio: 70% | Overhead reserva memória para modelo, ativações e buffers; budget_ratio evita fragmentação | Memória efetivamente disponível para KV cache por nó. Ratio >75% aumenta risco de instabilidade. |
| **Sessões por nó** | floor(Budget_KV / KV_per_session) | Budget: 1418.0 GiB, KV/sessão: 2.25 GiB | Capacidade limitada por memória, não por compute | Máximo de sessões que cada nó suporta simultaneamente. Exceder causa recusa de conexões. |
| **Nós necessários** | ceil(concurrency × (1 + headroom) / sessions_per_node) + HA | Concorrência: 1,000, Headroom: 20%, Sessões/nó: 629, HA: 1 | Headroom para picos; HA garante continuidade em falhas | Número de nós DGX a provisionar. Subdimensionamento gera throttling; superdimensionamento desperdiça capex. |

---

## 5. Análise Comparativa dos Cenários

| Critério | Mínimo | Recomendado | Ideal |
|----------|--------|-------------|-------|
| **Número de nós DGX** | 2 | 3 | 5 |
| **Tolerância a falhas** | Nenhuma | 1 nó (N+1) | 2 nós (N+2) |
| **Capacidade para picos** | 0% | 20% | 30% |
| **Risco de indisponibilidade** | Alto | Médio | Baixo |
| **Complexidade operacional** | Baixa | Média | Média-Alta |
| **CapEx relativo** | Baseline | +50% | +150% |

**Conclusão Comparativa:** O **cenário Recomendado** oferece o melhor equilíbrio entre custo e resiliência. Com 3 nós (+50% vs Mínimo), garante operação estável em produção, tolerando falhas simples e picos de demanda. O cenário Mínimo (2 nó(s)) reduz capex, mas expõe a operação a risco de indisponibilidade não gerenciável. O cenário Ideal (5 nós) é justificável apenas para cargas de missão crítica com SLA > 99.95% ou em ambientes com histórico de falhas múltiplas.

---

## 6. Principais Riscos e Alertas

**Riscos de Operação no Limite (Cenário Mínimo):**

- **Indisponibilidade imediata em falhas:** Com 2 nó(s) e sem HA, qualquer falha de hardware paralisa o serviço
- **Degradação em picos:** Sem headroom, variações de demanda causam throttling, latência elevada ou recusa de conexões
- **Impossibilidade de manutenção planejada:** Deploy de updates ou manutenção preventiva exigem janela de downtime

**Impactos de Decisões Técnicas:**

- **Precisão KV (FP16 vs FP8):** Uso de FP16 dobra o consumo de memória, reduzindo sessões/nó em ~50% e aumentando custos de infra proporcionalmente. FP8 oferece qualidade equivalente na maioria dos casos, sendo fortemente recomendado.
- **Contexto máximo liberado:** Liberar contexto de 131,072 tokens sem controle de governança pode levar a sobrecarga não planejada. Recomenda-se limite de 131,072 tokens com throttling para casos excepcionais.
- **Budget de HBM agressivo:** Alocar >75% da HBM para KV cache aumenta risco de fragmentação de memória e instabilidade em runtime. Cenário Ideal usa 65% (conservador) para maior estabilidade.

**Consequências Operacionais de Subdimensionamento:**

- **Filas de espera:** Sessões excedentes entram em fila, aumentando latência e degradando experiência do usuário
- **Degradação de SLA:** Percentis P95/P99 de latência ultrapassam limites aceitáveis, violando acordos de nível de serviço
- **Indisponibilidade parcial:** Em falhas, sistema opera em modo degradado, recusando novas conexões ou encerrando sessões existentes

**Alertas Técnicos Identificados:**

- ALERTA: Contexto longo (131,072 tokens) aumenta TTFT (Time To First Token) e pressiona I/O de storage durante prefill. Storage: profile_default (28 GB/s read, P99=0.15 ms).

---

## 7. Recomendação Final

Considerando os requisitos atuais (concorrência de 1,000 sessões, contexto de 131,072 tokens) e o perfil de uso esperado para operação em produção, **recomenda-se a adoção do cenário RECOMENDADO**.

**Justificativa:** Com **3 nós DGX** (2 operacionais + 1 para HA), este cenário equilibra eficiência de capital, estabilidade operacional e tolerância a falhas, sem comprometer a experiência do usuário. O sistema suporta picos de até 20% acima da carga nominal e mantém operação em caso de falha de 1 nó.

**Premissas que Devem Ser Mantidas Sob Governança:**

1. **Limite de Contexto:** Manter contexto efetivo em 131,072 tokens com controle de governança. Liberação de contexto máximo (131,072 tokens) deve exigir aprovação.
2. **Concorrência Monitorada:** Implementar alertas quando concorrência real ultrapassar 800 sessões (80% da capacidade), permitindo escala proativa.
3. **Precisão KV:** Manter FP8 como padrão. Mudança para FP16 requer reavaliação de capacidade e custo.
4. **Budget de HBM:** Respeitar 70% de alocação para KV cache. Ajustes devem ser validados via profiling.

**Próximos Passos:** Proceder com procurement de 3 nós dgx300, implementar monitoramento de capacidade em tempo real e estabelecer política de governança para controle de contexto e concorrência.

---

## 8. Dicionário de Parâmetros

Esta seção detalha os parâmetros utilizados no dimensionamento, sua origem e importância para o cálculo.

| Parâmetro | Origem | Descrição | Importância para o Cálculo |
|-----------|--------|-----------|---------------------------|
| **num_layers** | Arquitetura do Modelo | Número total de camadas (layers) do transformer no modelo LLM | Impacta linearmente o tamanho do KV cache |
| **num_key_value_heads** | Arquitetura do Modelo | Número de cabeças (heads) de atenção para Key e Value | Impacta diretamente o tamanho do KV cache |
| **attention_pattern** | Arquitetura do Modelo | Padrão de atenção usado pelo modelo: 'full' (todas camadas atendem contexto completo), 'sliding' (janela deslizante), ou 'hybrid' (mix de full e sliding) | Crítico para cálculo correto de KV cache |
| **effective_context** | NFR do Produto | Tamanho de contexto (em tokens) que sua aplicação efetivamente usará em runtime | Impacta diretamente o tamanho do KV cache por sessão |
| **concurrency** | NFR do Produto | Número de sessões/requisições simultâneas (concurrent users) que o sistema deve suportar | Define quantos nós você precisa |
| **kv_precision** | Runtime/Configuração | Precisão numérica usada para armazenar tensores Key e Value: fp8/int8 (1 byte/elemento) ou fp16/bf16 (2 bytes/elemento) | Impacta diretamente (2x) o tamanho do KV cache |
| **kv_budget_ratio** | Tuning de Infraestrutura | Fração da HBM total alocada para KV cache (ex: 0 | Define quantas sessões cabem por nó |
| **runtime_overhead_gib** | Estimativa de Runtime | Memória GPU (GiB) reservada para modelo (pesos), ativações de computação, e buffers do runtime de inferência | Subtrai da HBM disponível antes de calcular budget de KV |
| **peak_headroom_ratio** | NFR de Resiliência | Fração adicional de capacidade reservada para picos de tráfego (ex: 0 | Garante que sistema aguenta picos sem degradação de SLO |
| **ha_mode** | NFR de Disponibilidade | Modo de alta disponibilidade: 'none' (sem redundância), 'n+1' (tolera falha de 1 nó), 'n+2' (tolera 2 nós) | Define quantos nós extras alocar para redundância |

**Nota:** Parâmetros de arquitetura do modelo são fixos e não ajustáveis em runtime. Parâmetros de NFR e configuração devem ser definidos com base em requisitos de negócio e validados via testes de carga.

---

## Informações do Relatório

- **Sistema:** Sizing de Infraestrutura IA v2.0
- **Data de Geração:** 08/02/2026 13:14:57
- **Metodologia:** Dimensionamento baseado em memória GPU (KV cache)
- **Servidor de Referência:** dgx300 (8 GPUs, 2304 GB HBM)

*Este relatório foi gerado automaticamente e deve ser revisado por arquitetos de infraestrutura antes de decisões de investimento.*