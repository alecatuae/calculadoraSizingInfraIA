# RELATÓRIO EXECUTIVO
## Dimensionamento de Infraestrutura de Inferência LLM

**Data:** 08/02/2026
**Modelo Analisado:** opt-oss-20b
**Carga Operacional:** 500 sessões simultâneas × 65,536 tokens/contexto

---

## 1. Sumário Executivo

Este relatório dimensiona a infraestrutura necessária para operar o modelo **opt-oss-20b** em produção, sustentando **500 sessões simultâneas** com contexto de **65,536 tokens**. O principal limitador da operação é a **memória de GPU (HBM)**, especificamente o **KV cache** que mantém o contexto conversacional ativo.

A análise identifica impacto direto em três dimensões críticas:
- **Servidores**: 2 nós DGX dgx-b200 (cenário recomendado)
- **Energia**: 28.6 kW de consumo contínuo
- **Datacenter**: 20U de espaço em rack (0.5 racks padrão)

**Para sustentar a carga avaliada com estabilidade operacional, a plataforma exige múltiplos nós DGX, com impacto direto em energia (28.6 kW) e densidade de rack (20U).**

---

## 2. Cenários Avaliados

### Tabela – Visão Geral dos Cenários

| Cenário | Objetivo | Tolerância a Falhas | Risco Operacional |
|---------|----------|---------------------|-------------------|
| **Mínimo** | Atender no limite | Nenhuma | **Alto** — Falha causa indisponibilidade imediata |
| **Recomendado** | Produção estável | Falha simples (N+1) | **Médio** — Degradação gerenciável |
| **Ideal** | Alta resiliência | Falhas múltiplas (N+2) | **Baixo** — Sistema mantém SLA sob adversidades |

Os três cenários representam diferentes níveis de **investimento** versus **risco operacional**. O cenário **Mínimo** minimiza capex mas expõe a operação a risco de indisponibilidade não gerenciável. O cenário **Recomendado** equilibra custo e resiliência, adequado para produção com SLA 99.9%. O cenário **Ideal** maximiza disponibilidade, indicado para cargas críticas com requisitos de SLA > 99.95%.

---

## 3. Informações do Modelo Avaliado

### Tabela – Perfil do Modelo

| Item | Valor |
|------|-------|
| **Modelo** | opt-oss-20b |
| **Número de camadas** | 24 |
| **KV heads** | 8 |
| **Contexto máximo** | 131,072 tokens |
| **Contexto efetivo usado** | 65,536 tokens |
| **Padrão de atenção** | Hybrid |
| **Precisão do KV cache** | FP8 (1 byte/elemento) |

O modelo opt-oss-20b consome **memória viva** durante a operação para armazenar o **KV cache** — tensores Key e Value que mantêm o contexto conversacional. Este consumo é proporcional ao **contexto efetivo** (65,536 tokens) e à **concorrência** (500 sessões), dominando a capacidade de infraestrutura necessária. Diferente dos pesos do modelo (fixos), o KV cache escala linearmente com o número de sessões ativas.

---

## 4. Consumo Unitário do Modelo

### Tabela – Consumo por Sessão

| Recurso | Consumo por Sessão | Significado Operacional |
|---------|-------------------|------------------------|
| **KV cache** | 0.75 GiB | Memória GPU ocupada enquanto a sessão está ativa |
| **GPU HBM (%)** | 0.1% de um nó | Fração da capacidade de um servidor consumida |
| **Energia estimada** | 13 W | Impacto incremental por sessão ativa (aproximado) |
| **Rack** | N/A | Sessão não consome rack diretamente; nó sim (10U/nó) |

**Cada sessão ativa "reserva" 0.75 GiB de HBM (0.1% do budget do nó).** A soma dessas reservas define o **limite físico** do servidor: com 854.8 GiB disponíveis para KV, cada nó suporta no máximo **1137 sessões simultâneas**. Exceder este limite causa recusa de novas conexões ou degradação de performance.

---

## 5. Consumo Agregado da Plataforma

### Tabela – Consumo Total (Cenário Recomendado)

| Recurso | Total Consumido |
|---------|----------------|
| **Sessões simultâneas** | 500 |
| **KV total** | 0.37 TiB (375.7 GiB) |
| **Nós DGX** | 2 |
| **Energia total** | 28.6 kW (251 MWh/ano) |
| **Espaço em rack** | 20U (0.5 racks) |

O **consumo agregado** demonstra a diferença entre consumo unitário e impacto total: enquanto uma sessão consome 0.75 GiB, 500 sessões simultâneas consomem 0.37 TiB distribuídos entre 2 nós. **O crescimento de usuários impacta linearmente a infraestrutura**: dobrar concorrência para 1,000 sessões dobraria energia para 57.2 kW e rack para 40U.

---

## 6. Resultados por Cenário

### Cenário MÍNIMO

| Métrica | Valor |
|---------|-------|
| **Nós DGX** | 1 |
| **Sessões por nó** | 1137 |
| **KV por sessão** | 0.75 GiB |
| **KV total** | 0.37 TiB |
| **Energia total** | **14.3 kW** (125 MWh/ano) |
| **Espaço em rack** | **10U** (0.2 racks) |
| **Arquitetura** | Sem redundância |
| **Headroom para picos** | 0% |

**Significado Operacional:** Este cenário dimensiona a infraestrutura no limite absoluto (1 nós, 14.3 kW, 10U). **Sem tolerância a falhas**: qualquer evento de manutenção ou falha de hardware resulta em indisponibilidade imediata. **Sem headroom**: picos de tráfego causam throttling ou recusa de conexões. **Impacto físico mínimo** mas **risco operacional alto**. Adequado apenas para PoC ou ambientes de desenvolvimento.

---

### Cenário RECOMENDADO

| Métrica | Valor |
|---------|-------|
| **Nós DGX** | 2 |
| **Sessões por nó** | 1137 |
| **KV por sessão** | 0.75 GiB |
| **KV total** | 0.37 TiB |
| **Energia total** | **28.6 kW** (251 MWh/ano) |
| **Espaço em rack** | **20U** (0.5 racks) |
| **Arquitetura** | N+1 |
| **Headroom para picos** | 20% |

**Significado Operacional:** Dimensionado para produção com resiliência (2 nós, 28.6 kW, 20U). **Tolera falha de 1 nó** sem perda de capacidade crítica. Headroom de 20% absorve picos de demanda. **Impacto físico:** 28.6 kW requer PDU com capacidade adequada e UPS dimensionado; 20U equivale a 0.5 racks, gerenciável em datacenter padrão. **Recomendado para produção com SLA 99.9%**.

---

### Cenário IDEAL

| Métrica | Valor |
|---------|-------|
| **Nós DGX** | 3 |
| **Sessões por nó** | 1056 |
| **KV por sessão** | 0.75 GiB |
| **KV total** | 0.37 TiB |
| **Energia total** | **42.9 kW** (376 MWh/ano) |
| **Espaço em rack** | **30U** (0.7 racks) |
| **Arquitetura** | N+2 |
| **Headroom para picos** | 30% |

**Significado Operacional:** Máxima resiliência operacional (3 nós, 42.9 kW, 30U). **Tolera falhas simultâneas de até 2 nós**, cenário raro mas possível em eventos de rack ou rede. Headroom de 30% e budget conservador (65% HBM) garantem estabilidade máxima. **Impacto físico significativo:** 42.9 kW pode exigir upgrade de PDU/UPS; 30U requer planejamento de densidade de rack. Indicado para cargas de missão crítica (financeiro, healthcare, SLA > 99.95%).

---

## 7. Racional de Cálculo

### Tabela – Metodologia de Dimensionamento

| Resultado | Fórmula | Parâmetros do Cálculo | Suposição Aplicada | Significado Operacional |
|-----------|---------|----------------------|-------------------|------------------------|
| **KV por sessão** | 2 × [(full_layers × context) + (sliding_layers × window)] × kv_heads × head_dim × bytes | Camadas: 24, Context: 65,536, KV heads: 8, Precisão: fp8 | Padrão 'hybrid' determina seq_length por camada | Memória reservada por sessão; subdimensionar causa OOM |
| **Sessões por nó** | floor(Budget_KV / KV_per_session) | Budget: 854.8 GiB, KV/sessão: 0.75 GiB | Budget = (HBM - overhead) × ratio; limitado por memória | Capacidade máxima do servidor; exceder causa recusa de conexões |
| **Nós DGX** | ceil(concurrency × (1 + headroom) / sessões_per_nó) + HA | Concorrência: 500, Headroom: 20%, Sessões/nó: 1137, HA: +1 | Headroom para picos; HA garante continuidade em falhas | Número de servidores a provisionar; define capex e opex |
| **Energia (kW)** | nodes_final × power_kw_max | Nós: 2, Power/nó: 14.3 kW | Consumo máximo contínuo do sistema | Dimensiona PDU, UPS, contrato de energia; considerar PUE (~1.4x) |
| **Rack (U)** | nodes_final × rack_units_u | Nós: 2, U/nó: 10U | Cada servidor ocupa 10U; racks padrão = 42U | Define densidade e capacidade física; adicionar ~20% para infra |

---

## 8. Comparação Executiva dos Cenários

### Tabela – Comparativo

| Critério | Mínimo | Recomendado | Ideal |
|----------|--------|-------------|-------|
| **Nós DGX** | 1 | 2 | 3 |
| **Energia (kW)** | 14.3 | 28.6 | 42.9 |
| **Rack (U)** | 10 | 20 | 30 |
| **Tolerância a falhas** | Nenhuma | 1 nó (N+1) | 2 nós (N+2) |
| **Headroom** | 0% | 20% | 30% |
| **Risco operacional** | Alto | Médio | Baixo |
| **CapEx relativo** | Baseline | +100% | +200% |
| **Energia relativa** | Baseline | +100% | +200% |

**O cenário RECOMENDADO oferece o melhor equilíbrio custo × risco.** Com 2 nós (+100% vs Mínimo), garante operação estável, tolerando falhas e picos. **O impacto físico muda significativamente entre cenários:** Mínimo usa 14.3 kW, Recomendado 28.6 kW (+100%), Ideal 42.9 kW (+200%). A escolha deve considerar não apenas servidores, mas capacidade elétrica e densidade de datacenter.

---

## 9. Recomendação Final

**Recomenda-se o cenário RECOMENDADO**, que equilibra capacidade, consumo energético e tolerância a falhas sem sobrecarregar o datacenter.

**Justificativa:**
- **Estabilidade:** 2 nós com N+1 toleram falha de 1 servidor mantendo 1,137 sessões (suficiente para carga nominal)
- **Energia:** 28.6 kW requer PDU/UPS padrão de datacenter; PUE 1.4x = 40.0 kW total facility
- **Datacenter:** 20U (0.5 racks) é gerenciável e não exige reconfiguração física
- **Risco:** Médio, com degradação gerenciável em falhas; adequado para produção com SLA 99.9%

**Premissas sob governança:**
- Limite de contexto: 65,536 tokens (não liberar contexto máximo sem validação)
- Monitoramento: Alertas quando concorrência ultrapassar 1,819 sessões (80% capacidade)
- Precisão KV: Manter FP8 (mudança para FP16 dobraria energia e rack)

---

## 10. Dicionário de Parâmetros

### Tabela – Dicionário

| Parâmetro | Origem | Descrição | Importância |
|-----------|--------|-----------|------------|
| **num_layers** | Arquitetura do Modelo | Número de camadas do transformer (24) | Impacta linearmente o KV cache |
| **num_key_value_heads** | Arquitetura do Modelo | Cabeças de atenção para K/V (8) | Redução via GQA economiza memória |
| **attention_pattern** | Arquitetura do Modelo | Padrão de atenção: hybrid | Crítico para cálculo correto de KV |
| **total_hbm_gb** | Hardware do Servidor | HBM total do servidor (1440 GB) | Define capacidade bruta de memória |
| **power_kw_max** | Hardware do Servidor | Consumo máximo (14.3 kW) | Define impacto elétrico por nó |
| **rack_units_u** | Hardware do Servidor | Espaço em rack (10U) | Define densidade física |
| **concurrency** | NFR do Produto | Sessões simultâneas (500) | Define escala e número de nós |
| **effective_context** | NFR do Produto | Contexto efetivo (65,536 tokens) | Impacta KV por sessão linearmente |
| **kv_precision** | Configuração de Runtime | Precisão do KV (FP8) | FP8=1 byte, FP16=2 bytes (dobra memória) |
| **peak_headroom_ratio** | NFR de Resiliência | Folga para picos (20%) | Garante absorção de variações de carga |
| **ha_mode** | NFR de Disponibilidade | Alta disponibilidade (N+1) | N+1 tolera 1 falha; N+2 tolera 2 falhas |

**Nota:** Parâmetros de modelo e servidor são fixos. Parâmetros de NFR e runtime são ajustáveis conforme requisitos de negócio.

---

## Informações do Relatório

- **Sistema:** Sizing de Infraestrutura IA v2.0
- **Data de Geração:** 08/02/2026 14:28:56
- **Metodologia:** Dimensionamento baseado em memória GPU (KV cache) com impacto físico de datacenter
- **Servidor de Referência:** dgx-b200 (8 GPUs, 1440 GB HBM, 14.3 kW, 10U)

*Este relatório foi gerado automaticamente. Decisões de investimento devem ser revisadas por arquitetos de infraestrutura e finance.*