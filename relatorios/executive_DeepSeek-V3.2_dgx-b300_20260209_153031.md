# Relatório Executivo - Sizing de Infraestrutura LLM

**Modelo:** DeepSeek-V3.2  
**Servidor:** dgx-b300  
**Data:** 2026-02-09 15:30:31  

---

## Sumário Executivo

Para sustentar **1,000 sessões simultâneas** com contexto de **131,072 tokens** 
utilizando o modelo **DeepSeek-V3.2**, a infraestrutura é dimensionada por **memória GPU (KV cache)** e **storage**.

O principal limitador de capacidade é o consumo de HBM para armazenar o estado de atenção (KV cache) de cada sessão ativa. 
Storage é dimensionado para operação contínua (pesos do modelo, cache de runtime, logs e auditoria), 
garantindo resiliência, tempo de recuperação e governança operacional.

**Recomendação:** 4 nós DGX dgx-b300 
(58.0 kW, 40U rack, 22.1 TB storage) 
com tolerância a falhas N+1.

---

## Cenários Avaliados

| Cenário | Objetivo | Tolerância a Falhas | Risco Operacional |
|---------|----------|---------------------|-------------------|
| **Mínimo** | Atender no limite | Nenhuma | Alto |
| **Recomendado** | Produção estável | Falha simples (N+1) | Médio |
| **Ideal** | Alta resiliência | Falhas múltiplas (N+2) | Baixo |

Avaliar múltiplos cenários é essencial para equilibrar custo de investimento com risco operacional.

---

## Informações do Modelo Avaliado

| Item | Valor |
|------|-------|
| Modelo | DeepSeek-V3.2 |
| Número de camadas | 61 |
| Contexto máximo | 163,840 tokens |
| Padrão de atenção | sliding |
| Precisão KV cache | FP8 |

O modelo consome memória viva (KV cache) proporcional ao contexto e concorrência.

---

## Consumo Unitário do Modelo

| Recurso | Consumo por Sessão | Significado Operacional |
|---------|-------------------|------------------------|
| KV cache | 1.67 GiB | Memória ocupada enquanto sessão está ativa |
| GPU HBM | 0.1% de um nó | Fração da capacidade GPU consumida |

Cada sessão ativa 'reserva' parte do servidor. A soma das reservas define o limite físico do nó.

---

## Resultados por Cenário

### Cenário MÍNIMO

| Métrica | Valor |
|---------|-------|
| Nós DGX | 2 |
| Sessões por nó (capacidade) | 582 |
| Sessões por nó (operando) | 500 |
| KV por sessão | 1.67 GiB |
| VRAM total por nó | 1591.9 GiB (74.2% HBM) |
| **Energia (Compute + Storage)** | **29.5 kW** (29.0 + 0.5) |
| **Rack (Compute + Storage)** | **22U** (20 + 2) |
| **Storage total** | **17.07 TB** |
| Storage (modelo) | 3.12 TB |
| Storage (cache) | 1.85 TB |
| Storage (logs) | 12.07 TB |
| IOPS (pico R/W) | 75,000 / 3,000 |
| Throughput (pico R/W) | 39.9 / 0.0 GB/s |
| Arquitetura HA | NONE |

**Análise Computacional:** Opera no limite da capacidade sem margem para picos ou falhas. 
Risco operacional **alto** - qualquer indisponibilidade de hardware afeta o serviço diretamente. 
**Análise Storage:** Volumetria mínima (17.1 TB) para operação steady-state. 
IOPS e throughput dimensionados sem margem. Risco de gargalo em scale-out ou restart simultâneo.

### Cenário RECOMENDADO

| Métrica | Valor |
|---------|-------|
| Nós DGX | 4 |
| Sessões por nó (capacidade) | 582 |
| Sessões por nó (operando) | 250 |
| KV por sessão | 1.67 GiB |
| VRAM total por nó | 1175.0 GiB (54.8% HBM) |
| **Energia (Compute + Storage)** | **58.5 kW** (58.0 + 0.5) |
| **Rack (Compute + Storage)** | **42U** (40 + 2) |
| **Storage total** | **22.06 TB** |
| Storage (modelo) | 6.23 TB |
| Storage (cache) | 3.70 TB |
| Storage (logs) | 12.07 TB |
| IOPS (pico R/W) | 75,000 / 3,000 |
| Throughput (pico R/W) | 39.9 / 0.0 GB/s |
| Arquitetura HA | N+1 |

**Análise Computacional:** Equilibra eficiência e resiliência. Suporta picos de até 20% 
e tolera falha de 1 nó sem degradação do serviço. **Adequado para produção.** 
**Análise Storage:** 22.1 TB com margem operacional (1.5x). 
IOPS e throughput suportam restart de 25% dos nós + burst de logs. Tempo de recuperação aceitável.

### Cenário IDEAL

| Métrica | Valor |
|---------|-------|
| Nós DGX | 5 |
| Sessões por nó (capacidade) | 540 |
| Sessões por nó (operando) | 200 |
| KV por sessão | 1.67 GiB |
| VRAM total por nó | 1091.6 GiB (50.9% HBM) |
| **Energia (Compute + Storage)** | **73.0 kW** (72.5 + 0.5) |
| **Rack (Compute + Storage)** | **52U** (50 + 2) |
| **Storage total** | **49.86 TB** |
| Storage (modelo) | 7.79 TB |
| Storage (cache) | 5.76 TB |
| Storage (logs) | 36.21 TB |
| IOPS (pico R/W) | 100,000 / 4,000 |
| Throughput (pico R/W) | 53.2 / 0.0 GB/s |
| Arquitetura HA | N+2 |

**Análise Computacional:** Máxima resiliência com margem para múltiplas falhas e picos elevados. 
Custo maior, mas risco operacional **mínimo**. Ideal para serviços críticos. 
**Análise Storage:** 49.9 TB com margem ampla (2x). 
IOPS e throughput suportam falhas em cascata. Retenção estendida de logs (90 dias). Máxima resiliência.

---

## Comparação Executiva dos Cenários

| Critério | Mínimo | Recomendado | Ideal |
|----------|--------|-------------|-------|
| Nós DGX | 2 | 4 | 5 |
| Energia Total (kW) | 29.5 | 58.5 | 73.0 |
| Rack Total (U) | 22 | 42 | 52 |
| Storage (TB) | 17.1 | 22.1 | 49.9 |
| IOPS pico (R) | 75,000 | 75,000 | 100,000 |
| Throughput pico (R) | 39.9 GB/s | 39.9 GB/s | 53.2 GB/s |
| Tolerância a falhas | Nenhuma | 1 nó | 2 nós |
| Risco operacional | Alto | Médio | Baixo |

**Conclusão:** O cenário **RECOMENDADO** oferece o melhor equilíbrio custo-risco para operação em produção. 
Storage subdimensionado compromete resiliência e tempo de recuperação, mesmo com GPUs suficientes.

---

## Recomendação Final

Recomenda-se o **cenário RECOMENDADO** com **4 nós DGX dgx-b300**, que:

- Atende os requisitos de capacidade (1,000 sessões)
- Suporta picos de até 20%
- Tolera falha de 1 nó sem degradação (N+1)
- Consome 58.0 kW e ocupa 40U de rack
- Requer 22.1 TB de storage (profile_default)
  - IOPS pico: 75,000 leitura / 3,000 escrita
  - Throughput pico: 39.9 GB/s leitura / 0.0 GB/s escrita
- Mantém risco operacional em nível **aceitável** para produção

**Governança:** Storage é recurso crítico. Subdimensionamento impacta:
- Tempo de recuperação (restart lento)
- Escalabilidade (gargalo em scale-out)
- Auditoria e conformidade (retenção inadequada de logs)

---

*Relatório gerado automaticamente pelo sistema de sizing de infraestrutura LLM*
