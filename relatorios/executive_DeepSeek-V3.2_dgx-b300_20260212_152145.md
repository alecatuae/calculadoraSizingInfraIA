# Relatório Executivo - Sizing de Infraestrutura LLM

**Modelo:** DeepSeek-V3.2  
**Servidor:** dgx-b300  
**Data:** 2026-02-12 15:21:45  

---

## Sumário Executivo

Para sustentar **1,000 sessões simultâneas** com contexto de **131,072 tokens** 
utilizando o modelo **DeepSeek-V3.2**, a infraestrutura é dimensionada por **memória GPU (KV cache)** e **storage**.

O principal limitador de capacidade é o consumo de HBM para armazenar o estado de atenção (KV cache) de cada sessão ativa. 
Storage é dimensionado para operação contínua (pesos do modelo, cache de runtime, logs e auditoria), 
garantindo resiliência, tempo de recuperação e governança operacional.

**Recomendação:** 4 nós DGX dgx-b300 
(58.0 kW, 40U rack, 37.0 TB storage) 
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
| **Storage total** | **27.58 TB** |
| Storage (modelo) | 4.67 TB |
| Storage (cache) | 2.78 TB |
| Storage (logs) | 18.10 TB |
| IOPS (pico R/W) | 75,000 / 3,000 |
| Throughput (pico R/W) | 353.0 / 0.0 GB/s |
| Arquitetura HA | NONE |

**Análise Computacional:** Opera no limite da capacidade sem margem para picos ou falhas. 
Risco operacional **alto** - qualquer indisponibilidade de hardware afeta o serviço diretamente. 
**Análise Storage:** Volumetria recomendada 27.6 TB (base: 18.4 TB) para operação steady-state. 
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
| **Storage total** | **37.05 TB** |
| Storage (modelo) | 9.35 TB |
| Storage (cache) | 5.55 TB |
| Storage (logs) | 18.10 TB |
| IOPS (pico R/W) | 75,000 / 3,000 |
| Throughput (pico R/W) | 237.1 / 0.0 GB/s |
| Arquitetura HA | N+1 |

**Análise Computacional:** Equilibra eficiência e resiliência. Suporta picos de até 20% 
e tolera falha de 1 nó sem degradação do serviço. **Adequado para produção.** 
**Análise Storage:** 37.0 TB recomendado (base: 24.7 TB) com margem de capacidade. 
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
| **Storage total** | **79.73 TB** |
| Storage (modelo) | 11.68 TB |
| Storage (cache) | 8.64 TB |
| Storage (logs) | 54.31 TB |
| IOPS (pico R/W) | 100,000 / 4,000 |
| Throughput (pico R/W) | 544.3 / 0.0 GB/s |
| Arquitetura HA | N+2 |

**Análise Computacional:** Máxima resiliência com margem para múltiplas falhas e picos elevados. 
Custo maior, mas risco operacional **mínimo**. Ideal para serviços críticos. 
**Análise Storage:** 79.7 TB recomendado (base: 53.2 TB) com margem ampla para máxima resiliência. 
IOPS e throughput suportam falhas em cascata. Retenção estendida de logs (90 dias). Máxima resiliência.

---

## Comparação Executiva dos Cenários

| Critério | Mínimo | Recomendado | Ideal |
|----------|--------|-------------|-------|
| Nós DGX | 2 | 4 | 5 |
| Energia Total (kW) | 29.5 | 58.5 | 73.0 |
| Rack Total (U) | 22 | 42 | 52 |
| Storage (TB) | 27.6 | 37.0 | 79.7 |
| IOPS pico (R) | 75,000 | 75,000 | 100,000 |
| Throughput pico (R) | 353.0 GB/s | 237.1 GB/s | 544.3 GB/s |
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
- Requer 37.0 TB de storage (profile_default, incluindo margem de capacidade)
  - IOPS pico: 75,000 leitura / 3,000 escrita
  - Throughput pico: 237.1 GB/s leitura / 0.0 GB/s escrita
- Mantém risco operacional em nível **aceitável** para produção

**Governança:** Storage é recurso crítico. Subdimensionamento impacta:
- Tempo de recuperação (restart lento)
- Escalabilidade (gargalo em scale-out)
- Auditoria e conformidade (retenção inadequada de logs)

---

## Glossário Executivo de Termos

| Métrica | O que significa | Por que importa para a decisão | Impacto se estiver errado |
| ------------------------------- | --------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| **Nós DGX** | Quantidade de servidores de IA necessários para atender a carga analisada. | Define investimento em hardware e influencia energia, rack e custo total. | Subdimensionamento causa indisponibilidade; superdimensionamento aumenta custo. |
| **Sessões por nó (capacidade)** | Número máximo teórico de conversas simultâneas que um servidor suporta. | Indica o limite físico do servidor antes de atingir saturação de memória. | Operar no limite reduz margem para picos e aumenta risco de instabilidade. |
| **Sessões por nó (operando)** | Número real de sessões em uso no cenário avaliado. | Mostra a folga operacional disponível. | Se muito próximo do limite, o sistema fica vulnerável a picos de uso. |
| **KV por sessão** | Memória de GPU consumida por cada conversa ativa. | É o principal fator que determina quantas sessões cabem por servidor. | Conversas mais longas aumentam consumo e reduzem capacidade total. |
| **VRAM total por nó** | Memória total da GPU utilizada pelo modelo, runtime e sessões. | Indica quão próximo o servidor está do limite físico. | Uso excessivo pode causar falhas ou degradação de performance. |
| **Energia (Compute + Storage)** | Consumo total de energia dos servidores de IA e do storage. | Impacta custo operacional mensal e capacidade elétrica do datacenter. | Subdimensionar pode causar sobrecarga elétrica; superdimensionar eleva custo. |
| **Rack (Compute + Storage)** | Espaço físico ocupado por servidores e storage no datacenter. | Define viabilidade física de implantação e expansão futura. | Espaço insuficiente limita crescimento. |
| **Storage total** | Capacidade total de armazenamento necessária para rodar o modelo e sustentar o sistema (modelo + cache + logs). | Representa o espaço mínimo necessário para operar o ambiente com segurança. | Falta de espaço pode impedir inicialização, gravação de logs ou escala do sistema. Recomenda-se dimensionar ~50% acima do mínimo calculado. |
| **Storage (modelo)** | Espaço necessário para armazenar os arquivos do modelo (pesos e artefatos). | Essencial para subir o sistema e permitir reinicializações rápidas. | Se insuficiente, o sistema pode não iniciar corretamente. Recomenda-se margem adicional. |
| **Storage (cache)** | Espaço para arquivos temporários e dados intermediários usados na execução. | Garante funcionamento contínuo e estável do ambiente. | Pode gerar falhas ou degradação se o espaço se esgotar. |
| **Storage (logs)** | Espaço destinado ao armazenamento de logs operacionais e auditoria. | Fundamental para rastreabilidade, análise de incidentes e governança. | Falta de espaço compromete auditoria e diagnóstico de problemas. |
| **IOPS (pico R/W)** | Número máximo de operações de leitura e escrita por segundo no pico. | Determina se o storage suporta eventos como subida simultânea de múltiplos servidores. | Gargalo de IOPS aumenta tempo de recuperação e escala. |
| **Throughput (pico R/W)** | Volume máximo de dados transferidos por segundo no pico de uso. | Afeta tempo de carregamento do modelo e recuperação após falhas. | Throughput insuficiente aumenta tempo de indisponibilidade. |
| **Arquitetura HA** | Nível de tolerância a falhas adotado (ex.: NONE, N+1, N+2). | Define o quanto o sistema continua operando mesmo após falhas de hardware. | Ausência de HA pode causar interrupção total do serviço. |

---

*Relatório gerado automaticamente pelo Calculadora de Sizing de Infraestrutura para Inferência, desenvolvido pelo time de InfraCore de CLOUD.*
