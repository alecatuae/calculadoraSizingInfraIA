# Relatório Executivo - Sizing de Infraestrutura LLM

**Modelo:** DeepSeek-V3.2  
**Servidor:** dgx-b300  
**Data:** 2026-02-12 12:41:38  

---

## Sumário Executivo

Para sustentar **1,000 sessões simultâneas** com contexto de **131,072 tokens** 
utilizando o modelo **DeepSeek-V3.2**, a infraestrutura é dimensionada por **memória GPU (KV cache)** e **storage**.

O principal limitador de capacidade é o consumo de HBM para armazenar o estado de atenção (KV cache) de cada sessão ativa. 
Storage é dimensionado para operação contínua (pesos do modelo, cache de runtime, logs e auditoria), 
garantindo resiliência, tempo de recuperação e governança operacional.

**Recomendação:** 6 nós DGX dgx-b300 
(87.0 kW, 60U rack, 36.8 TB storage) 
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
| Precisão KV cache | BF16 |

O modelo consome memória viva (KV cache) proporcional ao contexto e concorrência.

---

## Consumo Unitário do Modelo

| Recurso | Consumo por Sessão | Significado Operacional |
|---------|-------------------|------------------------|
| KV cache | 3.34 GiB | Memória ocupada enquanto sessão está ativa |
| GPU HBM | 0.2% de um nó | Fração da capacidade GPU consumida |

Cada sessão ativa 'reserva' parte do servidor. A soma das reservas define o limite físico do nó.

---

## Resultados por Cenário

### Cenário MÍNIMO

| Métrica | Valor |
|---------|-------|
| Nós DGX | 4 |
| Sessões por nó (capacidade) | 291 |
| Sessões por nó (operando) | 250 |
| KV por sessão | 3.34 GiB |
| VRAM total por nó | 1591.9 GiB (74.2% HBM) |
| **Energia (Compute + Storage)** | **58.5 kW** (58.0 + 0.5) |
| **Rack (Compute + Storage)** | **42U** (40 + 2) |
| **Storage total** | **30.53 TB** |
| Storage (modelo) | 9.35 TB |
| Storage (cache) | 3.00 TB |
| Storage (logs) | 18.10 TB |
| IOPS (pico R/W) | 75,000 / 3,000 |
| Throughput (pico R/W) | 195.4 / 0.0 GB/s |
| Arquitetura HA | NONE |

**Análise Computacional:** Opera no limite da capacidade sem margem para picos ou falhas. 
Risco operacional **alto** - qualquer indisponibilidade de hardware afeta o serviço diretamente. 
**Análise Storage:** Volumetria recomendada 30.5 TB (base: 20.4 TB) para operação steady-state. 
IOPS e throughput dimensionados sem margem. Risco de gargalo em scale-out ou restart simultâneo.

### Cenário RECOMENDADO

| Métrica | Valor |
|---------|-------|
| Nós DGX | 6 |
| Sessões por nó (capacidade) | 291 |
| Sessões por nó (operando) | 167 |
| KV por sessão | 3.34 GiB |
| VRAM total por nó | 1315.1 GiB (61.3% HBM) |
| **Energia (Compute + Storage)** | **87.5 kW** (87.0 + 0.5) |
| **Rack (Compute + Storage)** | **62U** (60 + 2) |
| **Storage total** | **36.75 TB** |
| Storage (modelo) | 14.02 TB |
| Storage (cache) | 4.50 TB |
| Storage (logs) | 18.10 TB |
| IOPS (pico R/W) | 75,000 / 3,000 |
| Throughput (pico R/W) | 156.8 / 0.0 GB/s |
| Arquitetura HA | N+1 |

**Análise Computacional:** Equilibra eficiência e resiliência. Suporta picos de até 20% 
e tolera falha de 1 nó sem degradação do serviço. **Adequado para produção.** 
**Análise Storage:** 36.8 TB recomendado (base: 24.5 TB) com margem de capacidade. 
IOPS e throughput suportam restart de 25% dos nós + burst de logs. Tempo de recuperação aceitável.

### Cenário IDEAL

| Métrica | Valor |
|---------|-------|
| Nós DGX | 8 |
| Sessões por nó (capacidade) | 270 |
| Sessões por nó (operando) | 125 |
| KV por sessão | 3.34 GiB |
| VRAM total por nó | 1175.0 GiB (54.8% HBM) |
| **Energia (Compute + Storage)** | **116.5 kW** (116.0 + 0.5) |
| **Rack (Compute + Storage)** | **82U** (80 + 2) |
| **Storage total** | **80.74 TB** |
| Storage (modelo) | 18.69 TB |
| Storage (cache) | 7.50 TB |
| Storage (logs) | 54.31 TB |
| IOPS (pico R/W) | 200,000 / 4,000 |
| Throughput (pico R/W) | 689.0 / 0.0 GB/s |
| Arquitetura HA | N+2 |

**Análise Computacional:** Máxima resiliência com margem para múltiplas falhas e picos elevados. 
Custo maior, mas risco operacional **mínimo**. Ideal para serviços críticos. 
**Análise Storage:** 80.7 TB recomendado (base: 53.8 TB) com margem ampla para máxima resiliência. 
IOPS e throughput suportam falhas em cascata. Retenção estendida de logs (90 dias). Máxima resiliência.

---

## Comparação Executiva dos Cenários

| Critério | Mínimo | Recomendado | Ideal |
|----------|--------|-------------|-------|
| Nós DGX | 4 | 6 | 8 |
| Energia Total (kW) | 58.5 | 87.5 | 116.5 |
| Rack Total (U) | 42 | 62 | 82 |
| Storage (TB) | 30.5 | 36.8 | 80.7 |
| IOPS pico (R) | 75,000 | 75,000 | 200,000 |
| Throughput pico (R) | 195.4 GB/s | 156.8 GB/s | 689.0 GB/s |
| Tolerância a falhas | Nenhuma | 1 nó | 2 nós |
| Risco operacional | Alto | Médio | Baixo |

**Conclusão:** O cenário **RECOMENDADO** oferece o melhor equilíbrio custo-risco para operação em produção. 
Storage subdimensionado compromete resiliência e tempo de recuperação, mesmo com GPUs suficientes.

---

## Recomendação Final

Recomenda-se o **cenário RECOMENDADO** com **6 nós DGX dgx-b300**, que:

- Atende os requisitos de capacidade (1,000 sessões)
- Suporta picos de até 20%
- Tolera falha de 1 nó sem degradação (N+1)
- Consome 87.0 kW e ocupa 60U de rack
- Requer 36.8 TB de storage (profile_default, incluindo margem de capacidade)
  - IOPS pico: 75,000 leitura / 3,000 escrita
  - Throughput pico: 156.8 GB/s leitura / 0.0 GB/s escrita
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

*Relatório gerado automaticamente pelo sistema de sizing de infraestrutura LLM*
