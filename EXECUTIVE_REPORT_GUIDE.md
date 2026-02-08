# GUIA DO RELATÓRIO EXECUTIVO

## 📊 Visão Geral

O **Relatório Executivo** é uma versão especializada do relatório de sizing, projetado especificamente para **diretoria, comitê executivo e líderes de tecnologia**. Ele transforma dados técnicos em informações estratégicas orientadas à decisão.

## 🎯 Diferenças Entre os Formatos

| Aspecto | Relatório Técnico | Relatório Executivo |
|---------|------------------|---------------------|
| **Público-alvo** | Engenheiros, arquitetos, SREs | Diretoria, C-level, VP de Tecnologia |
| **Foco** | Detalhes técnicos, fórmulas | Capacidade, risco, custo, decisão |
| **Linguagem** | Técnica, detalhada | Executiva, estratégica |
| **Estrutura** | Dados → Análise → Resultados | Sumário → Cenários → Recomendação |
| **Formato** | Texto + JSON | Markdown formatado para apresentação |
| **Racional** | Texto corrido | Tabelas estruturadas |

## 🚀 Como Gerar

### Comando Básico

```bash
python3 sizing.py \
  --model opt-oss-120b \
  --server dgx300 \
  --storage profile_default \
  --concurrency 1000 \
  --effective-context 131072 \
  --executive-report \
  --output-markdown-file executive_report.md
```

### Opções de Saída

```bash
# Apenas exibir no terminal (sem salvar)
python3 sizing.py ... --executive-report

# Salvar em arquivo específico
python3 sizing.py ... --executive-report --output-markdown-file relatório_diretoria.md

# Gerar relatório executivo + JSON de dados
python3 sizing.py ... --executive-report --output-json-file dados.json
```

## 📋 Estrutura do Relatório Executivo

### 1. Sumário Executivo
- **Objetivo:** Contextualizar o problema em 1 página
- **Conteúdo:** 
  - Qual problema de capacidade está sendo analisado
  - Modelo, carga e premissas
  - Principal fator limitante (memória)
  - Conclusão diretiva clara

**Exemplo:**
> "Para sustentar 1.000 sessões simultâneas com contexto de 128k tokens,
> a infraestrutura passa a ser limitada por memória de GPU,
> exigindo 3 nós DGX para garantir estabilidade e continuidade operacional."

### 2. Cenários Avaliados

**Tabela de Visão Geral:**

| Cenário | Objetivo | Característica | Risco |
|---------|----------|----------------|-------|
| Mínimo | Atender no limite | Sem tolerância a falhas | Alto |
| Recomendado | Produção estável | N+1, 20% headroom | Médio |
| Ideal | Operação resiliente | N+2, 30%+ headroom | Baixo |

**Explicação:** Por que avaliar múltiplos cenários é essencial para decisões de investimento.

### 3. Resultado Consolidado por Cenário

Para cada cenário (Mínimo, Recomendado, Ideal):

**Tabela de Métricas:**
- Modelo avaliado
- Servidor base
- Contexto efetivo
- Concorrência alvo
- KV cache por sessão
- KV total necessário
- Budget HBM por nó
- Sessões por nó
- **Nós DGX necessários** (destaque)
- Arquitetura de HA

**Parágrafo Executivo:**
- O que significa na prática
- Como se comporta em falhas e picos
- Adequação para produção

### 4. Racional de Cálculo

**Formato Obrigatório: TABELA**

| Resultado | Fórmula | Parâmetros | Suposição | Significado Operacional |
|-----------|---------|------------|-----------|------------------------|

**Colunas:**
- **Fórmula Utilizada:** Como foi calculado
- **Parâmetros do Cálculo:** Valores de entrada
- **Suposição Aplicada:** Premissas e políticas
- **Significado Operacional:** Impacto real na operação

### 5. Análise Comparativa

**Tabela Comparativa:**

| Critério | Mínimo | Recomendado | Ideal |
|----------|--------|-------------|-------|
| Número de nós | | | |
| Tolerância a falhas | | | |
| Capacidade para picos | | | |
| Risco de indisponibilidade | | | |
| Complexidade operacional | | | |
| CapEx relativo | Baseline | +X% | +Y% |

**Parágrafo Conclusivo:**
- Qual cenário equilibra melhor custo e risco
- Quando usar Mínimo ou Ideal

### 6. Principais Riscos e Alertas

**Formato: Bullets executivos**

**Riscos de Operação no Limite:**
- Indisponibilidade imediata em falhas
- Degradação em picos
- Impossibilidade de manutenção planejada

**Impactos de Decisões Técnicas:**
- Precisão KV (FP16 vs FP8)
- Contexto máximo liberado
- Budget de HBM agressivo

**Consequências Operacionais:**
- Filas de espera
- Degradação de SLA
- Indisponibilidade parcial

### 7. Recomendação Final

**Formato: Decisão clara e acionável**

**Conteúdo:**
- Qual cenário adotar
- Por quê (justificativa)
- Premissas sob governança
- Próximos passos

**Exemplo:**
> "Recomenda-se a adoção do cenário RECOMENDADO (3 nós DGX),
> por equilibrar eficiência de capital, estabilidade operacional
> e tolerância a falhas, sem comprometer a experiência do usuário."

### 8. Dicionário de Parâmetros

**Formato: Tabela**

| Parâmetro | Origem | Descrição | Importância |
|-----------|--------|-----------|-------------|

**Parâmetros cobertos:**
- Arquitetura do Modelo (fixos)
- NFR do Produto (requisitos)
- Runtime/Configuração (ajustáveis)
- Tuning de Infraestrutura (políticas)

## 🎨 Princípios de Design

### Linguagem Executiva

✅ **BOM:**
> "Com 3 nós DGX, o sistema tolera a falha de 1 nó sem perda de capacidade crítica."

❌ **RUIM:**
> "Aplicando ceil((1000 × 1.2) / 629) + 1 na fórmula de dimensionamento..."

### Foco em Impacto

✅ **BOM:**
> "Uso de FP16 dobra custos de infraestrutura."

❌ **RUIM:**
> "FP16 consome 2 bytes por elemento vs 1 byte do FP8."

### Orientação à Decisão

✅ **BOM:**
> "Recomenda-se cenário RECOMENDADO: 3 nós, N+1, SLA 99.9%."

❌ **RUIM:**
> "Há múltiplas opções possíveis, cada uma com trade-offs..."

### Tabelas, Não Texto

✅ **BOM:**
```
| Cenário | Nós | CapEx Relativo |
|---------|-----|----------------|
| Mínimo  | 2   | Baseline       |
| Recomendado | 3 | +50%        |
```

❌ **RUIM:**
> "O cenário mínimo usa 2 nós, o recomendado 3 nós (+50% de capex)..."

## 📊 Casos de Uso

### 1. Apresentação para Comitê de Investimento

**Objetivo:** Aprovar budget para nova infra de IA

**Como usar:**
1. Gere relatório executivo
2. Extraia Seção 2 (Cenários) + Seção 5 (Comparativa)
3. Adicione slides com custos estimados por nó
4. Apresente Seção 7 (Recomendação) como proposta

**Foco:** CapEx relativo, risco operacional, SLA

### 2. Planejamento de Capacidade Anual

**Objetivo:** Dimensionar crescimento de infraestrutura

**Como usar:**
1. Gere múltiplos relatórios com projeções de concorrência
   - Q1: 1.000 sessões
   - Q2: 2.000 sessões
   - Q3: 5.000 sessões
   - Q4: 10.000 sessões
2. Compare "Nós DGX necessários" por cenário
3. Planeje aquisições escalonadas

**Foco:** Escalabilidade, janelas de procurement

### 3. Avaliação de Fornecedores GPU

**Objetivo:** Comparar DGX B300 vs H200 vs cloud

**Como usar:**
1. Gere relatório executivo para cada servidor:
   - `--server dgx300`
   - `--server dgx200`
2. Compare Seção 3 (Resultados por Cenário)
3. Calcule TCO: nós × custo_unitário × 3_anos

**Foco:** Eficiência de HBM, sessões/nó, TCO

### 4. Resposta a Incidentes de Capacidade

**Objetivo:** Explicar para diretoria por que sistema atingiu limite

**Como usar:**
1. Gere relatório com parâmetros atuais
2. Mostre Seção 6 (Riscos) se estiver em cenário Mínimo
3. Apresente Seção 7 (Recomendação) como plano de remediação

**Foco:** Risco atual, necessidade de investimento urgente

### 5. Governança de Recursos

**Objetivo:** Estabelecer políticas de uso (contexto, concorrência)

**Como usar:**
1. Gere relatório com diferentes valores de contexto:
   - 32k, 64k, 128k, 200k tokens
2. Extraia "Nós DGX necessários" para cada
3. Mostre Seção 7 → Premissas sob Governança

**Foco:** Custo por contexto, limites operacionais

## 🔍 Dicas de Uso

### Para Diretoria (C-level)

- **Leia apenas:** Seções 1, 2, 5, 7
- **Foco:** Sumário Executivo, Comparativa, Recomendação
- **Tempo:** 5-10 minutos

### Para VP/Diretor de Tecnologia

- **Leia:** Todas as seções
- **Foco especial:** Seções 4 (Racional), 6 (Riscos), 8 (Parâmetros)
- **Tempo:** 20-30 minutos

### Para Gerentes de Infraestrutura

- **Leia:** Seções 3, 4, 6, 8
- **Combine com:** Relatório técnico detalhado + JSON
- **Tempo:** 30-45 minutos

### Para Arquitetos

- **Gere ambos:** Relatório executivo + técnico
- **Use executivo para:** Discussões com liderança
- **Use técnico para:** Implementação, validações

## 📝 Checklist de Qualidade

Antes de apresentar o relatório executivo, verifique:

- [ ] **Sumário Executivo** tem conclusão clara e diretiva
- [ ] **Cenários** apresentados logo no início (Seção 2)
- [ ] **Tabelas** usadas para todos os dados numéricos
- [ ] **Racional** está em formato de tabela (não texto corrido)
- [ ] **Linguagem** é executiva, não acadêmica
- [ ] **Recomendação** é específica e acionável
- [ ] **Riscos** focam em impacto operacional, não detalhes técnicos
- [ ] **CapEx relativo** está presente na análise comparativa

## 🚨 Erros Comuns a Evitar

### ❌ Erro 1: Detalhamento Excessivo

**Problema:** Explicar fórmulas matemáticas passo a passo

**Solução:** Use Seção 4 (Racional em tabela) e mantenha objetivo

### ❌ Erro 2: Falta de Direcionamento

**Problema:** "Existem 3 opções possíveis, cabe à diretoria decidir"

**Solução:** Sempre recomende um cenário específico com justificativa

### ❌ Erro 3: Linguagem Técnica

**Problema:** "O KV cache cresce linearmente com num_layers e seq_length"

**Solução:** "Contextos longos consomem mais memória, aumentando custos"

### ❌ Erro 4: Dados em Texto Corrido

**Problema:** "O cenário mínimo usa 2 nós, o recomendado 3, e o ideal 5..."

**Solução:** Use tabela comparativa (Seção 5)

### ❌ Erro 5: Falta de Contexto de Custo

**Problema:** "Você precisa de 3 nós"

**Solução:** "Você precisa de 3 nós (+50% vs mínimo, mas com N+1 e 20% headroom)"

## 📚 Documentação Relacionada

- **README_v2.md:** Documentação técnica completa
- **QUICKREF.md:** Referência rápida de comandos
- **SCENARIO_GUIDE.md:** Guia de escolha de cenários
- **USE_CASES.md:** Exemplos técnicos de uso

## 🆘 Suporte

Para dúvidas ou sugestões sobre o relatório executivo:

1. Leia esta documentação completa
2. Consulte exemplos em `executive_report.md`
3. Gere um relatório de teste e valide com sua equipe

---

**Versão:** 2.0  
**Data:** 2026-02-08  
**Público:** Diretoria, C-level, VP de Tecnologia
