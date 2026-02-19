# Prompts da Calculadora de Sizing

Este diretório contém prompts estruturados para desenvolvimento de funcionalidades adicionais da Calculadora de Sizing de Infraestrutura para Inferência.

## 📁 Prompts Disponíveis

### 1. Análise Comparativa de Modelos
**Arquivo**: `analise_comparativa_modelos.md`  
**Objetivo**: Gerar script Python que compara múltiplos relatórios de sizing e identifica o modelo mais eficiente em diferentes dimensões.

### 2. Response Time SLO
**Arquivo**: `response_time_slo.md`  
**Objetivo**: Integrar validação de tempo de resposta (latência) no sistema de sizing, permitindo definir e validar SLOs de performance.

**Principais funcionalidades**:
- ✅ Ranking de eficiência de KV cache
- ✅ Comparativo de infraestrutura (nós, VRAM, energia, rack)
- ✅ Análise de custo-benefício (TCO 3 anos)
- ✅ Breakdown de VRAM (modelo fixo vs KV cache vs overhead)
- ✅ Comparação de storage (volumetria, IOPS, throughput)
- ✅ Recomendações executivas por caso de uso
- ✅ Saída em Markdown e JSON

**Casos de uso**:
- Escolher qual modelo LLM adotar para produção
- Avaliar trade-offs entre eficiência de KV e tamanho do modelo
- Estimar TCO para diferentes arquiteturas
- Justificar decisões de infraestrutura para liderança executiva

**Exemplo de uso**:
```bash
python analise_comparativa.py --models "DeepSeek-V3.2,opt-oss-120b" --scenario recommended
```

### 2. Response Time SLO
**Arquivo**: `response_time_slo.md`  
**Objetivo**: Integrar parâmetro `--responsetime` (em millisegundos) para validar se a infraestrutura consegue atender SLOs de latência.

**Principais funcionalidades**:
- ✅ Novo parâmetro `--responsetime` (tempo de resposta alvo em ms)
- ✅ Cálculo de latência end-to-end (network + prefill + decode + queuing)
- ✅ Breakdown detalhado de componentes de latência
- ✅ Validação automática contra SLO definido (P50 e P99)
- ✅ Identificação de gargalos (network, compute, queuing)
- ✅ Recomendações acionáveis para atingir SLO
- ✅ Alertas com impacto quantitativo
- ✅ Nova seção em relatórios técnico e executivo
- ✅ Integração com dados de performance em `models.json`

**Casos de uso**:
- Validar se infraestrutura atende requisitos de latência (ex: 200ms P50)
- Identificar gargalos de performance (rede, compute, fila)
- Dimensionar infraestrutura baseada em SLO de latência
- Calcular quantos nós adicionais são necessários para atingir SLO
- Comparar modelos por tempo de resposta esperado

**Exemplo de uso**:
```bash
# Validar se consegue atender 1000 requisições com 200ms de resposta
python main.py --model DeepSeek-V3.2 --server dgx-b300 \
  --storage netapp_a_series --concurrency 1000 \
  --effective-context 131072 --kv-precision fp8 \
  --responsetime 200 --responsetime-p99 500
```

**Output esperado**:
```
⚠️  ALERTA: SLO de Response Time NÃO ATENDIDO [RECOMENDADO]

📊 MÉTRICA: Response Time P50
   • SLO definido: 200 ms
   • Esperado: 225 ms
   • Déficit: 25 ms (+12.5% acima do SLO)

🔍 BREAKDOWN DE LATÊNCIA:
   • Network Latency P50: 10 ms
   • Prefill Time: 80 ms
   • Decode Time: 120 ms
   • Queuing Delay P50: 15 ms
   • Utilização: 62.5%

🎯 GARGALO IDENTIFICADO: DECODE_COMPUTE

💡 AÇÃO RECOMENDADA:
   Considerar modelo com decode mais rápido ou ajustar SLO para 250ms.
```

---

## 🚀 Como Usar os Prompts

1. **Leia o prompt completo**: Cada arquivo `.md` contém especificações detalhadas
2. **Use como entrada para LLM**: Copie o conteúdo e forneça a um modelo de linguagem (GPT-4, Claude, etc.)
3. **Revise o código gerado**: Valide a implementação e adapte conforme necessário
4. **Teste extensivamente**: Execute os testes sugeridos no próprio prompt
5. **Integre ao projeto**: Adicione o script ao repositório e documente no README principal

---

## 🎯 Boas Práticas

### Ao Criar Novos Prompts

1. **Estrutura Clara**:
   - Objetivo (O quê?)
   - Contexto (Por quê?)
   - Requisitos funcionais (Como?)
   - Exemplos de entrada/saída
   - Casos de uso

2. **Especificações Técnicas**:
   - Linguagem e dependências
   - Arquitetura do código (módulos, funções)
   - Validações obrigatórias
   - Formato de saída (JSON schema)

3. **Testes e Validação**:
   - Casos de teste obrigatórios
   - Casos de erro esperados
   - Exemplos de execução

4. **Restrições**:
   - O que NÃO fazer
   - Limitações conhecidas
   - Trade-offs de design

### Ao Implementar a Partir de Prompts

1. ✅ **Valide o prompt**: Certifique-se de que está completo e sem ambiguidades
2. ✅ **Gere incrementalmente**: Não tente implementar tudo de uma vez
3. ✅ **Teste cada módulo**: Valide funções individuais antes de integrar
4. ✅ **Documente divergências**: Se precisar adaptar, documente o motivo
5. ✅ **Atualize o prompt**: Se encontrar melhorias, atualize o prompt original

---

## 📋 Backlog de Prompts Futuros

Ideias para próximos prompts:

### 3. Dashboard Web Interativo
- Interface web para visualizar relatórios de sizing
- Filtros dinâmicos (modelo, servidor, cenário)
- Gráficos comparativos (Chart.js)
- Exportação de relatórios personalizados

### 4. Benchmark de Latência Integrado
- Script para executar benchmarks de TTFT/TPOT
- Integração com vLLM, TensorRT-LLM, TGI
- Correlação entre sizing e performance real
- Validação de premissas da calculadora

### 5. CI/CD para Validação de Modelos
- Pipeline automatizado para testar novos modelos
- Validação de schema do `models.json`
- Sizing automático em múltiplos servidores
- Geração de relatório de compatibilidade

### 6. Estimador de Custo Cloud
- Tradução de sizing on-premise para cloud (AWS, GCP, Azure)
- Comparação de custos entre provedores
- Recomendação de instâncias (p5.48xlarge, etc.)
- TCO on-prem vs cloud

### 7. Otimizador de Configuração
- Algoritmo para encontrar melhor combinação (TP, PP, batch, context)
- Maximizar throughput ou minimizar latência
- Considerar restrições de orçamento
- Sugerir ajustes de `parameters.json`

### 8. Gerador de Relatórios Executivos Personalizados
- Templates customizáveis por organização
- Branded reports (logo, cores)
- Seções opcionais (incluir/excluir métricas)
- Exportação em PDF

### 9. API REST para Sizing
- Endpoint HTTP para sizing via API
- Autenticação e rate limiting
- Cache de resultados
- Documentação OpenAPI/Swagger

---

## 🤝 Contribuindo com Novos Prompts

Se você criar um novo prompt, siga este template:

```markdown
# PROMPT: <Nome Descritivo>

## OBJETIVO
[O que o script/feature deve fazer]

## CONTEXTO
[Por que isso é necessário]

## REQUISITOS FUNCIONAIS
[Especificações detalhadas]

## REQUISITOS TÉCNICOS
[Linguagem, dependências, arquitetura]

## ESTRUTURA DO CÓDIGO
[Módulos, funções principais]

## VALIDAÇÕES OBRIGATÓRIAS
[Testes e casos de erro]

## CASOS DE USO
[Exemplos de uso]

## RESULTADO ESPERADO
[Output esperado]

## RESTRIÇÕES
[O que NÃO fazer]
```

Depois, adicione uma entrada neste README e envie um PR.

---

## 📚 Recursos Adicionais

- **README Principal**: `/README.md`
- **Documentação de Schemas**: `/README_SCHEMAS.md`
- **Arquitetura do Sistema**: `/ARCHITECTURE.md`
- **Guia de Início Rápido**: `/QUICKSTART.md`
- **Schema de Servidores**: `/servers.schema.md`

---

**Última atualização**: 2026-02-13
